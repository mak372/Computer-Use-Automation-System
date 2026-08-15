"""Loop controller: the observe -> decide -> act orchestrator (decision #6),
tying together perception, guardrails, action_executor, llm_client,
checkpoints, operator, and logger.

This is the discovery-time driver - an LLM proposes each action. The
replay engine (not yet built) will reuse perception/guardrails/
action_executor/logger/checkpoints identically, swapping this module out
for deterministic, pre-recorded action selection.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Optional

from playwright.sync_api import Page

from agent import checkpoints, config, operator
from agent import llm_client as llm_module
from agent.action_executor import execute_action
from agent.guardrails import evaluate_action
from agent.logger import RunLogger, redact
from agent.perception import Observation, build_observation, capture_screenshot


@dataclass
class RunResult:
    outcome_type: str  # "success" | "business_outcome" | "escalated" | "failed"
    outcome_label: Optional[str]
    outputs: Optional[dict]
    total_steps: int
    run_id: str


def run_discovery(
    goal: str,
    goal_key: str,
    page: Page,
    client: llm_module.LLMClient,
    logger: RunLogger,
) -> RunResult:
    logger.log_run_start(
        goal=goal, target_url=page.url, model=config.MODEL_NAME, model_version=None
    )

    system_instruction = llm_module.build_system_instruction(goal)
    history_lines: list[str] = []
    previous_fingerprint: Optional[str] = None
    no_progress_count = 0
    run_start = time.monotonic()
    step_number = 0

    while True:
        step_number += 1

        if step_number > config.MAX_STEPS:
            return _escalate_and_end(
                logger, "max_steps_exhausted", {"max_steps": config.MAX_STEPS}, step_number - 1
            )
        if time.monotonic() - run_start > config.OVERALL_TIMEOUT_SECONDS:
            return _escalate_and_end(
                logger,
                "overall_timeout",
                {"timeout_seconds": config.OVERALL_TIMEOUT_SECONDS},
                step_number - 1,
            )

        try:
            observation = build_observation(page)
        except Exception as exc:
            # session/browser-level breakage - no live session left to escalate into
            return _resolve_forced_fail(logger, f"perception failed: {exc}", step_number - 1)

        pre_reasons = []
        if step_number == 1:
            pre_reasons.append("first_step")
        if observation.is_empty:
            pre_reasons.append("empty_interactive_list")
        if observation.has_duplicates:
            pre_reasons.append("duplicate_role_name")
        if step_number % config.SCREENSHOT_PERIODIC_INTERVAL == 0:
            pre_reasons.append("periodic")

        screenshot_taken = bool(pre_reasons)
        screenshot_reason = _maybe_screenshot(logger, page, step_number, pre_reasons)

        observation_text = llm_module.render_observation(observation)
        history_text = llm_module.render_history(history_lines)

        try:
            decision = client.decide(
                system_instruction,
                history_text,
                observation_text,
                capture_screenshot(page) if screenshot_taken else None,
            )
        except llm_module.LLMCallFailed as exc:
            for fa in exc.failed_attempts:
                logger.log_llm_call_attempt_failed(
                    step_number=step_number,
                    attempt=fa["attempt"],
                    result=fa["result"],
                    duration_ms=fa["duration_ms"],
                )
            return _resolve_forced_fail(logger, "LLM call failed after retries", step_number)

        for fa in decision.failed_attempts:
            logger.log_llm_call_attempt_failed(
                step_number=step_number,
                attempt=fa["attempt"],
                result=fa["result"],
                duration_ms=fa["duration_ms"],
            )

        action_log = {
            "type": decision.action,
            "reasoning": redact(decision.args.get("reasoning", "")),
            "model_version": decision.model_version,
        }

        # --- finish ---
        if decision.action == "finish":
            if not screenshot_taken:
                screenshot_reason = _maybe_screenshot(logger, page, step_number, ["pre_finish"])
                screenshot_taken = True

            outcome = decision.args.get("outcome", "")
            output_refs = decision.args.get("output_refs", []) or []
            action_log["outcome"] = outcome
            action_log["output_refs"] = output_refs

            result = checkpoints.verify_finish(
                goal_key, outcome, output_refs, observation, page.url
            )

            if not result.verified:
                action_log["verification_reason"] = result.reason
                _log_step(
                    logger,
                    step_number,
                    observation,
                    action_log,
                    guardrail_result={"decision": "not_applicable"},
                    execution_result="checkpoint_verification_failed",
                    fingerprint=observation.fingerprint,
                    screenshot_reason=screenshot_reason,
                )
                history_lines.append(
                    f"Step {step_number}: called finish(outcome={outcome!r}) -> rejected: {result.reason}"
                )
                no_progress_count, previous_fingerprint = _track_fingerprint(
                    observation.fingerprint, previous_fingerprint, no_progress_count
                )
                if no_progress_count >= config.NO_PROGRESS_THRESHOLD:
                    return _escalate_and_end(
                        logger,
                        "no_progress",
                        {"consecutive_unchanged_steps": no_progress_count},
                        step_number,
                    )
                continue

            _log_step(
                logger,
                step_number,
                observation,
                action_log,
                guardrail_result={"decision": "not_applicable"},
                execution_result="success",
                fingerprint=observation.fingerprint,
                screenshot_reason=screenshot_reason,
            )
            outcome_type = "success" if outcome == "success" else "business_outcome"
            logger.log_run_end(
                outcome_type=outcome_type,
                outputs=result.outputs,
                checkpoint_verification={"verified": True, "outcome": outcome},
                total_steps=step_number,
            )
            return RunResult(outcome_type, outcome, result.outputs, step_number, logger.run_id)

        # --- escalate (LLM-initiated) ---
        if decision.action == "escalate":
            _log_step(
                logger,
                step_number,
                observation,
                action_log,
                guardrail_result={"decision": "not_applicable"},
                execution_result="escalated",
                fingerprint=observation.fingerprint,
                screenshot_reason=screenshot_reason,
            )
            reasoning = redact(decision.args.get("reasoning", ""))
            return _escalate_and_end(
                logger, "llm_initiated", {"reasoning": reasoning}, step_number
            )

        # --- click / type / select ---
        index = decision.args.get("index")
        value = decision.args.get("value")
        action_log["index"] = index
        action_log["value"] = redact(value) if isinstance(value, str) else value

        element = None
        if isinstance(index, int) and 1 <= index <= len(observation.interactive_elements):
            element = observation.interactive_elements[index - 1]

        if element is None:
            _log_step(
                logger,
                step_number,
                observation,
                action_log,
                guardrail_result={"decision": "not_applicable"},
                execution_result="stale_or_invalid_index",
                fingerprint=observation.fingerprint,
                screenshot_reason=screenshot_reason,
            )
            history_lines.append(
                f"Step {step_number}: {decision.action}(index={index}) -> invalid index, ignored"
            )
            no_progress_count, previous_fingerprint = _track_fingerprint(
                observation.fingerprint, previous_fingerprint, no_progress_count
            )
            if no_progress_count >= config.NO_PROGRESS_THRESHOLD:
                return _escalate_and_end(
                    logger,
                    "no_progress",
                    {"consecutive_unchanged_steps": no_progress_count},
                    step_number,
                )
            continue

        action_log["target"] = {"role": element.role, "name": element.name}

        guardrail = evaluate_action(element, page)
        guardrail_log = {
            "decision": guardrail.decision,
            "reason": guardrail.reason,
            "amount": guardrail.amount,
        }

        if guardrail.decision == "blocked":
            _log_step(
                logger,
                step_number,
                observation,
                action_log,
                guardrail_result=guardrail_log,
                execution_result="guardrail_blocked",
                fingerprint=observation.fingerprint,
                screenshot_reason=screenshot_reason,
            )
            history_lines.append(
                f'Step {step_number}: {decision.action} "{element.name}" -> blocked by policy: {guardrail.reason}'
            )
            no_progress_count, previous_fingerprint = _track_fingerprint(
                observation.fingerprint, previous_fingerprint, no_progress_count
            )
            if no_progress_count >= config.NO_PROGRESS_THRESHOLD:
                return _escalate_and_end(
                    logger,
                    "no_progress",
                    {"consecutive_unchanged_steps": no_progress_count},
                    step_number,
                )
            continue

        hitl_event_id = None
        if guardrail.decision == "hitl_required":
            hitl_event_id = secrets.token_hex(4)
            logger.log_hitl_event(
                hitl_event_id=hitl_event_id,
                phase="raised",
                trigger="guardrail_threshold",
                context={
                    "action": decision.action,
                    "element": element.name,
                    "amount": guardrail.amount,
                },
            )
            _log_step(
                logger,
                step_number,
                observation,
                action_log,
                guardrail_result=guardrail_log,
                execution_result="pending_hitl",
                fingerprint=observation.fingerprint,
                screenshot_reason=screenshot_reason,
                hitl_event_id=hitl_event_id,
            )
            approved = operator.request_approval(
                f'{decision.action} "{element.name}"', guardrail.amount
            )
            logger.log_hitl_event(
                hitl_event_id=hitl_event_id,
                phase="resolved",
                decision="approved" if approved else "denied",
                operator_id="local-operator",
            )
            if not approved:
                history_lines.append(
                    f'Step {step_number}: {decision.action} "{element.name}" -> denied by human approver'
                )
                no_progress_count, previous_fingerprint = _track_fingerprint(
                    observation.fingerprint, previous_fingerprint, no_progress_count
                )
                if no_progress_count >= config.NO_PROGRESS_THRESHOLD:
                    return _escalate_and_end(
                        logger,
                        "no_progress",
                        {"consecutive_unchanged_steps": no_progress_count},
                        step_number,
                    )
                continue
            # Resumed execution after approval is its own step event.
            step_number += 1

        exec_result = execute_action(element, decision.action, value)

        # Re-observe after execution so the logged fingerprint is the same
        # one the no-progress algorithm actually compares below - logging
        # the pre-action fingerprint here would silently diverge from what
        # drove the real decision for any action that changed page state.
        try:
            post_observation = build_observation(page)
        except Exception as exc:
            return _resolve_forced_fail(logger, f"perception failed after action: {exc}", step_number)
        new_fingerprint = post_observation.fingerprint

        post_reasons = [] if exec_result.success else ["on_error"]
        post_screenshot_reason = None
        if post_reasons and not (screenshot_taken and hitl_event_id is None):
            post_screenshot_reason = _maybe_screenshot(logger, page, step_number, post_reasons)

        _log_step(
            logger,
            step_number,
            observation,
            action_log,
            guardrail_result=guardrail_log,
            execution_result=("success" if exec_result.success else exec_result.error_type),
            fingerprint=new_fingerprint,
            screenshot_reason=(post_screenshot_reason or (screenshot_reason if hitl_event_id is None else None)),
            resumed_from_hitl_event_id=hitl_event_id,
        )

        if exec_result.success:
            history_lines.append(
                f'Step {step_number}: {decision.action} "{element.name}"'
                + (f" = {value!r}" if value is not None else "")
                + " -> ok"
            )
        else:
            history_lines.append(
                f'Step {step_number}: {decision.action} "{element.name}" -> '
                f"{exec_result.error_type}: {exec_result.message}"
            )

        no_progress_count, previous_fingerprint = _track_fingerprint(
            new_fingerprint, previous_fingerprint, no_progress_count
        )
        if no_progress_count >= config.NO_PROGRESS_THRESHOLD:
            return _escalate_and_end(
                logger,
                "no_progress",
                {"consecutive_unchanged_steps": no_progress_count},
                step_number,
            )


def _maybe_screenshot(logger: RunLogger, page: Page, step_number: int, reasons: list[str]) -> Optional[str]:
    if not reasons:
        return None
    path = logger.screenshot_path(step_number)
    path.write_bytes(capture_screenshot(page))
    return ",".join(reasons)


def _track_fingerprint(
    new_fp: str, previous_fp: Optional[str], count: int
) -> tuple[int, str]:
    if new_fp == previous_fp:
        return count + 1, new_fp
    return 0, new_fp


def _log_step(
    logger: RunLogger,
    step_number: int,
    observation: Observation,
    action_log: dict,
    guardrail_result: dict,
    execution_result: str,
    fingerprint: str,
    screenshot_reason: Optional[str] = None,
    hitl_event_id: Optional[str] = None,
    resumed_from_hitl_event_id: Optional[str] = None,
) -> None:
    observation_summary = {
        "url": observation.url,
        "interactive_elements": [
            {"index": el.index, "role": el.role, "name": el.name}
            for el in observation.interactive_elements
        ],
        "static_text": [redact(t) for t in observation.static_text],
        "screenshot_path": (f"step_{step_number}.png" if screenshot_reason else None),
        "screenshot_reason": screenshot_reason,
    }
    logger.log_step(
        step_number=step_number,
        observation_summary=observation_summary,
        action=action_log,
        guardrail_result=guardrail_result,
        execution_result=execution_result,
        state_fingerprint=fingerprint,
        hitl_event_id=hitl_event_id,
        resumed_from_hitl_event_id=resumed_from_hitl_event_id,
    )


def _escalate_and_end(
    logger: RunLogger, trigger: str, context: dict, total_steps: int
) -> RunResult:
    hitl_event_id = secrets.token_hex(4)
    logger.log_hitl_event(
        hitl_event_id=hitl_event_id, phase="raised", trigger=trigger, context=context
    )
    resumed = operator.request_manual_handoff(trigger, context)
    logger.log_hitl_event(
        hitl_event_id=hitl_event_id,
        phase="resolved",
        decision="resolved_manually" if resumed else "aborted",
        operator_id="local-operator",
    )
    logger.log_run_end(
        outcome_type="escalated", outputs=None, checkpoint_verification=None, total_steps=total_steps
    )
    return RunResult("escalated", trigger, None, total_steps, logger.run_id)


def _resolve_forced_fail(logger: RunLogger, reason: str, total_steps: int) -> RunResult:
    logger.log_run_end(
        outcome_type="failed", outputs=None, checkpoint_verification=None, total_steps=total_steps
    )
    return RunResult("failed", reason, None, total_steps, logger.run_id)
