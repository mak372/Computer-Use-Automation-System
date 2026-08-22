"""Loop controller"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Optional

from playwright.sync_api import Page

from agent import checkpoints, config, dialogs, operator
from agent import llm_client as llm_module
from agent.action_executor import execute_action
from agent.guardrails import evaluate_action
from agent.logger import RunLogger
from agent.perception import Observation, build_observation, capture_screenshot, diff_observations


@dataclass
class RunResult:
    outcome_type: str  # "success" | "business_outcome" | "escalated" | "failed"
    outcome_label: Optional[str]
    outputs: Optional[dict]
    total_steps: int
    run_id: str
    # None only for the rare case classify_goal_key itself never resolved a
    # capability (see run_discovery's opening block) - always set once a
    # goal_key exists, including on every ordinary "failed"/"escalated" run.
    goal_key: Optional[str]


def run_discovery(
    goal: str,
    page: Page,
    client: llm_module.LLMClient,
    logger: RunLogger,
) -> RunResult:
    # Capability selection: the LLM's own one-shot choice of which
    # checkpoints.REGISTRIES entry this goal maps to, made once before any
    # page is observed - replaces a human-supplied --goal-key CLI argument
    # (see llm_client.LLMClient.classify_goal_key). Not a step-loop action:
    # doesn't consume step_number/MAX_STEPS budget, and isn't logged as a
    # "step" event (log_capability_selection is its own event type, so
    # artifact_builder.py's step-scanning loop never sees it).
    # classification_failure_reason stays None on the normal path (a valid
    # goal_key came back from the LLM); either abnormal path below (the call
    # itself failing after retries, or the model returning something the
    # tool schema's enum should have prevented but isn't a guarantee of) is
    # routed through the same human backup rather than two different dead
    # ends, since both are really the same situation: automated
    # classification didn't produce a trustworthy goal_key.
    classification_failure_reason: Optional[str] = None
    goal_key: Optional[str] = None
    capability_reasoning = ""
    model_version: Optional[str] = None

    try:
        classification = client.classify_goal_key(goal)
        for fa in classification.failed_attempts:
            logger.log_llm_call_attempt_failed(
                step_number=0, attempt=fa["attempt"], result=fa["result"], duration_ms=fa["duration_ms"]
            )
        goal_key = classification.args.get("goal_key")
        capability_reasoning = classification.args.get("reasoning", "")
        model_version = classification.model_version
        # The tool schema enum-constrains this, but a schema constraint
        # isn't a proof - this is basic malformed-output handling, not an
        # attempt to detect a *plausible but wrong* capability choice
        # (that's deliberately left uncaught: checkpoints.verify_finish
        # will essentially never confirm success against the wrong
        # registry, the same downstream-catches-a-bad-self-report pattern
        # every other claim in this system goes through).
        if goal_key == llm_module.UNSUPPORTED_GOAL_KEY:
            classification_failure_reason = (
                f"LLM reports no supported capability matches this goal: {capability_reasoning}"
            )
        elif goal_key not in checkpoints.REGISTRIES:
            classification_failure_reason = f"select_capability returned unknown goal_key {goal_key!r}"
    except llm_module.LLMCallFailed as exc:
        for fa in exc.failed_attempts:
            logger.log_llm_call_attempt_failed(
                step_number=0, attempt=fa["attempt"], result=fa["result"], duration_ms=fa["duration_ms"]
            )
        classification_failure_reason = "capability classification failed after retries (LLM unreachable)"

    if classification_failure_reason is not None:
        goal_key = _handle_capability_classification_failure(logger, page, goal, classification_failure_reason)
        if goal_key is None:
            logger.log_run_end(
                outcome_type="escalated", outputs=None, checkpoint_verification=None, total_steps=0
            )
            return RunResult("escalated", "capability_classification_failed", None, 0, logger.run_id, None)
        capability_reasoning = f"human-selected after classification failure: {classification_failure_reason}"
        model_version = None

    logger.log_capability_selection(
        goal=goal, goal_key=goal_key, reasoning=capability_reasoning, model_version=model_version
    )

    dialogs.install_dialog_handler(page, goal_key, logger)

    logger.log_run_start(
        goal=goal,
        goal_key=goal_key,
        target_url=page.url,
        model=config.MODEL_NAME,
        model_version=None,
    )

    system_instruction = llm_module.build_system_instruction(goal)
    history_lines: list[str] = []
    previous_fingerprint: Optional[str] = None
    no_progress_count = 0
    run_start = time.monotonic()
    step_number = 0
    # Both budgets can be extended, once per escalation, if a human resumes
    # rather than aborts - otherwise "resume" would just immediately
    # re-trigger the same budget-exhaustion escalation on the very next
    # iteration, making resume meaningless for these two triggers.
    max_steps_limit = config.MAX_STEPS
    deadline = run_start + config.OVERALL_TIMEOUT_SECONDS

    while True:
        step_number += 1

        if step_number > max_steps_limit:
            result = _handle_escalation(
                logger, page, goal_key, "max_steps_exhausted",
                {"max_steps": max_steps_limit}, step_number - 1,
            )
            if result is not None:
                return result
            max_steps_limit += config.MAX_STEPS
            no_progress_count, previous_fingerprint = 0, None
            continue
        if time.monotonic() > deadline:
            result = _handle_escalation(
                logger, page, goal_key, "overall_timeout",
                {"timeout_seconds": config.OVERALL_TIMEOUT_SECONDS}, step_number - 1,
            )
            if result is not None:
                return result
            deadline = time.monotonic() + config.OVERALL_TIMEOUT_SECONDS
            no_progress_count, previous_fingerprint = 0, None
            continue

        try:
            observation = build_observation(page)
        except Exception as exc:
            # session/browser-level breakage - no live session left to escalate into
            return _resolve_forced_fail(logger, f"perception failed: {exc}", step_number - 1, goal_key)

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
            return _resolve_forced_fail(logger, "LLM call failed after retries", step_number, goal_key)

        for fa in decision.failed_attempts:
            logger.log_llm_call_attempt_failed(
                step_number=step_number,
                attempt=fa["attempt"],
                result=fa["result"],
                duration_ms=fa["duration_ms"],
            )

        action_log = {
            "type": decision.action,
            # Not redacted here - RunLogger._write() redacts every string in
            # every logged event centrally now; redacting again here would
            # be redundant (see logger.py's _redact_value docstring).
            "reasoning": decision.args.get("reasoning", ""),
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
                    escalation_result = _handle_escalation(
                        logger, page, goal_key, "no_progress",
                        {"consecutive_unchanged_steps": no_progress_count}, step_number,
                    )
                    if escalation_result is not None:
                        return escalation_result
                    no_progress_count, previous_fingerprint = 0, None
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
            return RunResult(outcome_type, outcome, result.outputs, step_number, logger.run_id, goal_key)

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
            # Raw (unredacted) deliberately: this also reaches the human
            # operator's own screen via operator.request_manual_handoff's
            # printed context, not just the log - the log gets redacted
            # centrally regardless (RunLogger._write()). The operator
            # already has full, unredacted access to the live browser page
            # during this exact handoff, so hiding an email/phone in this
            # one summary string while showing it plainly on the page a
            # click away would be inconsistent, not safer.
            reasoning = decision.args.get("reasoning", "")
            result = _handle_escalation(
                logger, page, goal_key, "llm_initiated", {"reasoning": reasoning}, step_number
            )
            if result is not None:
                return result
            no_progress_count, previous_fingerprint = 0, None
            continue

        # --- click / type / select ---
        index = decision.args.get("index")
        value = decision.args.get("value")
        action_log["index"] = index
        # Not redacted here - RunLogger._write() handles it centrally now.
        action_log["value"] = value

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
                escalation_result = _handle_escalation(
                    logger, page, goal_key, "no_progress",
                    {"consecutive_unchanged_steps": no_progress_count}, step_number,
                )
                if escalation_result is not None:
                    return escalation_result
                no_progress_count, previous_fingerprint = 0, None
            continue

        action_log["target"] = {
            "role": element.role,
            "name": element.name,
            "nth": element.nth,
            # (method, path) as resolved live at record time - not the
            # safety-relevant value (guardrails re-derives ground truth
            # itself), just the raw material artifact_builder.py needs to
            # populate each step's grounding.expected_target.
            "effective_target": element.effective_target,
            # The app's own declared <input type>, textbox only - the
            # authoritative seed for a parameter's draft type in
            # artifact_builder.py. Must be logged, not just captured in
            # perception.py's in-memory InteractiveElement, since the
            # artifact builder only ever reads log.jsonl (decision: build
            # post-hoc, not inline during the live run).
            "html_input_type": element.html_input_type,
        }

        guardrail = evaluate_action(element, page, goal=goal)
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
                escalation_result = _handle_escalation(
                    logger, page, goal_key, "no_progress",
                    {"consecutive_unchanged_steps": no_progress_count}, step_number,
                )
                if escalation_result is not None:
                    return escalation_result
                no_progress_count, previous_fingerprint = 0, None
            continue

        hitl_event_id = None
        if guardrail.decision == "hitl_required":
            # Guaranteed, not incidental: previously this only carried a
            # screenshot if some UNRELATED condition (first step,
            # duplicates, periodic interval) happened to already trigger
            # one this same step - a risky-amount approval on any other
            # step showed the operator "Screenshot: None". Same pattern
            # already used to guarantee one before `finish`, below.
            if not screenshot_taken:
                screenshot_reason = _maybe_screenshot(logger, page, step_number, ["hitl_approval"])
                screenshot_taken = True

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
                f'{decision.action} "{element.name}"',
                guardrail.amount,
                guardrail.reason,
                goal_key=goal_key,
                step_number=step_number,
                screenshot_path=(
                    str(logger.screenshot_path(step_number)) if screenshot_taken else None
                ),
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
                    escalation_result = _handle_escalation(
                        logger, page, goal_key, "no_progress",
                        {"consecutive_unchanged_steps": no_progress_count}, step_number,
                    )
                    if escalation_result is not None:
                        return escalation_result
                    no_progress_count, previous_fingerprint = 0, None
                continue

            # Re-verify nothing changed underneath the approval before
            # acting on it. request_approval() blocks for an arbitrary
            # amount of time, during which a human can - by design, this
            # is the real handoff mechanism - freely navigate the SAME
            # live session. `element`/`guardrail` were captured before
            # that wait; a live re-query of the button the human approved
            # can now resolve against completely different page state
            # (e.g. a different amount someone typed while the approval
            # was pending). Re-running the exact check that gated this
            # action in the first place is the same "never trust a stale
            # value, read live" principle guardrails.py already applies to
            # the amount field itself - applied here at the one point that
            # principle didn't yet cover. A live incident during testing
            # (not hypothetical) showed exactly this: a human approved a
            # $12,000 withdrawal, manually changed the live amount to
            # $5,000 while the approval prompt was still pending, and the
            # stale approval executed a real $5,000 withdrawal that was
            # never itself approved.
            fresh_guardrail = evaluate_action(element, page, goal=goal)
            approval_still_valid = fresh_guardrail.decision == guardrail.decision and (
                (fresh_guardrail.amount is None and guardrail.amount is None)
                or (
                    fresh_guardrail.amount is not None
                    and guardrail.amount is not None
                    and abs(fresh_guardrail.amount - guardrail.amount) < 0.01
                )
            )
            if not approval_still_valid:
                logger.log_hitl_event(
                    hitl_event_id=hitl_event_id,
                    phase="stale_after_approval",
                    trigger="live_state_changed_during_approval_wait",
                    context={
                        "approved_decision": guardrail.decision,
                        "approved_amount": guardrail.amount,
                        "live_decision": fresh_guardrail.decision,
                        "live_amount": fresh_guardrail.amount,
                    },
                )
                history_lines.append(
                    f'Step {step_number}: {decision.action} "{element.name}" -> refused: '
                    f"approved {guardrail.decision} (${guardrail.amount}) no longer matches "
                    f"live state ({fresh_guardrail.decision}, ${fresh_guardrail.amount}) - "
                    "not executing a stale approval"
                )
                no_progress_count, previous_fingerprint = _track_fingerprint(
                    observation.fingerprint, previous_fingerprint, no_progress_count
                )
                if no_progress_count >= config.NO_PROGRESS_THRESHOLD:
                    escalation_result = _handle_escalation(
                        logger, page, goal_key, "no_progress",
                        {"consecutive_unchanged_steps": no_progress_count}, step_number,
                    )
                    if escalation_result is not None:
                        return escalation_result
                    no_progress_count, previous_fingerprint = 0, None
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
            return _resolve_forced_fail(logger, f"perception failed after action: {exc}", step_number, goal_key)
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
            escalation_result = _handle_escalation(
                logger, page, goal_key, "no_progress",
                {"consecutive_unchanged_steps": no_progress_count}, step_number,
            )
            if escalation_result is not None:
                return escalation_result
            no_progress_count, previous_fingerprint = 0, None


def _maybe_screenshot(logger: RunLogger, page: Page, step_number: int, reasons: list[str]) -> Optional[str]:
    if not reasons:
        return None
    path = logger.screenshot_path(step_number)
    # A screenshot is a secondary/cosmetic perception channel (Section 3.1) -
    # a broken/half-loaded page or a disk write failure here must fail safe,
    # same as every other live Playwright read on this system's failure
    # paths (action_executor.execute_action, guardrails._amount_bearing_
    # field's live read, the HITL-context screenshots below), not crash the
    # whole run with no run_end logged. Returning None (not the reasons
    # string) on failure matters beyond just not crashing: _log_step derives
    # the logged screenshot_path from truthiness of this return value, so
    # returning a reason here when the write actually failed would log a
    # screenshot_path that doesn't exist on disk.
    try:
        path.write_bytes(capture_screenshot(page))
    except Exception:
        return None
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
        # Not redacted here - RunLogger._write() handles it centrally now.
        "static_text": list(observation.static_text),
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


def _handle_capability_classification_failure(
    logger: RunLogger, page: Page, goal: str, reason: str
) -> Optional[str]:
    """Backup path for run_discovery's opening classify_goal_key call: asks
    a human to pick the capability directly, via
    operator.request_capability_selection, instead of dead-ending the whole
    run when automated classification can't produce a trustworthy goal_key
    (the LLM call failed after retries, or it returned something outside
    checkpoints.REGISTRIES). Returns the chosen goal_key, or None if the
    human aborts - the caller treats an abort as a terminal "escalated"
    result, the same "a human was asked and declined" semantics
    _handle_escalation's abort path already uses, not "failed" (reserved
    for a genuine no-human-recourse dead end)."""
    hitl_event_id = secrets.token_hex(4)

    screenshot_path = logger.screenshot_path(0)
    try:
        screenshot_path.write_bytes(capture_screenshot(page))
    except Exception:
        screenshot_path = None

    logger.log_hitl_event(
        hitl_event_id=hitl_event_id,
        phase="raised",
        trigger="capability_classification_failed",
        context={"goal": goal, "reason": reason},
    )

    valid_goal_keys = sorted(checkpoints.REGISTRIES.keys())
    goal_key = operator.request_capability_selection(
        goal,
        valid_goal_keys,
        reason=reason,
        screenshot_path=str(screenshot_path) if screenshot_path else None,
    )

    logger.log_hitl_event(
        hitl_event_id=hitl_event_id,
        phase="resolved",
        decision=(f"selected:{goal_key}" if goal_key is not None else "aborted"),
        operator_id="local-operator",
    )
    return goal_key


def _handle_escalation(
    logger: RunLogger,
    page: Page,
    goal_key: str,
    trigger: str,
    context: dict,
    step_number: int,
) -> Optional[RunResult]:
    """Raises a real human-handoff event and blocks on a real decision.

    Captures a before/after snapshot of the live page across the handoff
    window - not a click-by-click recording of what the human did (out of
    scope per the brief's own framing), but a real, if coarse, record of
    what actually changed while they had control.

    Returns `None` if the human resumed (the caller should reset its
    no-progress tracking and continue the loop - the next iteration
    re-observes the page fresh, so if the human already finished the goal
    manually, the LLM will see the terminal state itself and go through
    the same checkpoint verification as any other `finish` call - the
    human's own claim of "done" is never trusted directly, consistent with
    every other self-report in this system). Returns a terminal
    `RunResult` if they aborted.
    """
    hitl_event_id = secrets.token_hex(4)

    try:
        pre_observation: Optional[Observation] = build_observation(page)
    except Exception:
        pre_observation = None

    screenshot_path = logger.screenshot_path(step_number)
    try:
        screenshot_path.write_bytes(capture_screenshot(page))
    except Exception:
        screenshot_path = None

    logger.log_hitl_event(
        hitl_event_id=hitl_event_id,
        phase="raised",
        trigger=trigger,
        context={
            **context,
            "pre_handoff_url": pre_observation.url if pre_observation else None,
            "pre_handoff_fingerprint": pre_observation.fingerprint if pre_observation else None,
        },
    )

    resumed = operator.request_manual_handoff(
        trigger,
        context,
        goal_key=goal_key,
        step_number=step_number,
        screenshot_path=str(screenshot_path) if screenshot_path else None,
    )

    try:
        post_observation: Optional[Observation] = build_observation(page)
    except Exception:
        post_observation = None

    logger.log_hitl_event(
        hitl_event_id=hitl_event_id,
        phase="resolved",
        decision="resolved_manually" if resumed else "aborted",
        operator_id="local-operator",
        context={
            "post_handoff_url": post_observation.url if post_observation else None,
            "post_handoff_fingerprint": post_observation.fingerprint if post_observation else None,
            # Section 3.6's "record what the human did" - a real, before/
            # after account of what changed on the page during the handoff
            # window (see perception.diff_observations' docstring for why
            # this is a state diff, not an action log).
            "handoff_diff": diff_observations(pre_observation, post_observation),
        },
    )

    if not resumed:
        logger.log_run_end(
            outcome_type="escalated", outputs=None, checkpoint_verification=None, total_steps=step_number
        )
        return RunResult("escalated", trigger, None, step_number, logger.run_id, goal_key)

    return None


def _resolve_forced_fail(
    logger: RunLogger, reason: str, total_steps: int, goal_key: Optional[str]
) -> RunResult:
    logger.log_run_end(
        outcome_type="failed", outputs=None, checkpoint_verification=None, total_steps=total_steps
    )
    return RunResult("failed", reason, None, total_steps, logger.run_id, goal_key)
