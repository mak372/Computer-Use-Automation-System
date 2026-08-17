"""Replay controller (Section 3.3): deterministic, non-LLM execution of a
previously-built artifact (agent/artifact_builder.py). Reuses perception.py,
guardrails.py, action_executor.py, checkpoints.py, logger.py, and
operator.py exactly as discovery does (see loop_controller.py's own
docstring) - only the action-selection logic is replaced: instead of an LLM
deciding what to do next, this module walks a fixed, pre-recorded steps
list.

This piece: artifact loading + preflight validation + the result contract.
No live browser/Playwright interaction yet - that's the step-execution loop
(added next), kept deliberately separate so validation can be built and
exercised against an artifact file alone, independent of a live target_app.

Preflight order (all before the browser is ever touched, settled in the
replay design conversation):
  1. schema_version must match what this build understands - a version
     mismatch means "I don't know how to safely interpret this file's
     shape," worse to guess at than to refuse outright.
  2. status must be "reviewed" - a "draft" artifact's parameter candidacy/
     naming/typing/description haven't been human-confirmed (artifact_
     builder.py's own docstring: "nothing here should be trusted").
  3. If built_from_degraded_log is True, degraded_grounding_reviewed must
     also be True. status == "reviewed" alone only proves someone reviewed
     the description - it says nothing about whether they also re-verified
     a degraded artifact's possibly-wrong-by-default step grounding (nth
     defaults to 0, which could silently target the wrong element if a
     real duplicate existed at that step). Both flags gate the same "is
     this artifact trustworthy" question, not two unrelated concerns.
  4. If the caller passes expected_capability_version, it must match the
     artifact's own capability_version - catches a caller relying on a
     stale understanding of this capability's shape (parameter names/count
     can differ across a capability_version bump).
  5. Every declared parameter must be supplied by the caller, no
     caller-supplied key may be unknown, and each value's shape must match
     its declared type - fail closed, one clear error before step 1, not a
     confusing mid-flow app-side failure three steps in.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from playwright.sync_api import Page

from agent import artifact_builder, checkpoints, config, operator
from agent.action_executor import execute_action
from agent.guardrails import evaluate_action
from agent.logger import RunLogger
from agent.perception import (
    InteractiveElement,
    Observation,
    build_observation,
    capture_screenshot,
    resolve_element_target,
)


class ReplayValidationError(Exception):
    """Raised for any preflight failure - artifact incompatible, unreviewed,
    or caller-supplied parameters invalid. Always raised before any live
    browser/Playwright interaction; never produced by a partially-executed
    run (that's ReplayResult.outcome_type == "failed" instead, added with
    the step-execution loop)."""


@dataclass
class ReplayResult:
    outcome_type: str  # "success" | "business_outcome" | "escalated" | "failed"
    outcome_label: Optional[str]
    outputs: Optional[dict]
    total_steps: int
    run_id: str
    # Diagnostics (Section 3.3: "what step, what was expected, what was
    # observed") - populated whenever the divergence has a specific step
    # attached; None on success or a whole-run outcome that isn't
    # step-specific (e.g. a preflight-level concern would never reach here
    # at all, since preflight raises before a run_id even exists).
    failed_at_step: Optional[int] = None
    expected: Optional[Any] = None
    observed: Optional[Any] = None


def load_artifact(goal_key: str, repo_root: Path | None = None) -> dict:
    repo_root = repo_root or Path.cwd()
    path = repo_root / config.ARTIFACTS_DIR / f"{goal_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"no artifact found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact_preflight(
    artifact: dict, expected_capability_version: int | None = None
) -> None:
    if artifact.get("schema_version") != artifact_builder.SCHEMA_VERSION:
        raise ReplayValidationError(
            f"artifact schema_version {artifact.get('schema_version')!r} does not match "
            f"the version this replay engine understands ({artifact_builder.SCHEMA_VERSION})"
        )

    if artifact.get("status") != "reviewed":
        raise ReplayValidationError(
            f"artifact status is {artifact.get('status')!r}, not 'reviewed' - a draft "
            "artifact's parameters/description have not been human-confirmed"
        )

    if artifact.get("built_from_degraded_log") and not artifact.get("degraded_grounding_reviewed"):
        raise ReplayValidationError(
            "artifact was built from a degraded log and its grounding has not been "
            "re-reviewed (degraded_grounding_reviewed is not True) - recorded nth values "
            "may silently target the wrong element"
        )

    if expected_capability_version is not None:
        actual = artifact.get("capability_version")
        if actual != expected_capability_version:
            raise ReplayValidationError(
                f"caller expected capability_version {expected_capability_version}, "
                f"artifact is at capability_version {actual} - this capability's shape "
                "(parameters, steps) may have changed since the caller last recorded it"
            )


def _validate_param_shape(name: str, declared_type: str, value: Any) -> None:
    # Reject bool up front - isinstance(True, int) is True in Python, so a
    # naive numeric check would silently accept it as 1.0/0.0. Reject any
    # non-primitive (dict/list/None) too, since it's about to be str()'d
    # verbatim into a Playwright fill()/select_option() call, and a
    # nonsensical value like "None" or "{'a': 1}" landing in a form field
    # should fail here, not three steps into a live run.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ReplayValidationError(
            f"parameter {name!r} must be a plain string or number, got {type(value).__name__}"
        )
    if declared_type == "number":
        try:
            float(value)
        except (TypeError, ValueError):
            raise ReplayValidationError(f"parameter {name!r} must be a number, got {value!r}")


def bind_parameters(artifact: dict, parameters: dict[str, Any]) -> dict[str, str]:
    """Validates the caller-supplied parameter dict against the artifact's
    declared `parameters[]` and returns a dict of param_name -> str(value),
    ready to bind into each step's {"param": name} reference. Playwright's
    fill()/select_option() always take strings regardless of the declared
    contract type (see action_executor.py) - the declared type is a
    calling-agent-facing contract concern, checked here once, never
    branched on again during execution."""
    declared = {p["name"]: p["type"] for p in artifact.get("parameters", [])}

    missing = declared.keys() - parameters.keys()
    if missing:
        raise ReplayValidationError(f"missing required parameter(s): {sorted(missing)}")

    unknown = parameters.keys() - declared.keys()
    if unknown:
        raise ReplayValidationError(
            f"unknown parameter(s) not declared by this artifact: {sorted(unknown)}"
        )

    for name, value in parameters.items():
        _validate_param_shape(name, declared[name], value)

    return {name: str(value) for name, value in parameters.items()}


def _rehydrate_path(templated_path: str, bound_params: dict[str, str]) -> str:
    """Literal inverse of artifact_builder._template_path: splits on '/' and
    substitutes each whole segment that reads exactly "{param_name}" with
    this run's bound value for that parameter. Segment-exact-match, not a
    naive whole-string replace, for the same reason the forward direction
    is segment-exact: a bound value containing '/' (or coincidentally
    matching other path text) must never be able to corrupt a fixed route
    segment on either side of the templating operation."""
    segments = templated_path.split("/")
    for i, seg in enumerate(segments):
        if seg.startswith("{") and seg.endswith("}"):
            param_name = seg[1:-1]
            if param_name in bound_params:
                segments[i] = bound_params[param_name]
    return "/".join(segments)


@dataclass
class RelocationResult:
    # None only when drift == "grounding_drift" (element genuinely absent).
    element: Optional[InteractiveElement]
    # None | "grounding_drift" | "target_drift"
    drift: Optional[str]
    expected: Optional[Any] = None
    observed: Optional[Any] = None


def relocate_step_element(
    page: Page, step: dict, bound_params: dict[str, str]
) -> RelocationResult:
    """Decision 1's fast path: address the step's element directly via
    get_by_role(...).nth(n) and check .count() before ever touching it -
    cheaper than a full build_observation() walk, and gives a clean,
    explicit "recorded nth doesn't exist among current matches" signal
    instead of discovering absence only via a Playwright timeout at click
    time (which would be indistinguishable from a live execution glitch).

    Also runs Decision 1's second check: diff the located element's live
    effective_target against the artifact's recorded expected_target
    (rehydrated with this run's actual parameter values, not the original
    recording's) - catches the subtler case where the element is still
    there, at the same nth, but now resolves somewhere different.
    """
    grounding = step["grounding"]
    role, name, nth = grounding["role"], grounding["name"], grounding["nth"]

    count = page.get_by_role(role, name=name, exact=True).count()
    if count <= nth:
        return RelocationResult(
            element=None,
            drift="grounding_drift",
            expected=f"at least {nth + 1} match(es) for role={role!r} name={name!r}",
            observed=f"{count} match(es) found",
        )

    locator = page.get_by_role(role, name=name, exact=True).nth(nth)
    effective_target, html_input_type = resolve_element_target(role, locator)

    element = InteractiveElement(
        index=0,
        role=role,
        name=name,
        nth=nth,
        locator=locator,
        preceding_context="",
        value=None,
        effective_target=effective_target,
        html_input_type=html_input_type,
    )

    recorded_expected_target = grounding.get("expected_target")
    if recorded_expected_target is not None:
        expected_method, expected_path_template = recorded_expected_target
        expected_path = _rehydrate_path(expected_path_template, bound_params)
        observed = list(effective_target) if effective_target else None
        if observed != [expected_method, expected_path]:
            return RelocationResult(
                element=element,
                drift="target_drift",
                expected=[expected_method, expected_path],
                observed=observed,
            )

    return RelocationResult(element=element, drift=None)


@dataclass
class DiagnosticSnapshot:
    observation: Optional[Observation]
    # Present only when a fast-path element was passed in and it's a link -
    # buttons/textboxes already call the exact same resolution code on both
    # paths (see resolve_element_target), so comparing them would be
    # trivially true and uninformative. Links are the one case
    # resolve_element_target's docstring flags as "should agree, not
    # proven": the fast path reads raw `href` via get_attribute, the walk
    # reads the aria-snapshot's /url metadata. Whenever both a fast-path
    # element and a fresh walk happen to be available together (i.e.
    # whenever the step loop already needed a diagnostic snapshot for some
    # other reason), this closes that gap with real evidence instead of
    # leaving it as a manual spot-check.
    link_resolution_cross_check: Optional[dict] = None


def capture_diagnostic_snapshot(
    page: Page, element: Optional[InteractiveElement] = None
) -> DiagnosticSnapshot:
    """Best-effort full observation, called by the step loop (not this
    module) whenever something abnormal happens - drift, a failed
    execution, or a mismatched final checkpoint. Purely for evidence-log
    diagnostic detail; never used to re-attempt relocation or to decide
    whether to retry."""
    try:
        observation = build_observation(page)
    except Exception:
        return DiagnosticSnapshot(observation=None)

    cross_check = None
    if element is not None and element.role == "link":
        match = next(
            (
                el
                for el in observation.interactive_elements
                if el.role == "link" and el.name == element.name and el.nth == element.nth
            ),
            None,
        )
        if match is not None:
            cross_check = {
                "name": element.name,
                "nth": element.nth,
                "fast_path_href": element.effective_target[1] if element.effective_target else None,
                "walk_href": match.effective_target[1] if match.effective_target else None,
                "agree": element.effective_target == match.effective_target,
            }

    return DiagnosticSnapshot(observation=observation, link_resolution_cross_check=cross_check)


def _match_signature(observation: Observation, url_pattern: str, text_signature: str) -> bool:
    path = urlparse(observation.url).path
    if not re.match(url_pattern, path):
        return False
    return any(text_signature in text for text in observation.static_text)


def _extract_outputs_from_observation(spec, observation: Observation) -> tuple[Optional[dict], Optional[str]]:
    """Discovery's finish(output_refs=[...]) lets the LLM point at exactly
    which static_text entry holds each output value; replay has no LLM to
    ask. Instead: scan every static_text entry, try the extractor against
    each as a lone candidate ([text]), and require exactly one to succeed.
    Zero or multiple matches both fail closed (returns None) rather than
    guessing which one is right - same never-guess-on-ambiguity principle
    that forced explicit preceding_context disambiguation for duplicate
    buttons in discovery. Returns ({}, None) when the outcome has no
    output_extractor at all (a legitimate no-outputs success)."""
    if spec.output_extractor is None:
        return {}, None

    successes = []
    for text in observation.static_text:
        try:
            successes.append(spec.output_extractor([text]))
        except ValueError:
            continue

    if len(successes) == 1:
        return successes[0], None
    if len(successes) == 0:
        return None, "no static text entry matched the expected output shape"
    return None, f"{len(successes)} static text entries ambiguously matched the expected output shape"


@dataclass
class CheckpointMatch:
    outcome_type: Optional[str]  # "success" | "business_outcome" | None (nothing matched)
    outcome_label: Optional[str]  # None for success, the outcome name for business_outcome
    outputs: Optional[dict] = None
    detail: Optional[str] = None  # set when success text/url matched but output extraction failed


def resolve_checkpoint(artifact: dict, goal_key: str, observation: Observation) -> CheckpointMatch:
    """Checks live state against the artifact's full frozen checkpoint set,
    success first, then every known non-success outcome - the same
    business-outcome-vs-genuine-failure distinction checkpoints.py already
    enforces in discovery (Section 3.3's most common design mistake to
    avoid). Only outcome_type is None does the caller get to conclude
    genuine checkpoint drift - nothing recognized at all."""
    success_spec = artifact["checkpoint"]["success"]
    if _match_signature(observation, success_spec["url_pattern"], success_spec["text_signature"]):
        spec = checkpoints.REGISTRIES[goal_key]["success"]
        outputs, detail = _extract_outputs_from_observation(spec, observation)
        if outputs is None:
            # Signature matched but outputs couldn't be unambiguously
            # re-extracted - reject the whole claimed outcome rather than
            # partially accept it, mirroring discovery's verify_finish()
            # (decision #7): neither check substitutes for the other.
            return CheckpointMatch(outcome_type=None, outcome_label=None, detail=detail)
        return CheckpointMatch(outcome_type="success", outcome_label=None, outputs=outputs)

    for known in artifact["checkpoint"]["known_outcomes"]:
        if _match_signature(observation, known["url_pattern"], known["text_signature"]):
            return CheckpointMatch(outcome_type="business_outcome", outcome_label=known["outcome"])

    return CheckpointMatch(outcome_type=None, outcome_label=None)


def _handle_unrecoverable(
    page: Page,
    logger: RunLogger,
    goal_key: str,
    artifact: dict,
    step_index: int,
    trigger: str,
    context: dict,
) -> Optional[CheckpointMatch]:
    """Replay's equivalent of loop_controller._handle_escalation - a real
    human handoff for the exact scenario the brief names explicitly ("a
    replay hits a condition it can't recover from"). Called only after
    every other recovery has already failed: resolve_checkpoint() matched
    nothing at all against the artifact's full checkpoint set, whether
    that's mid-run (drift, or execution failure surviving its one retry -
    both reach this via _try_checkpoint_override) or at the very end of a
    clean run that still landed somewhere unrecognized. Before this, replay
    had no handoff path here at all - only the HITL-threshold approval gate
    (operator.request_approval), which is a different mechanism (a yes/no
    gate before automation acts, not a human taking manual control of a
    stuck run).

    Captures a before/after screenshot + observation across the handoff
    window, same minimal-but-real "what changed while the human had
    control" record discovery's own handler captures - not a click-by-click
    trace, which is out of scope per the brief's own framing.

    Returns None if the human aborted (caller finalizes as failed, as
    before this existed). Returns a CheckpointMatch if they resumed - the
    live page is re-checked against the same checkpoint set the rest of
    this module already trusts, never the human's own claim of "done"
    directly, same principle as every other self-report in this system. A
    resumed handoff whose re-check still matches nothing (outcome_type
    None) means the manual fix didn't land on a recognized state either -
    the caller still finalizes as failed in that case, just now with real
    evidence that a human looked at it first.
    """
    hitl_event_id = secrets.token_hex(4)

    try:
        pre_observation: Optional[Observation] = build_observation(page)
    except Exception:
        pre_observation = None

    screenshot_path = logger.screenshot_path(step_index)
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
        step_number=step_index,
        screenshot_path=str(screenshot_path) if screenshot_path else None,
    )

    try:
        post_observation: Optional[Observation] = build_observation(page)
    except Exception:
        post_observation = None
    state_changed = (
        pre_observation is not None
        and post_observation is not None
        and pre_observation.fingerprint != post_observation.fingerprint
    )

    logger.log_hitl_event(
        hitl_event_id=hitl_event_id,
        phase="resolved",
        decision="resolved_manually" if resumed else "aborted",
        operator_id="local-operator",
        context={
            "post_handoff_url": post_observation.url if post_observation else None,
            "post_handoff_fingerprint": post_observation.fingerprint if post_observation else None,
            "state_changed_during_handoff": state_changed,
        },
    )

    if not resumed or post_observation is None:
        return None

    return resolve_checkpoint(artifact, goal_key, post_observation)


def _guardrail_gate(
    element: InteractiveElement,
    page: Page,
    logger: RunLogger,
    step: dict,
    step_index: int,
    skip_hitl_prompt: bool,
) -> tuple[Any, Optional["ReplayResult"], bool]:
    """Returns (guardrail_decision, terminal_result_or_None, just_approved).
    Always re-evaluates the guardrail live against the current page (never
    trust a stale check, same rule guardrails.py itself already follows for
    the amount field).

    skip_hitl_prompt must only be True when the CALLER has already recorded
    a real approval for this exact step_index earlier this run - it is not
    "we've retried this step before." If the first attempt at a step was
    auto_approved (no prompt shown at all) and a retry's live guardrail
    re-evaluation newly comes back hitl_required, that must still prompt: no
    human has approved it yet, regardless of how many times this step has
    been attempted. just_approved is True only when this call itself just
    obtained a fresh "yes" from the operator - the caller is responsible for
    remembering that for any subsequent retry of the same step_index.
    """
    guardrail = evaluate_action(element, page)

    if guardrail.decision == "blocked":
        return guardrail, ReplayResult(
            outcome_type="escalated",
            outcome_label="guardrail_blocked",
            outputs=None,
            total_steps=step_index,
            run_id=logger.run_id,
            failed_at_step=step_index,
            expected=None,
            observed=guardrail.reason,
        ), False

    if guardrail.decision == "hitl_required":
        if skip_hitl_prompt:
            # Caller has already recorded a real approval for this exact
            # step_index - trust it, do not prompt again.
            return guardrail, None, False

        hitl_event_id = secrets.token_hex(4)
        logger.log_hitl_event(
            hitl_event_id=hitl_event_id,
            phase="raised",
            trigger="guardrail_threshold",
            context={"action": step["action"], "element": element.name, "amount": guardrail.amount},
        )
        approved = operator.request_approval(f'{step["action"]} "{element.name}"', guardrail.amount)
        logger.log_hitl_event(
            hitl_event_id=hitl_event_id,
            phase="resolved",
            decision="approved" if approved else "denied",
            operator_id="local-operator",
        )
        if not approved:
            return guardrail, ReplayResult(
                outcome_type="escalated",
                outcome_label="hitl_denied",
                outputs=None,
                total_steps=step_index,
                run_id=logger.run_id,
                failed_at_step=step_index,
                expected=None,
                observed="human denied approval",
            ), False
        return guardrail, None, True

    return guardrail, None, False


def _log_replay_step(
    logger: RunLogger,
    step_index: int,
    step: dict,
    value: Optional[str],
    execution_result: str,
    retried: bool,
    guardrail: Any,
) -> None:
    logger.log_step(
        step_number=step_index,
        observation_summary={},
        action={
            "source": "artifact_step",
            "artifact_step_index": step["step_index"],
            "action_type": step["action"],
            "grounding": step["grounding"],
            "value": value,
            "retried": retried,
        },
        guardrail_result={
            "decision": guardrail.decision,
            "reason": guardrail.reason,
            "amount": guardrail.amount,
        },
        execution_result=execution_result,
        # No-progress fingerprint tracking doesn't apply to replay the way
        # it does discovery - that mechanism exists to catch an LLM stuck
        # looping indefinitely, but replay walks a fixed, finite steps
        # list and can only ever make exactly one attempt (plus one bounded
        # retry) per step, deterministically. Left empty deliberately, not
        # an oversight.
        state_fingerprint="",
    )


def _try_checkpoint_override(
    page: Page,
    logger: RunLogger,
    artifact: dict,
    goal_key: str,
    element: Optional[InteractiveElement],
    step_index: int,
) -> Optional["ReplayResult"]:
    """Whenever something abnormal happens mid-run (grounding/target drift,
    or an execution failure), check live page state against the artifact's
    full checkpoint set before finalizing a failure - a client-side error
    (a timed-out click, an element that hasn't re-appeared yet) can mask a
    real server-side effect that already happened (Decision 2, point 2). A
    positive match here always overrides the mechanical failure that
    triggered this check. Also logs the diagnostic snapshot itself, a
    screenshot (discovery captures one on every on_error trigger - this is
    replay's equivalent, since a text-only observation summary alone loses
    exactly the kind of visual context a human reviewer would want when
    something abnormal happened), and, opportunistically, the
    link-resolution cross-check (Piece 3)."""
    snapshot = capture_diagnostic_snapshot(page, element)

    screenshot_reason = None
    try:
        path = logger.screenshot_path(step_index)
        path.write_bytes(capture_screenshot(page))
        screenshot_reason = "replay_diagnostic"
    except Exception:
        # Best-effort, same as discovery's own screenshot capture - never
        # let a logging/diagnostic failure crash the run itself.
        screenshot_reason = None

    logger.log_step(
        step_number=step_index,
        observation_summary={
            "url": snapshot.observation.url if snapshot.observation else None,
            "link_resolution_cross_check": snapshot.link_resolution_cross_check,
            "screenshot_path": (f"step_{step_index}.png" if screenshot_reason else None),
            "screenshot_reason": screenshot_reason,
        },
        action={"diagnostic_snapshot_for_step": step_index},
        guardrail_result={"decision": "not_applicable"},
        execution_result="diagnostic_snapshot",
        state_fingerprint=(snapshot.observation.fingerprint if snapshot.observation else ""),
    )
    if snapshot.observation is None:
        return None

    match = resolve_checkpoint(artifact, goal_key, snapshot.observation)
    if match.outcome_type is None:
        # Every mechanical recovery has now failed (drift's relocation, or
        # the bounded execution retry) and the checkpoint set still
        # recognizes nothing - this is exactly "a condition replay can't
        # recover from" on its own. Offer a real human handoff before
        # finalizing as a failure.
        handoff_match = _handle_unrecoverable(
            page, logger, goal_key, artifact, step_index,
            trigger="replay_unrecoverable",
            context={"url": snapshot.observation.url},
        )
        if handoff_match is None or handoff_match.outcome_type is None:
            return None
        match = handoff_match

    return ReplayResult(
        outcome_type=match.outcome_type,
        outcome_label=match.outcome_label,
        outputs=match.outputs,
        total_steps=step_index,
        run_id=logger.run_id,
    )


def _locate_or_terminal(
    page: Page,
    logger: RunLogger,
    artifact: dict,
    goal_key: str,
    step: dict,
    bound_params: dict[str, str],
    step_index: int,
) -> tuple[Optional[InteractiveElement], Optional["ReplayResult"]]:
    relocation = relocate_step_element(page, step, bound_params)
    if relocation.drift is None:
        return relocation.element, None

    override = _try_checkpoint_override(page, logger, artifact, goal_key, relocation.element, step_index)
    if override is not None:
        return None, override

    return None, ReplayResult(
        outcome_type="failed",
        outcome_label=relocation.drift,
        outputs=None,
        total_steps=step_index,
        run_id=logger.run_id,
        failed_at_step=step_index,
        expected=relocation.expected,
        observed=relocation.observed,
    )


def _execute_one_step(
    page: Page,
    logger: RunLogger,
    artifact: dict,
    goal_key: str,
    step: dict,
    bound_params: dict[str, str],
    step_index: int,
) -> Optional["ReplayResult"]:
    """Returns None if the step succeeded (proceed to the next one), or a
    terminal ReplayResult to stop the whole run now."""
    element, terminal = _locate_or_terminal(page, logger, artifact, goal_key, step, bound_params, step_index)
    if terminal is not None:
        return terminal

    guardrail, refusal, just_approved = _guardrail_gate(
        element, page, logger, step, step_index, skip_hitl_prompt=False
    )
    if refusal is not None:
        return refusal
    # Scoped to this one step_index only - _execute_one_step handles exactly
    # one step per call, retries never cross into a different step, so this
    # is the exact-scoped equivalent of tracking approved_step_indices.
    step_hitl_approved = just_approved

    value = bound_params[step["value"]["param"]] if step["value"] is not None else None
    exec_result = execute_action(element, step["action"], value)
    if exec_result.success:
        _log_replay_step(logger, step_index, step, value, "success", retried=False, guardrail=guardrail)
        return None

    override = _try_checkpoint_override(page, logger, artifact, goal_key, element, step_index)
    if override is not None:
        return override

    # Bounded retry: re-relocate (the failed attempt may have partially
    # navigated the page), re-check guardrails live. Only skip re-prompting
    # if THIS step_index was actually approved above - never just because
    # this is a retry (a first attempt that was auto_approved carries no
    # approval to reuse if the retry's live re-evaluation newly comes back
    # hitl_required).
    element, terminal = _locate_or_terminal(page, logger, artifact, goal_key, step, bound_params, step_index)
    if terminal is not None:
        return terminal

    guardrail, refusal, just_approved = _guardrail_gate(
        element, page, logger, step, step_index, skip_hitl_prompt=step_hitl_approved
    )
    if refusal is not None:
        return refusal
    step_hitl_approved = step_hitl_approved or just_approved

    exec_retry = execute_action(element, step["action"], value)
    if exec_retry.success:
        _log_replay_step(logger, step_index, step, value, "success", retried=True, guardrail=guardrail)
        return None

    override = _try_checkpoint_override(page, logger, artifact, goal_key, element, step_index)
    if override is not None:
        return override

    _log_replay_step(
        logger, step_index, step, value, exec_retry.error_type or "execution_error",
        retried=True, guardrail=guardrail,
    )
    return ReplayResult(
        outcome_type="failed",
        outcome_label="execution_failure",
        outputs=None,
        total_steps=step_index,
        run_id=logger.run_id,
        failed_at_step=step_index,
        expected="successful execution after one retry",
        observed=f"{exec_retry.error_type}: {exec_retry.message}",
    )


def run_replay(
    goal_key: str,
    page: Page,
    logger: RunLogger,
    parameters: dict[str, Any],
    expected_capability_version: int | None = None,
    repo_root: Path | None = None,
) -> ReplayResult:
    """Top-level entry point - the replay equivalent of loop_controller.
    run_discovery(). Preflight (load, validate, bind parameters) all raise
    ReplayValidationError before this ever touches the browser or creates a
    run_id; everything after that point is captured in a ReplayResult and
    logged through the same RunLogger/log.jsonl schema discovery uses."""
    artifact = load_artifact(goal_key, repo_root)
    validate_artifact_preflight(artifact, expected_capability_version)
    bound_params = bind_parameters(artifact, parameters)

    logger.log_run_start(
        goal=f"Replay of artifact '{goal_key}' (capability_version={artifact['capability_version']})",
        goal_key=goal_key,
        target_url=page.url,
        model="none (deterministic replay)",
        model_version=None,
    )

    for step in artifact["steps"]:
        result = _execute_one_step(
            page, logger, artifact, goal_key, step, bound_params, step["step_index"]
        )
        if result is not None:
            logger.log_run_end(
                outcome_type=result.outcome_type,
                outputs=result.outputs,
                checkpoint_verification=None,
                total_steps=result.total_steps,
            )
            return result

    total_steps = len(artifact["steps"])
    final_observation = build_observation(page)
    match = resolve_checkpoint(artifact, goal_key, final_observation)

    if match.outcome_type is None:
        # Every recorded step executed successfully, but the final page
        # still matches nothing in the checkpoint set - the same
        # unrecoverable-condition scenario as the mid-run drift/execution-
        # failure paths, just discovered only at the very end. Same real
        # handoff before finalizing as checkpoint_drift.
        handoff_match = _handle_unrecoverable(
            page, logger, goal_key, artifact, total_steps,
            trigger="replay_unrecoverable",
            context={"url": final_observation.url},
        )
        if handoff_match is not None:
            match = handoff_match
            try:
                final_observation = build_observation(page)
            except Exception:
                pass

    if match.outcome_type is not None:
        result = ReplayResult(
            outcome_type=match.outcome_type,
            outcome_label=match.outcome_label,
            outputs=match.outputs,
            total_steps=total_steps,
            run_id=logger.run_id,
        )
    else:
        result = ReplayResult(
            outcome_type="failed",
            outcome_label="checkpoint_drift",
            outputs=None,
            total_steps=total_steps,
            run_id=logger.run_id,
            failed_at_step=total_steps,
            expected="the artifact's success checkpoint or a known outcome",
            observed={
                "url": final_observation.url,
                "static_text_sample": final_observation.static_text[:5],
            },
        )

    logger.log_run_end(
        outcome_type=result.outcome_type,
        outputs=result.outputs,
        checkpoint_verification={"matched": match.outcome_type is not None},
        total_steps=total_steps,
    )
    return result
