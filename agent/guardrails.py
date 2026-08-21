"""Guardrail enforcement: allowlist + risk tiering + amount provenance.

Layer 2 risk tiering was originally amount-size-only (amount >= threshold
-> hitl_required). A real discovery run surfaced a gap that pure size
tiering can't catch: a goal that never actually states a dollar amount at
all, where the LLM types a small guessed placeholder instead of
escalating - a guessed $100 would sail through auto-approve untouched,
even though its provenance is exactly as unverified as an untraceable
$50,000 would be. Layer 2a below closes that: an amount is only eligible
for auto-approval if it can be traced back to a figure the goal itself
declared, regardless of size. An untraceable amount is routed to
hitl_required rather than blocked outright, since a human may well glance
at it and clear it in seconds - the same review lane the size threshold
already uses, not a new terminal failure mode. See REPORT.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from agent import config
from agent.perception import InteractiveElement

# Requires the literal `$` anchor before any digits count as a "declared
# amount" - without it, this would also match unrelated digit runs in the
# goal text (a member ID like "M-1007" contains "1007", indistinguishable
# from a genuine amount declaration without the anchor). Tolerant of
# commas/decimals after the anchor ("$15,000", "$500.00"). A goal that
# states an amount in words or as a bare number without `$` will not be
# recognized - deliberately fails toward escalation, not a silent false
# match.
_DECLARED_AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")


@dataclass
class GuardrailDecision:
    # "blocked" | "not_applicable" | "auto_approved" | "hitl_required"
    decision: str
    reason: str
    amount: float | None = None


def _route_allowed(method: str, path: str) -> bool:
    for pattern, methods in config.ALLOWED_ROUTES:
        if method in methods and re.match(pattern, path):
            return True
    return False


def _amount_bearing_field(method: str, path: str) -> str | None:
    if method != "POST":
        return None
    for pattern, field_name in config.AMOUNT_BEARING_ROUTES:
        if re.match(pattern, path):
            return field_name
    return None


def _extract_declared_amounts(goal: str) -> set[float]:
    # Recomputed per call rather than cached - the goal string is short
    # and constant for the whole run, not worth adding state for.
    return {float(m.replace(",", "")) for m in _DECLARED_AMOUNT_RE.findall(goal)}


def _amount_declared_in_goal(goal: str, amount: float) -> bool:
    return any(abs(amount - declared) < 0.01 for declared in _extract_declared_amounts(goal))


def evaluate_action(
    element: InteractiveElement, page: Page, goal: str | None = None
) -> GuardrailDecision:
    # type/select never submit anything by themselves - they don't have an
    # effective_target at all, so there's nothing for the allowlist or
    # risk tier to evaluate.
    if element.role in ("textbox", "combobox"):
        return GuardrailDecision(decision="not_applicable", reason="not a submitting action")

    # Layer 1: allowlist, checked first, true short-circuit - risk tiering
    # below is never reached for a blocked target.
    if element.effective_target is None:
        return GuardrailDecision(decision="blocked", reason="no resolvable target (fail closed)")

    method, path = element.effective_target
    if not _route_allowed(method, path):
        return GuardrailDecision(decision="blocked", reason=f"{method} {path} not on allowlist")

    # Layer 2: risk tiering - only the two final "Confirm" actions carry a
    # committed dollar amount; everything else on the allowlist proceeds
    # with no amount check at all.
    field_name = _amount_bearing_field(method, path)
    if field_name is None:
        return GuardrailDecision(decision="not_applicable", reason="allowed, no amount involved")

    # Ground truth, not the LLM's earlier `type` argument - read the live
    # hidden field that will actually be submitted on this click. Wrapped
    # like every other live Playwright read on this system's failure paths
    # (action_executor.execute_action, replay_controller.
    # capture_diagnostic_snapshot) - a broken/half-loaded page (the amount-
    # bearing route matched, but the field itself never rendered, or the
    # app died mid-response) must fail closed here too, not crash the run
    # with an uncaught exception and no run_end logged.
    try:
        amount_str = page.locator(f'input[name="{field_name}"]').input_value()
    except (PlaywrightTimeoutError, PlaywrightError) as exc:
        return GuardrailDecision(
            decision="blocked",
            reason=f"could not read live amount field (fail closed): {exc}",
        )

    try:
        amount = float(amount_str)
    except (TypeError, ValueError):
        return GuardrailDecision(
            decision="blocked", reason=f"malformed amount value {amount_str!r} (fail closed)"
        )

    # Layer 2a: provenance. `goal` is discovery-only (None during replay -
    # replay's `bound_params` are already structured, caller-supplied,
    # type-validated values, a stronger provenance guarantee than parsing
    # free text, so this check would be redundant there).
    if goal is not None and not _amount_declared_in_goal(goal, amount):
        return GuardrailDecision(
            decision="hitl_required",
            reason=f"amount ${amount:,.2f} not found in the stated goal (fail closed to human review)",
            amount=amount,
        )

    # Layer 2b: size threshold.
    if amount < config.RISK_TIER_HITL_THRESHOLD:
        return GuardrailDecision(
            decision="auto_approved", reason="within auto-approve threshold", amount=amount
        )

    return GuardrailDecision(
        decision="hitl_required",
        reason=(
            f"amount ${amount:,.2f} meets or exceeds the "
            f"${config.RISK_TIER_HITL_THRESHOLD:,.2f} auto-approve threshold"
        ),
        amount=amount,
    )
