"""Guardrail enforcement: allowlist + risk tiering"""

from __future__ import annotations

import re
from dataclasses import dataclass

from playwright.sync_api import Page

from agent import config
from agent.perception import InteractiveElement


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


def evaluate_action(element: InteractiveElement, page: Page) -> GuardrailDecision:
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
    # hidden field that will actually be submitted on this click.
    amount_str = page.locator(f'input[name="{field_name}"]').input_value()
    try:
        amount = float(amount_str)
    except (TypeError, ValueError):
        return GuardrailDecision(
            decision="blocked", reason=f"malformed amount value {amount_str!r} (fail closed)"
        )

    if amount < config.RISK_TIER_HITL_THRESHOLD:
        return GuardrailDecision(
            decision="auto_approved", reason="within auto-approve threshold", amount=amount
        )

    return GuardrailDecision(
        decision="hitl_required", reason="amount requires human approval", amount=amount
    )
