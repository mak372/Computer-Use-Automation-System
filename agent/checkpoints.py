"""Checkpoint / outcome verification (decision #7).

Never trust the LLM's self-report of `finish`. Every claimed outcome is
checked against a hand-authored, per-goal registry of (URL pattern, static
text signature) - the terminal states of target_app are fully known since
we wrote it. For a success outcome specifically, the referenced output
values are also re-extracted from the actual observation text (via known,
exact-template regexes) rather than trusted from the LLM's own transcription.

This same signature registry is what the artifact's checkpoint (Section
3.2/3.3) will reuse for deterministic replay verification later.

Deliberately excluded from every registry: form validation errors
(recoverable - the agent corrects and retries, never a finish call), the
similar-account interstitial (recoverable, agent clicks through it), and
the "broken page" / system-error state (a hard failure the agent should
escalate on, not report as a completed goal - leaving it unregistered
means a finish() claiming it gets rejected, enforcing the business-outcome-
vs-hard-failure distinction structurally rather than only via prompting).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

from agent.perception import Observation


def _extract_balance(refs: list[str]) -> dict:
    if not refs:
        raise ValueError("success outcome requires at least one output_ref")
    match = re.match(r"^\$([\d,]+\.\d{2})$", refs[0].strip())
    if not match:
        raise ValueError(f"referenced text is not a dollar amount: {refs[0]!r}")
    return {"balance": float(match.group(1).replace(",", ""))}


def _extract_sub_account_created(refs: list[str]) -> dict:
    if not refs:
        raise ValueError("success outcome requires at least one output_ref")
    match = re.search(
        r'Sub-account "([^"]+)" \(([^)]+)\) created successfully '
        r"with an initial deposit of \$([\d,]+\.\d{2})",
        refs[0],
    )
    if not match:
        raise ValueError(
            f"referenced text does not match sub-account confirmation shape: {refs[0]!r}"
        )
    return {
        "nickname": match.group(1),
        "account_type": match.group(2),
        "deposit_amount": float(match.group(3).replace(",", "")),
    }


def _extract_withdrawal_result(refs: list[str]) -> dict:
    if not refs:
        raise ValueError("success outcome requires at least one output_ref")
    match = re.search(r"Remaining balance: \$([\d,]+\.\d{2})", refs[0])
    if not match:
        raise ValueError(
            f"referenced text does not match withdrawal confirmation shape: {refs[0]!r}"
        )
    return {"remaining_balance": float(match.group(1).replace(",", ""))}


@dataclass
class OutcomeSpec:
    url_pattern: str
    text_signature: str
    # Only for outcomes that carry outputs (success). Takes the raw
    # referenced static-text strings, returns typed outputs, or raises
    # ValueError if the text doesn't actually match the expected shape.
    output_extractor: Optional[Callable[[list[str]], dict]] = None


REGISTRIES: dict[str, dict[str, OutcomeSpec]] = {
    "lookup_balance": {
        "success": OutcomeSpec(r"^/member/[^/]+$", "Balance", _extract_balance),
        "member_not_found": OutcomeSpec(r"^/member/[^/]+$", "No member found with ID"),
        "restricted": OutcomeSpec(r"^/member/[^/]+$", "This account is restricted"),
        "session_expired": OutcomeSpec(r"^/member/[^/]+$", "Your session has expired"),
    },
    "open_sub_account": {
        "success": OutcomeSpec(
            r"^/member/[^/]+/sub-account/confirm$",
            "created successfully",
            _extract_sub_account_created,
        ),
        "member_not_found": OutcomeSpec(r"^/member/[^/]+$", "No member found with ID"),
        "restricted": OutcomeSpec(r"^/member/[^/]+$", "This account is restricted"),
        "session_expired": OutcomeSpec(r"^/member/[^/]+$", "Your session has expired"),
        "max_sub_accounts_reached": OutcomeSpec(
            r"^/member/[^/]+/sub-account/new$",
            "already has the maximum number of sub-accounts",
        ),
    },
    "withdraw_funds": {
        "success": OutcomeSpec(
            r"^/member/[^/]+/withdraw/confirm$",
            "completed. Remaining balance",
            _extract_withdrawal_result,
        ),
        "member_not_found": OutcomeSpec(r"^/member/[^/]+$", "No member found with ID"),
        "restricted": OutcomeSpec(r"^/member/[^/]+$", "This account is restricted"),
        "session_expired": OutcomeSpec(r"^/member/[^/]+$", "Your session has expired"),
    },
}


@dataclass
class VerificationResult:
    verified: bool
    reason: str
    outputs: Optional[dict] = None


def verify_finish(
    goal_key: str,
    claimed_outcome: str,
    output_refs: list,  # nominally list[int]; may contain str items - see coercion below
    observation: Observation,
    url: str,
) -> VerificationResult:
    registry = REGISTRIES.get(goal_key)
    if registry is None:
        return VerificationResult(verified=False, reason=f"unknown goal '{goal_key}'")

    spec = registry.get(claimed_outcome)
    if spec is None:
        return VerificationResult(
            verified=False,
            reason=f"'{claimed_outcome}' is not a recognized outcome for this goal",
        )

    # Defensive: accept either a full URL or a bare path so this doesn't
    # depend on every caller remembering to strip the origin first.
    path = urlparse(url).path
    if not re.match(spec.url_pattern, path):
        return VerificationResult(
            verified=False,
            reason=f"URL path '{path}' does not match expected pattern for '{claimed_outcome}'",
        )

    if not any(spec.text_signature in text for text in observation.static_text):
        return VerificationResult(
            verified=False,
            reason=f"page content does not contain the expected signature for '{claimed_outcome}'",
        )

    if spec.output_extractor is None:
        return VerificationResult(verified=True, reason="checkpoint matched", outputs=None)

    referenced_texts = []
    for ref in output_refs:
        # Defensive: Gemini's function-calling response has been observed
        # returning array items as strings (e.g. "2") even though the tool
        # schema declares them as integers - not something callable from
        # this side of the API, so coerce rather than trust the type.
        try:
            ref_int = int(ref)
        except (TypeError, ValueError):
            return VerificationResult(verified=False, reason=f"output_ref {ref!r} is not an integer")
        if ref_int < 1 or ref_int > len(observation.static_text):
            return VerificationResult(verified=False, reason=f"output_ref {ref_int} is out of range")
        referenced_texts.append(observation.static_text[ref_int - 1])

    try:
        outputs = spec.output_extractor(referenced_texts)
    except ValueError as exc:
        return VerificationResult(verified=False, reason=str(exc))

    return VerificationResult(verified=True, reason="checkpoint and outputs verified", outputs=outputs)
