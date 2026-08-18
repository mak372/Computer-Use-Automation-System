"""Known-interstitial registry for deterministic replay (Section 3.3): a
data-dependent confirmation page inserted between two steps of an otherwise
fixed flow (e.g. "this member already has a similar account, proceed?").
Not a business outcome (checkpoints.py deliberately excludes it - discovery's
LLM just clicks through it like any other page) and not a terminal state -
purely a recoverable detour, checked only when a step's recorded grounding
has already failed to resolve.

Matching follows the same fail-closed discipline as output extraction in
replay_controller.py (_extract_outputs_from_observation): zero matches and
multiple matches are both treated as "not recognized," never guessed at - an
ambiguous match between two registered interstitials is exactly the kind of
silent-wrong-choice this system refuses to make anywhere else. Not a live
case today (open_sub_account has exactly one registered entry), but cheap to
hold to before a second one is ever added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from agent.perception import Observation


@dataclass
class InterstitialSpec:
    url_pattern: str
    text_signature: str
    dismiss_role: str
    dismiss_name: str


REGISTRIES: dict[str, list[InterstitialSpec]] = {
    "open_sub_account": [
        InterstitialSpec(
            url_pattern=r"^/member/[^/]+/sub-account/new$",
            text_signature="This member already has a similar account",
            dismiss_role="button",
            dismiss_name="Proceed",
        ),
    ],
    "withdraw_funds": [],
    "lookup_balance": [],
}


def find_candidates(goal_key: str, observation: Observation) -> list[InterstitialSpec]:
    path = urlparse(observation.url).path
    return [
        spec
        for spec in REGISTRIES.get(goal_key, [])
        if re.match(spec.url_pattern, path)
        and any(spec.text_signature in text for text in observation.static_text)
    ]
