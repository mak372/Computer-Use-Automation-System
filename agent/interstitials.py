"""Known-interstitial registry for deterministic replay"""

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
