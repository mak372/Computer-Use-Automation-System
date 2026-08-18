"""Known-dialog registry for native browser dialogs (window.confirm/alert/
prompt) - Section 3.3's "unexpected dialog" runtime condition.

Playwright's own default (no listener registered) is to silently
auto-dismiss any dialog. That's a real failure mode worth naming explicitly:
an action that legitimately needs a dialog accepted to proceed would just
silently not happen - the triggering click still "succeeds" (no exception),
the page simply never changes, and there is no evidence anywhere explaining
why. Registering a handler closes two separate gaps at once: a *known*
dialog gets actively accepted so the legitimate flow can proceed at all
(without this, that flow is structurally broken, not just unobserved), and
an *unknown* dialog's message/type get logged before it's dismissed, so a
resulting stall is debuggable instead of a silent no-op.

Same fail-closed discipline as interstitials.py: a dialog message matching
exactly one registered pattern for this goal is accepted (the browser-native
equivalent of "dismiss a known interstitial" - a recoverable condition, not
a guess). Zero or multiple matches means "not a dialog recognized as safe" -
dismissed, and the resulting unchanged page surfaces through the normal
grounding/checkpoint machinery as a hard failure with real diagnostic detail
attached, exactly like any other unrecognized state in this system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from playwright.sync_api import Dialog, Page

from agent.logger import RunLogger


@dataclass
class DialogSpec:
    message_pattern: str


REGISTRIES: dict[str, list[DialogSpec]] = {
    "lookup_balance": [],
    "open_sub_account": [],
    "withdraw_funds": [
        DialogSpec(
            message_pattern=r"^This member's record was recently flagged for review\."
        ),
    ],
}


def find_candidates(goal_key: str, message: str) -> list[DialogSpec]:
    return [spec for spec in REGISTRIES.get(goal_key, []) if re.match(spec.message_pattern, message)]


def install_dialog_handler(page: Page, goal_key: str, logger: RunLogger) -> None:
    """Registered once per run, before any action that could trigger a
    dialog - both run_discovery and run_replay call this as their first
    statement, since both already have page/goal_key/logger in hand."""

    def handler(dialog: Dialog) -> None:
        candidates = find_candidates(goal_key, dialog.message)
        if len(candidates) == 1:
            logger.log_dialog_event(
                dialog_type=dialog.type,
                message=dialog.message,
                resolution="accepted",
                matched_count=1,
            )
            dialog.accept()
        else:
            logger.log_dialog_event(
                dialog_type=dialog.type,
                message=dialog.message,
                resolution="dismissed",
                matched_count=len(candidates),
            )
            dialog.dismiss()

    page.on("dialog", handler)
