"""Known-dialog registry for native browser dialogs (window.confirm/alert/
prompt)
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
