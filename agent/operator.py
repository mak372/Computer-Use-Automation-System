"""Mock human-operator surface (decision #6/#8 HITL implementation).

Section 3.6 explicitly allows mocking the operator UI as long as the
handoff mechanism and control-transfer model are real. Here, that means:
the harness pauses issuing Playwright commands, the browser window stays
open and interactive (non-headless) so a human genuinely could act on the
live session, and the human's decision is captured via a blocking terminal
prompt tied to this same run. There is no separate operator web console in
this build - the terminal prompt is the mock UI; the pause/resume/
control-transfer mechanics underneath it are real.
"""

from __future__ import annotations


def request_approval(
    action_description: str, amount: float | None, reason: str | None = None
) -> bool:
    print("\n=== HUMAN APPROVAL REQUIRED ===")
    print(f"Action: {action_description}")
    if amount is not None:
        print(f"Amount: ${amount:,.2f}")
    if reason:
        print(f"Reason: {reason}")
    print("The live browser session is open for inspection.")
    response = input("Approve this action? [y/n]: ").strip().lower()
    return response == "y"


def request_manual_handoff(trigger: str, context: dict) -> bool:
    """Returns True if the human indicates the goal was completed/handled
    manually, False if they abort the run."""
    print("\n=== AGENT ESCALATED - HUMAN HAND-OFF ===")
    print(f"Trigger: {trigger}")
    for key, value in context.items():
        print(f"{key}: {value}")
    print("The live browser session is open - you may take over and complete the goal manually.")
    response = input("Press Enter once done, or type 'abort': ").strip().lower()
    return response != "abort"
