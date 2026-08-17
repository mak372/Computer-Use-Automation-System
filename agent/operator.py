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
    action_description: str,
    amount: float | None,
    reason: str | None = None,
    goal_key: str | None = None,
    step_number: int | None = None,
    screenshot_path: str | None = None,
) -> bool:
    print("\n=== HUMAN APPROVAL REQUIRED ===")
    if goal_key is not None:
        print(f"Capability/Goal: {goal_key}")
    if step_number is not None:
        print(f"Step: {step_number}")
    print(f"Action: {action_description}")
    if amount is not None:
        print(f"Amount: ${amount:,.2f}")
    if reason:
        print(f"Reason: {reason}")
    if screenshot_path is not None:
        print(f"Screenshot: {screenshot_path}")
    print("The live browser session is open for inspection.")
    response = input("Approve this action? [y/n]: ").strip().lower()
    return response == "y"


def request_manual_handoff(
    trigger: str,
    context: dict,
    goal_key: str | None = None,
    step_number: int | None = None,
    screenshot_path: str | None = None,
) -> bool:
    """Returns True if the human indicates the goal was completed/handled
    manually and automation should resume/re-check, False if they abort
    the run outright."""
    print("\n=== AGENT ESCALATED - HUMAN HAND-OFF ===")
    if goal_key is not None:
        print(f"Capability/Goal: {goal_key}")
    if step_number is not None:
        print(f"Step: {step_number}")
    print(f"Trigger: {trigger}")
    for key, value in context.items():
        print(f"{key}: {value}")
    if screenshot_path is not None:
        print(f"Screenshot (state at handoff): {screenshot_path}")
    print("The live browser session is open - you may take over and complete the goal manually.")
    response = input(
        "Press Enter once done (automation will re-check state and may resume), "
        "or type 'abort': "
    ).strip().lower()
    return response != "abort"
