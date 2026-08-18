"""Mock human-operator surface (decision #6/#8 HITL implementation).

Section 3.6 explicitly allows mocking the operator UI as long as the
handoff mechanism and control-transfer model are real. Two surfaces are
live for every escalation: the original blocking terminal prompt, and a
small local web console (agent/operator_ui.py). Neither surface ever
touches the live Playwright `page` - the mechanism that's real is that the
harness pauses issuing Playwright commands, the browser window stays open
and interactive (non-headless) so a human genuinely can act on the live
session, and whichever surface the human answers first supplies the
decision that unblocks it.

The race is what keeps "who is in control" unambiguous despite two input
surfaces: both a terminal-reading thread and the web console are handed
the *same* single-slot queue, exactly one decision is ever consumed, and
the other surface's pending input is simply abandoned (the terminal thread
is a daemon thread; an extra stdin line or a stray button click after the
race is already decided is a harmless no-op, not a competing action). This
also means the web console is a genuinely optional convenience, not a new
point of failure: if operator_ui's Flask server fails to start for any
reason (port in use, etc.), the terminal path is completely unaffected and
this behaves exactly as it did before that module existed.
"""

from __future__ import annotations

import queue
import threading

from agent import config, operator_ui


def _read_terminal(prompt: str, decision_queue: "queue.Queue[tuple[str, str]]") -> None:
    try:
        response = input(prompt)
    except EOFError:
        # No terminal attached (or stdin closed) - just don't ever put
        # anything on the queue; the web console remains a live surface.
        return
    decision_queue.put(("terminal", response))


def _race_terminal_and_web(prompt: str, pending: dict) -> tuple[str, str]:
    decision_queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)
    threading.Thread(target=_read_terminal, args=(prompt, decision_queue), daemon=True).start()
    operator_ui.publish_and_wait(pending, decision_queue)
    try:
        return decision_queue.get()
    finally:
        operator_ui.clear_pending()


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
    print(f"Also open: http://127.0.0.1:{config.OPERATOR_UI_PORT} for the web console")

    pending = {
        "type": "approval",
        "goal_key": goal_key,
        "step_number": step_number,
        "action_description": action_description,
        "amount": amount,
        "reason": reason,
        "screenshot_path": screenshot_path,
    }
    _source, response = _race_terminal_and_web("Approve this action? [y/n]: ", pending)
    return response.strip().lower() in ("y", "approve")


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
    print(f"Also open: http://127.0.0.1:{config.OPERATOR_UI_PORT} for the web console")

    pending = {
        "type": "handoff",
        "goal_key": goal_key,
        "step_number": step_number,
        "trigger": trigger,
        "context": context,
        "screenshot_path": screenshot_path,
    }
    _source, response = _race_terminal_and_web(
        "Press Enter once done (automation will re-check state and may resume), "
        "or type 'abort': ",
        pending,
    )
    return response.strip().lower() != "abort"
