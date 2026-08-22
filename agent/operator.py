"""Mock human-operator surface"""

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


def request_capability_selection(
    goal: str,
    valid_goal_keys: list[str],
    reason: str | None = None,
    screenshot_path: str | None = None,
) -> str | None:
    """Backup path for loop_controller.run_discovery's opening
    classify_goal_key call: if the LLM couldn't produce a trustworthy
    goal_key (call failed after retries, or it returned something outside
    checkpoints.REGISTRIES), ask a human to pick the capability directly
    rather than dead-ending the whole run. Returns the chosen goal_key, or
    None if the human aborts. Re-prompts (re-races) on anything that isn't
    exactly one of valid_goal_keys or 'abort' - a typo at the terminal
    should never silently fail this the way it would a one-shot prompt."""
    print("\n=== CAPABILITY SELECTION FAILED - HUMAN INPUT REQUIRED ===")
    print(f"Goal: {goal}")
    if reason:
        print(f"Reason: {reason}")
    if screenshot_path is not None:
        print(f"Screenshot: {screenshot_path}")
    print(f"Valid capabilities: {', '.join(valid_goal_keys)}")
    print("The live browser session is open for inspection.")
    print(f"Also open: http://127.0.0.1:{config.OPERATOR_UI_PORT} for the web console")

    pending = {
        "type": "capability_selection",
        "goal": goal,
        "valid_goal_keys": valid_goal_keys,
        "reason": reason,
        "screenshot_path": screenshot_path,
    }
    prompt = f"Enter one of [{', '.join(valid_goal_keys)}] or 'abort': "
    while True:
        _source, response = _race_terminal_and_web(prompt, pending)
        response = response.strip()
        if response.lower() == "abort":
            return None
        if response in valid_goal_keys:
            return response
        print(
            f"Invalid input {response!r} - must be exactly one of {valid_goal_keys} "
            "or 'abort'. Try again."
        )


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
