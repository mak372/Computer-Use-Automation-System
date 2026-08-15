"""Action execution: role-validated, timeout-bounded Playwright calls.

Decision #3 (grounding is a stored Locator, not a re-query) + decision #4
(role validation before execution, structured results instead of raw
Playwright exceptions surfacing to the LLM).

Guardrail evaluation (agent.guardrails) and index/staleness resolution both
happen in the loop controller before this module is ever called -
execute_action assumes the caller has already decided this exact action is
permitted to run right now.
"""

from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from agent import config
from agent.perception import InteractiveElement

# Only click/type/select ever reach this function; finish/escalate have no
# target element at all and are handled by the loop controller directly.
_VALID_ROLES = {
    "click": {"button", "link"},
    "type": {"textbox"},
    "select": {"combobox"},
}


@dataclass
class ActionResult:
    success: bool
    # None on success; "invalid_action_for_element" | "missing_value" |
    # "timeout" | "execution_error" on failure.
    error_type: str | None = None
    message: str = ""


def execute_action(
    element: InteractiveElement, action: str, value: str | None = None
) -> ActionResult:
    valid_roles = _VALID_ROLES.get(action)
    if valid_roles is None:
        return ActionResult(
            success=False,
            error_type="invalid_action_for_element",
            message=f"unknown action type '{action}'",
        )

    if element.role not in valid_roles:
        return ActionResult(
            success=False,
            error_type="invalid_action_for_element",
            message=(
                f"action '{action}' is not valid for element role "
                f"'{element.role}' (index {element.index})"
            ),
        )

    if action in ("type", "select") and value is None:
        return ActionResult(
            success=False,
            error_type="missing_value",
            message=f"action '{action}' requires a value",
        )

    timeout_ms = config.ACTION_TIMEOUT_SECONDS * 1000
    try:
        if action == "click":
            element.locator.click(timeout=timeout_ms)
        elif action == "type":
            element.locator.fill(value, timeout=timeout_ms)
        elif action == "select":
            element.locator.select_option(value, timeout=timeout_ms)
        return ActionResult(success=True)
    except PlaywrightTimeoutError as exc:
        return ActionResult(success=False, error_type="timeout", message=str(exc))
    except PlaywrightError as exc:
        return ActionResult(success=False, error_type="execution_error", message=str(exc))
