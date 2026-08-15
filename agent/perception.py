"""Perception: turn the live page into a structured Observation each step."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from playwright.sync_api import Locator, Page

# Roles exposed to the LLM as valid action targets. Anything else in the
# accessibility tree (headings, plain text, table wrappers) is read-only
# context - excluded structurally by never being indexed here, not by a
# runtime check later.
INTERACTIVE_ROLES = {"button", "link", "textbox", "combobox"}


@dataclass
class InteractiveElement:
    index: int
    role: str
    name: str
    nth: int
    locator: Locator
    preceding_context: str
    value: str | None
    # (method, path) for click-capable elements whose target is statically
    # resolvable; None if unresolvable (fail-closed in the guardrail layer)
    # or not applicable (textbox/combobox don't submit anything by themselves).
    effective_target: tuple[str, str] | None


@dataclass
class Observation:
    url: str
    static_text: list[str]
    interactive_elements: list[InteractiveElement]
    fingerprint: str
    has_duplicates: bool
    is_empty: bool


def _resolve_effective_target(locator: Locator, role: str) -> tuple[str, str] | None:
    if role == "link":
        href = locator.get_attribute("href")
        return ("GET", href) if href else None

    if role == "button":
        form = locator.evaluate(
            "el => { const f = el.closest('form'); "
            "return f ? {action: f.getAttribute('action') || '', "
            "method: (f.getAttribute('method') || 'GET').toUpperCase()} : null; }"
        )
        if not form:
            return None
        return (form["method"], form["action"])

    return None


def _walk(
    node: dict,
    page: Page,
    counts: dict[tuple[str, str], int],
    static_text: list[str],
    elements: list[InteractiveElement],
    last_static: list[str],
) -> None:
    role = node.get("role", "")
    name = (node.get("name") or "").strip()

    if role in INTERACTIVE_ROLES and name:
        key = (role, name)
        nth = counts.get(key, 0)
        counts[key] = nth + 1

        locator = page.get_by_role(role, name=name, exact=True).nth(nth)
        value = None
        if role in ("textbox", "combobox"):
            try:
                value = locator.input_value()
            except Exception:
                value = None

        elements.append(
            InteractiveElement(
                index=len(elements) + 1,
                role=role,
                name=name,
                nth=nth,
                locator=locator,
                preceding_context=last_static[-1] if last_static else "",
                value=value,
                effective_target=_resolve_effective_target(locator, role),
            )
        )
    elif name:
        static_text.append(name)
        last_static.append(name)

    for child in node.get("children") or []:
        _walk(child, page, counts, static_text, elements, last_static)


def build_observation(page: Page) -> Observation:
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass  # best-effort safety margin; the triggering action's own
        # auto-wait already covers the common case, including the `slow`
        # member's server-side delay

    snapshot = page.accessibility.snapshot() or {}

    static_text: list[str] = []
    elements: list[InteractiveElement] = []
    _walk(snapshot, page, {}, static_text, elements, [])

    fingerprint_parts = [
        page.url,
        hashlib.sha256("|".join(static_text).encode()).hexdigest(),
    ]
    fingerprint_parts += [f"{el.role}:{el.name}:{el.value}" for el in elements]
    fingerprint = hashlib.sha256("|".join(fingerprint_parts).encode()).hexdigest()

    role_name_counts: dict[tuple[str, str], int] = {}
    for el in elements:
        key = (el.role, el.name)
        role_name_counts[key] = role_name_counts.get(key, 0) + 1
    has_duplicates = any(c > 1 for c in role_name_counts.values())

    return Observation(
        url=page.url,
        static_text=static_text,
        interactive_elements=elements,
        fingerprint=fingerprint,
        has_duplicates=has_duplicates,
        is_empty=len(elements) == 0,
    )


def capture_screenshot(page: Page) -> bytes:
    return page.screenshot()
