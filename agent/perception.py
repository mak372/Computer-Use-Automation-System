"""Perception: turn the live page into a structured Observation each step."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import yaml
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


# playwright.sync_api.Page has no `accessibility` attribute as of
# Playwright 1.62 (that API was removed, not just deprecated). The
# documented replacement is Locator.aria_snapshot(), which returns a YAML
# string rather than a nested dict - parsed below. Verified against the
# real target_app: <td> cells separate into individual leaf nodes (so
# label/value stay distinct static_text entries), <select> reports role
# "combobox", and duplicate "Confirm" buttons resolve via get_by_role(...)
# .nth() in the same order they appear in this YAML, confirming the
# grounding approach from decision #3 still holds against this API.
_KEY_PATTERN = re.compile(r'^(\S+)(?:\s+"(.*)")?$')


def _parse_key(key: str) -> tuple[str, str | None]:
    """'role "name"' -> (role, name); 'role' (no name) -> (role, None).

    The `"name"` here is Playwright's own compact notation, not real YAML
    quoting - aria_snapshot() backslash-escapes any quote characters inside
    the name (e.g. a cell's accessible name Sub-account "X" becomes
    cell "Sub-account \"X\" ..." in the raw snapshot text), but since it's
    a plain YAML scalar, yaml.safe_load() has no quoting to undo. Confirmed
    via a direct aria_snapshot() test against a cell containing embedded
    quotes - without this unescape, exact-template checkpoint regexes in
    checkpoints.py silently stop matching on any page whose text embeds a
    quoted value (e.g. the sub-account/withdraw confirmation messages).
    """
    match = _KEY_PATTERN.match(key)
    if not match:
        return key, None
    role, name = match.group(1), match.group(2)
    if name is not None:
        name = re.sub(r'\\(.)', r"\1", name)
    return role, name


def _resolve_button_target(locator: Locator) -> tuple[str, str] | None:
    form = locator.evaluate(
        "el => { const f = el.closest('form'); "
        "return f ? {action: f.getAttribute('action') || '', "
        "method: (f.getAttribute('method') || 'GET').toUpperCase()} : null; }"
    )
    if not form:
        return None
    return (form["method"], form["action"])


def _split_children(children: list) -> tuple[dict[str, str], list]:
    """Separates a node's YAML children into (metadata, real_child_nodes).
    Metadata is Playwright's own convention for keys like "/url" (a link's
    href) - not a genuine child element to recurse into."""
    metadata: dict[str, str] = {}
    real_children: list = []
    for child in children:
        if isinstance(child, dict) and len(child) == 1:
            key, value = next(iter(child.items()))
            if key.startswith("/"):
                metadata[key[1:]] = value
                continue
        real_children.append(child)
    return metadata, real_children


def _handle_node(
    role: str,
    name: str | None,
    children: list | None,
    page: Page,
    counts: dict[tuple[str, str], int],
    static_text: list[str],
    elements: list[InteractiveElement],
    last_static: list[str],
) -> None:
    if role == "text":
        # Handled by the caller (dict key "text" carries the string value
        # directly, not a nested node) - should not reach here.
        return

    metadata, real_children = _split_children(children or [])

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

        if role == "link":
            href = metadata.get("url")
            effective_target = ("GET", href) if href else None
        elif role == "button":
            effective_target = _resolve_button_target(locator)
        else:
            effective_target = None

        elements.append(
            InteractiveElement(
                index=len(elements) + 1,
                role=role,
                name=name,
                nth=nth,
                locator=locator,
                preceding_context=last_static[-1] if last_static else "",
                value=value,
                effective_target=effective_target,
            )
        )
        # Deliberately not recursing into an interactive node's own
        # children (e.g. <select> option nodes) - not independently
        # actionable or meaningful as standalone context.
        return

    if not real_children and name:
        # Leaf, non-interactive (e.g. a <td> cell's own text) - static context.
        static_text.append(name)
        last_static.append(name)
        return

    # Container node (table/rowgroup/row/cell wrapping other elements) -
    # its own bracketed name is just an aggregation of its children's
    # text, not new content, so only the children get walked.
    _walk_children(real_children, page, counts, static_text, elements, last_static)


def _walk_children(
    children: list,
    page: Page,
    counts: dict[tuple[str, str], int],
    static_text: list[str],
    elements: list[InteractiveElement],
    last_static: list[str],
) -> None:
    for item in children:
        if isinstance(item, str):
            role, name = _parse_key(item)
            _handle_node(role, name, None, page, counts, static_text, elements, last_static)
        elif isinstance(item, dict):
            for key, value in item.items():
                if key == "text":
                    if value:
                        static_text.append(value)
                        last_static.append(value)
                    continue
                if key.startswith("/"):
                    continue  # metadata belonging to a sibling, already consumed there
                role, name = _parse_key(key)
                child_list = value if isinstance(value, list) else None
                _handle_node(role, name, child_list, page, counts, static_text, elements, last_static)


def build_observation(page: Page) -> Observation:
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass  # best-effort safety margin; the triggering action's own
        # auto-wait already covers the common case, including the `slow`
        # member's server-side delay

    snapshot_yaml = page.locator("body").aria_snapshot()
    tree = yaml.safe_load(snapshot_yaml) or []

    static_text: list[str] = []
    elements: list[InteractiveElement] = []
    _walk_children(tree, page, {}, static_text, elements, [])

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
