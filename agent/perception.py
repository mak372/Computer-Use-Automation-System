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
    # The underlying <input type="..."> attribute, textbox role only (None
    # for combobox/<textarea>, which have no equivalent). A more authoritative
    # source than shape-inferring a parameter's type from its one recorded
    # value (see artifact_builder.py) - the app itself already declares
    # "number" vs "text" here, so a draft type should be seeded from this
    # first and only fall back to shape-inference when it's absent.
    html_input_type: str | None


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


def resolve_element_target(role: str, locator: Locator) -> tuple[tuple[str, str] | None, str | None]:
    """Resolves (effective_target, html_input_type) for an already-located
    element, without a full accessibility-tree walk - used by the replay
    engine's fast relocation path (agent/replay_controller.py), which
    addresses a step's element directly via get_by_role(...).nth(n) rather
    than re-walking build_observation()'s full tree every step.

    Button/textbox resolution is identical to _handle_node's (calls the
    same _resolve_button_target / get_attribute("type")) - no duplicated
    logic, so the two paths can't silently drift apart for those roles.
    Link resolution here reads the raw `href` attribute directly instead of
    the walk's aria_snapshot-derived `/url` metadata; both are expected to
    agree for this app's plain server-rendered relative hrefs (no
    client-side URL rewriting), but that hasn't been proven identical by
    construction the way button/textbox resolution has - worth a real
    spot-check the first time replay exercises a link-type step.
    """
    html_input_type = None
    if role == "textbox":
        try:
            html_input_type = locator.get_attribute("type")
        except Exception:
            html_input_type = None

    if role == "link":
        try:
            href = locator.get_attribute("href")
        except Exception:
            href = None
        effective_target = ("GET", href) if href else None
    elif role == "button":
        effective_target = _resolve_button_target(locator)
    else:
        effective_target = None

    return effective_target, html_input_type


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

        html_input_type = None
        if role == "textbox":
            try:
                html_input_type = locator.get_attribute("type")
            except Exception:
                html_input_type = None

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
                html_input_type=html_input_type,
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


# Label text (as rendered in target_app's own td.lbl2/td.lbl3 cells) that
# marks the adjacent value cell as regulated financial data / PII. Matched
# case-insensitively against the label's own text, not the value - target_app
# is a deliberately legacy/non-semantic surface (Section 1: "no test IDs"),
# so there's no data-sensitive attribute to select on directly. This mirrors
# how a human reviewer would find these fields: by reading the label next to
# them, not by a selector a real legacy vendor app would never provide.
_SENSITIVE_LABELS = {
    "name", "member", "member id", "balance", "available balance",
    "amount", "initial deposit", "email", "phone",
}

# Mirrors logger._REDACT_PATTERNS's phone-/email-shaped regexes. Needed as a
# second, independent signal because not every sensitive value sits in a
# label/value row at all - withdraw_confirm.html has a hardcoded
# "Phone on file: 555-0142" string under a "Contact Verification" section
# header, with no "Phone" label anywhere near it for signal 1 to catch.
_SENSITIVE_CONTENT_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{4}\b"),  # phone-shaped
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email-shaped
]


def _sensitive_mask_locators(page: Page) -> list[Locator]:
    """Locators for every on-screen region that should be blacked out before
    a screenshot is saved. Three independent signals:

    1. A td.lbl2/td.lbl3 label whose text is in _SENSITIVE_LABELS -> mask its
       sibling td.val2/td.val3 value cell (member detail, withdraw/sub-account
       forms and confirmation screens all use this label/value row shape).
    2. Any td.msg1 success banner - these interpolate a dollar figure directly
       into a sentence ("Withdrawal of $X completed... Remaining balance: $Y"),
       so the sensitive figure can't be isolated from the surrounding text
       without a template change; the whole banner is masked instead.
    3. Any td.val2/td.val3 cell whose own text is phone-/email-shaped,
       regardless of what label (if any) precedes it - catches values with
       no meaningful adjacent label, which signal 1 can't see by construction.

    Returns only locators that actually resolved to >=1 element on the
    current page, since Page.screenshot(mask=...) errors on a locator with
    zero matches.
    """
    masks: list[Locator] = []
    labels = page.locator("td.lbl2, td.lbl3")
    for i in range(labels.count()):
        cell = labels.nth(i)
        if cell.inner_text().strip().lower() in _SENSITIVE_LABELS:
            sibling = cell.locator("xpath=following-sibling::td[1]")
            if sibling.count() > 0:
                masks.append(sibling)
    banner = page.locator("td.msg1")
    if banner.count() > 0:
        masks.append(banner)
    values = page.locator("td.val2, td.val3")
    for i in range(values.count()):
        cell = values.nth(i)
        text = cell.inner_text()
        if any(pattern.search(text) for pattern in _SENSITIVE_CONTENT_PATTERNS):
            masks.append(cell)
    return masks


def capture_screenshot(page: Page) -> bytes:
    masks = _sensitive_mask_locators(page)
    if masks:
        return page.screenshot(mask=masks, mask_color="#000000")
    return page.screenshot()


def diff_observations(pre: "Observation | None", post: "Observation | None") -> dict:
    """A before/after account of what changed across a human handoff window
    (loop_controller._handle_escalation / replay_controller._handle_
    unrecoverable), Section 3.6's "record what the human did" - the
    minimal-but-real version of it. Deliberately a state diff, not an
    action log: literally recording individual clicks/keystrokes would mean
    injecting live event listeners into the page, which is functionally the
    same thing as the "full real-time co-browsing operator console"
    Section 3.6 explicitly puts out of scope. This stays a single before/
    after comparison computed at resolve time - a real, inspectable answer
    to "what did the human do" inferred from net effect on page state, not
    a stream of anything.

    `pre`/`post` are None when perception itself failed on either side
    (e.g. the page broke); that's reported as `comparable: False` rather
    than guessed at, same "don't assume, report what's actually known"
    principle as everywhere else in this system."""
    if pre is None or post is None:
        return {"comparable": False}

    pre_text = set(pre.static_text)
    post_text = set(post.static_text)
    pre_elements = {(el.role, el.name) for el in pre.interactive_elements}
    post_elements = {(el.role, el.name) for el in post.interactive_elements}

    return {
        "comparable": True,
        "changed": pre.fingerprint != post.fingerprint,
        "url_before": pre.url,
        "url_after": post.url,
        "url_changed": pre.url != post.url,
        "text_added": sorted(post_text - pre_text),
        "text_removed": sorted(pre_text - post_text),
        "elements_added": sorted(f'{role} "{name}"' for role, name in (post_elements - pre_elements)),
        "elements_removed": sorted(f'{role} "{name}"' for role, name in (pre_elements - post_elements)),
    }
