"""Gemini wrapper: tool schema, prompt construction, retry policy.

Decisions #1 (Gemini 2.5 Flash), #4 (action schema - every LLM tool call
carries a required `reasoning` field), #5 (system instruction set once,
compact text-only history + full current observation, single-turn calls
built fresh from run-log state rather than an accumulating SDK chat
object), #6 (retry-once on timeout), #9 (model_version captured for
run_start, failed attempts surfaced for llm_call_attempt_failed logging).

Interactive elements are numbered [1], [2]... (valid click/type/select
targets). Static text is numbered separately as [S1], [S2]... - a distinct
namespace used only by `finish` to reference where an output value was
read from, so it can never be confused with an action target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from agent import config
from agent.perception import Observation


class LLMCallFailed(Exception):
    def __init__(self, message: str, failed_attempts: list[dict]):
        super().__init__(message)
        self.failed_attempts = failed_attempts


class NoToolCallReturned(Exception):
    """Model responded with text instead of calling a tool."""

    pass


@dataclass
class LLMDecision:
    action: str  # "click" | "type" | "select" | "finish" | "escalate"
    args: dict
    attempt_count: int
    failed_attempts: list[dict] = field(default_factory=list)
    model_version: str | None = None


_CLICK_SCHEMA = {
    "name": "click",
    "description": "Click a button or link from the current numbered interactive elements list.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Element number from the current step's interactive elements list."},
            "reasoning": {"type": "string", "description": "Brief justification for this action."},
        },
        "required": ["index", "reasoning"],
    },
}

_TYPE_SCHEMA = {
    "name": "type",
    "description": "Fill a text input from the current numbered interactive elements list.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Element number from the current step's interactive elements list."},
            "value": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["index", "value", "reasoning"],
    },
}

_SELECT_SCHEMA = {
    "name": "select",
    "description": "Choose an option from a dropdown from the current numbered interactive elements list.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Element number from the current step's interactive elements list."},
            "value": {"type": "string", "description": "The option's value attribute to select."},
            "reasoning": {"type": "string"},
        },
        "required": ["index", "value", "reasoning"],
    },
}

_FINISH_SCHEMA = {
    "name": "finish",
    "description": (
        "Declare that the application produced a definitive terminal result for the goal - "
        "either true success or a legitimate business outcome (member not found, restricted, "
        "limit reached, validation error treated as terminal, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "description": "Label for the terminal state, e.g. 'success', 'member_not_found', 'restricted', 'max_sub_accounts_reached'.",
            },
            "output_refs": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Only for a success outcome: the S-numbers (e.g. 2 for [S2]) of the page "
                    "content entries containing the actual output values. Never retype a value "
                    "yourself - point at where it was read. Omit or leave empty for non-success outcomes."
                ),
            },
            "reasoning": {"type": "string"},
        },
        "required": ["outcome", "reasoning"],
    },
}

_ESCALATE_SCHEMA = {
    "name": "escalate",
    "description": "Request human hand-off because you cannot safely or confidently continue on your own.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
        },
        "required": ["reasoning"],
    },
}

_TOOLS = [
    types.Tool(
        function_declarations=[
            _CLICK_SCHEMA,
            _TYPE_SCHEMA,
            _SELECT_SCHEMA,
            _FINISH_SCHEMA,
            _ESCALATE_SCHEMA,
        ]
    )
]


def build_system_instruction(goal: str) -> str:
    return (
        f"You are operating a web application to accomplish this goal: {goal}\n\n"
        "Rules:\n"
        "- You may only click/type/select elements from the numbered interactive "
        "elements list shown for the CURRENT step. Numbers from earlier steps are "
        "no longer valid once the page has changed.\n"
        "- Lines under \"Page content\" (labeled [S1], [S2], ...) are read-only - "
        "you cannot click or type into them. Use them to understand the page and "
        "to identify where output values live.\n"
        "- If two elements share the same role and label (e.g. two buttons both "
        "named \"Confirm\"), a \"(near: ...)\" hint after the label shows nearby "
        "page text to tell them apart - use it to pick the one matching your "
        "intended action rather than escalating on ambiguity you can resolve.\n"
        "- Call finish once the application has produced a definitive result - this "
        "includes real success AND legitimate business outcomes you observe (member "
        "not found, restricted, limit reached, insufficient funds, etc.). For a "
        "success outcome, "
        "reference the [S#] entries containing the output values via output_refs - "
        "never retype a value from memory.\n"
        "- Call escalate only if you cannot safely or confidently continue - this "
        "hands off to a human. Prefer escalating over guessing.\n"
        "- Always include a brief reasoning string explaining your choice.\n"
    )


def render_observation(observation: Observation) -> str:
    lines = ["Interactive elements:"]
    if observation.interactive_elements:
        # Elements sharing a (role, name) are indistinguishable from the
        # label alone (e.g. two "Confirm" buttons on the same page) - only
        # for those, surface the nearby page text captured at perception
        # time so the model has something to disambiguate by, instead of
        # having to guess or escalate on every duplicate.
        name_counts: dict[tuple[str, str], int] = {}
        for el in observation.interactive_elements:
            key = (el.role, el.name)
            name_counts[key] = name_counts.get(key, 0) + 1

        for el in observation.interactive_elements:
            line = f'[{el.index}] {el.role} "{el.name}"'
            if name_counts[(el.role, el.name)] > 1 and el.preceding_context:
                line += f' (near: "{el.preceding_context}")'
            lines.append(line)
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("Page content (read-only):")
    if observation.static_text:
        for i, text in enumerate(observation.static_text, start=1):
            lines.append(f"[S{i}] {text}")
    else:
        lines.append("(none)")

    return "\n".join(lines)


def render_history(history_lines: list[str]) -> str:
    if not history_lines:
        return "(no actions taken yet)"
    return "\n".join(history_lines)


class LLMClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._recent_call_times: list[float] = []

    def _throttle(self) -> None:
        """Proactively paces calls to stay under the measured free-tier
        quota (config.LLM_MAX_REQUESTS_PER_MINUTE), rather than only
        reacting after a 429. Applies to every real API call - a fresh
        attempt or a retry both consume the same quota."""
        now = time.monotonic()
        self._recent_call_times = [t for t in self._recent_call_times if now - t < 60]
        if len(self._recent_call_times) >= config.LLM_MAX_REQUESTS_PER_MINUTE:
            wait_for = 60 - (now - self._recent_call_times[0]) + 0.5
            if wait_for > 0:
                time.sleep(wait_for)
            now = time.monotonic()
            self._recent_call_times = [t for t in self._recent_call_times if now - t < 60]
        self._recent_call_times.append(time.monotonic())

    def decide(
        self,
        system_instruction: str,
        history_text: str,
        observation_text: str,
        screenshot: bytes | None,
    ) -> LLMDecision:
        failed_attempts: list[dict] = []
        last_failure_type: str | None = None

        for attempt in range(1, config.LLM_CALL_MAX_RETRIES + 2):
            start = time.monotonic()
            try:
                response = self._call_gemini(
                    system_instruction,
                    history_text,
                    observation_text,
                    screenshot,
                    corrective_nudge=(last_failure_type == "no_tool_call"),
                )
                action, args, model_version = _parse_tool_call(response)
                return LLMDecision(
                    action=action,
                    args=args,
                    attempt_count=attempt,
                    failed_attempts=failed_attempts,
                    model_version=model_version,
                )
            except NoToolCallReturned:
                duration_ms = int((time.monotonic() - start) * 1000)
                failed_attempts.append(
                    {"attempt": attempt, "result": "no_tool_call", "duration_ms": duration_ms}
                )
                last_failure_type = "no_tool_call"
            except Exception as exc:  # network/timeout/malformed response - decision #6
                duration_ms = int((time.monotonic() - start) * 1000)
                failed_attempts.append(
                    {"attempt": attempt, "result": str(exc), "duration_ms": duration_ms}
                )
                last_failure_type = "infra_error"
                if attempt < config.LLM_CALL_MAX_RETRIES + 1:
                    # A retry is coming - give a transient capacity/quota
                    # blip a short chance to clear before trying again.
                    time.sleep(config.LLM_INFRA_RETRY_DELAY_SECONDS)

        raise LLMCallFailed("LLM call failed after retries", failed_attempts)

    def _call_gemini(
        self,
        system_instruction: str,
        history_text: str,
        observation_text: str,
        screenshot: bytes | None,
        corrective_nudge: bool = False,
    ):
        user_text = f"History so far:\n{history_text}\n\nCurrent step:\n{observation_text}"
        if corrective_nudge:
            user_text += (
                "\n\nYour previous response did not call one of the provided tools. "
                "You must respond by calling exactly one tool (click, type, select, "
                "finish, or escalate) - do not respond with plain text."
            )

        parts = [types.Part.from_text(text=user_text)]
        if screenshot is not None:
            parts.append(types.Part.from_bytes(data=screenshot, mime_type="image/png"))

        self._throttle()
        return self.client.models.generate_content(
            model=config.MODEL_NAME,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=_TOOLS,
                http_options=types.HttpOptions(timeout=config.LLM_CALL_TIMEOUT_SECONDS * 1000),
            ),
        )


def _parse_tool_call(response) -> tuple[str, dict, str | None]:
    model_version = getattr(response, "model_version", None)
    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if getattr(part, "function_call", None):
            return part.function_call.name, dict(part.function_call.args), model_version
    raise NoToolCallReturned("model did not return a tool call")
