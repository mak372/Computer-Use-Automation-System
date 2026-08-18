"""Structured JSONL run logging + evidence storage (decision #9).

One run = one directory under EVIDENCE_DIR, named after the run:
  evidence/{run_id}/log.jsonl
  evidence/{run_id}/step_{n}.png   (only for steps that captured a screenshot)

Shared by discovery and replay - both produce logs through this same module
so /evidence/ contains directly comparable output for either run type.

Callers must pass plain JSON-serializable dicts (str/int/float/bool/None/
list/dict) into the log_* methods - dataclass instances and Playwright
Locator objects are not serializable and must be converted first.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from agent import config

_REDACT_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{4}\b"),  # phone-number-shaped
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email-shaped
]


def redact(text: str) -> str:
    """Applied to the logged representation only - never to data used for
    verification/output-extraction logic, which must operate on real values."""
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _new_run_id(run_type: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{run_type}-{timestamp}-{suffix}"


class RunLogger:
    def __init__(self, run_type: str, repo_root: Path | None = None):
        self.run_type = run_type
        self.run_id = _new_run_id(run_type)
        base = (repo_root or Path.cwd()) / config.EVIDENCE_DIR / self.run_id
        base.mkdir(parents=True, exist_ok=True)
        self.dir = base
        self._file = open(base / "log.jsonl", "a", encoding="utf-8")

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def _write(self, event: dict) -> None:
        event["logged_at"] = datetime.now(timezone.utc).isoformat()
        try:
            line = json.dumps(event)
        except TypeError as exc:
            # A caller passed something non-serializable (e.g. a dataclass
            # or Locator instead of a plain dict) - never let a logging bug
            # crash the run itself, same "structured, not raw exception"
            # pattern used in guardrails/action_executor.
            line = json.dumps(
                {
                    "event": "log_serialization_error",
                    "run_id": self.run_id,
                    "original_event_type": event.get("event"),
                    "error": str(exc),
                    "logged_at": event["logged_at"],
                }
            )
        self._file.write(line + "\n")
        self._file.flush()

    def screenshot_path(self, step_number: int) -> Path:
        return self.dir / f"step_{step_number}.png"

    def log_run_start(
        self,
        goal: str,
        goal_key: str,
        target_url: str,
        model: str,
        model_version: str | None,
    ) -> None:
        self._write(
            {
                "event": "run_start",
                "run_id": self.run_id,
                "run_type": self.run_type,
                "goal": goal,
                # checkpoints.REGISTRIES key - not derivable from `goal`
                # (free-form NL text) alone. Needed so artifact_builder.py
                # can look up the right registry entry from log.jsonl
                # without any input beyond the log itself.
                "goal_key": goal_key,
                "target_url": target_url,
                "model": model,
                "model_version": model_version,
            }
        )

    def log_step(
        self,
        step_number: int,
        # Caller must pre-redact (agent.logger.redact) before passing:
        # observation_summary's static-text-derived fields, and action's
        # "reasoning" string. Nothing downstream of this call re-checks it.
        observation_summary: dict,
        action: dict,
        guardrail_result: dict,
        execution_result: str,
        state_fingerprint: str,
        retry_count: int = 0,
        hitl_event_id: str | None = None,
        resumed_from_hitl_event_id: str | None = None,
    ) -> None:
        self._write(
            {
                "event": "step",
                "run_id": self.run_id,
                "step_number": step_number,
                "observation": observation_summary,
                "action": action,
                "guardrail_result": guardrail_result,
                "execution_result": execution_result,
                "state_fingerprint": state_fingerprint,
                "retry_count": retry_count,
                "hitl_event_id": hitl_event_id,
                "resumed_from_hitl_event_id": resumed_from_hitl_event_id,
            }
        )

    def log_llm_call_attempt_failed(
        self, step_number: int, attempt: int, result: str, duration_ms: int
    ) -> None:
        self._write(
            {
                "event": "llm_call_attempt_failed",
                "run_id": self.run_id,
                "step_number": step_number,
                "attempt": attempt,
                "result": result,
                "duration_ms": duration_ms,
            }
        )

    def log_hitl_event(
        self,
        hitl_event_id: str,
        phase: str,
        trigger: str | None = None,
        context: dict | None = None,
        decision: str | None = None,
        operator_id: str | None = None,
    ) -> None:
        self._write(
            {
                "event": "hitl_event",
                "run_id": self.run_id,
                "hitl_event_id": hitl_event_id,
                "phase": phase,
                "trigger": trigger,
                "context": context,
                "decision": decision,
                "operator_id": operator_id,
            }
        )

    def log_dialog_event(
        self,
        dialog_type: str,
        message: str,
        resolution: str,  # "accepted" | "dismissed"
        matched_count: int,
    ) -> None:
        self._write(
            {
                "event": "dialog",
                "run_id": self.run_id,
                "dialog_type": dialog_type,
                "message": message,
                "resolution": resolution,
                "matched_count": matched_count,
            }
        )

    def log_run_end(
        self,
        outcome_type: str,
        outputs: dict | None,
        checkpoint_verification: dict | None,
        total_steps: int,
    ) -> None:
        self._write(
            {
                "event": "run_end",
                "run_id": self.run_id,
                "outcome_type": outcome_type,
                "outputs": outputs,
                "checkpoint_verification": checkpoint_verification,
                "total_steps": total_steps,
            }
        )

    def close(self) -> None:
        self._file.close()
