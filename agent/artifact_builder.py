"""Artifact builder (Section 3.2): turns a successful discovery run's
evidence/{run_id}/log.jsonl into a versioned, reviewable capability
definition, written to artifacts/{goal_key}.json.

Deliberately post-hoc, not built inline into loop_controller.py's success
path: reads only the already-persisted log, never live Playwright/LLM
objects, so a bug here can never corrupt or block a run that already
succeeded, and so it can be developed/tested against the evidence/ runs
already on disk without spending any more LLM quota (see config.py /
PROGRESS.md decision #1).

Schema shape (settled by design conversation, not re-derived here):
  - Two audiences: a calling agent reads only the top-level contract
    (parameters / outputs / checkpoint); a human reviewer or the future
    replay engine reads the full `steps` list plus `grounding_strategy`.
  - Every `type`/`select` step becomes a candidate parameter (name from
    the element's accessible label, type seeded from the app's own
    declared <input type> and falling back to shape-inference); every
    `click` stays a literal, structural action. This is a draft: `status`
    starts "draft" and a human is expected to review/correct parameter
    candidacy and write the real `description` before treating the
    artifact as final - nothing here should be trusted as reviewed.
  - `checkpoint`'s url_pattern/text_signature are copied from
    checkpoints.REGISTRIES at build time and become the artifact's own
    frozen, authoritative copy for replay matching - REGISTRIES is never
    re-consulted for the match condition, only for output_extractor /
    output_schema (the one piece that can't serialize), keyed by
    (goal_key, outcome). This includes every OTHER outcome in the goal's
    registry too (checkpoint.known_outcomes), not just success - so replay
    can distinguish a legitimate business outcome (member_not_found,
    restricted, ...) from genuine checkpoint drift (the final page matches
    nothing the artifact recognizes at all), the same business-outcome-vs-
    failure distinction checkpoints.py already enforces in discovery.
  - `degraded_grounding_reviewed` closes a gap `built_from_degraded_log`
    alone left open: `status: "reviewed"` only proves a human looked at
    parameter candidacy/naming/typing/description (decision #4) - it says
    nothing about whether they also re-verified a degraded artifact's
    possibly-wrong-by-default grounding (nth defaults to 0, which could
    silently target the wrong element if a real duplicate existed at that
    step). Defaults to False whenever built_from_degraded_log is True, and
    must be hand-flipped to True by whoever reviews the artifact - not by
    this builder - once they've actually re-checked step grounding, not
    just the description.
  - Nothing sensitive/literal is persisted: parameters carry no raw
    recorded example value, and any expected_target path segment that
    exactly matches a known parameter's recorded value is templated back
    to {param_name} before being written out.
  - `built_from_degraded_log` self-flags an artifact built from a
    pre-instrumentation log (missing nth/effective_target/html_input_type
    on a step target, or missing goal_key on run_start) so degraded
    provenance is visible in the artifact itself, not something to track
    externally by run_id.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from agent import checkpoints, config

SCHEMA_VERSION = 1

GROUNDING_STRATEGY_NOTE = (
    "Each step's target is (role, name, nth): a live Playwright Locator via "
    "get_by_role(role, name=name, exact=True).nth(nth), matched in the same "
    "document order the accessibility tree reports elements in. nth "
    "disambiguates true duplicate (role, name) pairs. expected_target "
    "records the (method, path) resolved at discovery time, with any known "
    "parameter's recorded value templated back to {param_name} - it is a "
    "reviewer/drift-detection signal only, never authoritative: guardrails "
    "re-derive ground truth live from the DOM at replay time."
)


def _read_events(log_path: Path) -> list[dict]:
    if not log_path.exists():
        raise FileNotFoundError(f"no log.jsonl found at {log_path}")
    events = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _label_to_param_name(label: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", (label or "").strip()).strip("_").lower()
    return slug or "value"


def _infer_type(html_input_type: str | None, recorded_value) -> str:
    if html_input_type == "number":
        return "number"
    if isinstance(recorded_value, (int, float)) and not isinstance(recorded_value, bool):
        return "number"
    if isinstance(recorded_value, str):
        try:
            float(recorded_value)
            return "number"
        except ValueError:
            pass
    return "string"


def _bare_path(path: str) -> str:
    return urlparse(path).path or path


def _template_path(path: str, known_values: dict[str, str]) -> str:
    segments = path.split("/")
    for i, seg in enumerate(segments):
        for param_name, value in known_values.items():
            if seg == value:
                segments[i] = f"{{{param_name}}}"
                break
    return "/".join(segments)


def _unique_param_name(
    base: str, source: tuple[str, str, int], used_sources: dict[str, tuple[str, str, int]]
) -> str:
    """base is the mechanically-derived name; source is (role, label, nth),
    used to tell 'the same element referenced again' (reuse the name) apart
    from 'a different element that happens to slug to the same name'
    (genuine collision - disambiguate with a numeric suffix). nth must be
    part of the source key, not just (role, label): two elements sharing a
    label but distinguished only by nth (the duplicate-label case the whole
    grounding design exists to handle) are genuinely different fields, not
    the same one revisited."""
    if base not in used_sources or used_sources[base] == source:
        used_sources[base] = source
        return base
    i = 2
    while f"{base}_{i}" in used_sources and used_sources[f"{base}_{i}"] != source:
        i += 1
    candidate = f"{base}_{i}"
    used_sources[candidate] = source
    return candidate


def build_artifact(
    run_id: str,
    goal_key: str,
    repo_root: Path | None = None,
    capability_version: int = 1,
) -> dict:
    repo_root = repo_root or Path.cwd()
    log_path = repo_root / config.EVIDENCE_DIR / run_id / "log.jsonl"
    events = _read_events(log_path)

    run_start = next((e for e in events if e.get("event") == "run_start"), None)
    if run_start is None:
        raise ValueError(f"no run_start event found in {log_path}")

    degraded = "goal_key" not in run_start
    logged_goal_key = run_start.get("goal_key")
    if logged_goal_key is not None and logged_goal_key != goal_key:
        raise ValueError(
            f"--goal-key {goal_key!r} does not match the log's recorded "
            f"goal_key {logged_goal_key!r} for run {run_id!r}"
        )

    run_end = next((e for e in events if e.get("event") == "run_end"), None)
    if run_end is None:
        raise ValueError(f"no run_end event found in {log_path} - run did not complete")
    if run_end.get("outcome_type") != "success":
        raise ValueError(
            f"run {run_id!r} ended with outcome_type={run_end.get('outcome_type')!r}, "
            "not 'success' - only a fully successful run can become an artifact"
        )

    registry = checkpoints.REGISTRIES.get(goal_key)
    if registry is None:
        raise ValueError(f"unknown goal_key {goal_key!r} - not in checkpoints.REGISTRIES")
    spec = registry.get("success")
    if spec is None:
        raise ValueError(f"checkpoints.REGISTRIES[{goal_key!r}] has no 'success' outcome")
    # output_schema is only required when there's actually an extractor to
    # describe - a success outcome with no output_extractor legitimately has
    # no outputs (e.g. a bare confirmation with nothing to extract), and
    # that's not an error. It IS an error for an extractor to exist without
    # a declared shape, since that's exactly the drift risk output_schema
    # was added to close.
    if spec.output_extractor is not None and spec.output_schema is None:
        raise ValueError(
            f"checkpoints.REGISTRIES[{goal_key!r}]['success'] has an output_extractor "
            "but no output_schema - add one (see checkpoints.py) before building an "
            "artifact for this goal"
        )

    parameters: dict[str, dict] = {}
    param_sources: dict[str, tuple[str, str, int]] = {}
    known_values: dict[str, str] = {}
    steps: list[dict] = []

    for event in events:
        if event.get("event") != "step" or event.get("execution_result") != "success":
            continue
        action = event.get("action", {})
        action_type = action.get("type")
        if action_type not in ("click", "type", "select"):
            continue

        target = action.get("target", {})
        role = target.get("role")
        name = target.get("name")
        nth = target.get("nth", 0)
        if "nth" not in target:
            degraded = True
        raw_effective_target = target.get("effective_target")
        if "effective_target" not in target:
            degraded = True
        html_input_type = target.get("html_input_type")
        if "html_input_type" not in target:
            degraded = True

        value_field = None
        if action_type in ("type", "select"):
            recorded_value = action.get("value")
            param_name = _unique_param_name(
                _label_to_param_name(name), (role, name, nth), param_sources
            )
            if param_name not in parameters:
                parameters[param_name] = {
                    "name": param_name,
                    "type": _infer_type(html_input_type, recorded_value),
                    "description": (
                        f"Value {'typed into' if action_type == 'type' else 'selected in'} "
                        f"the '{name}' {role}."
                    ),
                }
            # A None value (e.g. an LLM-issued type with no real value) must
            # never become the literal string "None" in known_values - that
            # would make it eligible to falsely match a path segment that
            # happens to literally read "None".
            if recorded_value is not None:
                known_values[param_name] = str(recorded_value)
            value_field = {"param": param_name}

        expected_target = None
        if raw_effective_target:
            method, raw_path = raw_effective_target
            bare_path = _bare_path(raw_path)
            expected_target = [method, _template_path(bare_path, known_values)]

            if action_type == "click":
                for pattern, field_name in config.AMOUNT_BEARING_ROUTES:
                    if re.match(pattern, bare_path) and field_name in parameters:
                        parameters[field_name]["may_require_approval"] = {
                            "threshold": config.RISK_TIER_HITL_THRESHOLD
                        }

        steps.append(
            {
                "step_index": len(steps) + 1,
                "action": action_type,
                "value": value_field,
                "grounding": {
                    "role": role,
                    "name": name,
                    "nth": nth,
                    "expected_target": expected_target,
                },
            }
        )

    outputs = [
        {
            "name": out_name,
            "type": out_type,
            "description": f"'{out_name}', extracted from the success confirmation page.",
        }
        for out_name, out_type in (spec.output_schema or {}).items()
    ]

    checkpoint = {
        "success": {
            "url_pattern": spec.url_pattern,
            "text_signature": spec.text_signature,
            "extractor_ref": {"goal_key": goal_key, "outcome": "success"},
        },
        # Frozen copies of every other known outcome for this goal, so
        # replay can tell "hit a legitimate business outcome" apart from
        # "the app changed in a way this artifact doesn't recognize at
        # all" - see module docstring.
        "known_outcomes": [
            {
                "outcome": outcome_name,
                "url_pattern": outcome_spec.url_pattern,
                "text_signature": outcome_spec.text_signature,
            }
            for outcome_name, outcome_spec in registry.items()
            if outcome_name != "success"
        ],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "capability_version": capability_version,
        "goal_key": goal_key,
        "status": "draft",
        "description": "DRAFT: auto-generated placeholder, pending human review.",
        "source_run_id": run_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "built_from_degraded_log": degraded,
        # Only meaningful when built_from_degraded_log is True (see module
        # docstring) - None signals "not applicable" for a non-degraded
        # artifact rather than a misleading False.
        "degraded_grounding_reviewed": False if degraded else None,
        "grounding_strategy": GROUNDING_STRATEGY_NOTE,
        "parameters": list(parameters.values()),
        "outputs": outputs,
        "checkpoint": checkpoint,
        "steps": steps,
    }


def save_artifact(
    artifact: dict,
    goal_key: str,
    repo_root: Path | None = None,
    force: bool = False,
) -> tuple[Path, bool]:
    """Writes artifact to artifacts/{goal_key}.json, unless an artifact
    already there is status: "reviewed" and force is False - a reviewed
    artifact represents completed human review work (decision #4/#12) that
    an automatic rebuild (main.py, after every successful discovery run)
    must not silently discard back to draft. A "draft" or missing artifact
    is always safe to (over)write, since nothing has reviewed it yet.

    force=True is for a deliberate, explicit rebuild - the manual CLI below
    always passes it, since typing the exact `run_id --goal-key` command *is*
    the deliberate human action the guard exists to require; only main.py's
    automatic post-success call uses the default (force=False), since that's
    the call site with no human in the loop to have decided anything.
    Returns (path, written) - written is False when the skip fired."""
    repo_root = repo_root or Path.cwd()
    out_dir = repo_root / config.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{goal_key}.json"

    if out_path.exists() and not force:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
        if existing.get("status") == "reviewed":
            return out_path, False

    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return out_path, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a versioned artifact from a successful discovery run's evidence log."
    )
    parser.add_argument("run_id", help="evidence/{run_id} directory to build from")
    parser.add_argument(
        "--goal-key", required=True, help="checkpoints.REGISTRIES key for this run's goal"
    )
    parser.add_argument("--capability-version", type=int, default=1)
    args = parser.parse_args()

    artifact = build_artifact(
        args.run_id, args.goal_key, capability_version=args.capability_version
    )
    # force=True: running this CLI with an explicit run_id/goal_key already
    # is the deliberate rebuild - see save_artifact()'s docstring.
    out_path, _ = save_artifact(artifact, args.goal_key, force=True)
    print(f"wrote {out_path} (built_from_degraded_log={artifact['built_from_degraded_log']})")


if __name__ == "__main__":
    main()
