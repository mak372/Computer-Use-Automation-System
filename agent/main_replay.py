"""Replay CLI entry point.

Usage:
    python -m agent.main_replay --goal-key withdraw_funds --param member_id=M-1007 --param amount=150

Runs one deterministic replay of a previously-built, human-reviewed
artifact (agent/artifact_builder.py output, artifacts/{goal_key}.json)
against the live target_app - no LLM call anywhere in this path, unlike
python -m agent.main. Requires target_app running separately
(python target_app/app.py). No GEMINI_API_KEY needed.
"""

from __future__ import annotations

import argparse
import sys

from playwright.sync_api import sync_playwright

from agent import config, operator_ui
from agent.checkpoints import REGISTRIES
from agent.logger import RunLogger
from agent.replay_controller import (
    ReplayValidationError,
    bind_parameters,
    load_artifact,
    run_replay,
    validate_artifact_preflight,
)


def _parse_param(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--param must be name=value, got {raw!r}")
    name, value = raw.split("=", 1)
    return name, value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically replay a previously-built artifact against target_app."
    )
    parser.add_argument(
        "--goal-key",
        required=True,
        choices=sorted(REGISTRIES.keys()),
        help="Which artifact (artifacts/{goal_key}.json) to replay.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=_parse_param,
        metavar="name=value",
        help="A parameter declared by the artifact, repeatable (e.g. --param amount=150).",
    )
    parser.add_argument(
        "--expected-capability-version",
        type=int,
        default=None,
        help="Refuse to run if the artifact's capability_version doesn't match this.",
    )
    args = parser.parse_args()

    parameters = dict(args.param)

    # Fail fast, before ever launching a browser: preflight is cheap and
    # pure (one file read, no I/O beyond that), so checking it here first
    # avoids opening Chromium for a run that was always going to be
    # refused. run_replay() below re-validates internally regardless - this
    # is purely a CLI-experience shortcut, not a substitute for that (any
    # other caller invoking run_replay() directly still gets the real
    # guarantee without needing to know to call this first).
    try:
        artifact = load_artifact(args.goal_key)
        validate_artifact_preflight(artifact, args.expected_capability_version)
        bind_parameters(artifact, parameters)
    except (ReplayValidationError, FileNotFoundError) as exc:
        print(f"Replay refused before starting: {exc}", file=sys.stderr)
        return 1

    # Eager start (not lazy-on-first-escalation) - see main.py for why.
    operator_ui.ensure_started()

    result = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        try:
            page = browser.new_page()
            try:
                page.goto(config.TARGET_BASE_URL)
            except Exception as exc:
                print(
                    f"Could not reach target_app at {config.TARGET_BASE_URL} - "
                    f"is it running? ({exc})",
                    file=sys.stderr,
                )
                return 1

            with RunLogger(run_type="replay") as logger:
                try:
                    result = run_replay(
                        args.goal_key,
                        page,
                        logger,
                        parameters,
                        expected_capability_version=args.expected_capability_version,
                    )
                except ReplayValidationError as exc:
                    # Should not happen given the pre-check above, unless
                    # the artifact file changed between the two calls -
                    # handled rather than assumed impossible.
                    print(f"Replay refused: {exc}", file=sys.stderr)
                    return 1
        finally:
            browser.close()

    if result is None:
        print("Replay did not complete (unexpected error before a result was produced).", file=sys.stderr)
        return 1

    print(f"\nReplay finished: {result.outcome_type} ({result.outcome_label})")
    print(f"Steps: {result.total_steps}")
    if result.outputs:
        print(f"Outputs: {result.outputs}")
    if result.failed_at_step is not None:
        print(f"Failed at step: {result.failed_at_step}")
    if result.expected is not None:
        print(f"Expected: {result.expected}")
    if result.observed is not None:
        print(f"Observed: {result.observed}")
    print(f"Evidence: {config.EVIDENCE_DIR}/{result.run_id}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
