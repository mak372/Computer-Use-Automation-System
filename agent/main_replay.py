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
        help="Which artifact (evidence/artifacts/{goal_key}.json) to replay.",
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
    parser.add_argument(
        "--target",
        default=config.TARGET_BASE_URL,
        help=f"Base URL of the target application (default: {config.TARGET_BASE_URL}).",
    )
    args = parser.parse_args()

    parameters = dict(args.param)

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
                page.goto(args.target)
            except Exception as exc:
                print(
                    f"Could not reach target_app at {args.target} - "
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
