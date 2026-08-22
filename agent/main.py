from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from agent import config, operator_ui
from agent.artifact_builder import build_artifact, save_artifact
from agent.llm_client import LLMClient
from agent.logger import RunLogger
from agent.loop_controller import run_discovery


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run a goal-driven discovery agent against target_app."
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="Natural language goal, e.g. 'Look up member M-1001 and report their balance.'",
    )
    parser.add_argument(
        "--target",
        default=config.TARGET_BASE_URL,
        help=f"Base URL of the target application (default: {config.TARGET_BASE_URL}).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. Add it to .env before running.", file=sys.stderr)
        return 1

    client = LLMClient(api_key=api_key)

    # Eager start (not lazy-on-first-escalation): the console must be
    # reachable the moment this process starts, same as target_app's port
    # being live the instant it's run - opening the tab before any
    # escalation happens should show "no action needed", never a
    # connection error.
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

            with RunLogger(run_type="discovery") as logger:
                result = run_discovery(args.goal, page, client, logger)
        finally:
            browser.close()

    if result is None:
        print("Run did not complete (unexpected error before a result was produced).", file=sys.stderr)
        return 1

    print(f"\nRun finished: {result.outcome_type} ({result.outcome_label})")
    print(f"Steps: {result.total_steps}")
    if result.outputs:
        print(f"Outputs: {result.outputs}")
    print(f"Evidence: {config.EVIDENCE_DIR}/{result.run_id}/")

    if result.outcome_type == "success":
        try:
            # result.goal_key is always set on a "success" outcome - success
            # can only be reached via checkpoints.verify_finish, which itself
            # requires a resolved, valid goal_key (see loop_controller.
            # run_discovery's opening block).
            artifact = build_artifact(result.run_id, result.goal_key)
            out_path, written = save_artifact(artifact, result.goal_key)
            if written:
                print(f"[artifact_builder] wrote {out_path} (status: draft, pending human review)")
                for warning in artifact["build_warnings"]:
                    print(f"[artifact_builder] WARNING: {warning}", file=sys.stderr)
            else:
                print(
                    f"[artifact_builder] {out_path} already exists with status='reviewed' - "
                    f"skipping auto-build. Run `python -m agent.artifact_builder "
                    f"{result.run_id} --goal-key {result.goal_key}` explicitly if you intend "
                    f"to rebuild it."
                )
        except Exception as exc:
            print(
                f"[artifact_builder] ERROR: discovery run succeeded but artifact build "
                f"failed - no artifact was saved for goal_key={result.goal_key!r}, "
                f"run_id={result.run_id!r}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
