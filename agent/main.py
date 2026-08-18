"""CLI entry point.

Usage:
    python -m agent.main --goal "Look up member M-1001 and report their balance." --goal-key lookup_balance

Runs one genuine LLM-driven discovery run against the live target_app and
writes structured evidence (JSONL log + screenshots) to evidence/{run_id}/.
Requires target_app running separately (python target_app/app.py) and a
real GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from agent import config, operator_ui
from agent.artifact_builder import build_artifact, save_artifact
from agent.checkpoints import REGISTRIES
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
        "--goal-key",
        required=True,
        choices=sorted(REGISTRIES.keys()),
        help="Which outcome registry applies to this goal.",
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
                page.goto(config.TARGET_BASE_URL)
            except Exception as exc:
                print(
                    f"Could not reach target_app at {config.TARGET_BASE_URL} - "
                    f"is it running? ({exc})",
                    file=sys.stderr,
                )
                return 1

            with RunLogger(run_type="discovery") as logger:
                result = run_discovery(args.goal, args.goal_key, page, client, logger)
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
        # Section 3.2: "after a successful run, emit ... an artifact" -
        # automatic, not a separate manual step a human has to remember to
        # run. build_artifact() itself stays post-hoc/decoupled (reads only
        # the already-persisted log.jsonl, per decision #1) so a bug here
        # can't corrupt or invalidate the discovery run that already
        # succeeded and was already logged - only this auto-save step is
        # new, wrapped so its failure surfaces as a warning, not a crash.
        try:
            artifact = build_artifact(result.run_id, args.goal_key)
            out_path, written = save_artifact(artifact, args.goal_key)
            if written:
                print(f"[artifact_builder] wrote {out_path} (status: draft, pending human review)")
            else:
                print(
                    f"[artifact_builder] {out_path} already exists with status='reviewed' - "
                    f"skipping auto-build. Run `python -m agent.artifact_builder "
                    f"{result.run_id} --goal-key {args.goal_key}` explicitly if you intend "
                    f"to rebuild it."
                )
        except Exception as exc:
            print(
                f"[artifact_builder] ERROR: discovery run succeeded but artifact build "
                f"failed - no artifact was saved for goal_key={args.goal_key!r}, "
                f"run_id={result.run_id!r}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
