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

from agent import config
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
