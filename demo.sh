#!/usr/bin/env bash
# One-command demo: installs deps, starts target_app, runs discovery on one
# goal, then replays the resulting artifact. Run from the project root.
#
# Requires a .env file with GEMINI_API_KEY set (discovery calls the LLM;
# replay does not). See README.md for the manual, step-by-step version.
set -e

if [ ! -f .env ] || ! grep -q '^GEMINI_API_KEY=.\+' .env; then
  echo "Missing .env with GEMINI_API_KEY=<your key> - create it before running this." >&2
  exit 1
fi

echo "==> Installing dependencies"
pip install -r requirements.txt
playwright install chromium

echo "==> Starting target_app on http://127.0.0.1:5000"
python target_app/app.py &
TARGET_PID=$!
trap 'kill "$TARGET_PID" 2>/dev/null' EXIT

echo "==> Waiting for target_app to be ready"
for _ in $(seq 1 20); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=1)" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

echo "==> Running discovery: 'Look up member M-1001 and report their balance.'"
echo "    (a real browser window will open - this step calls the LLM and may pause"
echo "    for human input at http://127.0.0.1:5050 if the run needs it)"
python -m agent.main --goal "Look up member M-1001 and report their balance."

echo "==> Replaying the resulting artifact (no LLM call)"
python -m agent.main_replay --goal-key lookup_balance --param member_id=M-1001

echo "==> Done. Evidence: evidence/  Artifact: artifacts/lookup_balance.json"
