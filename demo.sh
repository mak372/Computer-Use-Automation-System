#!/usr/bin/env bash
# One-command demo: creates a virtual environment, installs deps, starts
# target_app, runs discovery on one goal, then replays the resulting
# artifact. Run from the project root.
#
# Requires a .env file with GEMINI_API_KEY set (discovery calls the LLM;
# replay does not). See README.md for the manual, step-by-step version.
set -e

if [ ! -f .env ] || ! grep -q '^GEMINI_API_KEY=.\+' .env; then
  echo "Missing .env with GEMINI_API_KEY=<your key> - create it before running this." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> Creating virtual environment (.venv)"
  python3 -m venv .venv 2>/dev/null || python -m venv .venv
fi
VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY=".venv/Scripts/python.exe"  # Git Bash on Windows lays out venvs under Scripts/, not bin/
fi

echo "==> Installing dependencies into .venv"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt
"$VENV_PY" -m playwright install chromium

echo "==> Starting target_app on http://127.0.0.1:5000"
"$VENV_PY" target_app/app.py &
TARGET_PID=$!
trap 'kill "$TARGET_PID" 2>/dev/null' EXIT

echo "==> Waiting for target_app to be ready"
for _ in $(seq 1 20); do
  if "$VENV_PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=1)" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

# The replay step below needs evidence/artifacts/lookup_balance.json to already be
# "status": "reviewed" - it ships that way in this repo. Discovery's own
# auto-save will NOT overwrite an already-reviewed artifact with a fresh
# draft, so this is safe to re-run; but if that committed artifact is ever
# deleted, discovery will only produce a draft, and replay will then refuse
# to run it (by design - drafts require real human review, and this script
# deliberately does not bypass that).
echo "==> Running discovery: 'Look up member M-1001 and report their balance.'"
echo "    (a real browser window will open - this step calls the LLM and may pause"
echo "    for human input at http://127.0.0.1:5050 if the run needs it)"
"$VENV_PY" -m agent.main --goal "Look up member M-1001 and report their balance."

echo "==> Replaying the resulting artifact (no LLM call)"
"$VENV_PY" -m agent.main_replay --goal-key lookup_balance --param member_id=M-1001

echo "==> Done. Evidence: evidence/  Artifact: evidence/artifacts/lookup_balance.json"
