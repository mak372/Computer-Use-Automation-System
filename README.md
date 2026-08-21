# interface-ai

A goal-driven browser automation agent with two modes: **discovery**, where an LLM figures out how to complete a natural-language goal against a live web app, and **replay**, where a previously discovered and human-reviewed procedure is executed deterministically, with no LLM involved.

See `REPORT.md` for the full design write-up (architecture, artifact schema, safety model, and known limitations).

## Setup

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Set your Gemini API key** (only needed for discovery, not replay — see below)

   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your-key-here
   ```

3. **Start the target application** (a local mock banking app — this always has to be running, in a separate terminal, for either mode):
   ```
   python target_app/app.py
   ```
   Runs on `http://127.0.0.1:5000`.

That's the only "live service" this project talks to besides the LLM — `target_app` is a local Flask app included in this repo, not a third-party service.

## Running without the LLM

**Replay never calls the LLM and never needs `GEMINI_API_KEY`.** It executes a fixed, previously human-reviewed artifact (see `artifacts/*.json`) deterministically. If you just want to see the automation actually work without setting up an API key, skip straight to the replay command in the demo path below — the three capability artifacts already committed in `artifacts/` are ready to replay.

Discovery (`agent.main`) does require a real `GEMINI_API_KEY`, since an LLM call is what decides each action.

## Demo path

**One command, does everything below:** installs dependencies, installs the Chromium browser, starts `target_app`, runs discovery on one goal, then replays the resulting artifact.

On Windows, in PowerShell:
```
.\demo.ps1
```
On macOS/Linux, or Windows Git Bash:
```
./demo.sh
```
On Windows, don't run `demo.sh` via a bare `bash demo.sh` if you also have WSL installed — `bash` on your `PATH` may resolve to WSL's bash instead of Git Bash, which runs against a completely separate Linux Python environment (and will fail with an "externally-managed-environment" pip error). Use `demo.ps1` instead, or launch a Git Bash terminal directly (right-click the folder → "Git Bash Here") rather than typing `bash` from PowerShell.

Needs `.env` with `GEMINI_API_KEY` set first (see Setup above) — discovery calls the LLM. A real browser window opens during the discovery step; if the run needs human input it'll pause and prompt in the terminal (and at `http://127.0.0.1:5050`), same as running the commands manually.

Or, step by step, with `target_app` running in one terminal, in another terminal:

**1. Run discovery on a natural-language goal** (requires `GEMINI_API_KEY`):
```
python -m agent.main --goal "Look up member M-1001 and report their balance."
```
This opens a real (visible, non-headless) browser window, lets the LLM drive it step by step, and on success automatically builds and saves a draft artifact to `artifacts/lookup_balance.json`. A small operator console also starts at `http://127.0.0.1:5050` — it stays idle unless the run needs human input (a risky-amount approval, or a stuck condition), in which case both that console and the terminal prompt for a decision.

A freshly auto-built artifact starts as `"status": "draft"` — replay refuses to run a draft artifact. The three artifacts already committed in this repo have been manually reviewed and marked `"status": "reviewed"`; if you build a new one from your own discovery run, edit its `status` field to `"reviewed"` before replaying it.

**2. Replay the resulting artifact** (no API key needed):
```
python -m agent.main_replay --goal-key lookup_balance --param member_id=M-1001
```
This drives the same browser flow again, but deterministically — no LLM, just the recorded steps re-executed against the live target app and verified against the artifact's recorded checkpoint.

Other supported goals, if you want to try discovery → replay for each:
```
python -m agent.main --goal "Withdraw 150 from member M-1001's account."
python -m agent.main_replay --goal-key withdraw_funds --param member_id=M-1001 --param amount=150 --param withdrawal_method=transfer

python -m agent.main --goal "Open a new sub-account for member M-1001."
python -m agent.main_replay --goal-key open_sub_account --param member_id=M-1001 --param nickname=Vacation --param initial_deposit=500
```


