# Steps to run 

**Prerequisites:** Python 3.10+ and `git` to clone this repo. Everything else (Flask, Playwright, Chromium, etc.) is installed by the steps below.

## Quickstart (one command)

Creates a virtual environment (`.venv`), installs dependencies into it, installs the Chromium browser, starts `target_app`, runs discovery on one goal, then replays the resulting artifact. This script only exercises the lookup_balance capability. See the Manual demo path below to try discovery and replay for withdraw_funds and open_sub_account.

Needs `.env` with `GEMINI_API_KEY` set first, discovery calls the LLM (replay doesn't). Create a `.env` file in the project root with following content:
```
GEMINI_API_KEY=your-key-here
```

On Windows, in PowerShell:
```
.\demo.ps1
```
On macOS/Linux, or Windows Git Bash:
```
./demo.sh
```
On Windows, don't run `demo.sh` via a bare `bash demo.sh` if you also have WSL installed  `bash` on your `PATH` may resolve to WSL's bash instead of Git Bash, which runs against a completely separate Linux Python/browser environment than the rest of this setup expects. Use `demo.ps1` instead, or launch a Git Bash terminal directly (right-click the folder → "Git Bash Here") rather than typing `bash` from PowerShell.

A real browser window opens during the discovery step; if the run needs human input it'll pause and prompt in the terminal (and at `http://127.0.0.1:5050`), same as running the commands manually.



## Manual setup (if you'd rather not run the script)

1. **Create a virtual environment and install dependencies**
   ```
   python -m venv .venv
   ```
   Activate it PowerShell: `.venv\Scripts\Activate.ps1`, Git Bash: `source .venv/Scripts/activate`, macOS/Linux: `source .venv/bin/activate` then:
   ```
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Set your Gemini API key**

   Create a `.env` file in the project root with content given below by putting your own gemini api key:
   ```
   GEMINI_API_KEY=your-key-here
   ```

3. **Start the target application** (This always has to be running, in a separate terminal):
   ```
   python target_app/app.py
   ```
   Runs on `http://127.0.0.1:5000`.

That's the only "live service" this project talks to besides the LLM. `target_app` is a local Flask app made for mocking the surface UI.

Since this is a mock platform, only the following member IDs exist. Any `member_id` parameter used below must be one of these:
- `M-1001` (active, balance 3000)
- `M-1002` (restricted)
- `M-1003` (active, balance 2000)
- `M-1005` (active, balance 4000)
- `M-1006` (active, balance 6000)
- `M-1007` (active, balance 100000)
- `M-1010` (active, balance 8000)
- `M-1098` (active, balance 1000 — for testing session-expiry behavior)
- `M-1099` (active, balance 1500 — for testing a broken/error page)


## Manual demo path

With `target_app` running in one terminal, in another terminal:

**1. Run discovery on a natural-language goal** (requires `GEMINI_API_KEY`):
### Command parameters

**Discovery** (`python -m agent.main`):
- `--goal` (required) - the natural-language goal you want the agent to complete, e.g. `"Look up member M-1001 and report their balance."`
```
python -m agent.main --goal "Look up member M-1001 and report their balance."
```
A small operator console also starts at `http://127.0.0.1:5050` it stays idle unless the run needs human input (a risky-amount approval, or a stuck condition), in which case both that console and the terminal prompt for a decision.

A freshly auto-built artifact starts as `"status": "draft"` replay refuses to run a draft artifact. The three artifacts already committed in this repo have been manually reviewed and marked `"status": "reviewed"`; if you build a new one from your own discovery run, edit its `status` field to `"reviewed"` before replaying it.

**Rebuilding an artifact that already exists for a capability:** discovery won't auto-overwrite one that's already `"reviewed"` — it just skips saving and tells you so. To rebuild it on purpose:
```
python -m agent.artifact_builder <run_id> --goal-key <goal_key> --capability-version <N>
```
- `<run_id>` is the evidence folder your discovery run just created, e.g. `evidence/2026-08-21T160512Z-lookup_balance/` → `run_id` is `2026-08-21T160512Z-lookup_balance`.
- `<N>` must be a different number from the existing artifact's current `capability_version` — the rebuild is refused otherwise, so a caller pinned to the old version doesn't silently get different behavior.
- The rebuilt artifact is written as `"status": "draft"` again, so it still needs a human to review it and flip `status` to `"reviewed"` before replay will use it.

**2. Replay the resulting artifact** (no API key needed):
### Command parameters
**Replay** (`python -m agent.main_replay`):
- `--goal-key` (required) - which artifact to replay: `lookup_balance`, `withdraw_funds`, or `open_sub_account`.
- `--param name=value` (repeatable) - one flag per parameter the artifact declares, e.g. `--param member_id=M-1001 --param amount=150`. Which parameters are required depends on the goal you're replaying (see the examples below).
- `--expected-capability-version` (optional) - if set, replay refuses to run unless the artifact's `capability_version` matches this exactly; use it if you want to guard against the artifact having changed underneath you since you last checked.
- `--target` (optional, defaults to `http://127.0.0.1:5000`) - base URL of the target application.

**Which `--param` flags each goal needs while replay:**
- `lookup_balance` - `member_id` (string, e.g. `M-1001`)
- `withdraw_funds` - `member_id`, `amount` (number), `withdrawal_method` (one of `cash`, `check`, `transfer`)
- `open_sub_account` - `member_id`, `nickname` (string, name for the new sub-account), `initial_deposit` (number)
```
python -m agent.main_replay --goal-key lookup_balance --param member_id=M-1001
```
This drives the same browser flow again, but deterministically no LLM, just the recorded steps re-executed against the live target app and verified against the artifact's recorded checkpoint.

Other supported goals, if you want to try discovery → replay for each:
```
python -m agent.main --goal "Withdraw 150 from member M-1001's account."
python -m agent.main_replay --goal-key withdraw_funds --param member_id=M-1001 --param amount=150 --param withdrawal_method=transfer

python -m agent.main --goal "Open a new sub-account for member M-1001."
python -m agent.main_replay --goal-key open_sub_account --param member_id=M-1001 --param nickname=Vacation
```

Evidence (a structured JSONL log plus screenshots) for every run, discovery or replay, is written to `evidence/{run_id}/`.
