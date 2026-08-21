# One-command demo: installs deps, starts target_app, runs discovery on one
# goal, then replays the resulting artifact. Run from the project root.
#
# Requires a .env file with GEMINI_API_KEY set (discovery calls the LLM;
# replay does not). See README.md for the manual, step-by-step version.

$ErrorActionPreference = "Stop"

if (-not (Test-Path .env) -or -not (Select-String -Path .env -Pattern '^GEMINI_API_KEY=.+' -Quiet)) {
    Write-Error "Missing .env with GEMINI_API_KEY=<your key> - create it before running this."
    exit 1
}

Write-Host "==> Installing dependencies"
pip install -r requirements.txt
playwright install chromium

Write-Host "==> Starting target_app on http://127.0.0.1:5000"
$targetProc = Start-Process -FilePath python -ArgumentList "target_app/app.py" -PassThru -NoNewWindow

try {
    Write-Host "==> Waiting for target_app to be ready"
    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:5000/" -UseBasicParsing -TimeoutSec 1 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) {
        Write-Warning "target_app did not respond within 10s - continuing anyway."
    }

    # The replay step below needs artifacts/lookup_balance.json to already be
    # "status": "reviewed" - it ships that way in this repo. Discovery's own
    # auto-save will NOT overwrite an already-reviewed artifact with a fresh
    # draft, so this is safe to re-run; but if that committed artifact is
    # ever deleted, discovery will only produce a draft, and replay will
    # then refuse to run it (by design - drafts require real human review,
    # and this script deliberately does not bypass that).
    Write-Host "==> Running discovery: 'Look up member M-1001 and report their balance.'"
    Write-Host "    (a real browser window will open - this step calls the LLM and may pause"
    Write-Host "    for human input at http://127.0.0.1:5050 if the run needs it)"
    python -m agent.main --goal "Look up member M-1001 and report their balance."

    Write-Host "==> Replaying the resulting artifact (no LLM call)"
    python -m agent.main_replay --goal-key lookup_balance --param member_id=M-1001

    Write-Host "==> Done. Evidence: evidence/  Artifact: artifacts/lookup_balance.json"
}
finally {
    if ($targetProc -and -not $targetProc.HasExited) {
        Stop-Process -Id $targetProc.Id -Force
    }
}
