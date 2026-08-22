"""Lightweight local web console for HITL escalations"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory

from agent import config

_app = Flask(__name__)

_server_thread: Optional[threading.Thread] = None
_server_lock = threading.Lock()

_state_lock = threading.Lock()
_pending: Optional[dict] = None
_active_queue: "Optional[queue.Queue]" = None


def _screenshot_url(screenshot_path: str | None) -> str | None:
    """Converts an absolute screenshot path (always constructed internally
    by RunLogger.screenshot_path(), never external input) into a URL under
    /screenshot/. Fails safe to None rather than raising - a missing
    screenshot in the console is a cosmetic loss, not worth risking the
    escalation flow itself over."""
    if not screenshot_path:
        return None
    try:
        rel = Path(screenshot_path).resolve().relative_to(
            (Path.cwd() / config.EVIDENCE_DIR).resolve()
        )
    except (ValueError, OSError):
        return None
    return f"/screenshot/{rel.as_posix()}"


def ensure_started() -> None:
    """Starts the console's Flask server if it isn't already running.
    Idempotent and safe to call from multiple places: the CLI entry points
    (main.py/main_replay.py) call this eagerly at process start - so the
    console is reachable the moment the run starts, the same way
    target_app's port is live the instant you start that process, rather
    than only appearing once an escalation happens - and operator.py's
    request_approval()/request_manual_handoff() also call it (a no-op by
    then in the normal CLI path, but keeps this module correct on its own
    for any other caller that invokes it without going through the CLI)."""
    global _server_thread
    with _server_lock:
        if _server_thread is not None:
            return
        _server_thread = threading.Thread(
            target=lambda: _app.run(
                host="127.0.0.1",
                port=config.OPERATOR_UI_PORT,
                use_reloader=False,
                debug=False,
                threaded=True,
            ),
            daemon=True,
        )
        _server_thread.start()
        print(f"[operator_ui] console: http://127.0.0.1:{config.OPERATOR_UI_PORT}")


def publish_and_wait(pending: dict, decision_queue: "queue.Queue") -> None:
    """Makes `pending` visible to the console and wires decision_queue up
    to /resolve. Does not itself block - the caller (operator.py) still
    owns the actual queue.get() so it can race the terminal thread too."""
    global _pending, _active_queue
    ensure_started()
    with _state_lock:
        _pending = {**pending, "screenshot_url": _screenshot_url(pending.get("screenshot_path"))}
        _active_queue = decision_queue


def clear_pending() -> None:
    global _pending, _active_queue
    with _state_lock:
        _pending = None
        _active_queue = None


@_app.get("/")
def index():
    return _PAGE_HTML


@_app.get("/status")
def status():
    with _state_lock:
        if _pending is None:
            return jsonify({"pending": False})
        return jsonify({"pending": True, **_pending})


@_app.get("/screenshot/<path:relpath>")
def screenshot(relpath: str):
    return send_from_directory(str((Path.cwd() / config.EVIDENCE_DIR).resolve()), relpath)


@_app.post("/resolve")
def resolve():
    decision = request.form.get("decision", "")
    with _state_lock:
        q = _active_queue
    if q is not None:
        q.put(("web", decision))
    return ("", 204)


_PAGE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Operator Console</title>
<style>
  body { font-family: system-ui, sans-serif; background: #f4f5f7; margin: 0; padding: 2rem; color: #1a1a1a; }
  .banner { padding: 0.9rem 1.2rem; border-radius: 6px; font-weight: 600; margin-bottom: 1.2rem; }
  .banner.ok { background: #e3f7e9; color: #1a7a3c; border: 1px solid #b8e6c6; }
  .banner.paused { background: #fdecea; color: #a3231c; border: 1px solid #f5b6b0; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem 1.5rem; max-width: 640px; }
  .row { display: flex; padding: 0.35rem 0; border-bottom: 1px solid #eee; }
  .row .label { width: 160px; color: #666; flex-shrink: 0; }
  .shot { max-width: 100%; margin-top: 1rem; border: 1px solid #ddd; border-radius: 4px; }
  .hint { background: #fff8e1; color: #6b4e00; border: 1px solid #ffe082; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 0.95rem; }
  .actions { margin-top: 1.2rem; }
  .btn { padding: 0.6rem 1.4rem; margin-right: 0.6rem; border: none; border-radius: 5px; font-size: 1rem; cursor: pointer; }
  .btn.approve { background: #1a7a3c; color: #fff; }
  .btn.deny { background: #a3231c; color: #fff; }
  .btn:disabled { opacity: 0.5; cursor: default; }
  h1 { font-size: 1.3rem; }
</style>
</head>
<body>
  <h1>Operator Console</h1>
  <div id="root"><div class="banner ok">Loading...</div></div>

<script>
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

let resolving = false;

async function resolveDecision(decision) {
  if (resolving) return;
  resolving = true;
  await fetch('/resolve', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'decision=' + encodeURIComponent(decision),
  });
  resolving = false;
  poll();
}

function render(data) {
  const root = document.getElementById('root');
  if (!data.pending) {
    root.innerHTML = '<div class="banner ok">Automation running - no action needed</div>';
    return;
  }

  let fields = '';
  let buttons;

  if (data.type === 'approval') {
    fields += ''
      + '<div class="row"><span class="label">Capability/Goal</span><span>' + esc(data.goal_key ?? '-') + '</span></div>'
      + '<div class="row"><span class="label">Step</span><span>' + esc(data.step_number ?? '-') + '</span></div>'
      + '<div class="row"><span class="label">Action</span><span>' + esc(data.action_description ?? '-') + '</span></div>'
      + '<div class="row"><span class="label">Amount</span><span>' + (data.amount != null ? '$' + Number(data.amount).toLocaleString(undefined, {minimumFractionDigits: 2}) : '-') + '</span></div>'
      + '<div class="row"><span class="label">Reason</span><span>' + esc(data.reason ?? '-') + '</span></div>';
    buttons = ''
      + '<button class="btn approve" onclick="resolveDecision(\\'approve\\')">Approve</button>'
      + '<button class="btn deny" onclick="resolveDecision(\\'deny\\')">Deny</button>';
  } else if (data.type === 'capability_selection') {
    // No goal_key/step_number yet - that's exactly what this prompt is
    // resolving - so this branch shows the raw goal text instead.
    fields += '<div class="row"><span class="label">Goal</span><span>' + esc(data.goal ?? '-') + '</span></div>';
    if (data.reason) {
      fields += '<div class="row"><span class="label">Reason</span><span>' + esc(data.reason) + '</span></div>';
    }
    buttons = (data.valid_goal_keys || []).map(function(k) {
      return '<button class="btn approve" onclick="resolveDecision(\\'' + k + '\\')">' + esc(k) + '</button>';
    }).join('') + '<button class="btn deny" onclick="resolveDecision(\\'abort\\')">Abort</button>';
  } else {
    fields += ''
      + '<div class="hint">&#9888; This screen only shows context - it cannot take actions on the site itself. '
      + 'Switch to the live automation browser window (the separate Chromium window this run opened, '
      + 'usually on port 5000) and complete the needed step there by hand - see Trigger/details below for '
      + 'what\\'s needed (e.g. filling in a Member ID or similar). Then come back to THIS page and click Resume.</div>'
      + '<div class="row"><span class="label">Capability/Goal</span><span>' + esc(data.goal_key ?? '-') + '</span></div>'
      + '<div class="row"><span class="label">Step</span><span>' + esc(data.step_number ?? '-') + '</span></div>'
      + '<div class="row"><span class="label">Trigger</span><span>' + esc(data.trigger ?? '-') + '</span></div>';
    if (data.context) {
      for (const [k, v] of Object.entries(data.context)) {
        fields += '<div class="row"><span class="label">' + esc(k) + '</span><span>' + esc(v) + '</span></div>';
      }
    }
    buttons = ''
      + '<button class="btn approve" onclick="resolveDecision(\\'resume\\')">Resume</button>'
      + '<button class="btn deny" onclick="resolveDecision(\\'abort\\')">Abort</button>';
  }

  const shot = data.screenshot_url
    ? '<img class="shot" src="' + data.screenshot_url + '" alt="screenshot at handoff">'
    : '';

  root.innerHTML = ''
    + '<div class="banner paused">AUTOMATION PAUSED - awaiting your decision</div>'
    + '<div class="card">' + fields + shot + '<div class="actions">' + buttons + '</div></div>';
}

async function poll() {
  try {
    const res = await fetch('/status');
    render(await res.json());
  } catch (e) {
    // transient - next poll will retry
  }
}

setInterval(poll, 1500);
poll();
</script>
</body>
</html>
"""
