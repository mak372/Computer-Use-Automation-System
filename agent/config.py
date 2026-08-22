"""Configuration constants for the agent loop."""

# --- Target application ---
TARGET_BASE_URL = "http://127.0.0.1:5000"

MODEL_NAME = "gemini-flash-lite-latest"

# --- Loop control ---
MAX_STEPS = 20
OVERALL_TIMEOUT_SECONDS = 300
LLM_CALL_TIMEOUT_SECONDS = 30
LLM_CALL_MAX_RETRIES = 1
LLM_INFRA_RETRY_DELAY_SECONDS = 5
LLM_MAX_REQUESTS_PER_MINUTE = 5
ACTION_TIMEOUT_SECONDS = 15
NO_PROGRESS_THRESHOLD = 3

# --- Perception ---
SCREENSHOT_PERIODIC_INTERVAL = 3

# --- Guardrails: allowlist ---
# Only the routes needed for the 3 supported goals. Deliberately excludes
# /admin/delete-member/<id> (off-scope decoy) and
# /member/<id>/contact/confirm (ambiguous-button decoy) - both are
# guardrail-blocked by omission, not by an explicit denylist entry.
# Each entry is (regex pattern, {allowed HTTP methods}).
ALLOWED_ROUTES = [
    (r"^/$", {"GET"}),
    (r"^/lookup$", {"GET"}),
    (r"^/member/[^/]+$", {"GET"}),
    (r"^/member/[^/]+/sub-account/new$", {"GET", "POST"}),
    (r"^/member/[^/]+/sub-account/confirm$", {"POST"}),
    (r"^/member/[^/]+/withdraw$", {"GET", "POST"}),
    (r"^/member/[^/]+/withdraw/confirm$", {"POST"}),
]

# Routes whose click carries a committed dollar amount and must go through
# risk-tier evaluation before executing (the final "Confirm" actions only -
# the earlier form POSTs are non-committal, just "continue to confirmation").
# Maps (regex pattern, hidden field name holding the ground-truth amount).
# Sub-account opening is deliberately excluded: unlike withdraw_funds, it
# never deducts from member.balance (target_app/app.py's sub_account_confirm
# only appends a new SubAccount record) - no funds leave the member's
# relationship with the institution, so amount-based risk tiering doesn't
# apply the way it does to a real withdrawal outflow.
AMOUNT_BEARING_ROUTES = [
    (r"^/member/[^/]+/withdraw/confirm$", "amount"),
]

# --- Guardrails: risk tiering---
# Applies to withdrawal amount only - see AMOUNT_BEARING_ROUTES above.
# amount < RISK_TIER_HITL_THRESHOLD -> auto-approved
# amount >= RISK_TIER_HITL_THRESHOLD -> requires HITL approval, no ceiling
RISK_TIER_HITL_THRESHOLD = 10000

# --- Evidence / logging---
EVIDENCE_DIR = "evidence"

# --- HITL operator console (Section 3.6) ---
# Local-only web console for approving/resolving escalations - a second,
# additive surface alongside the terminal prompt (operator.py), never a
# replacement for it. See operator.py's module docstring for the
# race-to-first-response mechanism.
OPERATOR_UI_PORT = 5050

# --- Artifacts (Section 3.2) ---
# One canonical file per capability, evidence/artifacts/{goal_key}.json,
# overwritten on rebuild - git history is the version history, not multiple
# on-disk files.
ARTIFACTS_DIR = "evidence/artifacts"
