"""Configuration constants for the agent loop."""

# --- Target application ---
TARGET_BASE_URL = "http://127.0.0.1:5000"

# --- LLM ---
MODEL_NAME = "gemini-2.5-flash"

# --- Loop control ---
MAX_STEPS = 20
OVERALL_TIMEOUT_SECONDS = 300
LLM_CALL_TIMEOUT_SECONDS = 30
LLM_CALL_MAX_RETRIES = 1
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
AMOUNT_BEARING_ROUTES = [
    (r"^/member/[^/]+/withdraw/confirm$", "amount"),
    (r"^/member/[^/]+/sub-account/confirm$", "deposit_amount"),
]

# --- Guardrails: risk tiering---
# Applied uniformly to withdrawal amount and sub-account initial deposit.
# amount < RISK_TIER_HITL_THRESHOLD -> auto-approved
# amount >= RISK_TIER_HITL_THRESHOLD -> requires HITL approval, no ceiling
RISK_TIER_HITL_THRESHOLD = 10000

# --- Evidence / logging---
EVIDENCE_DIR = "evidence"
