"""Configuration constants for the agent loop."""

# --- Target application ---
TARGET_BASE_URL = "http://127.0.0.1:5000"

# --- LLM ---
# "gemini-2.5-flash" (the originally chosen model, decision #1) was
# retired for new API keys between design and first live test - see
# REPORT.md. Switched to "gemini-flash-latest" ("-latest" alias instead of
# a pinned version string so this doesn't hit the same wall again as models
# get retired), which then resolved to "gemini-3.7-flash" - but that model's
# free tier turned out to be capped at 20 requests/DAY on this key (a real
# 429 body showed quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
# quotaValue 20), not just a per-minute limit - a single test day of manual
# runs exhausts it completely, and no amount of pacing within a run fixes a
# per-day bucket that's already empty. Switched again to
# "gemini-flash-lite-latest" (resolves to "gemini-3.5-flash-lite"), whose
# free tier is far more generous since lite-tier models get their own,
# larger quota bucket separate from flagship/preview models. See REPORT.md.
MODEL_NAME = "gemini-flash-lite-latest"

# --- Loop control ---
MAX_STEPS = 20
OVERALL_TIMEOUT_SECONDS = 300
LLM_CALL_TIMEOUT_SECONDS = 30
LLM_CALL_MAX_RETRIES = 1
# Short fixed pause before retrying an infra-type failure (503/429/network).
# Deliberately not honoring the API's own suggested retry-after delay
# (which can be 30s+ for quota exhaustion) - that would eat
# disproportionately into one step's budget for uncertain benefit. This
# only gives genuinely transient capacity blips a chance to clear;
# quota exhaustion needs spacing out test runs, not a longer sleep here.
LLM_INFRA_RETRY_DELAY_SECONDS = 5
# Originally set to 5 after a 429 against gemini-3.7-flash, assumed at the
# time to be a per-minute limit. A later 429 (same model) showed the real
# quotaId was "GenerateRequestsPerDayPerProjectPerModel-FreeTier" - a
# 20-requests-PER-DAY cap, not per-minute - which no amount of within-run
# pacing can fix once exhausted; see the MODEL_NAME comment and REPORT.md
# for the resulting model switch. Kept as a conservative per-minute pacer
# regardless (cheap insurance against real per-minute limits on whichever
# model is configured), just not the fix for that specific incident.
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

# --- Artifacts (Section 3.2) ---
# One canonical file per capability, artifacts/{goal_key}.json, overwritten
# on rebuild - git history is the version history, not multiple on-disk files.
ARTIFACTS_DIR = "artifacts"
