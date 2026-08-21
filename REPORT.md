## Architecture
The implementation targets a web application. The system receives a natural-language goal. Before the task starts, one LLM call maps the goal to one of the three supported capabilities: looking up a member's balance, withdrawing funds, or opening a sub-account. If the call fails, returns an unsupported capability, or cannot find a match, the system does not guess; it asks a human to choose. This makes uncertain classification a human decision rather than an incorrect automated action.

**Discovery**
At each step, the page is observed, the LLM decides the next action (click, type, hover, etc.), a deterministic guardrail checks whether that action is allowed before anything happens, only then is it executed, and the result is verified independently rather than trusting the LLM's own claim of success. This repeats step by step until the capability's checkpoint confirms the goal is done. The goal's parameters (e.g. member_id, amount) are injected into the LLM's context at every step, not just the first, so the model can use them at whichever step actually needs them. The page is represented primarily as accessibility-tree representation, each element is identified by its type of control (e.g. button, textbox) and its visible text label (e.g. "Submit"), plus which one it is in order if more than one element shares both. I chose this over screenshots because text is cheaper since it consumes less tokens and easier for the model to reason about. Screenshots are captured only at defined triggers (first step, an empty or duplicate-labeled element list, periodically, before declaring the goal complete, before risky-amount approval, and after execution errors). Each LLM call is also rebuilt fresh with system instruction, a one-line log of prior steps, and the current observation rather than an accumulating conversation, keeping token cost less as a run grows, at the cost of the model having no memory of *why* it made an earlier choice.

**Replay**
Replay is given a goal and its parameters. It locates the artifact for that goal and follows its recorded steps in order, re-grounding each element live on the page right before acting the same role/label/position description discovery recorded. The same guardrail and verification checks from discovery still run on every step. If a step can't proceed as recorded, the element has drifted, an action gets blocked or execution fails then the system tries to resolve it automatically first and only hands off to a human if that doesn't work.

## Artifact schema
A minimal fragment of a reviewed `withdraw_funds` artifact is shown below. The other capabilities follow the same structure, with differences in their inputs, actions, and expected outcomes.
{
      "schema_version": 1,                // Tracks changes to the artifact's format
      "capability_version": 1,            //  Tracks changes to the workflow itself
      "goal_key": "withdraw_funds",       // tells what underlying goal is being trying to achieved
      "parameters": ["member_id"],        // inputs the caller must supply to run this capability
      "status" : "draft" / "reviewed"     // change status from draft to review once all steps in artifact are checked for correctness
      "steps": [{
        "action": "type",                 // the browser action to perform - type, click, select, hover, etc.
        "value": {"param": "member_id"},  // value to use - here, substitute the caller's member_id parameter
        "grounding": {"role": "textbox", "name": "Member ID", "nth": 0}   // identifies the element on the current page
      }],
      "checkpoint": {                     // how replay decides what actually happened once each action is done
        "success": {                      // the page evidence that means this run succeeded
          "url_pattern": "^/member/[^/]+/withdraw/confirm$", // regex the page URL must match
          "text_signature": "completed. Remaining balance"   // text that must appear on the page
        },
        "known_outcomes": [{                // other legitimate, non-success outcomes replay can recognize instead of treating them as failures
          "outcome": "insufficient_funds", // label for this recognized outcome
          "url_pattern": "^/member/[^/]+/withdraw$",                  // regex the page URL must match for this outcome
          "text_signature": "Insufficient funds for this withdrawal" // text that must appear on the page for this outcome
        }]
      }
    }
A successful discovery run does not automatically become a trusted artifact. New artifacts start as `draft` because discovery may not always correctly determine which values should become reusable inputs or which page elements an action should use. A human reviews these decisions before the artifact is approved for replay. This review happens once, when the artifact is built not on every replay run.The final artifact stores the reusable workflow, not the specific values from that run. A reviewed workflow cannot be changed without giving it a new `capability_version`. If replay is given an older version, it refuses to run instead of silently using a newer workflow. This makes changes explicit and prevents unexpected behavior.

## Determinism & Error Handling
Given the same artifact and inputs, replay performs the same actions in the same order every time. The application's data can still change between runs, so replay is deterministic about the procedure, not the final result for example, the same withdrawal for the same member may succeed today but return insufficient_funds later if the member's balance has changed. The same principle applies to errors: the system responds based on what actually went wrong, rather than treating every problem as an automation failure.

**If the caller provides an invalid value**, such as a withdrawal method that the page does not offer, the run fails immediately and reports the valid options. Retrying cannot make an invalid input valid.
**If the caller leaves an optional value out**, the system waits until it reaches the step that needs it and then asks a human for that value. This avoids requesting information the workflow may never need.
**If the page has changed**, such as a recorded button disappearing or leading somewhere different, replay detects this immediately before acting by checking that the element still exists and behaves as expected. 
**If an action appears to fail**, such as a timeout, the system first checks whether the application actually completed it. A known case is handled automatically; if the result is still unclear, a human can inspect the live page. 
**If the problem remains unresolved**, the system gives the same step one final retry, re-finding the element and re-checking its safety before trying it again. The outcome is checked once more before recording a structured failure.

## Heterogeneity & Multi-tenant
Only two parts of the system are browser-specific, observing the page and performing actions. Decision-making, safety checks, outcome verification, human handoffs, and logging works with the information those parts provide, regardless of the underlying surface. Supporting a desktop or legacy application would therefore mainly require replacing those two parts rather than rebuilding the core pipeline. The same principle carries over to different tenants, but the boundary shifts: instead of a browser-specific component absorbing the variation, it's the artifact itself. The system doesn't automatically detect or adapt to differences between tenants' workflows that judgment happens during human artifact review, and gets recorded as a separate `capability_version`. A caller can optionally pin the `capability_version` it expects; if it does, replay refuses to run against a different version rather than silently assuming compatibility. A caller that doesn't pin one just gets whatever version the artifact currently is.

## Escalation & Handoff
The system stops and asks a human whenever it has observable evidence that continuing isn't safe, rather than trying to detect "confusion" with one complex mechanism. The flow has three parts: what triggers a handoff, how the human actually steps in, and what happens once they're done.
**Detection** - A handoff triggers when the page stops changing across several attempts, the run exceeds its step/time budget, the model explicitly says it can't safely continue, the goal can't be mapped to a known capability, or a monetary action is at or above the approval threshold. These are different failure modes, but all get the same response: ask a human rather than guess.
**Intervention** - The human gets control of the exact same live browser window the automation was just driving, so nothing is copied or transferred, and the human can act on the page directly for example, if a lookup was missing its `member_id`, the human can type it into the field themselves. Once the human approves the change, automation resumes from whatever state the page is now in.
**Follow-up** - Once the flow resumes, the system re-observes the live page and runs the same outcome checks it would after any automated step, so the human's action isn't trusted blindly either. A before-and-after snapshot is recorded around the handoff, not a full click-by-click trace, since that is not really needed to check what change human intervention did. An explicit human abort is logged separately from an unresolved automation failure.

## Safety
A decision to act, whether the LLM proposes it in discovery, or an artifact step supplies it in replay, never becomes a browser action automatically. Every action passes through the same deterministic guardrail first, which checks two things before letting anything execute: is this action even part of what the current capability is permitted to do, and if it involves money withdrawal, is the amount below the approval threshold. If either check fails, the action never happens. This is the same guardrail code in both modes; only who's proposing the action differs.

**During discovery:**
- The guardrail runs before every LLM-proposed action. The allowed-action check uses an explicit allowlist of permitted pairs and anything outside it is rejected by default, which gives an inspectable boundary instead of relying on the model to avoid every possible bad action.
- For monetary actions like withdrawing money in my case, the guardrail reads the amount directly from the live page immediately before execution rather than trusting the model's claimed value; amounts below the threshold proceed automatically, amounts at or above it require human approval, and an unreadable value blocks execution outright.
- A separate progress check guards against the model simply looping instead of acting unsafely: after each step, the system hashes the same URL + interactive-element text the model reasons over, and if it's unchanged across 3 consecutive attempts, the run hands off to a human. It exists to catch an LLM stuck making no real progress, a failure mode replay doesn't have since nothing is deciding anything.

**During replay:**
- The exact same allowlist and monetary-threshold checks run again before every recorded step, re-read live rather than trusted from the artifact, a step being pre-reviewed doesn't exempt it from the guardrail, since the live page can still differ from what was recorded.
- There's no progress-hash check here; replay's failure modes (grounding drift, a blocked action, an execution failure) are instead handled by the layered recovery process described in Determinism & Error Handling.


## Cuts
Several limitations are intentionally deferred because the current system does not yet justify their complexity; others are genuine extensions as the system grows.
**New capability setup:** Checkpoints which tell what counts as success or a known outcome are currently written manually by a human for trust, so adding a new capability means writing those for each new capability. Future tooling could instead draft candidate checkpoints from an actual discovery run's observed pages, leaving a human to just review and approve them rather than write them blind. With only 3 capabilities so far, this cost hasn't been felt yet, but it won't scale as more are added.
**Comparing valid paths:** Discovery step currently follows the first sensible path rather than comparing alternatives. With three capabilities there is little reason to explore alternatives; as institutional customization creates multiple valid routes, discovery step should compare them and prefer the more efficient or safer one.
**Canvas-only interfaces:** The current grounding approach depends on an accessibility tree existing at all, a canvas-rendered UI (e.g. a `<canvas>`-based app  with no real DOM elements) exposes none, so role/name-based grounding has nothing to bind to. That surface would need a different, visual-only fallback like taking screenshots at every step rather than an extension of the current approach.
**Guardrail Threshold:** The threshold for withdrawing money requiring human approval is one fixed value shared by the whole system, not something that can be set differently per institution even though different institutions would realistically want different limits. To fix this each institute will get its own threshold value in its withdrawal artifact.
**URL-dependent progress detection:** The hash calculation currently includes the page URL, which is a web-specific concept  a desktop or canvas-only surface wouldn't have one. Extending to those surfaces means the hash would need to fall back to just the neutral element/observation representation, dropping URL as an input rather than assuming it always exists.
**Guardrail scope:** The guardrail only evaluates single monetary amounts against a threshold — it doesn't flag other irreversible actions that have no dollar figure (e.g. permanently closing an account), and it doesn't track cumulative effects across multiple actions in one run (e.g. five withdrawals that individually clear the threshold but add up to a large total). Covering either would need a broader risk-policy model, not just an adjustment to the existing check.