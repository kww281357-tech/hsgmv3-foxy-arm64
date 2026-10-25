# Execution, information and spending boundaries

## Public policy vs privileged environment

The policy is given the original instruction, present raw RGB, supplied current
observations, measured executed actions, legal action IDs, aliases of visible
helpers, offer/budget state and an explicitly delivered English answer. It does
not receive Episode, environment handles, goal coordinates, full paths, private
pose, anchor coordinates, true-goal distance, goal images or success metrics.
The lost detector uses observed repetition/progress/collision/uncertainty only.

The offline route author has scene and annotated-goal privilege. The evaluator
also has goal distance and success privilege. The runtime oracle bridge verifies
the frozen card, discards the goal after binding, and retains the local connector
and text, not a whole-path planning handle. Its environment still performs trusted
safety checks. This is an auditable in-process design boundary, **not an OS
security sandbox against malicious Python plugins**. Do not inject private
information through supposedly public free text.

Only `accept` authorizes an answer; arrival/eligibility/offer is not acceptance.
A 1 m inclusive horizontal disk uses dummy foot centre, never just disk-edge
intersection or body surface. Same-floor tolerance is0.25 m in this adapter;
LOS, current visibility and short safe reachability must also pass. The state
machine uses10-step cooldown,1.2 m strict exit threshold,5-step offer expiry and
30-step pending-answer expiry. Budget≤3 is updated atomically on valid delivery;
failed/expired/duplicate callbacks, decline and ignore are free of **question**
charges. Whole reviewed answers are bounded to8192 characters without truncation.

The current added-person protection combines real scene NavMesh motion with a
conservative swept body proxy. Debug circle is not a collision object. Coarse
radial rays/contact/approach checks do not prove every scene route remains
connected after adding people. A staged clear position is not full topology proof.
Do not claim the imported noncollidable mesh alone prevents walking through people.

## Provider gate

Go official documentation checked on2026-10-25:
- https://opencode.ai/v2/docs/console/go/
- https://opencode.ai/v2/docs/console/api/inference/

The selected model is `deepseek-v4.1-flash`, not a free-tier substitute.
Fixed endpoint: `https://opencode.ai/inference/go/openai/v1/chat/completions`.
Docs describe Go as designed for OpenCode/coding agents and typical coding-agent
traffic; direct applications are documented. This is not an explicit blanket
ban on visual navigation, nor proof of provider approval for every workload.
Do not impersonate OpenCode/coding traffic. Our identifying User-Agent is
`r2rce-navigation-research/0.3`, with stable session ID. No unknown second provider.

A public model listing is not vision validation. Require one real image probe,
schema/returned-model checks and human image comparison before a live trial.
Model/region access may require account settings; the program never changes them.

**Extra Usage OFF must be verified in the actual account before any request.**
The CLI boolean records operator confirmation; it cannot read the billing switch.
Go may otherwise fall back to paid balance after plan limits. A response cost
check is too late to prevent the first charge; it is only a secondary stop check.
The current cloud account gate remains pending, and this source organization has
made zero authenticated model calls.

Transport: one request at a time, ≥3 seconds between requests, probe cap1,
trial cap≤160, timeout90s, no redirects, no automatic retry, no provider/model
fallback, no top-up or extra purchase. It checks billing_source go/go-plus,
zero reported balance cost and valid allowance. A0.50 USD allowance-consumption
threshold is checked between responses; this is **not an atomic provider spend
limit**. Timeout has uncertain outcome and stops rather than resending.

Only an interactive no-echo terminal may supply an explicitly authorized
transient key. Never pass it in arguments, environment variables, shell history,
source, logs, repository or cloud files. Reports omit raw provider errors and
headers. Close clears references but does not promise cryptographic memory erasure.
Exposed chat keys should normally be revoked/replaced; this project does not
retrieve chat messages or configure persistent credentials.

## Evaluation boundary

`survey` is privileged offline path following. `smoke` is a legacy scripted
interaction diagnostic. `trial --fixture-preflight` is real-simulator/canned-model
wiring. None is autonomous VLM navigation. The single live trial overrides the
original episode start to stand by a helper and therefore sets benchmark
success/SPL to null. A private staged-goal-region check must not be presented as
an R2RCE benchmark score. Original-start paired evaluations are separate future
runs with same episodes/seeds and explicit fresh output directories.

### Legacy baseline vs latest full-route trial

`runner.run_episode` and the generic `harness.run_paired` still default to the
legacy `RuleDirectionOracle`: vehicle-origin, at most3 local segments/8 m. That
baseline is retained for regression/diagnostic use, NOT presented as the user's
latest full-route design. Only `trial` explicitly binds the reviewed helper-side
anchor card. A future20-episode full-route paired run needs independently reviewed
cards and an explicit oracle factory per episode; this source does not claim
those20 cards exist or silently use the legacy oracle as if they did.

### GAME profile recognition

The game CLI uses current semantic-instance pixels rather than blue-color
recognition. This is simulator-assisted helper identification, not an actual
YOLO person detector. It exposes no semantic map or hidden actor location to
policy. At least8 current instance pixels, real Bullet first-hit LOS and all
range/floor/path/consent checks still apply. The existing v7 neutral visual mesh
must fit the unchanged conservative .32 m radius /1.6 m height proxy, and feet
use0 offset. Its yaw faces the fixed safe anchor, independent of the goal. Blue
remains an explicit reference profile, never a missing-game fallback.
