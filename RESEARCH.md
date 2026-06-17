# TRONformer Research Workflow

You are a professional ML researcher improving the Transformer variant (TRONformer) of TRON,
a recurrent DQN agent for limit-order-book market making in PyMarketSim.

**Primary goal:** Maximize `% over ZI` as reported by `eval_only.py --episodes 10000 --seed 42`.  
**Current best Transformer result:** +16.1% over ZI (checkpoint: `out_exp_gtrxl_best.pt`).  
**LSTM baseline for reference:** +14.7% over ZI (checkpoint: `tron_env_c_v2_final.pt`).

Improvement is measured against the previous best Transformer result, not the LSTM.

---

## Directory structure (create on first run)

```
research/
  notes/               # one file per topic, paper/technique summaries
  experiments/         # hypothesis docs and train logs, one per experiment
  log.md               # running results table, append-only
  baseline_metrics.md  # written once at Step 0, never modified
```

---

## Step 0: Scaffold refactor (do this ONCE on master before any experiments)

The two network classes must share identical input projection and dueling heads.
Only the recurrent block may differ between LSTM and Transformer variants.

1. Refactor `pymarketsim/marketsim/tron/network.py`:
   - Extract a shared `InputProjection` module (Linear→ReLU, input_dim→hidden_dim).
   - Extract a shared `FactoredDuelingHeads` module (adv + val heads for s and eta).
   - `TRONNet` and `TRONTransformerNet` both compose these shared modules; only their
     recurrent block (LSTM cell vs. TransformerEncoder) differs.
   - The `build_network` factory signature is unchanged externally.

2. Verify reproduction: run both eval commands from CLAUDE.md and confirm numbers
   match existing README results within ±0.5% over ZI (noise from eval stochasticity).

3. Write `research/baseline_metrics.md` with the verified eval numbers.

4. Commit to master:
   ```bash
   git add -p
   git commit -m "refactor: shared InputProjection + FactoredDuelingHeads scaffold"
   ```

Do not proceed to experiments until Step 0 is complete and verified.

---

## Phase 1: Recurrent block improvements (primary phase)

**In scope:** Only the recurrent block of `TRONTransformerNet` may change.
Input projection, dueling heads, factored action space, and observation space are frozen.
Training recipe (LR, batch size, epsilon schedule, reward clipping, buffer size) and
reward shaping in `tron_env.py` are also in scope.

**Always out of scope:** observation features (the 10-float obs vector), ZI agent
parameters, environment parameters (lam, shock_var, T, N, etc.), order book /
market microstructure.

### Per-experiment workflow

**Research (before touching code):**
1. Search for relevant 2024–2026 papers and techniques using `mcp__exa__web_search_exa`
   and `mcp__fetch__fetch`. Prioritize recurrent sequence models and RL stabilization techniques.
2. Write a summary to `research/notes/<topic>.md`: core idea, applicability to this setup,
   expected impact, implementation cost, citation/link.
3. Select the highest expected-impact / lowest-cost idea. Write a hypothesis doc to
   `research/experiments/<exp_name>_hypothesis.md`: what is changing, why, and the
   expected effect on `% over ZI`.

**Implementation:**
1. Create a branch: `git checkout -b exp/<short-name>`
2. Implement as a small, focused diff — no large rewrites.
3. Confirm the change runs: `./.venv/bin/python -c "import marketsim.wrappers; import gymnasium as gym; print(gym.make('PyMarketSim-TRON-v0').reset(seed=0)[0].shape)"`

**Training:**
```bash
./.venv/bin/python -u -m marketsim.tron.train \
  --arch transformer --episodes 100000 --eps-decay 40000 --buffer-size 50000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --checkpoint out_exp_<name>_final.pt --best-checkpoint out_exp_<name>_best.pt \
  2>&1 | tee research/experiments/<exp_name>_train.log
```

**Evaluation:**
```bash
./.venv/bin/python -u -m marketsim.tron.eval_only \
  --checkpoint out_exp_<name>_best.pt --episodes 10000 --seed 42
```

**After eval:**
1. Append a row to `research/log.md`:
   ```
   | <exp_name> | <change description> | <prev best %> | <new %> | <z-score> | <date> | <decision> |
   ```
2. Explicitly state: **keep** (merge to master), **discard** (leave branch), or **iterate**.
3. If improvement over previous best Transformer %:
   - Merge branch to master.
   - Update `research/baseline_metrics.md` with the new best.
   - Update the "Current best Transformer result" line at the top of this file.
4. If no improvement: do not merge. Leave branch for reference.
5. Commit all research notes and log updates to the experiment branch before deciding.

**Phase 2 trigger:** After 3 consecutive iterations with no improvement over the current
best Transformer `% over ZI`, switch to Phase 2.

---

## Phase 2: Broader architecture changes

Unlocked only after 3 consecutive non-improving Phase 1 iterations.

**Additional scope:** Architecture beyond the recurrent block (e.g., different head designs,
skip connections, normalization strategies, alternative sequence model families).  
**Still always frozen:** observation space, environment parameters, ZI agents, market microstructure.

Same per-experiment workflow as Phase 1 applies.

---

## Checkpoint protocol (every 3 training cycles)

After completing 3 training runs, write `research/checkpoint_N.md` and push to master.
Do **not** start a 4th training run until the human approves continuation.

The checkpoint file must follow this template exactly:

```markdown
# Checkpoint N — YYYY-MM-DD

## Results table
| Experiment | Change | Before % | After % | z | Decision |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

## Current best
Checkpoint file: <path>
% over ZI: <value> (z=<value>)

## What was tried
<2-3 sentences per experiment: what changed, what happened, why>

## Phase status
Phase 1 or Phase 2 — <N> consecutive non-improving iterations (Phase 2 trigger at 3)

## What's next
<Ranked list of 2-3 ideas to try in the next autonomous block, with brief rationale>

## Resume prompt
Paste the following into a new Claude Code session to resume:

---
Continue the TRONformer research workflow defined in RESEARCH.md.
Current best Transformer checkpoint: <path> at <% over ZI> (z=<value>).
Phase: <1 or 2>. Consecutive non-improving iterations: <N>.
Start from the "What's next" section of research/checkpoint_<N>.md.
Proceed autonomously for 3 training cycles, then write checkpoint_<N+1>.md and push to master.
---
```

Push the checkpoint:
```bash
git add research/checkpoint_N.md research/log.md
git commit -m "checkpoint N: <one-line summary>"
git push origin master
```

---

## General guidelines

- One experiment per branch. Branch name: `exp/<short-name>`.
- Merge to master only when `% over ZI` beats the previous best Transformer result.
- Always commit research notes and log updates before ending a session.
- Flag when a technique's claimed results may not transfer (different scale, data
  distribution, compute budget, or non-stationarity of the market simulation).
- Training is ~2.5–3.5 hrs per run on CPU. Do not abort a run once started unless it
  diverges (loss spikes and does not recover within 20k episodes).
- If a run diverges, log it as a failed iteration, note the divergence episode, and count
  it toward the Phase 2 trigger.
- Always include links/citations in research notes for later verification.
