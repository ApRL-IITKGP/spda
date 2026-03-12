# REP: Running Thoughts

*Reward Engineering Playbook — CoRL 2026*
*Last updated: 2026-03-11*

---

## The Core Claim (what this paper is trying to say)

The mathematical geometry of a dense reward function — its curvature, support, monotonicity, smoothness — has a causal, first-order effect on PPO training dynamics for robot manipulation. This dependency is not captured by potential-based shaping theory, and no systematic empirical study exists. We want to produce one, and distil the findings into an actionable reward design playbook.

---

## Taxonomy of Reward Dimensions

### 1. Support Type *(finite / infinite)*
Is the reward exactly 0 beyond some radius R?
- **Finite**: `linear`, `concave_truncated`, `piecewise_cc`
- **Infinite**: `tanh`, `exponential`, `quadratic`, `tanh_sq`

**Mechanism hypotheses:**
- Finite support eliminates *hover attractors* — the policy cannot extract non-zero reward from hovering near the object, so it must commit to grasping.
- Finite support causes *signal starvation* — once the reaching phase is consistently solved, the truncated reward gives zero gradient and the policy drifts.

### 2. Near-Goal Gradient *(zero / non-zero at d=0)*
What is `∂r/∂d` as d→0?
- **Zero gradient at goal** (parabola vertex): `tanh_sq`, `concave_truncated`, `quadratic`
- **Non-zero gradient at goal**: `tanh`, `exponential`, `linear`

**Mechanism hypothesis (H2):** Zero gradient at goal means no pull once contact is made → cleaner stage handoff to grasping reward. Non-zero gradient keeps pulling the arm into the object even after contact.

### 3. Global Monotonicity *(monotone / non-monotone)*
- **Monotone**: all except `oscillatory`, `impulse_concave`
- **Non-monotone**: local minima/maxima create gradient traps → policy gets stuck at a local maximum of reward that isn't the task goal.

### 4. Smoothness / Differentiability *(C∞ / C¹ / C⁰ / discontinuous)*
- **C∞**: `tanh`, `exponential`, `quadratic`, `tanh_sq`, `oscillatory`
- **C⁰ not C¹**: `linear`, `piecewise_cc` (kink at d=0.2)
- **Discontinuous**: `sparse_threshold`

PPO updates through the reward function implicitly (via value targets). Discontinuities and kinks cause gradient variance spikes. `sparse_threshold` crashes.

### 5. Far-Field Curvature *(convex / concave away from goal)*
`∂²r/∂d²` for large d:
- **Convex** (gradient grows as you approach from far): `exponential`, `linear`
- **Concave** (gradient shrinks as you approach from far): `quadratic`, `tanh`

Affects exploration: convex far-field gives stronger guidance when far away; concave front-loads gradient near the goal.

### 6. Sign / Floor *(non-negative / can go negative)*
- Negative floor variants add a small constant negative reward beyond the support threshold.
- This is *not* a time penalty — it penalises distance at every step, not episode length per se.
- The `-0.05` shift also reduces peak reward from 1.0 → 0.95, breaking value function normalization assumptions.

---

## Experimental Results

### Phase 1 — Single-seed, 2M steps (seed=1, PickCube-v1, hard gate, hard_jump terminal)

| Variant | eval/success_once |
|---|---|
| `concave_truncated` | **0.875** |
| `oscillatory` | 0.625 → collapses to 0.0625 |
| `tanh_sq` | 0.1875 → collapses to 0 |
| `exponential` | 0.125 → collapses to 0 |
| `linear` | 0.0625 |
| `tanh` | 0.0 |
| `sparse_threshold` | crashed |

Explained variance ~0.96 for all variants — value function health rules out as cause of failure.

Initial interpretation: finite support is the key. H1 (hover trap elimination) appears confirmed.

### Phase 2 — Multi-seed, 5M steps (seeds 1, 42, 123)

| Variant | Seed 1 | Seed 42 | Seed 123 |
|---|---|---|---|
| `tanh` | 0 | **1.0** | **0.875** |
| `tanh_sq` | 0 | **1.0** | **0.9375** |
| `concave_truncated` | 0.625 (↓ from 0.875) | 0.3125 | 0.0625 |
| `concave_truncated_negative_noslope` | 0.125 | — | — |

**This broke the simple story.** Seed 1 is a "hard" initialisation — only finite-support survives it. Seeds 42/123 are "easy" — infinite-support variants dominate while concave_truncated collapses.

---

## Findings So Far

### F1 — Finite support prevents hover attractors (H1: conditionally confirmed)
On hard initialisations (seed 1), only `concave_truncated` reaches meaningful success. On easy initialisations, the advantage disappears entirely.

### F2 — Finite support causes oscillation / mode-switching, not monotone drift (revised finding)
`concave_truncated` full 5M trajectory (seed 1): 0→0→0→0.6875→0.6875→0.875→0.875→0.75→**0.4375**→0.5625→0.9375→0.9375→0.9375→0.9375→**0.625**.
The mid-training dip to 0.4375 at 2.88M followed by recovery to 0.9375 is a policy phase transition — not monotone drift. The policy unlearns and relearns a better grasping strategy. The final drop to 0.625 at 4.8M suggests the policy keeps cycling between behavioral modes rather than converging.

"Signal starvation" was the wrong framing. More accurate: finite support creates a **discontinuous incentive boundary** (reward abruptly goes to zero at d=0.2) that the policy repeatedly crosses, leading to mode-switching instability rather than smooth convergence or monotone degradation.

### F3 — Near-goal curvature does NOT independently predict success (H2: not confirmed)
Cleanest comparison: `tanh` (convex, non-zero gradient at d=0) vs `tanh_sq` (concave, zero gradient at d=0). Nearly identical across all seeds. Curvature at goal matters only conditionally (perhaps only in finite-support regime).

### F4 — Seed-dependent bifurcation (new finding)
The same reward geometry can lead to wildly different outcomes depending on initialisation. Finite support is *robust to hard initialisations*; infinite support is *robust to starvation*. This is a tradeoff, not a hierarchy.

### F5-new — Bimodal landscape (impulse_concave) flatlines at 0
`impulse_concave` (seed 1, 5M): 0.0 throughout. Primary bump at d=0 + secondary bump at d=0.3 creates a saddle region between d=0.2 and d=0.3 where gradient points in conflicting directions. The secondary bump does not help pull the robot from far away; it creates a trap. Bimodal gradient landscapes are harmful.

### F6 — Non-monotone gradients cause collapse (confirmed)
`oscillatory` peaks then collapses. Gradient traps from local maxima are genuinely harmful.

### F7 — Tighter support radius (R=0.1) slows learning ~2.5× but reaches same peak
`concave_truncated_10` (R=0.1, zero at d=0.1) vs `concave_truncated` (R=0.2, zero at d=0.2), seed 1:
- concave_truncated reaches 0.6875 at 1.28M; concave_truncated_10 doesn't hit that until ~3.2M.
- Both peak near 0.875–0.9375 but neither holds it — same oscillation instability.
- Narrower support delays but doesn't prevent success. The final 0.875 at 4.8M for concave_truncated_10 is real but was preceded by erratic oscillation the whole way.
- Implication for adaptive curriculum: starting narrow is like starting the hard end of the curriculum. Starting wide (R=0.5) and shrinking to narrow should give early speed + late precision without the instability.

### F8 — Negative floor hurts (likely via normalization break)
`concave_truncated_negative_noslope` drops significantly vs `concave_truncated`. The -0.05 shift reduces peak reward to 0.95, distorting value targets. May not be a fundamental finding about negative rewards per se — possibly just a normalisation artifact.

---

## The Revised Story: Robustness-Ceiling Tradeoff

*Original story (single seed)*: finite support wins.

*Revised story (multi-seed)*: reward geometry creates a **robustness-ceiling tradeoff**:
- **Finite support**: initialisation-robust (survives hard seeds), but convergence-unstable (discontinuous incentive boundary → mode-switching oscillation, policy never fully settles).
- **Infinite support**: convergence-stable (smooth gradient everywhere), but initialisation-sensitive (hover traps on hard seeds → never gets off the ground).

Neither is universally better. The operative question becomes: **can we get both?**

---

## The Proposed Fix: Adaptive Support Radius (Reward Geometry Curriculum)

Shrink the support radius R over training:
- **Early training**: wide support (R ≈ 0.5m) → coarse gradient landscape, signal everywhere, no starvation.
- **Late training**: narrow support (R ≈ 0.15m) → precise gradient, hover trap eliminated, forces commitment.

Schedule options:
1. **Step-based**: linearly decay R from 0.5 → 0.15 over N steps.
2. **Threshold-triggered** (most principled): shrink R when `eval/success_once > 0.5`. Adapts to actual learning progress.

This is a *reward geometry curriculum* — not task difficulty curriculum, but spatial precision curriculum. Analogous to successive approximation in operant conditioning (biological motivation: reinforcing progressively closer approximations to target behaviour).

**Linear schedule tested (seed=42, 5M steps): FAILED.**
Trajectory: 0→0→0→0→0→0.75→0.375→0.25→0.125→0.5625→0.3125→0→0→0→0.
The radius decays from step 0 regardless of learning progress. At 1.92M the policy reached 0.75 (R≈0.208), but the continuous shrink to R=0.1 undercut the policy while it was mid-training. Complete collapse by 3.84M. Worse than fixed `concave_truncated` on an easy seed.

**Root cause**: continuous decay creates a moving target — the policy can't converge on a landscape that keeps changing. Curriculum must be *progress-gated*, not time-gated.

**Threshold schedule tested (seed=42, 5M steps): SUCCESS.**
Trajectory: R=0.4 held until 1.28M (success=0.75), then stepped 0.4→0.3→0.2→0.1 across three consecutive evals. Brief dip to 0.5625 at R=0.1 transition, then stabilised. Final: **1.0**.
Compare: fixed `concave_truncated` (R=0.2) got 0.3125 on seed=42. Threshold adaptive got 1.0 — the wide early support let the policy get a foothold before narrowing.

**Seeds 1 and 123 results:**
- seed=123: R stepped 0.4→0.3→0.2→0.1 successfully, hit 1.0 at 3.52M, oscillates after. Good.
- seed=1: R=0.4 for the entire 5M run. Never crossed 0.5 threshold. Peak 0.125 at 1.28M, then 0 forever.

**Chicken-and-egg failure on hard seeds**: needs success to narrow R, but needs narrow R to get success. R=0.4 is actually *worse* than R=0.2 on hard seeds — wider support = stronger hover attractor = robot happily hovering at d=0.35m for the whole run.

**The tradeoff is fundamental.** No static or threshold-adaptive geometry wins across all seeds. The characterisation of this tradeoff IS the contribution.

**Possible remaining test**: `r_initial=0.2, r_final=0.1, n_stages=2` — start narrow enough to kill the wide hover trap (protecting hard seeds), threshold-shrink to R=0.1 for precision (helping easy seeds). Avoids the chicken-and-egg by using R=0.2 as the baseline rather than R=0.4.

---

## Open Hypotheses (not yet tested)

**H2-clean**: `quadratic` (concave, infinite support, zero gradient at d=0) vs `tanh` (convex, infinite support, non-zero gradient at d=0).
This is the cleanest test of whether near-goal curvature matters independent of support.

**H-starvation**: Does adaptive support radius eliminate the starvation problem while preserving initialisation robustness?

**H-smoothness**: Is `linear` underperforming (despite finite support) because of the C⁰ kink? Compare `linear` vs `concave_truncated` — both finite, different smoothness.

**H-negative-normalisation**: Does the negative floor hurt because of the reward shift (peak < 1.0) rather than the negativity per se? Test: rescale `concave_truncated_negative` so peak stays at 1.0.

---

## Experiments Still Needed

| Experiment | Purpose | Priority |
|---|---|---|
| `quadratic`, seeds 0 2 3, 5M | H2-clean: curvature at goal, infinite support | **High** |
| `concave_truncated` + `tanh`, seeds 0 2 3, 5M | Quantify hard-seed fraction | **High** |
| Adaptive support radius | Fix starvation-robustness tradeoff | **High** |
| `linear` vs `concave_truncated`, more seeds | H-smoothness | Medium |
| `concave_truncated_negative` rescaled to peak=1.0 | H-negative-normalisation | Medium |
| Gate ablation (soft, additive, curriculum) | Dimension 2 of original taxonomy | Medium |
| Transfer to PushCube, StackCube, PlaceSphere | Generalisability | Low (after above) |

---

## What's Required for CoRL Submission

**Minimum publishable story:**
1. The robustness-ceiling tradeoff (finite vs infinite support) with 3+ seeds.
2. Adaptive support radius experiment confirming it resolves the tradeoff.
3. Evidence on 2+ tasks (PickCube + one more).
4. Clean near-goal curvature ablation (quadratic vs tanh).

**Nice to have:**
- Smoothness ablation (linear vs concave_truncated).
- Gate dimension results.
- Gradient norm / landscape visualisation supporting the mechanistic claims.

---

## Biological / Cognitive Parallels (for intro/discussion)

- **Hull's goal-gradient hypothesis**: drive increases as animal approaches goal. Concave reward (steep far, near-zero at contact) matches this. But the data don't support curvature mattering independently — so use carefully.
- **Dopamine RPE silence at goal**: finite support = no prediction error for hovering near goal = policy not reinforced for hovering. Strong analog for H1. Well-supported.
- **Motor chunking**: clean sub-goal completion signal needed for stage transition to next chunk. Finite support provides exactly this — abrupt zero means "stage done, move on." Supports the stage-handoff framing.
- **Successive approximation (operant conditioning)**: adaptive support radius is the direct RL analog. Reinforcing progressively closer approximations. Use for motivation of the curriculum fix.
- **Signal starvation warning**: biological systems don't have this problem because next-stage motivation activates automatically. In the robot, there's a dead zone between reaching and grasping success. The biological analogy actually *predicts* this failure mode if taken seriously.

---

---

## Stage-Conditioned Reward Geometry (new theory, 2026-03-12)

### The Core Insight

Within-episode time ≠ within-episode task progress on hierarchical tasks.

`adaptive_tanh_episode` schedules k(t) = 2 + 18·(t/T) — time-based.
On PickCube (flat, one stage): time ≈ progress. Works.
On StackCube (two committed stages): at t=T/2 with k already ~11, the robot may still
be in stage 1 (reaching for lower cube). The tight gradient starves it mid-reach.
Never consolidates grasp. Collapses.

The fix is to **condition the schedule on task progress, not time**.

### `adaptive_tanh_stage` Design

```
k(t_stage, T_stage) = k_initial + (k_final - k_initial) * (t_stage / T_stage)
```

Where `t_stage` = steps elapsed since the current stage began (resets to 0 at each
transition), and `T_stage` = max_episode_steps (full budget per stage).

- Every stage starts with k=2 (wide, exploration-friendly)
- Every stage ends with k=20 (tight, forces commitment)
- On flat tasks (1 stage): identical to `adaptive_tanh_episode`
- On StackCube (2 stages): stage 1 gets full wide→tight curriculum; stage 2 RESETS
  to wide (k=2) targeting placement. Full exploration budget for placement.

### Why the reset matters

`adaptive_tanh_episode` fails on StackCube because by the time the robot grasps
(mid-episode), k is already ~11 (half-tight). The robot enters the placement stage
with a tight reaching gradient toward the wrong target (the lower cube), starving
placement exploration.
With reset: the moment grasp succeeds, k resets to 2 (wide) targeting the new sub-goal.
Full exploration budget for placement.

### The Universal Reward Function

Reward geometry should be scheduled by **task progress stage**, not by time.
The universal reward function is:

```
r(d, stage) = 1 - tanh(k(stage) * d)
where k(stage) = k_initial + (k_final - k_initial) · progress(stage)
```

where `progress(stage)` is:
- 0 at the start of a new sub-task
- 1 at sub-task completion

This unifies `adaptive_tanh_episode` (flat tasks: progress = t/T ≈ stage progress)
and `adaptive_tanh_stage` (hierarchical tasks: progress = task-specific boolean).

Any task can be decomposed into stages (even flat tasks have one stage).
The only task-specific input is: "what is the current stage?" (already in evaluate()).

### F11 (Predicted) — `adaptive_tanh_stage` is universally optimal

Expected prediction:
| Task | `tanh` | `concave_truncated` | `adaptive_tanh_episode` | `adaptive_tanh_stage` |
|---|---|---|---|---|
| PickCube | 0.625 | 0.960 | **0.979** | ~0.98 (identical) |
| PushCube | 0.854 | — | **0.958** | ~0.96 (identical) |
| StackCube | 0.271 | **0.896** | 0.021 | **~0.9+** (fixes mismatch) |
| LiftPegUpright | 0.792 | — | **0.854** | ~0.86 (identical) |

If StackCube > 0.85 across seeds, `adaptive_tanh_stage` becomes the universal best
variant — a single reward function that works across flat AND hierarchical tasks.

### Relation to Option-Critic

Loosely related but we don't need it. Option-Critic learns temporal abstractions
(options + termination conditions). Our case: task stages ARE given (is_grasped,
is_lifted come from task evaluate()). We don't need to learn when to switch — we
know from the task definition. This is a finite automaton over task stages where each
state has its own reward geometry curriculum.

More relevant works:
- **DrS (2024)**: learns stage-specific dense rewards from sparse signal — closest,
  but studies magnitude not geometry.
- **HPRS (2024)**: hierarchical potential-based shaping — potential functions, not curvature.

The gap REP fills: none of these study *functional form* (curvature, support radius,
convexity) as a function of task stage.

### `concave_truncated` static approximation

`concave_truncated` with its zero-gradient zone is approximately doing
stage-conditioned geometry: once tcp enters d < R, the reaching reward goes silent
(k effectively → ∞), forcing the policy to commit to the next stage.
It's a static approximation to stage-conditioned annealing.
The limitation: same R in both stages; no reset of the exploration budget.
`adaptive_tanh_stage` is strictly better in theory.

### Implementation

`AdaptiveStageReward` is a marker subclass of `AdaptiveScaleReward(schedule="episode")`.
The env tracks `_stage_steps` (per-env counter, reset when stage changes) and calls
`reach_fn.prepare(_stage_steps, max_episode_steps)` instead of
`reach_fn.prepare(elapsed_steps, max_episode_steps)`.

StackCube: stage transition = `is_cubeA_grasped` flip False→True (or True→False).

---

## Rough Paper Structure (draft)

1. **Introduction**: reward geometry matters, but we don't understand how. Summarise gap.
2. **Taxonomy**: 6 dimensions (support, near-goal gradient, monotonicity, smoothness, far-field curvature, sign).
3. **Experimental setup**: PickCube-v1, PPO, 256 envs, 5M steps, 5 seeds.
4. **Result 1**: Finite support prevents hover traps — initialisation robustness (H1).
5. **Result 2**: Finite support causes signal starvation — long-run degradation (F2).
6. **Result 3**: Robustness-ceiling tradeoff — neither regime universally better.
7. **Result 4**: Near-goal curvature does not independently predict success (H2 refuted).
8. **Proposed fix**: Adaptive support radius (reward geometry curriculum).
9. **Transfer**: same patterns hold on PushCube / StackCube (to be confirmed).
10. **Playbook**: actionable decision tree for practitioners.
11. **Discussion**: biological parallels, limitations, future work.
