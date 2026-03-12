# Reward Engineering Playbook (REP)

Systematic study of how dense reward geometry affects PPO training dynamics in robot manipulation (ManiSkill3).

**Paper:** `paper/rep.tex` | **W&B:** [swami2004/REP](https://wandb.ai/swami2004/REP)

---

## Setup

```bash
pip install -e .
pip install wandb torch
```

---

## Reward Variants

### Reach shape (`--reach_variant`)
| Variant | Description |
|---|---|
| `tanh` | Baseline: `1 - tanh(5d)` |
| `exponential` | `exp(-5d)` |
| `concave_truncated` | Concave parabola, zero beyond d=0.2m |
| `adaptive_tanh_episode` | **Best:** tanh with k ramping 2→20 within each episode |
| `adaptive_tanh_sq_episode` | tanh_sq with within-episode annealing |
| `adaptive_tanh_threshold` | tanh, k steps up when eval success ≥ 0.5 |
| `adaptive_concave_threshold` | concave_truncated, radius shrinks at success threshold |

### Gate (`--gate_variant`): `hard` (default), `soft`, `additive`, `curriculum`
### Terminal (`--terminal_variant`): `hard_jump` (default), `smooth`, `none`

---

## Running Experiments

### Single run
```bash
python examples/baselines/ppo/ppo_fast.py \
  --env_id PickCube-v1 \
  --reach_variant adaptive_tanh_episode \
  --gate_variant hard \
  --terminal_variant hard_jump \
  --num_envs 256 \
  --total_timesteps 5000000 \
  --track \
  --wandb_entity swami2004 \
  --wandb_project_name REP \
  --wandb_group REP \
  --seed 1
```

### Full sweep (all variants × seeds)
```bash
# Dry run to preview commands
python examples/reward_sweep.py --env_id PickCube-v1 --dry_run

# Baseline only
python examples/reward_sweep.py --env_id PickCube-v1 \
  --reach_variants tanh --gate_variants hard --terminal_variants hard_jump --seeds 1

# Full sweep
python examples/reward_sweep.py --env_id PickCube-v1
```

### Multi-seed replication (3 seeds)
```bash
for SEED in 1 42 123; do
  python examples/baselines/ppo/ppo_fast.py \
    --env_id PickCube-v1 \
    --reach_variant adaptive_tanh_episode \
    --gate_variant hard --terminal_variant hard_jump \
    --num_envs 256 --total_timesteps 5000000 \
    --track --wandb_entity swami2004 --wandb_project_name REP --wandb_group REP \
    --seed $SEED
done
```

### Supported envs
`PickCube-v1`, `PushCube-v1`, `StackCube-v1`, `PlaceSphere-v1`

---

## Analysis

Export runs from W&B as CSV into `paper/`, then:

```bash
python paper/plot_figures.py   # generates all figures into paper/
```

---

## Key Results (PickCube-v1, 3 seeds)

| Variant | Seed 1 | Seed 42 | Seed 123 | Mean |
|---|---|---|---|---|
| tanh (baseline) | 0.0 | 1.0 | 1.0 | 0.667 |
| concave_truncated | 0.88 | 1.0 | 1.0 | 0.960 |
| adaptive_tanh_episode | **1.0** | **1.0** | **0.938** | **0.979** |

Within-episode annealing solves the robustness-ceiling tradeoff: wide support on episode start (bootstraps hard seeds), tight support at episode end (prevents hover attractor).
