# Stage-Posterior Demo Augmentation (SPDA)

State-based Soft Actor-Critic extended with a **dual replay buffer** and **distance-gated critic-threshold action replacement**, inspired by the data-mixing strategy in HIL-SERL. Strictly state-based, uses the environment's default dense reward. Designed for ManiSkill3 manipulation tasks.

---

## Setup

```bash
pip install -e .
pip install wandb torch imageio tqdm tyro
```

---

## Demo Collection

Collect ground-truth trajectories using ManiSkill3's built-in motion planner before training.

```bash
# Collect 500 expert episodes, save 10 GIFs for visual verification
python examples/baselines/sac/collect_demos.py --num_demos 500 --num_gif_episodes 10 --seed 0

# Quick sanity check (5 demos, all rendered as GIFs)
python examples/baselines/sac/collect_demos.py --num_demos 5 --num_gif_episodes 5 --out_dir ./quick_test

# Verify an existing demo file without re-collecting
python examples/baselines/sac/collect_demos.py --verify demos/demos.pt
```

### Output files

| File | Description |
|---|---|
| `demos/demos.pt` | Primary demo buffer — float32 tensors consumed by `DemoReplayBuffer` |
| `demos/demos.npz` | Identical data as a numpy archive (compatibility format) |
| `demos/demos_meta.json` | Per-episode statistics: length, return, success flag |
| `demos/gifs/ep_XXXX.gif` | Rendered rollouts for visual quality inspection |

### Demo tensor format

| Key | Shape | Description |
|---|---|---|
| `obs` | `[N, obs_dim]` | Flat state observation at step t |
| `next_obs` | `[N, obs_dim]` | Flat state observation at step t+1 |
| `actions` | `[N, act_dim]` | Action executed at step t |
| `rewards` | `[N]` | Dense reward received at step t |
| `dones` | `[N]` | 1.0 when the episode ended (matches `stop_bootstrap` in training) |

> **Note:** `control_mode` must match between `collect_demos.py` and `dual_buffer_sac.py`. Both default to `pd_joint_pos` — the mode ManiSkill's motion planner is designed for.

---

## Training

### Single run

```bash
python examples/baselines/sac/spda_sac.py \
  --demo_path demos/demos.pt \
  --demo_sampling_ratio 0.5 \
  --critic_threshold -5.0 \
  --demo_distance_threshold 0.1 \
  --num_envs 16 \
  --total_timesteps 1000000 \
  --track \
  --wandb_entity <your_entity> \
  --wandb_project_name SPDA \
  --wandb_group SPDA \
  --seed 42
```

### Vanilla SAC baseline (no demos)

```bash
python examples/baselines/sac/sac.py \
  --num_envs 16 \
  --total_timesteps 1000000 \
  --seed 1
```

### Multi-seed replication (3 seeds)

```bash
for SEED in 1 42 123; do
  python examples/baselines/sac/spda_sac.py \
    --demo_path ./demos/demos.pt \
    --demo_sampling_ratio 0.5 \
    --critic_threshold -5.0 \
    --demo_distance_threshold 0.1 \
    --num_envs 16 --total_timesteps 1000000 \
    --track --wandb_entity <your_entity> \
    --wandb_project_name SPDA --wandb_group SPDA \
    --seed $SEED
done
```

---

## Architecture

### Dual-buffer sampling

Every gradient step draws a **mixed batch** from two sources:

| Source | Fraction | Content |
|---|---|---|
| Online buffer | `1 - demo_sampling_ratio` | Agent's real-time exploration transitions |
| Demo buffer | `demo_sampling_ratio` | Expert transitions loaded from `demos.pt` |

The two sub-batches are concatenated along the batch dimension before computing critic and actor losses. With no `--demo_path` provided the script falls back to standard single-buffer SAC.

### Distance-gated critic-threshold action replacement

During rollout, if `--critic_threshold` is set, each proposed actor action is evaluated against the critic before being executed. Replacement happens only when **both** gates pass:

1. **Q-gate** — `min(Q1, Q2) < critic_threshold`
2. **Distance gate** — L2 distance from current state to nearest demo state `≤ demo_distance_threshold`

When both conditions hold, the demo action from the nearest neighbour is substituted. When the Q-value is low but the agent is far from any demo state, its own action is kept — preserving exploratory behaviour in out-of-distribution regions.

---

## Key Arguments

### Demo buffer

| Argument | Default | Description |
|---|---|---|
| `--demo_path` | `None` | Path to `.pt` or `.npz` demo file. Omit for vanilla SAC. |
| `--demo_sampling_ratio` | `0.5` | Fraction of each training batch drawn from the demo buffer |
| `--critic_threshold` | `None` | Q-value threshold for action replacement. `None` disables gating entirely. |
| `--demo_distance_threshold` | `0.1` | L2 radius within which a demo action may replace the actor's action |

### Environment

| Argument | Default | Description |
|---|---|---|
| `--env_id` | `PickCube-v1` | Target environment |
| `--control_mode` | `pd_joint_pos` | Must match the mode used during demo collection |
| `--num_envs` | `16` | Number of parallel training environments |
| `--total_timesteps` | `1_000_000` | Total environment steps |

### SAC

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `1024` | Total batch size per gradient step (online + demo combined) |
| `--demo_sampling_ratio` | `0.5` | Demo fraction of the batch |
| `--gamma` | `0.8` | Discount factor |
| `--tau` | `0.01` | Target network soft-update coefficient |
| `--utd` | `0.5` | Update-to-data ratio |
| `--autotune` | `True` | Automatic entropy coefficient tuning |

---

## Supported Environments

`PickCube-v1` · `StackCube-v1` · `PushCube-v1`

The demo collector (`collect_demos.py`) currently targets `PickCube-v1`. For other environments, swap in the corresponding solve function from `mani_skill.examples.motionplanning.panda.solutions`.

---

## Analysis

Export runs from W&B as CSV into `paper/`, then:

```bash
python paper/plot_figures.py   # generates all figures into paper/
```

---

## Key Results (PickCube-v1, 3 seeds)

| Configuration | Seed 1 | Seed 42 | Seed 123 | Mean |
|---|---|---|---|---|
| Vanilla SAC (no demos) | — | — | — | — |
| Dual-buffer, no gating | — | — | — | — |
| Dual-buffer + critic gate | — | — | — | — |

*Fill in after running the multi-seed sweep.*

The distance gate is the critical design choice: substituting demo actions only near demonstrated states prevents the agent from getting stuck imitating in novel regions where the nearest-neighbour lookup is unreliable.