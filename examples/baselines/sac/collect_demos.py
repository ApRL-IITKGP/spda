"""
collect_demos.py  —  Ground-truth demo collector for ManiSkill3 PickCube-v1
============================================================================
Generates expert demonstrations using ManiSkill3's built-in motion-planning
solution (solvePickCube), records every transition during the solve call via
a lightweight TransitionRecorder wrapper, and saves everything in the exact
format expected by DemoReplayBuffer in dual_buffer_sac.py.

Key design decisions (vs the old broken version)
-------------------------------------------------
* solvePickCube is a FUNCTION, not a class.  It is called exactly as in
  run.py:  res = solvePickCube(env, seed=seed, debug=False, vis=False)
  The function drives env.reset() + env.step() internally, so we intercept
  those calls with a wrapper instead of pre-planning and replaying.

* control_mode defaults to "pd_joint_pos" — the mode the ManiSkill motion
  planner was designed for.  Pass the same flag to dual_buffer_sac.py:
      python dual_buffer_sac.py --control_mode pd_joint_pos ...

* sim_backend defaults to "auto" so mplib (the CPU-based IK solver inside
  ManiSkill's motion planner) works correctly.  GPU tensors are squeezed
  and moved to CPU before being stored.

* GIFs are rendered DURING the solve call (not in a separate replay pass)
  by capturing env.render() inside the wrapper at every step.

Output files
------------
<out_dir>/demos.pt         — primary demo buffer (float32 tensors)
<out_dir>/demos.npz        — identical data as a numpy archive
<out_dir>/demos_meta.json  — per-episode statistics
<out_dir>/gifs/ep_XXXX.gif — visual verification GIFs

Demo .pt tensor keys  (all float32)
------------------------------------
  obs       [N, obs_dim]   — flat state observation at step t
  next_obs  [N, obs_dim]   — flat state observation at step t+1
  actions   [N, act_dim]   — action executed at step t
  rewards   [N]            — scalar reward received at step t
  dones     [N]            — 1.0 when the episode ended at step t
                             (matches stop_bootstrap logic in dual_buffer_sac.py)

Usage
-----
# Collect 200 successful demos, save 10 GIFs:
python collect_demos.py --num_demos 200 --num_gif_episodes 10

# Quick sanity check -- 5 demos, all as GIFs:
python collect_demos.py --num_demos 5 --num_gif_episodes 5 --out_dir ./quick_test

# Verify an existing .pt file without re-running collection:
python collect_demos.py --verify ./demos/demos.pt
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import imageio
import numpy as np
import torch
from tqdm import tqdm

import mani_skill.envs  # noqa: F401 -- registers all ManiSkill environments
from mani_skill.examples.motionplanning.panda.solutions import solvePickCube


# ---------------------------------------------------------------------------
# TransitionRecorder  --  gym.Wrapper that captures every transition
# ---------------------------------------------------------------------------

class TransitionRecorder(gym.Wrapper):
    """
    Thin wrapper around a ManiSkill gym environment that intercepts every
    reset() and step() call made by the motion-planning solve function.

    After a solve call completes, call .pop_episode() to retrieve all
    recorded transitions and rendered frames for that episode.

    Parameters
    ----------
    env           : the underlying gym environment
    capture_frames: if True, call env.render() at every step and store frames
    frame_skip    : store one frame every N steps (reduces GIF size)
    """

    def __init__(self, env: gym.Env, capture_frames: bool = False, frame_skip: int = 1):
        super().__init__(env)
        self.capture_frames = capture_frames
        self.frame_skip = max(frame_skip, 1)
        self._buf_reset()

    # ------------------------------------------------------------------
    def _buf_reset(self):
        """Clear episode buffers without touching the environment."""
        self._obs: List[np.ndarray] = []
        self._next_obs: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[float] = []
        self._frames: List[np.ndarray] = []
        self._step_count = 0
        self._current_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        self._buf_reset()
        self._current_obs = self._extract_obs(obs_raw)
        # Capture the very first frame (t=0 state)
        if self.capture_frames:
            self._capture_frame()
        return obs_raw, info

    # ------------------------------------------------------------------
    def step(self, action):
        action_np = self._extract_action(action)
        next_obs_raw, reward_raw, terminated, truncated, info = self.env.step(action)

        next_obs_np = self._extract_obs(next_obs_raw)
        reward = self._to_scalar(reward_raw)

        # done = True whenever episode ends (mirrors stop_bootstrap =
        # terminated | truncated used in dual_buffer_sac.py with
        # bootstrap_at_done="always")
        done = float(terminated or truncated)

        self._obs.append(self._current_obs.copy())
        self._next_obs.append(next_obs_np.copy())
        self._actions.append(action_np.copy())
        self._rewards.append(reward)
        self._dones.append(done)

        self._current_obs = next_obs_np
        self._step_count += 1

        # Capture frame after step
        if self.capture_frames and (self._step_count % self.frame_skip == 0):
            self._capture_frame()

        return next_obs_raw, reward_raw, terminated, truncated, info

    # ------------------------------------------------------------------
    def pop_episode(self) -> dict:
        """
        Return all transitions recorded since the last reset() and clear
        the internal buffers.
        """
        ep = {
            "obs":      list(self._obs),
            "next_obs": list(self._next_obs),
            "actions":  list(self._actions),
            "rewards":  list(self._rewards),
            "dones":    list(self._dones),
            "frames":   list(self._frames),
        }
        self._buf_reset()
        return ep

    # ------------------------------------------------------------------
    def _capture_frame(self):
        """Render and store a single RGB frame."""
        frame = self.env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, np.ndarray):
            # GPU envs may return (num_envs, H, W, 3) -- take first env
            if frame.ndim == 4:
                frame = frame[0]
            self._frames.append(frame.astype(np.uint8))

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_obs(obs_raw) -> np.ndarray:
        """
        Convert a raw observation (torch tensor or numpy array) to a flat
        float32 numpy vector.  Handles the (1, obs_dim) batch dimension that
        GPU envs add when num_envs=1.
        """
        if isinstance(obs_raw, torch.Tensor):
            obs = obs_raw.detach().cpu().numpy()
        else:
            obs = np.asarray(obs_raw, dtype=np.float32)
        # Squeeze leading batch dimension added by vectorised GPU envs
        if obs.ndim > 1 and obs.shape[0] == 1:
            obs = obs[0]
        return obs.flatten().astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_action(action) -> np.ndarray:
        """Convert action (tensor or array) to a flat float32 numpy vector."""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray) and action.ndim > 1 and action.shape[0] == 1:
            action = action[0]
        return np.asarray(action, dtype=np.float32).flatten()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_scalar(x) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.item())
        if isinstance(x, np.ndarray):
            return float(x.flatten()[0])
        return float(x)


# ---------------------------------------------------------------------------
# GIF writer
# ---------------------------------------------------------------------------

def save_gif(frames: List[np.ndarray], path: str, fps: int = 15) -> None:
    """Save a list of uint8 RGB (H, W, 3) frames as an animated GIF."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(control_mode: str, render_mode: str, sim_backend: str) -> TransitionRecorder:
    """
    Create a PickCube-v1 environment wrapped in a TransitionRecorder.

    Notes
    -----
    * obs_mode="state"  -- flat state vector, no images; required by dual_buffer_sac.py
    * render_mode is forwarded from CLI (use "rgb_array" to enable GIF capture)
    * sim_backend="auto" lets ManiSkill choose CPU/GPU; the mplib IK solver
      inside the motion planner is CPU-only, so "auto" is safer than "gpu"
    * num_envs is not passed -- defaults to 1 for single-env collection
    """
    base_env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode=control_mode,
        render_mode=render_mode,
        reward_mode="dense",
        sim_backend=sim_backend,
    )
    # TransitionRecorder is applied after env creation;
    # frame capture is toggled per-episode via wrapper attributes.
    recorder = TransitionRecorder(base_env, capture_frames=False, frame_skip=1)
    return recorder


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_demos(args) -> None:
    out_dir = Path(args.out_dir)
    gif_dir = out_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  ManiSkill3 -- PickCube-v1  Ground-Truth Demo Collector")
    print("=" * 65)
    print(f"  Target demos          : {args.num_demos}")
    print(f"  GIF episodes          : {args.num_gif_episodes}")
    print(f"  Success-only          : {not args.keep_failures}")
    print(f"  Control mode          : {args.control_mode}")
    print(f"  Sim backend           : {args.sim_backend}")
    print(f"  Output directory      : {out_dir.resolve()}")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------ env
    env = make_env(
        control_mode=args.control_mode,
        render_mode="rgb_array",   # always rgb_array so GIF capture works
        sim_backend=args.sim_backend,
    )

    # Determine obs / act dimensions from the wrapped env's spaces
    obs_space = (
        env.single_observation_space
        if hasattr(env, "single_observation_space")
        else env.observation_space
    )
    act_space = (
        env.single_action_space
        if hasattr(env, "single_action_space")
        else env.action_space
    )
    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))
    print(f"[Env] obs_dim={obs_dim}  act_dim={act_dim}\n")

    # ------------------------------------------------------------------ accumulators
    all_obs:      List[np.ndarray] = []
    all_next_obs: List[np.ndarray] = []
    all_actions:  List[np.ndarray] = []
    all_rewards:  List[float]      = []
    all_dones:    List[float]      = []
    meta:         List[dict]       = []

    num_collected = 0
    num_attempts  = 0
    num_gif_saved = 0
    failed_plans  = 0
    max_attempts  = args.num_demos * args.max_attempt_multiplier

    pbar = tqdm(total=args.num_demos, desc="Collecting demos")
    t0   = time.perf_counter()
    seed = args.seed

    while num_collected < args.num_demos and num_attempts < max_attempts:

        # Toggle GIF capture on the wrapper before each episode
        save_gif_this_ep = (
            num_gif_saved < args.num_gif_episodes
            and num_attempts % max(args.gif_stride, 1) == 0
        )
        env.capture_frames = save_gif_this_ep
        env.frame_skip     = args.gif_frame_skip

        # -------------------------------------------------------- call solver
        # This mirrors run.py exactly:
        #   res = solve(env, seed=seed, debug=False, vis=False)
        #
        # solvePickCube drives env.reset(seed=seed) and env.step(...) itself.
        # Our TransitionRecorder wrapper silently captures every transition
        # and frame during those internal calls.
        try:
            res = solvePickCube(env, seed=seed, debug=False, vis=False)
        except Exception as exc:
            tqdm.write(f"  [Planner] Exception at seed {seed}: {exc}")
            env.pop_episode()   # discard any partial transitions
            failed_plans += 1
            seed += 1
            num_attempts += 1
            continue

        # Pop transitions recorded by the wrapper during the solve call
        episode = env.pop_episode()
        num_attempts += 1

        if res == -1:
            # Hard planning failure -- no valid IK solution
            failed_plans += 1
            seed += 1
            continue

        # --------------------------------------------------- check success
        # res is (obs, reward, terminated, truncated, info) from the last step
        # Pattern taken from run.py: res[-1]["success"].item()
        final_info = res[-1]
        try:
            success_val = final_info["success"]
            if isinstance(success_val, torch.Tensor):
                success = bool(success_val.item())
            else:
                success = bool(success_val)
        except (KeyError, TypeError):
            # Fallback: non-zero cumulative reward means the cube was picked
            success = sum(episode["rewards"]) > 0.0

        ep_length = len(episode["obs"])
        ep_return = float(sum(episode["rewards"])) if episode["rewards"] else 0.0

        # Discard failed episodes unless --keep_failures is set
        if not args.keep_failures and not success:
            seed += 1
            continue

        # Skip degenerate empty episodes
        if ep_length == 0:
            seed += 1
            continue

        # -------------------------------------------- verify shapes before stacking
        obs_arr = np.stack(episode["obs"])       # [T, obs_dim]
        nobs_arr = np.stack(episode["next_obs"]) # [T, obs_dim]
        act_arr  = np.stack(episode["actions"])  # [T, act_dim]

        if obs_arr.shape[1] != obs_dim or act_arr.shape[1] != act_dim:
            tqdm.write(
                f"  [WARN] Shape mismatch at seed {seed}: "
                f"obs={obs_arr.shape}, act={act_arr.shape} -- skipping."
            )
            seed += 1
            continue

        # --------------------------------------------------- accumulate
        all_obs.extend(episode["obs"])
        all_next_obs.extend(episode["next_obs"])
        all_actions.extend(episode["actions"])
        all_rewards.extend(episode["rewards"])
        all_dones.extend(episode["dones"])

        meta.append({
            "episode_index": num_collected,
            "seed":          seed,
            "length":        ep_length,
            "return":        round(ep_return, 4),
            "success":       success,
        })

        # ------------------------------------------------------------- GIF
        if save_gif_this_ep and len(episode["frames"]) > 0:
            gif_path = str(gif_dir / f"ep_{num_collected:04d}.gif")
            save_gif(episode["frames"], gif_path, fps=args.gif_fps)
            num_gif_saved += 1
            tqdm.write(
                f"  [GIF] {gif_path}  "
                f"len={ep_length}  return={ep_return:.3f}  success={success}"
            )

        num_collected += 1
        seed += 1
        pbar.update(1)
        pbar.set_postfix(
            attempts=num_attempts,
            success_rate=f"{num_collected / num_attempts:.1%}",
            failed_plans=failed_plans,
            transitions=len(all_obs),
        )

    pbar.close()
    env.close()

    if num_collected == 0:
        print(
            "\n[ERROR] No demos collected. "
            "Check that mplib is installed and solvePickCube runs correctly."
        )
        return

    elapsed           = time.perf_counter() - t0
    total_transitions = len(all_obs)
    print(
        f"\n[Done] {num_collected} episodes | "
        f"{total_transitions} transitions | "
        f"{elapsed:.1f}s | "
        f"success rate {num_collected / num_attempts:.1%} | "
        f"failed plans {failed_plans}"
    )

    # ------------------------------------------------------------------ tensors
    print("[Save] Converting to float32 tensors ...")
    obs_t      = torch.tensor(np.stack(all_obs),      dtype=torch.float32)
    next_obs_t = torch.tensor(np.stack(all_next_obs), dtype=torch.float32)
    actions_t  = torch.tensor(np.stack(all_actions),  dtype=torch.float32)
    rewards_t  = torch.tensor(all_rewards,            dtype=torch.float32)
    dones_t    = torch.tensor(all_dones,              dtype=torch.float32)

    # Hard shape assertions -- catch regressions before writing to disk
    assert obs_t.shape      == (total_transitions, obs_dim), f"obs:      {obs_t.shape}"
    assert next_obs_t.shape == (total_transitions, obs_dim), f"next_obs: {next_obs_t.shape}"
    assert actions_t.shape  == (total_transitions, act_dim), f"actions:  {actions_t.shape}"
    assert rewards_t.shape  == (total_transitions,),         f"rewards:  {rewards_t.shape}"
    assert dones_t.shape    == (total_transitions,),         f"dones:    {dones_t.shape}"

    # ------------------------------------------------------------------ .pt
    pt_path = out_dir / "demos.pt"
    torch.save(
        {
            # Tensors consumed directly by DemoReplayBuffer
            "obs":      obs_t,       # [N, obs_dim]  float32
            "next_obs": next_obs_t,  # [N, obs_dim]  float32
            "actions":  actions_t,   # [N, act_dim]  float32
            "rewards":  rewards_t,   # [N]           float32
            "dones":    dones_t,     # [N]           float32
            # Embedded metadata for traceability
            "_meta": {
                "env_id":          "PickCube-v1",
                "obs_mode":        "state",
                "control_mode":    args.control_mode,
                "obs_dim":         obs_dim,
                "act_dim":         act_dim,
                "num_episodes":    num_collected,
                "num_transitions": total_transitions,
            },
        },
        str(pt_path),
    )
    print(f"[Save] demos.pt    -> {pt_path}")
    print(f"       obs={tuple(obs_t.shape)}  actions={tuple(actions_t.shape)}")

    # ------------------------------------------------------------------ .npz
    npz_path = out_dir / "demos.npz"
    np.savez_compressed(
        str(npz_path),
        obs      = obs_t.numpy(),
        next_obs = next_obs_t.numpy(),
        actions  = actions_t.numpy(),
        rewards  = rewards_t.numpy(),
        dones    = dones_t.numpy(),
    )
    print(f"[Save] demos.npz   -> {npz_path}")

    # ------------------------------------------------------------------ JSON
    meta_path = out_dir / "demos_meta.json"
    summary = {
        "env_id":             "PickCube-v1",
        "obs_mode":           "state",
        "control_mode":       args.control_mode,
        "obs_dim":            obs_dim,
        "act_dim":            act_dim,
        "num_episodes":       num_collected,
        "num_transitions":    total_transitions,
        "avg_episode_length": round(total_transitions / num_collected, 2),
        "avg_return":         round(float(np.mean([e["return"] for e in meta])), 4),
        "success_rate":       round(sum(e["success"] for e in meta) / len(meta), 4),
        "failed_plans":       failed_plans,
        "collection_time_s":  round(elapsed, 1),
        "episodes":           meta,
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Save] metadata    -> {meta_path}")

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 65)
    print("  COLLECTION SUMMARY")
    print("=" * 65)
    print(f"  Episodes collected   : {num_collected}")
    print(f"  Total transitions    : {total_transitions}")
    print(f"  Avg episode length   : {summary['avg_episode_length']:.1f} steps")
    print(f"  Avg episode return   : {summary['avg_return']:.4f}")
    print(f"  Success rate         : {summary['success_rate']:.1%}")
    print(f"  Failed IK plans      : {failed_plans}")
    print(f"  GIFs saved           : {num_gif_saved}  ->  {gif_dir}/")
    print(f"  Demo buffer (.pt)    : {pt_path}")
    print("=" * 65)
    print("\nTo train with these demos:")
    print(
        f"  python dual_buffer_sac.py \\\n"
        f"      --demo_path {pt_path} \\\n"
        f"      --control_mode {args.control_mode} \\\n"
        f"      --demo_sampling_ratio 0.5 \\\n"
        f"      --critic_threshold -5.0 \\\n"
        f"      --demo_distance_threshold 0.1\n"
    )


# ---------------------------------------------------------------------------
# Verify an existing .pt file (standalone utility)
# ---------------------------------------------------------------------------

def verify_demos(pt_path: str) -> None:
    """Load a .pt demo file and print diagnostics compatible with DemoReplayBuffer."""
    print(f"\n[Verify] Loading {pt_path} ...")
    data = torch.load(pt_path, map_location="cpu")

    required_keys = {"obs", "next_obs", "actions", "rewards", "dones"}
    missing = required_keys - set(data.keys())
    if missing:
        print(f"  [ERROR] Missing keys: {missing}")
        return

    N = data["obs"].shape[0]
    print(f"  Transitions   : {N}")
    for k in ("obs", "next_obs", "actions"):
        t = data[k]
        print(f"  {k:<10}: shape={tuple(t.shape)}  dtype={t.dtype}  "
              f"min={t.min():.4f}  max={t.max():.4f}")
    for k in ("rewards", "dones"):
        t = data[k]
        print(f"  {k:<10}: shape={tuple(t.shape)}  "
              f"min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")
    print(f"  terminal steps (dones=1): {data['dones'].sum().int().item()}")

    all_clean = True
    for k in required_keys:
        t = data[k]
        if torch.isnan(t).any():
            print(f"  [WARNING] NaN in '{k}'!")
            all_clean = False
        if torch.isinf(t).any():
            print(f"  [WARNING] Inf in '{k}'!")
            all_clean = False
    if all_clean:
        print("  [OK] No NaN or Inf values.")

    if "_meta" in data:
        m = data["_meta"]
        print(f"  env_id        : {m.get('env_id', '?')}")
        print(f"  obs_mode      : {m.get('obs_mode', '?')}")
        print(f"  control_mode  : {m.get('control_mode', '?')}")
        print(f"  num_episodes  : {m.get('num_episodes', '?')}")

    print("  [OK] File is compatible with DemoReplayBuffer in dual_buffer_sac.py.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect PickCube-v1 expert demos and generate verification GIFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Collection
    p.add_argument("--num_demos",              type=int, default=200,
                   help="Number of successful episodes to collect.")
    p.add_argument("--keep_failures",          action="store_true",
                   help="Also include failed episodes in the buffer.")
    p.add_argument("--max_attempt_multiplier", type=int, default=10,
                   help="Hard-stop after num_demos x this many total attempts.")
    p.add_argument("--seed",                   type=int, default=0,
                   help="Starting seed; incremented by 1 after each attempt.")

    # Environment
    p.add_argument("--control_mode",  type=str, default="pd_joint_pos",
                   help="Control mode.  Must match --control_mode in dual_buffer_sac.py. "
                        "The ManiSkill motion planner is designed for pd_joint_pos.")
    p.add_argument("--sim_backend",   type=str, default="auto",
                   help="Simulation backend: 'auto', 'cpu', or 'gpu'. "
                        "'auto' is recommended because mplib runs on CPU.")

    # Output
    p.add_argument("--out_dir",       type=str, default="./demos",
                   help="Output directory for .pt / .npz / GIFs / metadata.")

    # GIF
    p.add_argument("--num_gif_episodes", type=int, default=10,
                   help="Number of episodes to render as GIFs.")
    p.add_argument("--gif_stride",       type=int, default=1,
                   help="Save a GIF every N-th collected episode.")
    p.add_argument("--gif_fps",          type=int, default=15,
                   help="Frames per second in the output GIF.")
    p.add_argument("--gif_frame_skip",   type=int, default=1,
                   help="Capture every N-th step frame (reduces GIF file size).")

    # Utility
    p.add_argument("--verify",  type=str, default=None, metavar="PT_FILE",
                   help="Verify an existing .pt file and exit. No collection is run.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify is not None:
        verify_demos(args.verify)
    else:
        collect_demos(args)