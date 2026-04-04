"""
Dual-Buffer SAC for ManiSkill3 — PickCube-v1
=============================================
Architecture inspired by the data-mixing strategy in HIL-SERL, kept strictly
state-based and using the environment's default dense reward.

Key additions over the vanilla SAC baseline
-------------------------------------------
1. DemoReplayBuffer  — loads expert transitions from a .pt / .npz file.
2. Mixed-batch sampling — every gradient step draws
       (1 - demo_sampling_ratio) × batch_size  from the online buffer, and
           demo_sampling_ratio  × batch_size  from the demo buffer.
3. Distance-gated critic-threshold action replacement — during rollout, if a
   proposed action has a Q-value below `critic_threshold`, the agent looks up
   the nearest demo state.  If that state is within `demo_distance_threshold`
   (L2), the corresponding demo action is substituted; otherwise the agent's
   own action is kept so it can keep exploring novel regions.

Usage example
-------------
python dual_buffer_sac.py \
    --demo_path demos.pt \
    --demo_sampling_ratio 0.5 \
    --critic_threshold -5.0 \
    --demo_distance_threshold 0.1 \
    --track
"""

from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "DualBufferSAC"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # ── Environment ────────────────────────────────────────────────────────
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 100
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """reconfigure eval env each reset to randomise objects"""
    eval_freq: int = 1000
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = 100000
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # ── SAC hyper-parameters ───────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the online replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored ('cpu' or 'cuda')"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """total batch size drawn each gradient step (online + demo combined)"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in environment steps)"""
    utd: float = 0.5
    """update-to-data ratio"""
    bootstrap_at_done: str = "always"
    """bootstrap method on done: 'always' or 'never'"""

    # ── Dual-buffer arguments ──────────────────────────────────────────────
    demo_path: Optional[str] = None
    """path to the expert demonstration file (.pt or .npz).
    Expected keys: obs, next_obs, actions, rewards, dones (all float32 arrays
    of shape [N, dim]).  If None, the demo buffer is disabled and the script
    falls back to vanilla SAC."""
    demo_sampling_ratio: float = 0.5
    """fraction of each training batch drawn from the demo buffer.
    The remainder (1 - demo_sampling_ratio) comes from the online buffer."""
    critic_threshold: Optional[float] = None
    """Q-value threshold for the distance-gated action replacement.
    If None, action replacement is disabled entirely."""
    demo_distance_threshold: float = 0.1
    """L2 distance threshold: only replace the actor's action with a demo
    action when the nearest demo state is within this radius."""

    # ── Runtime fields (filled automatically) ─────────────────────────────
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


# ---------------------------------------------------------------------------
# Replay Buffer Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


# ---------------------------------------------------------------------------
# Online Replay Buffer  (unchanged from baseline)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular replay buffer that stores vectorised environment transitions."""

    def __init__(
        self,
        env,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device,
        sample_device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs

        shape_obs = env.single_observation_space.shape
        shape_act = env.single_action_space.shape

        self.obs      = torch.zeros((self.per_env_buffer_size, num_envs) + shape_obs).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, num_envs) + shape_obs).to(storage_device)
        self.actions  = torch.zeros((self.per_env_buffer_size, num_envs) + shape_act).to(storage_device)
        self.rewards  = torch.zeros((self.per_env_buffer_size, num_envs)).to(storage_device)
        self.dones    = torch.zeros((self.per_env_buffer_size, num_envs)).to(storage_device)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        if self.storage_device == torch.device("cpu"):
            obs, next_obs = obs.cpu(), next_obs.cpu()
            action, reward, done = action.cpu(), reward.cpu(), done.cpu()

        self.obs[self.pos]      = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        upper = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper,          size=(batch_size,))
        env_inds   = torch.randint(0, self.num_envs,  size=(batch_size,))
        return ReplayBufferSample(
            obs      = self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs = self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions  = self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards  = self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones    = self.dones[batch_inds, env_inds].to(self.sample_device),
        )


# ---------------------------------------------------------------------------
# Demo Replay Buffer
# ---------------------------------------------------------------------------

class DemoReplayBuffer:
    """
    Read-only buffer that holds expert demonstration transitions.

    Supported file formats
    ----------------------
    .pt  — a dict saved with ``torch.save`` containing float32 tensors with
            keys: obs, next_obs, actions, rewards, dones.
    .npz — a numpy archive with the same keys.

    All tensors are kept on ``storage_device`` and returned on
    ``sample_device``.  The obs tensor is also exposed as
    ``self.all_obs`` (shape [N, obs_dim]) for fast nearest-neighbour
    lookups during rollout.
    """

    def __init__(self, demo_path: str, storage_device: torch.device, sample_device: torch.device):
        self.storage_device = storage_device
        self.sample_device  = sample_device

        print(f"[DemoReplayBuffer] Loading demonstrations from: {demo_path}")
        data = self._load(demo_path)

        # Store tensors directly as 2-D float32 arrays  [N, dim]
        self.obs      = data["obs"].float().to(storage_device)
        self.next_obs = data["next_obs"].float().to(storage_device)
        self.actions  = data["actions"].float().to(storage_device)
        self.rewards  = data["rewards"].float().to(storage_device)
        self.dones    = data["dones"].float().to(storage_device)

        # Ensure rewards and dones are 1-D  [N]  (for uniform handling with online buffer)
        self.rewards = self.rewards.flatten()
        self.dones = self.dones.flatten()

        self.size = self.obs.shape[0]

        # Expose obs tensor for nearest-neighbour distance queries during rollout
        # Shape: [N, obs_dim]  — kept on storage_device for fast cdist
        self.all_obs     = self.obs          # alias (no copy)
        self.all_actions = self.actions      # corresponding actions

        print(f"[DemoReplayBuffer] Loaded {self.size} demo transitions. "
              f"obs_dim={self.obs.shape[1]}, act_dim={self.actions.shape[1]}")

    # ------------------------------------------------------------------
    def _load(self, path: str) -> dict:
        """Load a .pt or .npz file and return a dict of torch Tensors."""
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".pt":
            raw = torch.load(path, map_location="cpu")
            return {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v))
                    for k, v in raw.items() if k != "_meta"}
        elif ext == ".npz":
            raw = np.load(path, allow_pickle=False)
            return {k: torch.tensor(raw[k]) for k in raw.files if k != "_meta"}
        else:
            raise ValueError(f"Unsupported demo file format '{ext}'. Use .pt or .npz.")

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> ReplayBufferSample:
        """Uniformly sample `batch_size` transitions from the demo buffer."""
        idx = torch.randint(0, self.size, size=(batch_size,))
        return ReplayBufferSample(
            obs      = self.obs[idx].to(self.sample_device),
            next_obs = self.next_obs[idx].to(self.sample_device),
            actions  = self.actions[idx].to(self.sample_device),
            rewards  = self.rewards[idx].to(self.sample_device),
            dones    = self.dones[idx].to(self.sample_device),
        )

    # ------------------------------------------------------------------
    def find_nearest(
        self,
        query_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each query state, return the L2 distance to and action from the
        nearest demonstration state.

        Parameters
        ----------
        query_obs : Tensor of shape [B, obs_dim]

        Returns
        -------
        min_dists   : Tensor [B]  — L2 distance to nearest demo state
        demo_actions: Tensor [B, act_dim]  — corresponding demo action
        """
        # query_obs: [B, D],  all_obs: [N, D]  →  cdist: [B, N]
        dists      = torch.cdist(query_obs.to(self.storage_device),
                                  self.all_obs)        # [B, N]
        min_dists, nearest_idx = dists.min(dim=1)      # [B], [B]
        demo_actions = self.all_actions[nearest_idx]   # [B, act_dim]
        return min_dists.to(self.sample_device), demo_actions.to(self.sample_device)


# ---------------------------------------------------------------------------
# Networks  (standard MLPs — no CNNs)
# ---------------------------------------------------------------------------

class SoftQNetwork(nn.Module):
    """
    State-based Q-network: maps (obs, action) → scalar Q-value.
    Input is a flat 1-D state vector; no image encoders are used.
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),               nn.ReLU(),
            nn.Linear(256, 256),               nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, a], dim=1))


LOG_STD_MAX =  2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """
    State-based stochastic actor: maps obs → (action, log_prob, mean_action).
    Accepts a flat 1-D state vector.  No CNNs, ResNets, or augmentations.
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        # Action rescaling buffers (persisted in state_dict)
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",  torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        x       = self.backbone(x)
        mean    = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation (mean of the squashed Gaussian)."""
        x    = self.backbone(x)
        mean = self.fc_mean(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, x: torch.Tensor):
        """Stochastic action with reparameterisation trick."""
        mean, log_std = self(x)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Log-prob with Tanh change-of-variable
        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob  = log_prob.sum(1, keepdim=True)
        mean      = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias  = self.action_bias.to(device)
        return super().to(device)


# ---------------------------------------------------------------------------
# Unified Logger (TensorBoard + W&B)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_wandb: bool = False, tensorboard: SummaryWriter = None):
        self.writer   = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


# ---------------------------------------------------------------------------
# Mixed-batch helper
# ---------------------------------------------------------------------------

def sample_mixed_batch(
    rb: ReplayBuffer,
    demo_rb: Optional[DemoReplayBuffer],
    batch_size: int,
    demo_sampling_ratio: float,
) -> ReplayBufferSample:
    """
    Draw a mixed batch from the online buffer and (optionally) the demo buffer.

    When no demo buffer is available the full batch comes from the online buffer.
    """
    if demo_rb is None or demo_sampling_ratio <= 0.0:
        return rb.sample(batch_size)

    n_demo   = int(batch_size * demo_sampling_ratio)
    n_online = batch_size - n_demo

    online_batch = rb.sample(n_online)
    demo_batch   = demo_rb.sample(n_demo)

    # Concatenate along the batch dimension
    return ReplayBufferSample(
        obs      = torch.cat([online_batch.obs,      demo_batch.obs],      dim=0),
        next_obs = torch.cat([online_batch.next_obs, demo_batch.next_obs], dim=0),
        actions  = torch.cat([online_batch.actions,  demo_batch.actions],  dim=0),
        rewards  = torch.cat([online_batch.rewards,  demo_batch.rewards],  dim=0),
        dones    = torch.cat([online_batch.dones,     demo_batch.dones],    dim=0),
    )


# ---------------------------------------------------------------------------
# Distance-gated critic-threshold action replacement
# ---------------------------------------------------------------------------

def apply_critic_threshold_replacement(
    actor_actions: torch.Tensor,
    obs: torch.Tensor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    demo_rb: DemoReplayBuffer,
    critic_threshold: float,
    demo_distance_threshold: float,
) -> tuple[torch.Tensor, dict]:
    """
    Optionally replace actor actions with demo actions when both conditions hold:
      1. The actor's Q-value falls below `critic_threshold`.
      2. The nearest demo state is within `demo_distance_threshold` (L2).

    Replacing only when the distance is small ensures we don't substitute demo
    actions in out-of-distribution regions where the lookup is unreliable.

    Returns
    -------
    final_actions : Tensor [B, act_dim]
    info          : dict with diagnostic scalars for logging
    """
    with torch.no_grad():
        q1 = qf1(obs, actor_actions)   # [B, 1]
        q2 = qf2(obs, actor_actions)
        q  = torch.min(q1, q2).squeeze(1)  # [B]

    # Mask of environments whose Q-value is below the threshold
    low_q_mask = q < critic_threshold  # [B] bool

    num_replacements = 0
    num_low_q        = int(low_q_mask.sum().item())

    if low_q_mask.any():
        # Nearest-neighbour lookup  (only for envs that need it)
        min_dists, demo_actions = demo_rb.find_nearest(obs)  # [B], [B, act_dim]

        # Second gate: only replace if we are close enough to a demo state
        within_range  = min_dists <= demo_distance_threshold  # [B] bool
        replace_mask  = low_q_mask & within_range              # [B] bool

        if replace_mask.any():
            final_actions = actor_actions.clone()
            final_actions[replace_mask] = demo_actions[replace_mask]
            num_replacements = int(replace_mask.sum().item())
        else:
            final_actions = actor_actions
    else:
        final_actions = actor_actions

    info = {
        "rollout/num_low_q_envs":        num_low_q,
        "rollout/num_demo_replacements":  num_replacements,
        "rollout/mean_q_value":           q.mean().item(),
    }
    return final_actions, info


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    # Derived quantities
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env             = args.training_freq // args.num_envs

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup ──────────────────────────────────────────────────
    # obs_mode is always "state" — flat 1-D state vectors, no images.
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs      = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    envs      = ManiSkillVectorEnv(envs,      args.num_envs,      ignore_terminations=not args.partial_reset,      record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # ── Logging setup ──────────────────────────────────────────────────────
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["dual_buffer_sac", "state_based", "walltime_efficient"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ── Networks ───────────────────────────────────────────────────────────
    actor     = Actor(envs).to(device)
    qf1       = SoftQNetwork(envs).to(device)
    qf2       = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha     = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # ── Online replay buffer ───────────────────────────────────────────────
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    # ── Demo replay buffer (optional) ──────────────────────────────────────
    demo_rb: Optional[DemoReplayBuffer] = None
    if args.demo_path is not None:
        demo_rb = DemoReplayBuffer(
            demo_path=args.demo_path,
            storage_device=torch.device(args.buffer_device),
            sample_device=device,
        )
        # Validate that the demo obs dimension matches the environment
        env_obs_dim  = int(np.prod(envs.single_observation_space.shape))
        demo_obs_dim = demo_rb.obs.shape[1]
        if env_obs_dim != demo_obs_dim:
            raise ValueError(
                f"Demo obs dimension ({demo_obs_dim}) does not match "
                f"environment obs dimension ({env_obs_dim}). "
                "Check that the demo was collected with obs_mode='state' "
                "and the same control_mode."
            )
        print(f"[DualBufferSAC] Demo buffer ready. "
              f"demo_sampling_ratio={args.demo_sampling_ratio}, "
              f"critic_threshold={args.critic_threshold}, "
              f"demo_distance_threshold={args.demo_distance_threshold}")
    else:
        print("[DualBufferSAC] No demo_path provided — running as vanilla SAC.")

    use_critic_gate = (args.critic_threshold is not None) and (demo_rb is not None)

    # ── Main training loop ─────────────────────────────────────────────────
    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step   = 0
    global_update = 0
    learning_has_started = False
    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    # Placeholders so we can always reference these in the logging block
    actor_loss = qf1_loss = qf2_loss = qf_loss = torch.tensor(0.0)
    qf1_a_values = qf2_a_values = torch.zeros(1)
    alpha_loss = torch.tensor(0.0)

    while global_step < args.total_timesteps:

        # ── Periodic evaluation ────────────────────────────────────────────
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0

            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = actor.get_eval_action(eval_obs)
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)

            pbar.set_description(
                f"success_once: {eval_metrics_mean.get('success_once', 0):.2f}, "
                f"return: {eval_metrics_mean.get('return', 0):.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            # if args.save_model:
            #     model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            #     torch.save({
            #         "actor":     actor.state_dict(),
            #         "qf1":       qf1_target.state_dict(),
            #         "qf2":       qf2_target.state_dict(),
            #         "log_alpha": log_alpha,
            #     }, model_path)
            #     print(f"Model saved to {model_path}")

        # ── Environment rollout ────────────────────────────────────────────
        rollout_time = time.perf_counter()
        rollout_replacement_info = defaultdict(float)

        for local_step in range(args.steps_per_env):
            global_step += args.num_envs

            with torch.no_grad():
                if not learning_has_started:
                    # Random exploration before learning starts
                    actions = torch.tensor(
                        envs.action_space.sample(), dtype=torch.float32, device=device
                    )
                else:
                    actions, _, _ = actor.get_action(obs)
                    actions = actions.detach()

                    # ── Distance-gated critic-threshold action replacement ──
                    # If critic_threshold is set and a demo buffer is available,
                    # check whether the Q-value is low AND we are near a demo state.
                    # Only then substitute the demo action (see function docstring).
                    if use_critic_gate:
                        actions, rep_info = apply_critic_threshold_replacement(
                            actor_actions=actions,
                            obs=obs,
                            qf1=qf1,
                            qf2=qf2,
                            demo_rb=demo_rb,
                            critic_threshold=args.critic_threshold,
                            demo_distance_threshold=args.demo_distance_threshold,
                        )
                        for k, v in rep_info.items():
                            rollout_replacement_info[k] += v

            # Step all parallel environments
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()

            # Determine bootstrap masking
            if args.bootstrap_at_done == "never":
                need_final_obs  = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap  = truncations | terminations
            elif args.bootstrap_at_done == "always":
                need_final_obs  = truncations | terminations
                stop_bootstrap  = torch.zeros_like(terminations, dtype=torch.bool)
            else:  # "truncated"
                need_final_obs  = truncations & (~terminations)
                stop_bootstrap  = terminations

            if "final_info" in infos:
                final_info    = infos["final_info"]
                done_mask     = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # Log rollout replacement stats once per iteration
        if logger is not None and use_critic_gate:
            for k, v in rollout_replacement_info.items():
                logger.add_scalar(k, v / max(args.steps_per_env, 1), global_step)

        # ── Skip training until learning_starts ───────────────────────────
        if global_step < args.learning_starts:
            continue

        # ── Gradient update loop ───────────────────────────────────────────
        update_time = time.perf_counter()
        learning_has_started = True

        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1

            # ── Mixed-batch sampling ──────────────────────────────────────
            # Draws (1 - demo_sampling_ratio) × batch_size from the online
            # buffer and demo_sampling_ratio × batch_size from the demo buffer,
            # then concatenates along the batch dimension.
            data = sample_mixed_batch(rb, demo_rb, args.batch_size, args.demo_sampling_ratio)

            # ── Critic update ─────────────────────────────────────────────
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                # data.rewards / data.dones may be [B,1] from demo buffer — flatten to [B]
                rewards_flat = data.rewards.flatten()
                dones_flat   = data.dones.flatten()
                next_q_value = (
                    rewards_flat
                    + (1 - dones_flat) * args.gamma * min_qf_next_target.view(-1)
                )

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss     = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss     = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss      = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # ── Actor update ──────────────────────────────────────────────
            if global_update % args.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi    = qf1(data.obs, pi)
                qf2_pi    = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ── Entropy coefficient update ────────────────────────────
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # ── Target network soft update ────────────────────────────────
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # ── Periodic scalar logging ────────────────────────────────────────
        if logger is not None and (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_values",  qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values",  qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss",    qf1_loss.item(),            global_step)
            logger.add_scalar("losses/qf2_loss",    qf2_loss.item(),            global_step)
            logger.add_scalar("losses/qf_loss",     qf_loss.item() / 2.0,       global_step)
            logger.add_scalar("losses/actor_loss",  actor_loss.item(),          global_step)
            logger.add_scalar("losses/alpha",       alpha,                      global_step)
            logger.add_scalar("time/update_time",   update_time,                global_step)
            logger.add_scalar("time/rollout_time",  rollout_time,               global_step)
            logger.add_scalar(
                "time/rollout_fps",
                global_steps_per_iteration / max(rollout_time, 1e-9),
                global_step,
            )
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar(
                "time/total_rollout+update_time",
                cumulative_times["rollout_time"] + cumulative_times["update_time"],
                global_step,
            )
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            # Demo-buffer utilisation stats
            if demo_rb is not None:
                logger.add_scalar("demo/buffer_size",        demo_rb.size,          global_step)
                logger.add_scalar("demo/sampling_ratio",     args.demo_sampling_ratio, global_step)
                logger.add_scalar("demo/n_demo_per_batch",   int(args.batch_size * args.demo_sampling_ratio), global_step)

    # ── Final model save ───────────────────────────────────────────────────
    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            "actor":     actor.state_dict(),
            "qf1":       qf1_target.state_dict(),
            "qf2":       qf2_target.state_dict(),
            "log_alpha": log_alpha,
        }, model_path)
        print(f"Final model saved to {model_path}")
        if logger is not None:
            logger.close()

    envs.close()
    eval_envs.close()