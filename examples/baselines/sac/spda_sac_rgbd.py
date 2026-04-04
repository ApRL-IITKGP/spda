"""
HIL-SERL SAC Agent for ManiSkill3 — PyTorch + W&B
====================================================
Adapts the ManiSkill3 sac_rgbd.py baseline with the key algorithmic
innovations from HIL-SERL (Luo et al., 2024):

  1. ResNet10 visual encoder  (replaces PlainConv)
  2. DrQ random-shift augmentation applied at update time
  3. Dual replay buffers — online RL buffer + demo buffer with
     configurable mixing ratio (demo_sampling_ratio)
  4. Full Weights & Biases logging (no TensorBoard dependency)

Reference: https://github.com/rail-berkeley/hil-serl
Paper    : https://arxiv.org/abs/2410.21845
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb

import mani_skill.envs


# ---------------------------------------------------------------------------
# CLI Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # W&B — always enabled; set wandb_mode='disabled' to turn off
    wandb_project_name: str = "ManiSkill-HIL-SERL"
    """the wandb project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC-HILSERL"
    """the group of the run for wandb"""
    wandb_mode: str = "online"
    """wandb mode: 'online', 'offline', or 'disabled'"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Video / trajectory saving
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    save_trajectory: bool = False
    """whether to save trajectory data into the videos folder"""
    save_model: bool = True
    """whether to save model checkpoints"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""

    # Environment
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    obs_mode: str = "rgb"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include proprioceptive state in the observation"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination"""
    eval_partial_reset: bool = False
    """whether eval environments reset upon termination"""
    num_steps: int = 50
    """the number of steps per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps per evaluation rollout"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """how often to reconfigure the eval environment"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use"""
    render_mode: str = "all"
    """the environment rendering mode"""
    camera_width: Optional[int] = None
    """the width of the camera image"""
    camera_height: Optional[int] = None
    """the height of the camera image"""

    # SAC hyper-parameters
    total_timesteps: int = 1_000_000
    """total timesteps of the experiment"""
    buffer_size: int = 1_000_000
    """online replay buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored ('cpu' or 'cuda')"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size sampled from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q-network optimizer"""
    policy_frequency: int = 1
    """the frequency of policy updates"""
    target_network_frequency: int = 1
    """the frequency of target network updates"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 1.0
    """update-to-data ratio (HIL-SERL uses UTD≥1)"""
    bootstrap_at_done: str = "always"
    """bootstrap method when done is received: 'always', 'never', or 'truncated'"""

    # HIL-SERL specific
    # ---- DrQ augmentation ----
    use_augmentation: bool = True
    """whether to apply DrQ random-shift augmentation to pixel observations"""
    aug_pad: int = 4
    """padding size for DrQ random-shift augmentation"""

    # ---- Demo buffer ----
    demo_path: Optional[str] = None
    """path to a .pt / .npz file with demonstration transitions
       (dict with keys obs, next_obs, actions, rewards, dones)"""
    demo_buffer_size: int = 50_000
    """capacity of the demo replay buffer"""
    demo_sampling_ratio: float = 0.5
    """fraction of each batch drawn from the demo buffer (0 = no demos)"""
    critic_threshold: Optional[float] = None
    """if the Q value goes below this threshold, the policy will stop exploring and follow the gt traj from the replay buffer"""
    critic_activation_return_threshold: Optional[float] = None
    """enable critic-threshold action replacement only after recent avg episode return reaches this value"""
    critic_activation_return_window: int = 100
    """number of recent completed episodes used to compute avg episode return for critic activation"""

    # ---- Encoder ----
    encoder_type: str = "resnet"
    """visual encoder backbone: 'resnet' (ResNet10, HIL-SERL default) or 'plain' (original PlainConv)"""
    encoder_feature_dim: int = 256
    """dimension of the visual feature vector"""

    # runtime (filled in)
    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0


# ---------------------------------------------------------------------------
# DrQ Random-Shift Augmentation
# ---------------------------------------------------------------------------

def random_shift(imgs: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """
    DrQ-style random-shift augmentation.

    Args:
        imgs: (B, C, H, W) float tensor, values in [0, 1].
        pad:  number of pixels to pad on each side before random crop.

    Returns:
        Augmented tensor of the same shape.
    """
    B, C, H, W = imgs.shape
    imgs = F.pad(imgs, [pad] * 4, mode="replicate")
    h_start = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)
    w_start = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)
    # build grid
    arange_h = torch.arange(H, device=imgs.device).unsqueeze(0)  # (1, H)
    arange_w = torch.arange(W, device=imgs.device).unsqueeze(0)  # (1, W)
    h_idx = h_start.unsqueeze(1) + arange_h                      # (B, H)
    w_idx = w_start.unsqueeze(1) + arange_w                      # (B, W)
    # index manually
    out = imgs[
        torch.arange(B, device=imgs.device)[:, None, None],
        :,
        h_idx[:, :, None].expand(B, H, W),
        w_idx[:, None, :].expand(B, H, W),
    ]
    return out


def augment_obs(obs: dict, pad: int = 4) -> dict:
    """Apply random_shift to the 'rgb' (and 'depth') keys of an obs dict."""
    aug = {}
    for k, v in obs.items():
        if k in ("rgb", "depth"):
            # v is (B, C, H, W) float in [0,1] after encoder preprocessing
            aug[k] = random_shift(v, pad)
        else:
            aug[k] = v
    return aug


# ---------------------------------------------------------------------------
# Replay buffers
# ---------------------------------------------------------------------------

class DictArray:
    """Typed tensor container indexed by (pos, env) or arbitrary indices."""

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32 if v.dtype in (np.float32, np.float64) else
                        torch.uint8 if v.dtype == np.uint8 else
                        torch.int16 if v.dtype == np.int16 else
                        torch.int32 if v.dtype == np.int32 else
                        v.dtype
                    )
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
            return
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


@dataclass
class ReplayBufferSample:
    obs: dict
    next_obs: dict
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Online RL replay buffer."""

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

        self.obs = DictArray(
            (self.per_env_buffer_size, num_envs),
            env.single_observation_space,
            device=storage_device,
        )
        self.next_obs = DictArray(
            (self.per_env_buffer_size, num_envs),
            env.single_observation_space,
            device=storage_device,
        )
        self.actions = torch.zeros(
            (self.per_env_buffer_size, num_envs) + env.single_action_space.shape,
            device=storage_device,
        )
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(self, obs, next_obs, action, reward, done):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action, reward, done = action.cpu(), reward.cpu(), done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        max_idx = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, max_idx, (batch_size,))
        env_inds = torch.randint(0, self.num_envs, (batch_size,))
        obs_s = {k: v.to(self.sample_device) for k, v in self.obs[batch_inds, env_inds].items()}
        nobs_s = {k: v.to(self.sample_device) for k, v in self.next_obs[batch_inds, env_inds].items()}
        return ReplayBufferSample(
            obs=obs_s,
            next_obs=nobs_s,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )

    def __len__(self):
        return self.per_env_buffer_size * self.num_envs if self.full else self.pos * self.num_envs


class DemoReplayBuffer:
    """
    Fixed-size flat replay buffer for preloaded demonstration data.
    Loaded from a .pt file that is a dict:
        {obs: {rgb:(N,H,W,C), state:(N,D)},
         next_obs: {rgb:..., state:...},
         actions:(N,A), rewards:(N,), dones:(N,)}
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.obs: Optional[dict] = None
        self.next_obs: Optional[dict] = None
        self.actions: Optional[torch.Tensor] = None
        self.rewards: Optional[torch.Tensor] = None
        self.dones: Optional[torch.Tensor] = None
        self.size = 0

    def load(self, path: str, max_size: int = 50_000):
        data = torch.load(path, map_location=self.device)
        N = min(len(data["rewards"]), max_size)
        self.obs = {k: v[:N].to(self.device) for k, v in data["obs"].items()}
        self.next_obs = {k: v[:N].to(self.device) for k, v in data["next_obs"].items()}
        self.actions = data["actions"][:N].to(self.device)
        self.rewards = data["rewards"][:N].to(self.device)
        self.dones = data["dones"][:N].to(self.device)
        self.size = N
        print(f"[DemoBuffer] Loaded {N} transitions from {path}")

    def sample(self, batch_size: int) -> ReplayBufferSample:
        assert self.size > 0, "Demo buffer is empty — provide a valid --demo_path"
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return ReplayBufferSample(
            obs={k: v[idx] for k, v in self.obs.items()},
            next_obs={k: v[idx] for k, v in self.next_obs.items()},
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
        )

    def __len__(self):
        return self.size


def merge_samples(s1: ReplayBufferSample, s2: ReplayBufferSample) -> ReplayBufferSample:
    """Concatenate two ReplayBufferSamples along the batch dimension."""
    def cat_dict(d1, d2):
        return {k: torch.cat([d1[k], d2[k]], dim=0) for k in d1}

    return ReplayBufferSample(
        obs=cat_dict(s1.obs, s2.obs),
        next_obs=cat_dict(s1.next_obs, s2.next_obs),
        actions=torch.cat([s1.actions, s2.actions], dim=0),
        rewards=torch.cat([s1.rewards, s2.rewards], dim=0),
        dones=torch.cat([s1.dones, s2.dones], dim=0),
    )


# ---------------------------------------------------------------------------
# Visual Encoders
# ---------------------------------------------------------------------------

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    layers = []
    for idx, c_out in enumerate(mlp_channels):
        layers.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            layers.append(act_builder())
        c_in = c_out
    return nn.Sequential(*layers)


# --- PlainConv (original ManiSkill baseline) ---

class PlainConv(nn.Module):
    def __init__(self, in_channels=3, out_dim=256, image_size=(128, 128)):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4) if image_size[0] == 128 else nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 1), nn.ReLU(inplace=True),
        )
        self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.cnn(x).flatten(1))


# --- ResNet10 (HIL-SERL default encoder) ---

class BasicBlock(nn.Module):
    """Standard ResNet basic residual block."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(min(32, planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(32, planes), planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, planes), planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ResNet10(nn.Module):
    """
    ResNet-10 visual encoder as used in HIL-SERL.
    Architecture: 4 stages of [16, 32, 64, 128] channels, one block each.
    Global-average-pooling → linear projection.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(16, 16), 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)
        self.layer4 = BasicBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# --- Wrapper that handles obs dict → normalised image tensor ---

class EncoderObsWrapper(nn.Module):
    """
    Wraps a raw pixel encoder backbone.
    Accepts an obs dict (uint8 HxWxC layout), normalises to [0,1],
    permutes to CxHxW and applies random-shift augmentation if requested.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @property
    def out_dim(self):
        return self.encoder.out_dim

    def preprocess(self, obs: dict) -> torch.Tensor:
        """Convert obs dict to a float32 (B,C,H,W) tensor in [0,1]."""
        parts = []
        if "rgb" in obs:
            parts.append(obs["rgb"].float() / 255.0)          # (B,H,W,3k)
        if "depth" in obs:
            parts.append(obs["depth"].float())                 # (B,H,W,1k)
        img = torch.cat(parts, dim=-1)                         # (B,H,W,C)
        return img.permute(0, 3, 1, 2)                         # (B,C,H,W)

    def forward(self, obs: dict, augment: bool = False, aug_pad: int = 4) -> torch.Tensor:
        img = self.preprocess(obs)
        if augment:
            img = random_shift(img, pad=aug_pad)
        return self.encoder(img)


# ---------------------------------------------------------------------------
# Actor & Critic
# ---------------------------------------------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, sample_obs: dict, encoder_type: str = "resnet", feature_dim: int = 256):
        super().__init__()
        action_dim = int(np.prod(envs.single_action_space.shape))
        state_dim = envs.single_observation_space["state"].shape[0]

        # Count input channels
        in_channels = 0
        image_size = (128, 128)
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = tuple(sample_obs["rgb"].shape[1:3])
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = tuple(sample_obs["depth"].shape[1:3])

        if encoder_type == "resnet":
            backbone = ResNet10(in_channels=in_channels, out_dim=feature_dim)
        else:
            backbone = PlainConv(in_channels=in_channels, out_dim=feature_dim, image_size=image_size)

        self.encoder = EncoderObsWrapper(backbone)
        self.mlp = make_mlp(feature_dim + state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        high = torch.FloatTensor(envs.single_action_space.high)
        low = torch.FloatTensor(envs.single_action_space.low)
        self.action_scale = (high - low) / 2.0
        self.action_bias = (high + low) / 2.0

    def get_feature(self, obs: dict, augment: bool = False, aug_pad: int = 4, detach_encoder: bool = False):
        visual = self.encoder(obs, augment=augment, aug_pad=aug_pad)
        if detach_encoder:
            visual = visual.detach()
        x = torch.cat([visual, obs["state"]], dim=1)
        return self.mlp(x), visual

    def forward(self, obs: dict, augment: bool = False, aug_pad: int = 4, detach_encoder: bool = False):
        x, visual = self.get_feature(obs, augment=augment, aug_pad=aug_pad, detach_encoder=detach_encoder)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std, visual

    def get_action(self, obs: dict, augment: bool = False, aug_pad: int = 4):
        mean, log_std, visual = self(obs, augment=augment, aug_pad=aug_pad)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_out = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_out, visual

    def get_eval_action(self, obs: dict):
        mean, _, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder: EncoderObsWrapper):
        super().__init__()
        self.encoder = encoder
        action_dim = int(np.prod(envs.single_action_space.shape))
        state_dim = envs.single_observation_space["state"].shape[0]
        self.mlp = make_mlp(
            encoder.out_dim + action_dim + state_dim,
            [512, 256, 1],
            last_act=False,
        )

    def forward(self, obs: dict, action: torch.Tensor, visual_feature=None,
                detach_encoder: bool = False, augment: bool = False, aug_pad: int = 4):
        if visual_feature is None:
            visual_feature = self.encoder(obs, augment=augment, aug_pad=aug_pad)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# W&B Logger
# ---------------------------------------------------------------------------

class WandbLogger:
    """Thin wrapper around wandb.log that also maintains step."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def log(self, metrics: dict, step: int):
        if self.enabled:
            wandb.log(metrics, step=step)

    def add_scalar(self, tag: str, value, step: int):
        self.log({tag: value}, step=step)

    def close(self):
        if self.enabled:
            wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = max(1, int(args.training_freq * args.utd))
    args.steps_per_env = args.training_freq // args.num_envs

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

    # -----------------------------------------------------------------------
    # Environment setup
    # -----------------------------------------------------------------------
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        sim_backend="gpu",
        sensor_configs=dict(),
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height

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

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
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

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # -----------------------------------------------------------------------
    # W&B initialisation
    # -----------------------------------------------------------------------
    logger = None
    if not args.evaluate:
        print("Running training")
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
            config=config,
            name=run_name,
            save_code=True,
            group=args.wandb_group,
            mode=args.wandb_mode,
            tags=["sac", "hil-serl", "maniskill3"],
        )
        # Log hyper-parameters as a W&B table for quick reference
        hp_table = wandb.Table(
            columns=["param", "value"],
            data=[[k, str(v)] for k, v in vars(args).items()],
        )
        wandb.log({"hyperparameters": hp_table}, step=0)

        logger = WandbLogger(enabled=(args.wandb_mode != "disabled"))
    else:
        print("Running evaluation")

    # -----------------------------------------------------------------------
    # Replay buffers
    # -----------------------------------------------------------------------
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    demo_rb: Optional[DemoReplayBuffer] = None
    use_demos = args.demo_path is not None and (args.demo_sampling_ratio > 0.0 or args.critic_threshold is not None)
    if use_demos:
        demo_rb = DemoReplayBuffer(device=device)
        demo_rb.load(args.demo_path, max_size=args.demo_buffer_size)

    # -----------------------------------------------------------------------
    # Networks
    # -----------------------------------------------------------------------
    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)

    actor = Actor(
        envs,
        sample_obs=obs,
        encoder_type=args.encoder_type,
        feature_dim=args.encoder_feature_dim,
    ).to(device)

    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Encoder parameters are shared between actor and Q-networks.
    # We optimise encoder weights through the Q-networks only (as in DrQ).
    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) +
        list(qf2.mlp.parameters()) +
        list(actor.encoder.parameters()),
        lr=args.q_lr,
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        log_alpha = torch.tensor(np.log(alpha), device=device)

    # Log architecture to W&B
    if logger is not None:
        wandb.watch([actor, qf1, qf2], log="gradients", log_freq=args.log_freq)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = 0
    global_update = 0
    learning_has_started = False
    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times: dict = defaultdict(float)

    # Track last losses for logging
    actor_loss = qf1_loss = qf2_loss = alpha_loss = torch.tensor(0.0)
    qf1_a_values = qf2_a_values = torch.tensor(0.0)

    critic_gate_active = args.critic_activation_return_threshold is None
    recent_episode_returns = deque(maxlen=args.critic_activation_return_window)

    while global_step < args.total_timesteps:

        # -------------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------------
        if args.eval_freq > 0 and (
            (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq
        ):
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics: dict = defaultdict(list)
            num_episodes = 0

            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        actor.get_eval_action(eval_obs)
                    )
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)

            eval_metrics_mean = {}
            eval_log = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                eval_log[f"eval/{k}"] = mean.item()
            eval_log["eval/num_episodes"] = int(num_episodes)

            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                eval_log["time/eval_time"] = eval_time
                logger.log(eval_log, step=global_step)

            pbar.set_description(
                f"success_once={eval_metrics_mean.get('success_once', torch.tensor(0.0)):.2f} "
                f"return={eval_metrics_mean.get('return', torch.tensor(0.0)):.2f}"
            )

            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                os.makedirs(f"runs/{run_name}", exist_ok=True)
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qf1": qf1_target.state_dict(),
                        "qf2": qf2_target.state_dict(),
                        "log_alpha": log_alpha,
                    },
                    model_path,
                )
                print(f"Model saved → {model_path}")
                if logger is not None:
                    wandb.save(model_path)

        # -------------------------------------------------------------------
        # Collect rollouts
        # -------------------------------------------------------------------
        rollout_time = time.perf_counter()
        for _ in range(args.steps_per_env):
            global_step += args.num_envs

            if not learning_has_started:
                actions = torch.tensor(
                    envs.action_space.sample(), dtype=torch.float32, device=device
                )
                visual_feature = None
            else:
                with torch.no_grad():
                    actions, _, _, visual_feature = actor.get_action(obs)

            if (
                args.critic_threshold is not None
                and critic_gate_active
                and use_demos
                and demo_rb is not None
                and len(demo_rb) > 0
            ):
                with torch.no_grad():
                    if visual_feature is None:
                        visual_feature = actor.encoder(obs)
                    q1 = qf1(obs, actions, visual_feature)
                    q2 = qf2(obs, actions, visual_feature)
                    q = torch.min(q1, q2).squeeze(-1) # critic value

                    replace_mask = q < args.critic_threshold # based on low critic value, take actions from demo buffer
                    if replace_mask.any():
                        state_diff = obs["state"].unsqueeze(1) - demo_rb.obs["state"].unsqueeze(0)
                        dist = (state_diff ** 2).sum(-1)
                        min_idx = dist.argmin(-1)
                        actions[replace_mask] = demo_rb.actions[min_idx[replace_mask]] # replace with closest demo action

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = {k: v.clone() for k, v in next_obs.items()}

            if args.bootstrap_at_done == "never":
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            elif args.bootstrap_at_done == "always":
                need_final_obs = truncations | terminations
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:  # truncated
                need_final_obs = truncations & (~terminations)
                stop_bootstrap = terminations

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k in real_next_obs:
                    real_next_obs[k][need_final_obs] = infos["final_observation"][k][need_final_obs].clone()

                if "return" in final_info["episode"]:
                    done_returns = final_info["episode"]["return"][done_mask].float()
                    if done_returns.numel() > 0:
                        recent_episode_returns.extend(done_returns.detach().cpu().tolist())
                        if (not critic_gate_active and len(recent_episode_returns) == args.critic_activation_return_window and (sum(recent_episode_returns) / len(recent_episode_returns)) >= args.critic_activation_return_threshold
                        ):
                            critic_gate_active = True

                if logger is not None:
                    ep_log = {}
                    for k, v in final_info["episode"].items():
                        ep_log[f"train/{k}"] = v[done_mask].float().mean().item()
                    ep_log["train/critic_gate_active"] = float(critic_gate_active)
                    if len(recent_episode_returns) > 0:
                        ep_log["train/recent_avg_return"] = sum(recent_episode_returns) / len(recent_episode_returns)
                    logger.log(ep_log, step=global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # -------------------------------------------------------------------
        # SAC updates
        # -------------------------------------------------------------------
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True

        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1

            # Sample from online buffer (possibly mixed with demo buffer)
            if use_demos and len(demo_rb) > 0 and args.demo_sampling_ratio > 0.0:
                demo_batch_size = int(args.batch_size * args.demo_sampling_ratio)
                online_batch_size = args.batch_size - demo_batch_size
                online_data = rb.sample(online_batch_size)
                demo_data = demo_rb.sample(demo_batch_size)
                data = merge_samples(online_data, demo_data)
            else:
                data = rb.sample(args.batch_size)

            aug = args.use_augmentation
            pad = args.aug_pad

            # -- Critic update --
            with torch.no_grad():
                next_actions, next_log_pi, _, next_visual = actor.get_action(
                    data.next_obs, augment=aug, aug_pad=pad
                )
                qf1_next = qf1_target(data.next_obs, next_actions, next_visual)
                qf2_next = qf2_target(data.next_obs, next_actions, next_visual)
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                next_q_value = (
                    data.rewards.flatten()
                    + (1 - data.dones.flatten()) * args.gamma * min_qf_next.view(-1)
                )

            visual_feature = actor.encoder(data.obs, augment=aug, aug_pad=pad)
            qf1_a_values = qf1(data.obs, data.actions, visual_feature).view(-1)
            qf2_a_values = qf2(data.obs, data.actions, visual_feature).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # -- Actor update --
            if global_update % args.policy_frequency == 0:
                pi, log_pi, _, visual_feature = actor.get_action(
                    data.obs, augment=aug, aug_pad=pad
                )
                qf1_pi = qf1(data.obs, pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data.obs, pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # -- Alpha (entropy coeff) update --
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data.obs, augment=aug, aug_pad=pad)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # -- Target network soft update --
            if global_update % args.target_network_frequency == 0:
                for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # -------------------------------------------------------------------
        # W&B logging (every log_freq steps)
        # -------------------------------------------------------------------
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if logger is not None:
                log_dict = {
                    # losses
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
                    "losses/alpha": alpha,
                    # timing
                    "time/update_time": update_time,
                    "time/rollout_time": rollout_time,
                    "time/rollout_fps": global_steps_per_iteration / max(rollout_time, 1e-8),
                    "time/total_rollout_time": cumulative_times["rollout_time"],
                    "time/total_update_time": cumulative_times["update_time"],
                    "time/total_rollout_plus_update": (
                        cumulative_times["rollout_time"] + cumulative_times["update_time"]
                    ),
                    # buffer stats
                    "buffer/online_size": len(rb),
                    "buffer/demo_size": len(demo_rb) if demo_rb is not None else 0,
                    # misc
                    "train/global_step": global_step,
                    "train/global_update": global_update,
                    "train/learning_rate_q": args.q_lr,
                    "train/learning_rate_actor": args.policy_lr,
                }
                if args.autotune:
                    log_dict["losses/alpha_loss"] = alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss
                logger.log(log_dict, step=global_step)

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------
    if not args.evaluate and args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(
            {
                "actor": actor.state_dict(),
                "qf1": qf1_target.state_dict(),
                "qf2": qf2_target.state_dict(),
                "log_alpha": log_alpha,
            },
            model_path,
        )
        print(f"Final model saved → {model_path}")
        if logger is not None:
            wandb.save(model_path)

    if logger is not None:
        logger.close()
    envs.close()
    eval_envs.close()