"""
Reward variant components for the RewardGeom study.

Provides three independently-controllable reward dimensions:
  1. Reaching shape   → REACHING_VARIANTS dict  (distance → scalar reward)
  2. Gating mechanism → apply_gate()             (condition the place reward)
  3. Terminal bonus   → apply_terminal()         (reward at success)

Usage in task __init__:
    from mani_skill.envs.tasks.tabletop.reward_variants import (
        REACHING_VARIANTS, apply_gate, apply_terminal,
    )

    self.reach_fn = REACHING_VARIANTS[reach_variant]
    self.gate_variant = gate_variant
    self.terminal_variant = terminal_variant

Usage in compute_dense_reward:
    reaching_reward = self.reach_fn(tcp_to_obj_dist)
    gated_place = apply_gate(self.gate_variant, place_reward, is_grasped, ...)
    reward = apply_terminal(self.terminal_variant, reward, success, max_reward)
"""

from .reaching import (
    REACHING_VARIANTS,
    ADAPTIVE_REACHING_VARIANTS,
    VALID_REACHING_VARIANTS,
    AdaptiveStageReward,
    AdaptiveConcaveStageReward,
    AdaptiveDistanceReward,
    AdaptiveEMAReward,
    build_reach_fn,
)
from .gating import apply_gate, VALID_GATE_VARIANTS
from .terminal import apply_terminal, VALID_TERMINAL_VARIANTS

__all__ = [
    "REACHING_VARIANTS",
    "ADAPTIVE_REACHING_VARIANTS",
    "VALID_REACHING_VARIANTS",
    "AdaptiveStageReward",
    "AdaptiveConcaveStageReward",
    "AdaptiveDistanceReward",
    "AdaptiveEMAReward",
    "build_reach_fn",
    "apply_gate",
    "VALID_GATE_VARIANTS",
    "apply_terminal",
    "VALID_TERMINAL_VARIANTS",
]
