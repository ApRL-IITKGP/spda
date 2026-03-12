"""
Terminal bonus variants for reward at episode success.

Controls how reward is modified when the agent achieves success.
"""

import torch


def apply_terminal(
    variant: str,
    reward: torch.Tensor,
    success: torch.Tensor,
    max_reward: float,
) -> torch.Tensor:
    """Apply terminal bonus to reward at success.

    Args:
        variant: one of "hard_jump", "smooth", "none"
        reward: current reward tensor (num_envs,)
        success: boolean success tensor (num_envs,)
        max_reward: maximum reward value (used for hard_jump override)

    Returns:
        Modified reward tensor (num_envs,)
    """
    if variant == "hard_jump":
        # Override reward to max_reward at success (standard ManiSkill approach)
        reward = reward.clone()
        reward[success] = max_reward

    elif variant == "smooth":
        # Add a fixed bonus at success without overriding shaped reward.
        # Bonus = 50% of max_reward, additive so shaped info is preserved.
        reward = reward + success.float() * (max_reward * 0.5)

    elif variant == "none":
        # Pure shaping: no terminal override or bonus
        pass

    else:
        raise ValueError(
            f"Unknown terminal variant {variant!r}. "
            f"Valid: 'hard_jump', 'smooth', 'none'"
        )

    return reward


VALID_TERMINAL_VARIANTS = ["hard_jump", "smooth", "none"]
