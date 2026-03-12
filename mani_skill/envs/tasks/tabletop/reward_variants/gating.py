"""
Gating mechanisms for staged reward.

Controls how a place_reward is activated based on a boolean condition
(e.g., is_grasped for pick tasks, reached for push tasks).
"""

import torch


def apply_gate(
    variant: str,
    place_reward: torch.Tensor,
    condition: torch.Tensor,
    grasp_quality: torch.Tensor = None,
    elapsed_steps: torch.Tensor = None,
    max_steps: int = 50,
) -> torch.Tensor:
    """Apply gating mechanism to place_reward.

    Args:
        variant: one of "hard", "soft", "additive", "curriculum"
        place_reward: shaped reward tensor (num_envs,)
        condition: boolean condition tensor (is_grasped / reached), (num_envs,)
        grasp_quality: optional continuous signal in [0,1] for soft gate.
            If None in soft mode, falls back to condition.float().
        elapsed_steps: per-env step count tensor for curriculum gate.
            If None, curriculum falls back to hard gate.
        max_steps: max episode steps (for curriculum threshold).

    Returns:
        Gated place_reward tensor (num_envs,)
    """
    if variant == "hard":
        return place_reward * condition.float()

    elif variant == "soft":
        # Use continuous grasp quality if available, else binarize
        gate_signal = grasp_quality if grasp_quality is not None else condition.float()
        return place_reward * gate_signal

    elif variant == "additive":
        # No gate: placement reward always active regardless of grasp state
        return place_reward

    elif variant == "curriculum":
        # Hard gate for first 40% of episode, then additive (no gate)
        if elapsed_steps is None or max_steps is None:
            return place_reward * condition.float()
        threshold = int(0.4 * max_steps)
        if isinstance(elapsed_steps, torch.Tensor):
            unlocked = (elapsed_steps >= threshold).float()
        else:
            unlocked = float(elapsed_steps >= threshold)
        gated = place_reward * condition.float()
        return gated * (1.0 - unlocked) + place_reward * unlocked

    else:
        raise ValueError(
            f"Unknown gate variant {variant!r}. "
            f"Valid: 'hard', 'soft', 'additive', 'curriculum'"
        )


VALID_GATE_VARIANTS = ["hard", "soft", "additive", "curriculum"]
