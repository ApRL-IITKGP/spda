"""
Reaching reward shape functions: distance d → reward.

All variants map d=0 → 1.0 (at goal) and d→∞ → 0 (or near 0).
They differ in curvature, boundedness, and monotonicity.

Adaptive variants are stateful callables; see ADAPTIVE_REACHING_VARIANTS.
"""

import torch


class AdaptiveSupportReward:
    """Concave-truncated reward with a support radius that shrinks during training.

    Reward = clamp(1 - (d / R)^2, 0, 1), where R = current_radius.

    At R=0.5m (wide): coarse gradient everywhere, no signal starvation early on.
    At R=0.1m (narrow): precise gradient, hover trap eliminated, forces commitment.

    Two schedules:
      "linear"    — R decays linearly from r_initial to r_final over the first
                    `decay_fraction` of total training steps.
      "threshold" — R steps down discretely (n_stages equal steps) each time
                    eval_success crosses `threshold`.  Call update() with
                    eval_success after every evaluation rollout.
    """

    def __init__(
        self,
        r_initial: float = 0.4,
        r_final: float = 0.1,
        schedule: str = "linear",
        decay_fraction: float = 0.6,
        threshold: float = 0.5,
        n_stages: int = 3,
    ):
        self.r_initial = r_initial
        self.r_final = r_final
        self.schedule = schedule
        self.decay_fraction = decay_fraction
        self.threshold = threshold
        self.n_stages = n_stages

        self._current_radius = float(r_initial)
        # threshold schedule state
        self._stage = 0
        self._stage_radii = [
            r_initial - i * (r_initial - r_final) / n_stages
            for i in range(n_stages + 1)
        ]

    # ------------------------------------------------------------------
    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / self._current_radius
        return torch.clamp(1.0 - (scale * d) ** 2, 0.0, 1.0)

    # ------------------------------------------------------------------
    def update(
        self,
        global_step: int,
        total_steps: int,
        eval_success: float = None,
    ) -> None:
        """Update the support radius.  Call after each evaluation rollout."""
        if self.schedule == "linear":
            decay_steps = total_steps * self.decay_fraction
            t = min(global_step / max(decay_steps, 1), 1.0)
            self._current_radius = (
                self.r_initial - t * (self.r_initial - self.r_final)
            )
        elif self.schedule == "threshold":
            if (
                eval_success is not None
                and eval_success >= self.threshold
                and self._stage < self.n_stages
            ):
                self._stage += 1
                self._current_radius = self._stage_radii[self._stage]

    @property
    def current_radius(self) -> float:
        return self._current_radius

    def __repr__(self) -> str:
        return (
            f"AdaptiveSupportReward(schedule={self.schedule!r}, "
            f"R={self._current_radius:.3f}, "
            f"range=[{self.r_initial}, {self.r_final}])"
        )


class AdaptiveScaleReward:
    """Infinite-support reward (tanh or tanh_sq) with a scale k that grows during training.

    tanh mode:    r(d) = 1 - tanh(k * d)
    tanh_sq mode: r(d) = 1 - tanh(k * d^2)

    Small k → wide shallow gradient (pulls from far, hover attractor is weak).
    Large k → steep concentrated gradient (near-goal only, hover attractor reward shrinks).

    Unlike AdaptiveSupportReward there is no hard truncation — the gradient always
    exists everywhere, so there is no chicken-and-egg on hard seeds.  The hover
    attractor reward at distance d_hover decreases continuously as k grows:
      tanh mode:    r(d_hover) = 1 - tanh(k * d_hover)  → 0 as k → ∞
      tanh_sq mode: r(d_hover) = 1 - tanh(k * d_hover²) → 0 as k → ∞

    Three schedules:
      "linear"    — k grows linearly from k_initial to k_final over decay_fraction of training.
      "threshold" — k steps up discretely (n_stages equal steps) each time
                    eval_success crosses threshold.  Call update() after each eval rollout.
      "episode"   — k grows from k_initial to k_final within each episode based on
                    elapsed_steps / max_episode_steps.  No external update() call needed;
                    the env calls prepare(elapsed_steps, max_episode_steps) each reward step.
                    Every episode gets the full curriculum, so hard seeds benefit from the
                    wide gradient early in every episode regardless of training progress.
    """

    def __init__(
        self,
        k_initial: float = 2.0,
        k_final: float = 20.0,
        mode: str = "tanh",
        schedule: str = "threshold",
        decay_fraction: float = 0.6,
        threshold: float = 0.5,
        n_stages: int = 3,
    ):
        assert mode in ("tanh", "tanh_sq"), f"Unknown mode {mode!r}"
        self.k_initial = k_initial
        self.k_final = k_final
        self.mode = mode
        self.schedule = schedule
        self.decay_fraction = decay_fraction
        self.threshold = threshold
        self.n_stages = n_stages

        self._current_k = float(k_initial)
        self._episode_k = None   # set by prepare() for episode schedule; tensor [N]
        self._stage = 0
        self._stage_ks = [
            k_initial + i * (k_final - k_initial) / n_stages
            for i in range(n_stages + 1)
        ]

    def prepare(self, elapsed_steps: torch.Tensor, max_episode_steps: int) -> None:
        """Call at the start of compute_dense_reward for episode-schedule variants.

        elapsed_steps: int tensor of shape [N] (one per parallel env).
        Sets self._episode_k to a [N] tensor of per-env k values for this step.
        No-op for non-episode schedules.
        """
        if self.schedule == "episode":
            t = (elapsed_steps.float() / max(max_episode_steps or 50, 1)).clamp(0.0, 1.0)
            self._episode_k = self.k_initial + t * (self.k_final - self.k_initial)

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        # episode schedule uses a per-env k tensor; others use scalar _current_k
        k = self._episode_k if (self.schedule == "episode" and self._episode_k is not None) else self._current_k
        if self.mode == "tanh":
            return 1.0 - torch.tanh(k * d)
        else:
            return 1.0 - torch.tanh(k * d ** 2)

    def update(
        self,
        global_step: int,
        total_steps: int,
        eval_success: float = None,
    ) -> None:
        """Update k for linear/threshold schedules. No-op for episode schedule."""
        if self.schedule == "linear":
            decay_steps = total_steps * self.decay_fraction
            t = min(global_step / max(decay_steps, 1), 1.0)
            self._current_k = self.k_initial + t * (self.k_final - self.k_initial)
        elif self.schedule == "threshold":
            if (
                eval_success is not None
                and eval_success >= self.threshold
                and self._stage < self.n_stages
            ):
                self._stage += 1
                self._current_k = self._stage_ks[self._stage]

    @property
    def current_k(self) -> float:
        k = self._episode_k if (self.schedule == "episode" and self._episode_k is not None) else self._current_k
        return float(k.mean()) if hasattr(k, "mean") else k

    def __repr__(self) -> str:
        return (
            f"AdaptiveScaleReward(mode={self.mode!r}, schedule={self.schedule!r}, "
            f"k={self._current_k:.1f}, range=[{self.k_initial}, {self.k_final}])"
        )


class AdaptiveStageReward(AdaptiveScaleReward):
    """Stage-conditioned within-episode annealing.

    Identical to AdaptiveScaleReward(schedule="episode"), but designed to be
    driven by stage_steps (steps since the current task stage began) rather than
    elapsed_steps (total steps since episode start).

    This is a marker subclass — the env is responsible for tracking stage transitions
    and passing stage_steps to prepare() instead of elapsed_steps.

    On single-stage tasks (PickCube, PushCube): stage_steps == elapsed_steps, so
    behavior is identical to adaptive_tanh_episode.

    On hierarchical tasks (StackCube): stage_steps resets to 0 when the task
    stage changes (e.g., when is_grasped flips), giving each stage its own
    wide→tight curriculum independently of episode time.
    """

    def __init__(self, k_initial: float = 2.0, k_final: float = 20.0, mode: str = "tanh"):
        super().__init__(k_initial=k_initial, k_final=k_final, mode=mode, schedule="episode")


class AdaptiveConcaveStageReward:
    """Stage-conditioned within-episode annealing of the concave-truncated support radius.

    r(d) = max(0, 1 - (d / R(t_stage))^2)

    R shrinks from r_initial to r_final over each stage independently.
    When the task stage changes (e.g., is_grasped flips), t_stage resets to 0
    and R resets to r_initial, giving the new stage its own full R_wide→R_tight
    curriculum.

    Use r_initial=0.2 (same as plain concave_truncated) to avoid the wide-hover-attractor
    problem seen with r_initial=0.4 on hard seeds (PickCube F5 / chicken-and-egg).

    On single-stage tasks: R decays from r_initial to r_final within each episode,
    similar to adaptive_concave_linear but reset per episode.
    On hierarchical tasks (StackCube): each stage gets its own R curriculum.
    """

    def __init__(self, r_initial: float = 0.2, r_final: float = 0.1):
        self.r_initial = r_initial
        self.r_final = r_final
        self._episode_radius = None  # set by prepare(); tensor [N] or scalar

    def prepare(self, stage_steps: torch.Tensor, max_episode_steps: int) -> None:
        """Call at the start of compute_dense_reward with per-env stage step counts."""
        t = (stage_steps.float() / max(max_episode_steps or 50, 1)).clamp(0.0, 1.0)
        self._episode_radius = self.r_initial - t * (self.r_initial - self.r_final)

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        R = self._episode_radius if self._episode_radius is not None else self.r_initial
        scale = 1.0 / R
        return torch.clamp(1.0 - (scale * d) ** 2, 0.0, 1.0)

    @property
    def current_radius(self):
        R = self._episode_radius
        if R is None:
            return self.r_initial
        return float(R.mean()) if hasattr(R, "mean") else R

    def __repr__(self) -> str:
        return (
            f"AdaptiveConcaveStageReward(R={self.current_radius:.3f}, "
            f"range=[{self.r_initial}, {self.r_final}])"
        )


class AdaptiveDistanceReward:
    """k(d) = k_min + (k_max - k_min) * exp(-alpha * d)

    tanh mode:    r(d) = 1 - tanh(k(d) * d)
    concave mode: r(d) = clamp(1 - (k(d) * d)^2, 0, 1)   ← concave-truncated parabola

    Wide gradient far from goal (k→k_min), tight near goal (k→k_max).
    Stage resets are automatic: when d jumps to large on sub-goal switch, k drops to k_min.
    No state, no clock — purely reactive to current distance.
    """

    def __init__(self, k_min: float = 2.0, k_max: float = 20.0, alpha: float = 10.0, mode: str = "tanh"):
        assert mode in ("tanh", "concave"), f"Unknown mode {mode!r}"
        self.k_min = k_min
        self.k_max = k_max
        self.alpha = alpha
        self.mode = mode

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        k = self.k_min + (self.k_max - self.k_min) * torch.exp(-self.alpha * d)
        if self.mode == "concave":
            return torch.clamp(1.0 - (k * d) ** 2, 0.0, 1.0)
        return 1.0 - torch.tanh(k * d)

    def __repr__(self) -> str:
        return (
            f"AdaptiveDistanceReward(mode={self.mode!r}, k_min={self.k_min}, k_max={self.k_max}, alpha={self.alpha})"
        )


class AdaptiveEMAReward:
    """EMA of reaching reward drives k.

    k = k_min + (k_max - k_min) * ema

    tanh mode:    r(d) = 1 - tanh(k * d)
    concave mode: r(d) = clamp(1 - (k * d)^2, 0, 1)   ← concave-truncated parabola

    If the policy hovers at a constant low reward, EMA plateaus → k stays wide → no
    false tightening. When the policy improves (higher r), EMA rises → k tightens.
    EMA resets to 0 at episode start (detected via elapsed_steps == 1).
    """

    def __init__(self, k_min: float = 2.0, k_max: float = 20.0, beta: float = 0.95, mode: str = "tanh"):
        assert mode in ("tanh", "concave"), f"Unknown mode {mode!r}"
        self.k_min = k_min
        self.k_max = k_max
        self.beta = beta
        self.mode = mode
        self._ema = None  # lazily initialized to [num_envs] float tensor

    def prepare(self, elapsed_steps: torch.Tensor, max_episode_steps: int) -> None:
        if self._ema is None:
            self._ema = torch.zeros_like(elapsed_steps, dtype=torch.float32)
        # Reset EMA for envs on first step of a new episode
        reset_mask = elapsed_steps == 1
        self._ema[reset_mask] = 0.0

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        k = self.k_min + (self.k_max - self.k_min) * self._ema
        if self.mode == "concave":
            r = torch.clamp(1.0 - (k * d) ** 2, 0.0, 1.0)
        else:
            r = 1.0 - torch.tanh(k * d)
        self._ema = self.beta * self._ema + (1.0 - self.beta) * r.detach()
        return r

    def __repr__(self) -> str:
        ema_mean = float(self._ema.mean()) if self._ema is not None else 0.0
        return (
            f"AdaptiveEMAReward(mode={self.mode!r}, k_min={self.k_min}, k_max={self.k_max}, "
            f"beta={self.beta}, ema={ema_mean:.3f})"
        )


REACHING_VARIANTS = {
    # Baseline: smooth sigmoid, monotone decreasing, bounded [0,1]
    # k is parameterised at build time via build_reach_fn(k=...)
    "tanh": lambda d: 1.0 - torch.tanh(5.0 * d),

    # Linear with truncation: piecewise-linear, convex, bounded [0,1]
    "linear": lambda d: torch.clamp(1.0 - 5.0 * d, 0.0, 1.0),

    # Exponential decay: convex, never exactly zero, unbounded gradient at goal
    "exponential": lambda d: torch.exp(-5.0 * d),

    # Inverse quadratic (Cauchy kernel): smooth, heavy tail, concave peak
    "quadratic": lambda d: 1.0 / (1.0 + 25.0 * d ** 2),

    # Piecewise bimodal curvature: concave (sqrt) near goal, linear (convex) far.
    # Continuous at d=0.2: left=0.5, right=0.5.
    "piecewise_cc": lambda d: torch.where(
        d < 0.2,
        1.0 - 0.5 * torch.sqrt(d / 0.2 + 1e-8),
        torch.clamp(1.5 - 5.0 * d, 0.0, 0.5),
    ),

    # Concave-in-distance, piecewise-zero beyond threshold.
    # reward = max(0, 1 - (5d)^2): downward parabola, hits 0 at d=0.2.
    # Strictly concave, non-monotone gradient (high gradient near threshold).
    "concave_truncated": lambda d: torch.clamp(1.0 - (5.0 * d) ** 2, 0.0, 1.0),

    # Concave-in-distance, piecewise-zero beyond threshold.
    # reward = max(0, 1 - (5d)^2): downward parabola, hits 0 at d=0.2.
    # Strictly concave, non-monotone gradient (high gradient near threshold).
    "concave_truncated_10": lambda d: torch.clamp(1.0 - (10.0 * d) ** 2, 0.0, 1.0),

    # tanh of d² (not d): smooth bell, non-zero everywhere, BUT zero gradient at d=0.
    # Ablation isolating "concavity / d² inside" from "hard truncation" of concave_truncated.
    # gradient = -50d·sech²(25d²) — zero at goal, peaks around d≈0.14, then decays.
    "tanh_sq": lambda d: 1.0 - torch.tanh(25.0 * d ** 2),

    # Oscillatory: non-monotone, adds local minima/maxima to gradient landscape
    "oscillatory": lambda d: (1.0 - torch.tanh(5.0 * d)) + 0.05 * torch.sin(20.0 * d),

    # Sparse threshold: discontinuous, binary signal at d < 0.05
    "sparse_threshold": lambda d: (d < 0.05).float(),

    # Concave truncated with a linearly-growing negative floor beyond d=0.2.
    # At d=0.2: floor=0, at d=0.4: floor=-0.01, at d=1.0: floor=-0.04.
    # torch.clamp requires matching types; use torch.where to apply the floor only beyond threshold.
    "concave_truncated_negative": lambda d: torch.where(
        d < 0.2,
        torch.clamp(1.0 - (5.0 * d) ** 2, 0.0, 1.0),
        -0.01 * (5.0 * d - 1.0),
    ),

    # tanh_sq shifted down to bleed small negative rewards
    "tanh_sq_negative": lambda d: torch.clamp(1.0 - torch.tanh(25.0 * d ** 2) - 0.05, -0.05, 1.0),

    # concave truncated bleeding with no slope
    "concave_truncated_negative_noslope": lambda d: torch.clamp(1.0 - (5.0 * d) ** 2 - 0.05, -0.05, 1.0),

    # Two concave parabola bumps: primary at d=0 (peak=1, zero at d=0.2),
    # secondary at d=0.3 (peak=0.5, zero at d=0.3±0.1√2≈0.16m radius).
    # Takes the max of both bumps floored at 0 — creates a bimodal gradient landscape.
    # NOTE: original used d=1.0 for secondary bump which is outside any typical
    # manipulation range; shifted to d=0.3 so the secondary bump actually activates.
    "impulse_concave": lambda d: torch.clamp(
        torch.maximum(1.0 - (10.0 * d) ** 2, 0.5 - (10.0 * (d - 0.3)) ** 2),
        min=0.0,
    ),
}

# Adaptive variants: values are classes (factories), not callables.
# The env __init__ instantiates them so each env gets its own stateful object.
ADAPTIVE_REACHING_VARIANTS = {
    # --- AdaptiveSupportReward (finite support, shrinking radius) ---
    # Linear schedule: R decays from 0.4 → 0.1 over first 60% of training
    "adaptive_concave_linear": lambda: AdaptiveSupportReward(
        r_initial=0.4, r_final=0.1, schedule="linear", decay_fraction=0.6
    ),
    # Threshold schedule: R steps down in 3 stages each time success >= 0.5
    "adaptive_concave_threshold": lambda: AdaptiveSupportReward(
        r_initial=0.4, r_final=0.1, schedule="threshold", threshold=0.5, n_stages=3
    ),

    # --- AdaptiveScaleReward (infinite support, growing k) ---
    # tanh(k*d): k grows 2→20, threshold-triggered. No hard truncation.
    # Hover attractor reward at d=0.2 shrinks from 1-tanh(0.4)≈0.61 → 1-tanh(4.0)≈0.004
    "adaptive_tanh_threshold": lambda: AdaptiveScaleReward(
        k_initial=2.0, k_final=20.0, mode="tanh", schedule="threshold",
        threshold=0.5, n_stages=3
    ),
    # tanh(k*d²): k grows 5→50, threshold-triggered.
    # Zero gradient at d=0 throughout; gradient peak shifts inward as k grows.
    "adaptive_tanh_sq_threshold": lambda: AdaptiveScaleReward(
        k_initial=5.0, k_final=50.0, mode="tanh_sq", schedule="threshold",
        threshold=0.5, n_stages=3
    ),

    # --- Within-episode annealing (episode schedule) ---
    # k ramps from k_initial → k_final over the course of EACH episode.
    # Step 0: wide gradient (pulls from far, bootstraps exploration).
    # Final step: tight gradient (hover attractor reward → near zero).
    # Hard seeds get the wide phase at the start of every episode regardless of
    # overall training progress — no chicken-and-egg.
    "adaptive_tanh_episode": lambda: AdaptiveScaleReward(
        k_initial=2.0, k_final=20.0, mode="tanh", schedule="episode",
    ),
    "adaptive_tanh_sq_episode": lambda: AdaptiveScaleReward(
        k_initial=5.0, k_final=50.0, mode="tanh_sq", schedule="episode",
    ),

    # --- Stage-conditioned annealing ---
    # k ramps from k_initial → k_final over each STAGE (not the full episode).
    # stage_steps resets to 0 at each task-stage transition (e.g., when grasped flips).
    # On single-stage tasks: identical to adaptive_tanh_episode.
    # On hierarchical tasks (StackCube): each stage gets its own full wide→tight curriculum.
    "adaptive_tanh_stage": lambda: AdaptiveStageReward(
        k_initial=2.0, k_final=20.0, mode="tanh",
    ),
    "adaptive_tanh_sq_stage": lambda: AdaptiveStageReward(
        k_initial=5.0, k_final=50.0, mode="tanh_sq",
    ),

    # --- Stage-conditioned concave truncated (finite support + stage reset) ---
    # r_initial=0.2 matches plain concave_truncated to avoid wide-hover-attractor trap.
    # r_final=0.1 tightens within each stage for precision.
    "adaptive_concave_stage": lambda: AdaptiveConcaveStageReward(
        r_initial=0.2, r_final=0.1,
    ),

    # --- Distance-reactive annealing (no state, no clock) ---
    # k(d) = k_min + (k_max - k_min) * exp(-alpha * d)
    # Wide gradient far from goal, tight near goal. Stage resets automatic via d jump.
    "adaptive_tanh_distance": lambda: AdaptiveDistanceReward(
        k_min=2.0, k_max=20.0, alpha=10.0,
    ),
    # concave variant: uses d^2 inside tanh → zero gradient at goal, peak gradient at d>0
    "adaptive_concave_distance": lambda: AdaptiveDistanceReward(
        k_min=2.0, k_max=20.0, alpha=10.0, mode="concave",
    ),

    # --- EMA-driven annealing (stateful per-env) ---
    # k driven by EMA of recent reaching rewards. Stalls (constant low reward) keep k wide.
    # EMA resets to 0 at episode start.
    "adaptive_tanh_reward_ema": lambda: AdaptiveEMAReward(
        k_min=2.0, k_max=20.0, beta=0.95,
    ),
    # concave variant: uses d^2 inside tanh
    "adaptive_concave_reward_ema": lambda: AdaptiveEMAReward(
        k_min=2.0, k_max=20.0, beta=0.95, mode="concave",
    ),
}

VALID_REACHING_VARIANTS = (
    list(REACHING_VARIANTS.keys()) + list(ADAPTIVE_REACHING_VARIANTS.keys())
)


def build_reach_fn(
    variant: str,
    k: float = 5.0,
    k_min: float = 2.0,
    k_max: float = 20.0,
    alpha: float = 10.0,
):
    """Construct a reaching reward function by name.

    ``k`` parameterises the scale of static variants that support it:
      - ``tanh``              → ``1 - tanh(k * d)``          (default k=5)
      - ``concave_truncated`` → ``max(0, 1 - (k*d)^2)``      (default k=5, R=1/k=0.2m)

    ``k_min``, ``k_max``, ``alpha`` parameterise the distance-adaptive variants.
    All other variants ignore these kwargs.
    """
    if variant == "tanh":
        return lambda d: 1.0 - torch.tanh(k * d)
    if variant == "concave_truncated":
        return lambda d: torch.clamp(1.0 - (k * d) ** 2, 0.0, 1.0)
    if variant in ("adaptive_concave_distance", "adaptive_tanh_distance"):
        mode = "concave" if "concave" in variant else "tanh"
        return AdaptiveDistanceReward(k_min=k_min, k_max=k_max, alpha=alpha, mode=mode)
    if variant in ADAPTIVE_REACHING_VARIANTS:
        return ADAPTIVE_REACHING_VARIANTS[variant]()
    return REACHING_VARIANTS[variant]
