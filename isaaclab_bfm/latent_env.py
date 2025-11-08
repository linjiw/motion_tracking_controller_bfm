"""Latent-action wrapper for downstream PPO with a frozen BFM student.

This wraps an Isaac Lab vector env (already RslRlVecEnvWrapper) and exposes a
latent action space to PPO. It converts latent actions to joint actions via
BFM prior+decoder with a whitened latent residual. Observations include
real-state history and prior context, normalized online.

Notes:
- Mask semantics follow Stage-2: default `velocity_only` per episode.
- Goals are expressed in the robot body frame.
- Residual magnitude is clamped and lightly penalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_inv
try:
    from isaaclab.utils.math import matrix_from_quat, quat_mul
except ImportError:  # pragma: no cover - older Isaac builds
    matrix_from_quat = None
    quat_mul = None

try:
    from tensordict import TensorDict  # type: ignore
except Exception:  # pragma: no cover - tensordict might be unavailable in slim envs
    TensorDict = None  # type: ignore

from .state_builders import RealStateBuilder
from .control_interface import BFMControlInterface
from .models import BehaviorFoundationModel
from .inference import (
    build_velocity_targets_body,
    build_velocity_only_goal,
    act_with_latent_residual,
)
from whole_body_tracking.tasks.downstream.speed_env_cfg import (
    bad_body_tilt,
    bad_body_low_height,
)

try:
    from whole_body_tracking.tasks.downstream.kick_env_cfg import box_pos_body, box_lin_vel_body
except Exception:  # pragma: no cover - kick task optional during imports
    try:
        from whole_body_tracking.tasks.downstream.push_env_cfg import box_pos_body, box_lin_vel_body
    except Exception:  # pragma: no cover - object tasks optional during imports
        box_pos_body = None
        box_lin_vel_body = None

try:
    from whole_body_tracking.tasks.downstream.soccer_env_cfg import (
        ball_pos_body,
        ball_lin_vel_body,
        ball_ang_vel_body,
    )
except Exception:  # pragma: no cover - soccer task optional during imports
    ball_pos_body = None
    ball_lin_vel_body = None
    ball_ang_vel_body = None


_CANONICAL_OBS_COMPONENTS = {
    "base",
    "bfm_context",
    "goal",
    "base_action",
    "residual",
    "commands",
    "base_vels",
    "task_extras",
}

_OBS_COMPONENT_ALIASES = {
    "base": "base",
    "base_obs": "base",
    "original": "base",
    "env": "base",
    "bfm": "bfm_context",
    "bfm_obs": "bfm_context",
    "bfm_context": "bfm_context",
    "context": "bfm_context",
    "goal": "goal",
    "sg": "goal",
    "control": "goal",
    "base_action": "base_action",
    "action_base": "base_action",
    "bfm_action": "base_action",
    "residual": "residual",
    "residuals": "residual",
    "delta": "residual",
    "residual_stats": "residual",
    "commands": "commands",
    "command": "commands",
    "cmd": "commands",
    "base_vel": "base_vels",
    "base_vels": "base_vels",
    "velocities": "base_vels",
    "task_extras": "task_extras",
    "extras": "task_extras",
}

_OBS_COMPONENT_ORDER = [
    "base",
    "bfm_context",
    "goal",
    "base_action",
    "residual",
    "commands",
    "base_vels",
    "task_extras",
]


class _ObsDict(dict):
    """Lightweight observation container that mimics TensorDict.to() when tensordict is unavailable."""

    def to(self, *args, **kwargs):  # type: ignore[override]
        return _ObsDict({k: v.to(*args, **kwargs) for k, v in self.items()})


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-5, device: Optional[torch.device] = None):
        self.dim = dim
        self.eps = eps
        self.device = device or torch.device("cpu")
        self.count = torch.tensor(0.0, device=self.device)
        self.mean = torch.zeros(dim, device=self.device)
        self.M2 = torch.zeros(dim, device=self.device)
        self.frozen = False

    def update(self, x: torch.Tensor):
        if self.frozen:
            return
        x = x.detach()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        new_count = self.count + batch
        delta = x.mean(dim=0) - self.mean
        self.mean = self.mean + delta * (batch / (new_count + 1e-8))
        # Update M2 using batch variance + mean diff term
        var_batch = x.var(dim=0, unbiased=False)
        self.M2 = self.M2 + var_batch * batch + delta.pow(2) * self.count * batch / (new_count + 1e-8)
        self.count = new_count

    @property
    def var(self) -> torch.Tensor:
        return self.M2 / torch.clamp(self.count, min=1.0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def to(self, device: torch.device):
        self.device = device
        self.count = self.count.to(device)
        self.mean = self.mean.to(device)
        self.M2 = self.M2.to(device)
        return self

    def freeze(self):
        self.frozen = True


@dataclass
class LatentEnvCfg:
    latent_dim: int
    history_length: int
    alpha: float = 1.0  # Paper-faithful: direct residual addition (a' = a_base + α·Δa)
    residual_clip: float = 1.0
    residual_l2_weight: float = 0.0
    normalize_obs: bool = True
    z_dev_clip: Optional[float] = 1.0
    residual_mode: str = "delta_a"
    control_mode: str = "velocity_only"
    goal_source: str = "speed"
    obs_norm_freeze_steps: int = 20000
    guidance_lambda: float = 0.0
    use_student: bool = True
    # Speed-only: use legacy observation layout from commits 350cab5/c9fa450 for PPO
    # Layout: [sp_real, mu_p, (base_action if delta_a), last_residual, residual_norm, dir_b(2), speed(1), yaw_rate(1), base_lin_vel(3), base_ang_vel(3)]
    speed_obs_compat: bool = False
    # Observation exposure strategy for PPO actor
    # - 'inherit'   : forward the wrapped env's policy observation (default)
    # - 'control'   : expose Stage-2-style control interface features only (no base obs)
    # - 'augmented' : retain legacy augmented layout (base obs + control interface)
    obs_strategy: str = "inherit"
    # Optional custom component list (comma-separated via CLI).
    obs_components: tuple[str, ...] | None = None
    # RSL-RL observations API compatibility:
    # - 'tuple'  => get_observations() returns (obs, extras)  [e.g., Isaac Lab 0.45.1 era]
    # - 'single' => get_observations() returns obs tensor     [e.g., Isaac Lab 0.46.3 era]
    obs_api: str = "tuple"


class BFMLatentVecEnv(gym.Env):
    """Latent-action env driving a wrapped Isaac Lab vector env via BFM.

    The inner env must be an RslRlVecEnvWrapper so that stepping with joint actions
    advances the simulator correctly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_vec,  # RslRlVecEnvWrapper
        base_env,  # underlying .unwrapped Isaac env
        bfm: Optional[BehaviorFoundationModel],
        control: BFMControlInterface,
        action_dim: int,
        cfg: LatentEnvCfg,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.env_vec = env_vec
        self.base_env = base_env
        self.device = torch.device(device) if device is not None else torch.device(getattr(base_env, "device", "cuda"))
        self.num_envs = int(getattr(base_env, "num_envs", 1))
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.obs_components = self._canonicalize_obs_components(getattr(cfg, "obs_components", None))
        self.residual_mode = cfg.residual_mode.lower()
        if self.residual_mode not in {"delta_a", "blend"}:
            raise ValueError(
                f"Unknown residual_mode '{cfg.residual_mode}'. Expected one of: ['delta_a', 'blend']."
            )

        self.control_mode = cfg.control_mode.lower()
        valid_modes = {
            "velocity_only",
            "root_only",
            "keypoints_only",
            "joints_only",
            "all_on",
            "all_off",
        }
        if self.control_mode not in valid_modes:
            raise ValueError(
                f"Unknown control_mode '{cfg.control_mode}'. Expected one of: {sorted(valid_modes)}."
            )

        self.goal_source = cfg.goal_source.lower()
        valid_goal_sources = {"speed", "motion", "push", "kick", "soccer"}
        if self.goal_source not in valid_goal_sources:
            raise ValueError(
                f"Unknown goal_source '{cfg.goal_source}'. Expected one of: {sorted(valid_goal_sources)}."
            )

        self.use_student = bool(cfg.use_student)
        self.residual_dim = self.action_dim
        self.residual_clip = float(cfg.residual_clip)
        self._printed_goal_info = False
        self._warned_speed_term = False
        self.last_action_rms = 0.0
        self.last_residual_norm = 0.0
        self._printed_action_debug = False
        self.guidance_lambda = float(getattr(cfg, "guidance_lambda", 0.0) or 0.0)
        self.obs_api = str(getattr(cfg, "obs_api", "tuple")).lower()
        self._component_slices: Dict[str, Tuple[int, int]] = {}
        self._component_sequence: list[Tuple[str, int]] = []
        self._component_flags: Dict[str, bool] = {}
        self._resolved_obs_components: Tuple[str, ...] = tuple()
        self._printed_component_warnings: set[str] = set()
        self.obs_strategy = str(getattr(cfg, "obs_strategy", "inherit")).lower()
        valid_obs_strategies = {"inherit", "control", "augmented"}
        if self.obs_strategy not in valid_obs_strategies:
            raise ValueError(
                f"Unknown obs_strategy '{cfg.obs_strategy}'. Expected one of: {sorted(valid_obs_strategies)}."
            )
        self._effective_obs_strategy = self.obs_strategy
        self._printed_obs_strategy_fallback = False

        mode_note = (
            "a' = a_base + alpha * delta"
            if self.residual_mode == "delta_a"
            else "a' = (1-alpha) * a_base + alpha * delta"
        )
        print(
            "[BFM-Downstream] LatentVecEnv init | "
            f"residual_mode={self.residual_mode} ({mode_note}) residual_dim={self.residual_dim} clip={self.residual_clip} "
            f"alpha={self.cfg.alpha} control_mode={self.control_mode} goal_source={self.goal_source} "
            f"guidance_lambda={self.guidance_lambda} use_student={self.use_student} obs_strategy={self.obs_strategy}",
            flush=True,
        )

        # Frozen BFM
        if self.use_student:
            if bfm is None:
                raise ValueError("LatentEnvCfg.use_student=True requires a BehaviorFoundationModel instance")
            self.bfm = bfm.eval()
            for p in self.bfm.parameters():
                p.requires_grad = False
        else:
            self.bfm = bfm

        # Control interface and fixed mask
        self.control = control
        self.mask = self._sample_control_mask()

        # Real-state history builder
        self.real_builder = RealStateBuilder(action_dim=self.action_dim, history_length=self.cfg.history_length, device=self.device)
        self.real_builder.reset(self.base_env)

        # Observation layout extends original policy features with BFM context.
        self.dim_sp_real = self.real_builder.specs.dim_real_total
        self.dim_goal_unified = self.control.spec.dim_total
        self._push_have_box_state = bool(box_pos_body is not None and box_lin_vel_body is not None)
        self._soccer_have_ball_state = bool(ball_pos_body is not None and ball_lin_vel_body is not None)
        self._push_extra_obs = 7 if self.goal_source in {"push", "kick"} else 0
        self._soccer_extra_obs = 23 if self.goal_source == "soccer" else 0
        self._task_extra_obs = self._push_extra_obs + self._soccer_extra_obs
        inferred_base_dim = self._infer_base_obs_dim()
        self._configure_observation_layout(inferred_base_dim)
        self._obs_norm_freeze_steps = max(0, int(getattr(self.cfg, "obs_norm_freeze_steps", 0)))

        self.action_space = spaces.Box(
            low=-self.residual_clip,
            high=self.residual_clip,
            shape=(self.residual_dim,),
            dtype=np.float32,
        )

        # Book-keeping & logging
        self._last_obs: Optional[torch.Tensor] = None
        self._step_count = 0
        self._global_steps = 0
        self._log_every = 2048
        self._wandb = None  # lazy set via set_wandb
        self._metrics = {
            "speed_err_abs_sum": 0.0,
            "speed_forward_sum": 0.0,
            "speed_target_sum": 0.0,
            "lat_speed_abs_sum": 0.0,
            "yaw_z_abs_sum": 0.0,
            "yaw_cmd_sum": 0.0,
            "upright_cos_sum": 0.0,
            "delta_residual_l2_sum": 0.0,
            "z_dev_l2_sum": 0.0,
            "action_rms_sum": 0.0,
            "count": 0.0,
            "tilt_terms": 0.0,
            "low_terms": 0.0,
            "timeouts": 0.0,
        }
        if self.goal_source == "motion":
            self._metrics.update(
                {
                    "motion_error_anchor_pos_sum": 0.0,
                    "motion_error_anchor_rot_sum": 0.0,
                    "motion_error_anchor_lin_vel_sum": 0.0,
                    "motion_error_anchor_ang_vel_sum": 0.0,
                    "motion_error_body_pos_sum": 0.0,
                    "motion_error_body_rot_sum": 0.0,
                    "motion_error_joint_pos_sum": 0.0,
                    "motion_error_joint_vel_sum": 0.0,
                }
            )
        if self.goal_source == "push":
            self._metrics.update(
                {
                    "push_box_speed_proj_sum": 0.0,
                    "push_goal_speed_sum": 0.0,
                    "push_contact_gate_sum": 0.0,
                    "push_progress_raw_sum": 0.0,
                    "push_wrist_close_sum": 0.0,
                    "push_wrist_contact_gate_sum": 0.0,
                    "push_wrist_align_sum": 0.0,
                    "push_box_yaw_err_sum": 0.0,
                }
            )
        if self.goal_source == "soccer":
            self._metrics.update(
                {
                    "soccer_contact_gate_sum": 0.0,
                    "soccer_contact_error_sum": 0.0,
                    "soccer_ball_speed_proj_sum": 0.0,
                    "soccer_ball_progress_sum": 0.0,
                    "soccer_active_foot_sum": 0.0,
                    "soccer_stage_sum": 0.0,
                    "soccer_gate_distance_sum": 0.0,
                    "soccer_gate_width_sum": 0.0,
                    "soccer_gate_height_sum": 0.0,
                    "soccer_gate_yaw_sum": 0.0,
                    "soccer_gate_lateral_sum": 0.0,
                    "soccer_gate_success_sum": 0.0,
                    "soccer_ball_gate_forward_sum": 0.0,
                    "soccer_ball_gate_lat_abs_sum": 0.0,
                }
            )
        if self.goal_source == "kick":
            self._metrics.update(
                {
                    "kick_contact_gate_sum": 0.0,
                    "kick_contact_error_sum": 0.0,
                    "kick_box_progress_sum": 0.0,
                }
            )
        self._last_residual = torch.zeros(self.num_envs, self.residual_dim, device=self.device)
        self._last_residual_norm = torch.zeros(self.num_envs, 1, device=self.device)
        self._printed_get_obs = False
        self._printed_base_dim_adjust = False

    def _infer_base_obs_dim(self) -> int:
        """Infer base policy observation width resiliently across RSL-RL releases."""

        fallback = int(getattr(self.env_vec, "num_obs", 0))
        try:
            sample = self.env_vec.get_observations()
        except Exception:
            sample = None

        def _extract_dim(value) -> Optional[int]:
            if torch.is_tensor(value):
                if value.ndim >= 2:
                    return int(value.shape[-1])
                if value.ndim == 1:
                    return int(value.shape[0])
            return None

        dims: list[int] = []
        if sample is not None:
            if TensorDict is not None and isinstance(sample, TensorDict):
                try:
                    keys = list(sample.keys())
                except Exception:
                    keys = []

                if "policy" in keys:
                    dim = _extract_dim(sample["policy"])
                    if dim:
                        dims.append(dim)

                if "observations" in keys:
                    obs_group = sample["observations"]
                    if TensorDict is not None and isinstance(obs_group, TensorDict):
                        for nested_key in ("policy", "actor"):
                            if nested_key in obs_group.keys():
                                dim = _extract_dim(obs_group[nested_key])
                                if dim:
                                    dims.append(dim)
                    else:
                        dim = _extract_dim(obs_group)
                        if dim:
                            dims.append(dim)

                if not dims:
                    for key in keys:
                        try:
                            dim = _extract_dim(sample[key])
                        except Exception:
                            dim = None
                        if dim:
                            dims.append(dim)
                            break

            elif isinstance(sample, dict):
                if "policy" in sample:
                    dim = _extract_dim(sample["policy"])
                    if dim:
                        dims.append(dim)
                if not dims:
                    for value in sample.values():
                        dim = _extract_dim(value)
                        if dim:
                            dims.append(dim)
                            break
            else:
                dim = _extract_dim(sample)
                if dim:
                    dims.append(dim)

        for dim in dims:
            if dim and dim > 0:
                return int(dim)

        if fallback and fallback > 0:
            return fallback
        return self.real_builder.specs.dim_real_total

    def _canonicalize_obs_components(self, raw_components) -> tuple[str, ...] | None:
        if raw_components is None:
            return None

        if isinstance(raw_components, str):
            items = [part.strip() for part in raw_components.split(",")]
        else:
            try:
                items = [str(part).strip() for part in raw_components]
            except TypeError:
                items = [str(raw_components).strip()]

        canonical: list[str] = []
        for item in items:
            if not item:
                continue
            key = item.lower()
            mapped = _OBS_COMPONENT_ALIASES.get(key, key)
            if mapped not in _CANONICAL_OBS_COMPONENTS:
                raise ValueError(
                    f"Unknown observation component '{item}'. Valid options: {sorted(_CANONICAL_OBS_COMPONENTS)}."
                )
            if mapped not in canonical:
                canonical.append(mapped)
        return tuple(canonical) if canonical else None

    def _maybe_warn_component(self, component: str, message: str) -> None:
        if component not in self._printed_component_warnings:
            print(f"[BFM-Downstream][warn] {message}", flush=True)
            self._printed_component_warnings.add(component)

    def _default_components_for_strategy(self, speed_compat: bool) -> set[str]:
        strategy = self._effective_obs_strategy
        if strategy == "inherit":
            return {"base"}
        if strategy == "control":
            comps = {"bfm_context", "residual", "commands", "base_vels"}
            if not speed_compat:
                comps.add("goal")
            if speed_compat:
                if self.residual_mode == "delta_a":
                    comps.add("base_action")
            else:
                comps.add("base_action")
        elif strategy == "augmented":
            comps = {"base", "bfm_context", "goal", "residual", "commands", "base_vels", "base_action"}
        else:
            raise ValueError(f"Unsupported obs_strategy '{strategy}' for default components.")

        if self.goal_source in {"push", "kick", "soccer"} and not speed_compat and self._task_extra_obs > 0:
            comps.add("task_extras")
        return comps

    def _resolve_obs_components(self, speed_compat: bool) -> Dict[str, bool]:
        if self.obs_components:
            comps_source = set(self.obs_components)
        else:
            comps_source = self._default_components_for_strategy(speed_compat)

        invalid = comps_source - _CANONICAL_OBS_COMPONENTS
        if invalid:
            raise ValueError(f"Invalid observation components requested: {sorted(invalid)}")

        flags = {name: False for name in _CANONICAL_OBS_COMPONENTS}
        for name in comps_source:
            flags[name] = True

        if self.dim_base_obs <= 0 and flags["base"]:
            flags["base"] = False
            self._maybe_warn_component(
                "base",
                "Requested 'base' component but wrapped env returned 0-dim policy observation; dropping it.",
            )

        if self.cfg.latent_dim <= 0 and flags["bfm_context"]:
            flags["bfm_context"] = False
            self._maybe_warn_component(
                "bfm_context",
                "Requested 'bfm_context' component but latent_dim=0; dropping it.",
            )

        if self.dim_goal_unified <= 0 and flags["goal"]:
            flags["goal"] = False
            self._maybe_warn_component(
                "goal",
                "Requested 'goal' component but unified control vector is empty; dropping it.",
            )

        if self.action_dim <= 0 and flags["base_action"]:
            flags["base_action"] = False
            self._maybe_warn_component(
                "base_action",
                "Requested 'base_action' component but action_dim=0; dropping it.",
            )

        if self._task_extra_obs <= 0 and flags["task_extras"]:
            flags["task_extras"] = False
            self._maybe_warn_component(
                "task_extras",
                "Requested 'task_extras' component but no task-specific extras are available; dropping it.",
            )

        return flags

    def _configure_observation_layout(self, base_obs_dim: int) -> None:
        """Compute observation width, component slices, and normalization buffers."""

        base_obs_dim = int(max(0, base_obs_dim))
        self.dim_base_obs = base_obs_dim
        has_base_obs = base_obs_dim > 0

        self._effective_obs_strategy = "custom" if self.obs_components else self.obs_strategy
        if not self.obs_components and self.obs_strategy == "inherit" and not has_base_obs:
            self._effective_obs_strategy = "control"
            if not self._printed_obs_strategy_fallback:
                print(
                    "[BFM-Downstream][warn] Wrapped env returned empty policy observation. "
                    "Falling back to obs_strategy='control' (Stage-2 layout).",
                    flush=True,
                )
                self._printed_obs_strategy_fallback = True

        speed_compat = self.goal_source == "speed" and bool(getattr(self.cfg, "speed_obs_compat", False))
        self._component_flags = self._resolve_obs_components(speed_compat)
        self._include_base_action = self._component_flags.get("base_action", False)
        self._include_goal_in_obs = self._component_flags.get("goal", False)

        slices: Dict[str, Tuple[int, int]] = {}
        sequence: list[Tuple[str, int]] = []
        offset = 0

        if self._component_flags.get("base") and self.dim_base_obs > 0:
            slices["base_obs"] = (offset, offset + self.dim_base_obs)
            sequence.append(("base_obs", self.dim_base_obs))
            offset += self.dim_base_obs

        if self._component_flags.get("bfm_context") and self.dim_sp_real > 0:
            slices["sp_real"] = (offset, offset + self.dim_sp_real)
            sequence.append(("sp_real", self.dim_sp_real))
            offset += self.dim_sp_real

        if self._component_flags.get("goal") and self.dim_goal_unified > 0:
            slices["sg_real_masked"] = (offset, offset + self.dim_goal_unified)
            sequence.append(("sg_real_masked", self.dim_goal_unified))
            offset += self.dim_goal_unified

        if self._component_flags.get("bfm_context") and self.cfg.latent_dim > 0:
            slices["mu_p"] = (offset, offset + self.cfg.latent_dim)
            sequence.append(("mu_p", self.cfg.latent_dim))
            offset += self.cfg.latent_dim

        if self._component_flags.get("base_action") and self.action_dim > 0:
            slices["base_action"] = (offset, offset + self.action_dim)
            sequence.append(("base_action", self.action_dim))
            offset += self.action_dim

        if self._component_flags.get("residual"):
            slices["residual"] = (offset, offset + self.residual_dim)
            sequence.append(("residual", self.residual_dim))
            offset += self.residual_dim
            slices["residual_norm"] = (offset, offset + 1)
            sequence.append(("residual_norm", 1))
            offset += 1

        if self._component_flags.get("commands"):
            slices["dir_b"] = (offset, offset + 2)
            sequence.append(("dir_b", 2))
            offset += 2
            slices["speed"] = (offset, offset + 1)
            sequence.append(("speed", 1))
            offset += 1
            slices["yaw_rate"] = (offset, offset + 1)
            sequence.append(("yaw_rate", 1))
            offset += 1

        if self._component_flags.get("base_vels"):
            slices["base_lin_vel"] = (offset, offset + 3)
            sequence.append(("base_lin_vel", 3))
            offset += 3
            slices["base_ang_vel"] = (offset, offset + 3)
            sequence.append(("base_ang_vel", 3))
            offset += 3

        if self._component_flags.get("task_extras") and self._task_extra_obs > 0:
            slices["task_extras"] = (offset, offset + self._task_extra_obs)
            sequence.append(("task_extras", self._task_extra_obs))
            offset += self._task_extra_obs

        self._component_slices = slices
        self._component_sequence = sequence
        resolved_components = [name for name in _OBS_COMPONENT_ORDER if self._component_flags.get(name, False)]
        self._resolved_obs_components = tuple(resolved_components)
        self.dim_obs = offset

        if self.cfg.normalize_obs:
            self._obs_rnorm = RunningNorm(self.dim_obs, device=self.device)
        else:
            self._obs_rnorm = None

        high = np.ones(max(1, self.dim_obs), dtype=np.float32) * np.inf
        low = -high
        self.observation_space = spaces.Box(low=low[: self.dim_obs], high=high[: self.dim_obs], shape=(self.dim_obs,), dtype=np.float32)
        self._last_base_obs = torch.zeros(self.num_envs, self.dim_base_obs, device=self.device)

    def _wrap_obs_tensor(self, obs: torch.Tensor):
        """Package observation tensor into TensorDict-style container expected by RSL-RL 0.46."""

        if TensorDict is not None:
            return TensorDict({"policy": obs, "critic": obs}, batch_size=[self.num_envs])
        return _ObsDict({"policy": obs, "critic": obs})

    def set_wandb(self, wb):
        """Attach a WandB run for logging (wb.log-compatible)."""
        self._wandb = wb

    def set_log_every(self, steps: int):
        self._log_every = int(max(1, steps))

    @property
    def num_actions(self) -> int:
        return self.residual_dim

    @property
    def num_obs(self) -> int:
        return self.dim_obs

    def _sample_control_mask(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Sample a control mask for the configured mode.

        Args:
            batch_size: Optional batch size override. Defaults to ``num_envs``.

        Returns:
            A mask tensor on the same device as the control interface.
        """
        count = self.num_envs if batch_size is None else int(batch_size)
        mask = self.control.sample_mask(
            count,
            mask_prob=1.0,
            mode=self.control_mode,
        )
        return mask.to(self.device)

    def _build_motion_goal(self, command_name: str) -> torch.Tensor:
        try:
            cmd = self.base_env.command_manager.get_term(command_name)
        except Exception as exc:  # pragma: no cover - configuration issue
            raise RuntimeError(
                f"goal_source='{self.goal_source}' requires an environment CommandTerm named '{command_name}'."
            ) from exc

        N = self.num_envs
        dev = self.device
        q_inv = quat_inv(cmd.robot_anchor_quat_w)

        if matrix_from_quat is not None and quat_mul is not None:
            rel_q = quat_mul(q_inv, cmd.anchor_quat_w)
            rot = matrix_from_quat(rel_q)
            root_ori6_b = rot[..., :2].contiguous().view(N, 6)
        else:  # pragma: no cover - fallback for missing helpers
            root_ori6_b = torch.zeros(N, 6, device=dev)

        pos_diff = cmd.anchor_pos_w - cmd.robot_anchor_pos_w
        root_pos_b = quat_apply(q_inv, pos_diff)
        root_lin_b = quat_apply(q_inv, cmd.anchor_lin_vel_w)
        root_ang_b = quat_apply(q_inv, cmd.anchor_ang_vel_w)

        goal_w = cmd.body_pos_relative_w
        B = goal_w.shape[1]
        goal_rel = goal_w - cmd.robot_anchor_pos_w.unsqueeze(1)
        q_rep = q_inv.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
        goal_flat = goal_rel.reshape(-1, 3)
        body_b = quat_apply(q_rep, goal_flat).reshape(N, B, 3)

        joints = cmd.joint_pos

        target_bodies = 15
        if body_b.shape[1] != target_bodies:
            if body_b.shape[1] > target_bodies:
                body_b = body_b[:, :target_bodies]
            else:
                pad = torch.zeros(N, target_bodies - body_b.shape[1], 3, device=dev, dtype=body_b.dtype)
                body_b = torch.cat([body_b, pad], dim=1)

        return self.control.pack_control_vector(
            root_pos=root_pos_b,
            root_ori=root_ori6_b,
            root_lin_vel=root_lin_b,
            root_ang_vel=root_ang_b,
            keypoints=body_b,
            joints=joints,
            batch_size=N,
            device=dev,
        )

    def _build_goal_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.goal_source in {"motion", "kick"}:
            command_name = "motion"
            sg_real = self._build_motion_goal(command_name)
            dir_b = torch.zeros(self.num_envs, 2, device=self.device)
            speed = torch.zeros(self.num_envs, device=self.device)
            yaw_rate = torch.zeros(self.num_envs, device=self.device)
            if not self._printed_goal_info:
                print(
                    f"[BFM-Downstream] goal_source={self.goal_source} (reference command='{command_name}')",
                    flush=True,
                )
                self._printed_goal_info = True
            return sg_real, dir_b, speed, yaw_rate

        if self.goal_source == "push":
            sg_real, dir_b, speed, yaw_rate = self._build_push_goal()
            if not self._printed_goal_info:
                print(
                    "[BFM-Downstream] goal_source=push | "
                    f"speed_mean={speed.mean().item():.3f} yaw_rate_mean={yaw_rate.mean().item():.3f}",
                    flush=True,
                )
                self._printed_goal_info = True
            return sg_real, dir_b, speed, yaw_rate

        if self.goal_source == "soccer":
            sg_real, dir_b, speed, yaw_rate = self._build_soccer_goal()
            if not self._printed_goal_info:
                print(
                    "[BFM-Downstream] goal_source=soccer | keypoint guidance active",
                    flush=True,
                )
                self._printed_goal_info = True
            return sg_real, dir_b, speed, yaw_rate

        # Default: body-frame velocity goals
        dir_b, speed, yaw_rate = self._get_cmd_dir_speed()
        v_cmd_b, w_cmd_b = build_velocity_targets_body(dir_b, speed, yaw_rate=yaw_rate)
        sg_real = build_velocity_only_goal(self.control, v_cmd_b, w_cmd_b)
        if not self._printed_goal_info:
            print(
                "[BFM-Downstream] goal_source=speed | "
                f"speed_mean={speed.mean().item():.3f} yaw_rate_mean={yaw_rate.mean().item():.3f}",
                flush=True,
            )
            self._printed_goal_info = True
        return sg_real, dir_b, speed, yaw_rate

    def _get_cmd_dir_speed(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch current speed command (dir, speed, yaw-rate) in the robot body frame."""
        try:
            if self.goal_source == "push":
                term_name = "push"
            elif self.goal_source == "speed":
                term_name = "speed"
            else:
                term_name = None
            if term_name is None:
                raise LookupError
            cmd_term = self.base_env.command_manager.get_term(term_name)
            dir_b = cmd_term.goal_dir_b.to(self.device)
            speed = cmd_term.goal_speed[:, 0].to(self.device)
            yaw_attr = getattr(cmd_term, "goal_yaw_rate", None)
            if yaw_attr is None:
                yaw_rate = torch.zeros_like(speed)
            else:
                yaw_rate = yaw_attr[:, 0].to(self.device)
            return dir_b, speed, yaw_rate
        except Exception:
            if self.goal_source in {"speed", "push"} and not getattr(self, "_warned_speed_term", False):
                print(
                    "[BFM-Downstream][warn] speed command term missing; goals default to zeros",
                    flush=True,
                )
                self._warned_speed_term = True
            dir_b = torch.zeros(self.num_envs, 2, device=self.device)
            speed = torch.zeros(self.num_envs, device=self.device)
            yaw_rate = torch.zeros(self.num_envs, device=self.device)
            return dir_b, speed, yaw_rate

    def _get_base_vels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from whole_body_tracking.tasks.tracking import mdp as mdp_funcs
            lin = mdp_funcs.base_lin_vel(self.base_env).to(self.device)
            ang = mdp_funcs.base_ang_vel(self.base_env).to(self.device)
            return lin, ang
        except Exception:
            lin = torch.zeros(self.num_envs, 3, device=self.device)
            ang = torch.zeros(self.num_envs, 3, device=self.device)
            return lin, ang

    def _build_obs(self) -> torch.Tensor:
        base_policy_obs = self._last_base_obs
        sp_real = self.real_builder.get_current_state(self.base_env)  # [B, Dp]
        sg_real, dir_b, speed, yaw_rate = self._build_goal_state()
        sg_masked = sg_real * self.mask.to(sg_real.dtype)
        lin, ang = self._get_base_vels()

        flags = self._component_flags
        mu_p_tensor: torch.Tensor | None = None
        base_action_tensor: torch.Tensor | None = None

        if flags.get("bfm_context") or flags.get("base_action"):
            if self.use_student:
                prior = self.bfm.prior(sp_real, sg_masked)
                mu_p_tensor = prior["mu"]
                base_action_tensor = self.bfm.decoder(sp_real, mu_p_tensor).clamp(-1.0, 1.0)
            else:
                if flags.get("bfm_context"):
                    mu_p_tensor = torch.zeros(self.num_envs, self.cfg.latent_dim, device=self.device)
                if flags.get("base_action"):
                    base_action_tensor = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        if mu_p_tensor is None and flags.get("bfm_context"):
            mu_p_tensor = torch.zeros(self.num_envs, self.cfg.latent_dim, device=self.device)
        if base_action_tensor is None and flags.get("base_action"):
            base_action_tensor = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        obs_parts: list[torch.Tensor] = []
        if flags.get("base"):
            obs_parts.append(base_policy_obs)
        if flags.get("bfm_context"):
            obs_parts.append(sp_real)
        if flags.get("goal"):
            obs_parts.append(sg_masked)
        if flags.get("bfm_context") and mu_p_tensor is not None:
            obs_parts.append(mu_p_tensor)
        if flags.get("base_action") and base_action_tensor is not None:
            obs_parts.append(base_action_tensor)
        if flags.get("residual"):
            obs_parts.append(self._last_residual)
            obs_parts.append(self._last_residual_norm)
        if flags.get("commands"):
            obs_parts.extend([dir_b, speed.unsqueeze(-1), yaw_rate.unsqueeze(-1)])
        if flags.get("base_vels"):
            obs_parts.extend([lin, ang])
        if flags.get("task_extras") and self._task_extra_obs:
            if self.goal_source in {"push", "kick"}:
                if self._push_have_box_state:
                    try:
                        box_pos_b = box_pos_body(self.base_env).to(self.device)
                        box_lin_b = box_lin_vel_body(self.base_env).to(self.device)
                    except Exception:
                        box_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
                        box_lin_b = torch.zeros(self.num_envs, 3, device=self.device)
                else:
                    box_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
                    box_lin_b = torch.zeros(self.num_envs, 3, device=self.device)
                contact_gate = torch.zeros(self.num_envs, 1, device=self.device)
                try:
                    cmd_name = "push" if self.goal_source == "push" else "motion"
                    cmd_obj = self.base_env.command_manager.get_term(cmd_name)
                    metrics = getattr(cmd_obj, "metrics", {})
                    gate_key = "push/wrist_contact_gate" if self.goal_source == "push" else "kick/contact_gate"
                    gate_val = metrics.get(gate_key)
                    if torch.is_tensor(gate_val):
                        contact_gate = gate_val.to(self.device).unsqueeze(-1)
                except Exception:
                    pass
                obs_parts.extend([box_pos_b, box_lin_b, contact_gate])
            elif self.goal_source == "soccer":
                if self._soccer_have_ball_state:
                    try:
                        ball_pos_b = ball_pos_body(self.base_env).to(self.device)
                        ball_lin_b = ball_lin_vel_body(self.base_env).to(self.device)
                        ball_ang_b = ball_ang_vel_body(self.base_env).to(self.device)
                    except Exception:
                        ball_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
                        ball_lin_b = torch.zeros(self.num_envs, 3, device=self.device)
                        ball_ang_b = torch.zeros(self.num_envs, 3, device=self.device)
                else:
                    ball_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
                    ball_lin_b = torch.zeros(self.num_envs, 3, device=self.device)
                    ball_ang_b = torch.zeros(self.num_envs, 3, device=self.device)
                contact_gate = torch.zeros(self.num_envs, 1, device=self.device)
                contact_err = torch.zeros(self.num_envs, 1, device=self.device)
                speed_proj = torch.zeros(self.num_envs, 1, device=self.device)
                progress = torch.zeros(self.num_envs, 1, device=self.device)
                active = torch.zeros(self.num_envs, 1, device=self.device)
                stage = torch.zeros(self.num_envs, 1, device=self.device)
                gate_distance = torch.zeros(self.num_envs, 1, device=self.device)
                gate_width = torch.zeros(self.num_envs, 1, device=self.device)
                gate_height = torch.zeros(self.num_envs, 1, device=self.device)
                gate_yaw = torch.zeros(self.num_envs, 1, device=self.device)
                gate_lateral = torch.zeros(self.num_envs, 1, device=self.device)
                gate_success = torch.zeros(self.num_envs, 1, device=self.device)
                try:
                    cmd_obj = self.base_env.command_manager.get_term("soccer")
                    metrics = getattr(cmd_obj, "metrics", {})
                    gate_val = metrics.get("soccer/contact_gate")
                    err_val = metrics.get("soccer/contact_error")
                    speed_val = metrics.get("soccer/ball_speed_proj")
                    progress_val = metrics.get("soccer/ball_progress")
                    active_val = metrics.get("soccer/active_foot")
                    stage_val = metrics.get("soccer/curriculum_stage")
                    gate_dist_val = metrics.get("soccer/gate_distance")
                    gate_width_val = metrics.get("soccer/gate_width")
                    gate_height_val = metrics.get("soccer/gate_height")
                    gate_yaw_val = metrics.get("soccer/gate_yaw")
                    gate_lat_val = metrics.get("soccer/gate_lateral")
                    gate_success_val = metrics.get("soccer/gate_success")
                    gate_forward_val = metrics.get("soccer/ball_gate_forward")
                    gate_lat_abs_val = metrics.get("soccer/ball_gate_lateral_abs")
                    if torch.is_tensor(gate_val):
                        contact_gate = gate_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(err_val):
                        contact_err = err_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(speed_val):
                        speed_proj = speed_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(progress_val):
                        progress = progress_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(active_val):
                        active = active_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(stage_val):
                        stage = stage_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_dist_val):
                        gate_distance = gate_dist_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_width_val):
                        gate_width = gate_width_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_height_val):
                        gate_height = gate_height_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_yaw_val):
                        gate_yaw = gate_yaw_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_lat_val):
                        gate_lateral = gate_lat_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_success_val):
                        gate_success = gate_success_val.to(self.device).unsqueeze(-1)
                    if torch.is_tensor(gate_forward_val):
                        gate_distance_forward = gate_forward_val.to(self.device).unsqueeze(-1)
                    else:
                        gate_distance_forward = torch.zeros(self.num_envs, 1, device=self.device)
                    if torch.is_tensor(gate_lat_abs_val):
                        gate_lat_abs = gate_lat_abs_val.to(self.device).unsqueeze(-1)
                    else:
                        gate_lat_abs = torch.zeros(self.num_envs, 1, device=self.device)
                except Exception:
                    gate_distance_forward = torch.zeros(self.num_envs, 1, device=self.device)
                    gate_lat_abs = torch.zeros(self.num_envs, 1, device=self.device)
                obs_parts.extend(
                    [
                        ball_pos_b,
                        ball_lin_b,
                        ball_ang_b,
                        contact_gate,
                        contact_err,
                        speed_proj,
                        progress,
                        active,
                        stage,
                        gate_distance,
                        gate_width,
                        gate_height,
                        gate_yaw,
                        gate_lateral,
                        gate_success,
                        gate_distance_forward,
                        gate_lat_abs,
                    ]
                )

        obs = torch.cat(obs_parts, dim=-1) if obs_parts else torch.zeros(self.num_envs, 0, device=self.device)

        if self._obs_rnorm is not None:
            if self._obs_norm_freeze_steps and self._global_steps >= self._obs_norm_freeze_steps:
                self._obs_rnorm.freeze()
            self._obs_rnorm.update(obs)
            obs = self._obs_rnorm.normalize(obs)
        return obs

    def _build_push_goal(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            cmd = self.base_env.command_manager.get_term("push")
        except Exception as exc:  # pragma: no cover - configuration issue
            raise RuntimeError(
                "goal_source='push' requires an environment CommandTerm named 'push'."
            ) from exc

        lin_b, ang_b, keypoints_b, joints = cmd.get_control_goal()
        lin_b = lin_b.to(self.device)
        ang_b = ang_b.to(self.device)
        keypoints_b = keypoints_b.to(self.device)
        joints = joints.to(self.device)

        sg_real = self.control.pack_control_vector(
            root_lin_vel=lin_b,
            root_ang_vel=ang_b,
            keypoints=keypoints_b,
            joints=joints,
            batch_size=self.num_envs,
            device=self.device,
        )
        dir_b = cmd.goal_dir_b.to(self.device)
        speed = cmd.goal_speed[:, 0].to(self.device)
        yaw_attr = getattr(cmd, "goal_yaw_rate", None)
        if yaw_attr is None:
            yaw_rate = torch.zeros_like(speed)
        else:
            yaw_rate = yaw_attr[:, 0].to(self.device)
        return sg_real, dir_b, speed, yaw_rate

    def _build_soccer_goal(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            cmd = self.base_env.command_manager.get_term("soccer")
        except Exception as exc:
            raise RuntimeError(
                "goal_source='soccer' requires an environment CommandTerm named 'soccer'."
            ) from exc

        lin_b, ang_b, keypoints_b, joints = cmd.get_control_goal()
        lin_b = lin_b.to(self.device)
        ang_b = ang_b.to(self.device)
        keypoints_b = keypoints_b.to(self.device)
        joints = joints.to(self.device)

        sg_real = self.control.pack_control_vector(
            root_lin_vel=lin_b,
            root_ang_vel=ang_b,
            keypoints=keypoints_b,
            joints=joints,
            batch_size=self.num_envs,
            device=self.device,
        )
        dir_b = torch.zeros(self.num_envs, 2, device=self.device)
        speed = torch.zeros(self.num_envs, device=self.device)
        yaw_rate = torch.zeros(self.num_envs, device=self.device)
        return sg_real, dir_b, speed, yaw_rate

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_vec, info_vec = self.env_vec.reset()
        if not getattr(self, "_printed_reset_debug", False):
            try:
                if isinstance(obs_vec, dict):
                    keys = list(obs_vec.keys())
                else:
                    keys = None
                print(
                    f"[BFM-Downstream][debug] vec.reset -> type={type(obs_vec)} keys={keys} "
                    f"shape={getattr(obs_vec, 'shape', None)} mask_mode={self.control_mode}",
                    flush=True,
                )
            except Exception:
                pass
            self._printed_reset_debug = True
        # Extract base policy observation from wrapped reset output
        base_obs_tensor = None
        if TensorDict is not None and isinstance(obs_vec, TensorDict):
            try:
                if "policy" in obs_vec.keys():
                    base_obs_tensor = obs_vec["policy"].to(self.device, dtype=torch.float32)
            except Exception:
                base_obs_tensor = None
        if base_obs_tensor is None:
            if isinstance(obs_vec, dict):
                base = obs_vec.get("policy", None)
                if base is None:
                    base = next(iter(obs_vec.values())) if len(obs_vec) > 0 else None
            else:
                base = obs_vec
            if isinstance(base, torch.Tensor):
                base_obs_tensor = base.to(self.device, dtype=torch.float32)
        if base_obs_tensor is not None:
            base_dim = int(base_obs_tensor.shape[-1])
            if base_dim != self.dim_base_obs:
                prev_dim = self.dim_base_obs
                self._configure_observation_layout(base_dim)
                if not self._printed_base_dim_adjust:
                    try:
                        print(
                            f"[BFM-Downstream][compat] policy obs dim adjusted {prev_dim} -> {base_dim}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    self._printed_base_dim_adjust = True
        else:
            base_obs_tensor = torch.zeros(self.num_envs, self.dim_base_obs, device=self.device)
        self._last_base_obs = base_obs_tensor

        self.real_builder.reset(self.base_env)
        self.mask = self._sample_control_mask()
        self._last_residual.zero_()
        self._last_residual_norm.zero_()
        obs = self._build_obs()
        self._last_obs = obs
        self._step_count = 0
        extras = {"observations": {"actor": obs, "critic": obs}}
        if self.obs_api == "single":
            return self._wrap_obs_tensor(obs), extras
        return obs, extras

    def get_observations(self):
        """Return observations following the requested API compatibility.

        - For 'tuple': returns (obs, extras) where extras carries an observations payload
        - For 'single': returns obs tensor only
        """
        if self._last_obs is None:
            obs, _ = self.reset()
        else:
            obs = self._last_obs

        if self.obs_api == "single":
            if not self._printed_get_obs:
                try:
                    print(
                        f"[BFM-Downstream][debug] get_observations -> obs_shape={obs.shape} (single) residual_mode={self.residual_mode}",
                        flush=True,
                    )
                except Exception:
                    pass
                self._printed_get_obs = True
            return self._wrap_obs_tensor(obs)
        else:
            extras = {"observations": {"actor": obs, "critic": obs}}
            if not self._printed_get_obs:
                try:
                    print(
                        f"[BFM-Downstream][debug] get_observations -> obs_shape={obs.shape} (legacy tuple) residual_mode={self.residual_mode}",
                        flush=True,
                    )
                except Exception:
                    pass
                self._printed_get_obs = True
            return obs, extras

    def get_privileged_observations(self):
        if hasattr(self.env_vec, "get_privileged_observations"):
            try:
                priv = self.env_vec.get_privileged_observations()
                if isinstance(priv, dict) and "observations" in priv:
                    return priv
            except Exception:
                pass
        return {"observations": {}}

    def step(self, action: np.ndarray | torch.Tensor):
        self._step_count += 1
        if isinstance(action, np.ndarray):
            residual = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        else:
            residual = action.to(self.device, dtype=torch.float32)
        residual = residual.clamp(-self.residual_clip, self.residual_clip)
        if not self._printed_action_debug:
            print(
                "[BFM-Downstream][debug] latent_env.step | "
                f"residual_mean={residual.mean().item():.5f} residual_std={residual.std(unbiased=False).item():.5f}",
                flush=True,
            )

        # Build BFM inputs
        sp_real = self.real_builder.get_current_state(self.base_env)
        sg_real, dir_b, speed, yaw_rate = self._build_goal_state()
        if not self._printed_action_debug:
            print(
                "[BFM-Downstream][debug] goal state | "
                f"dir_mean={dir_b.mean(dim=0)} speed_mean={speed.mean().item():.4f} "
                f"yaw_mean={yaw_rate.mean().item():.4f}",
                flush=True,
            )

        if self.use_student:
            joint_action, mu_p, z, base_action = act_with_latent_residual(
                self.bfm,
                self.control,
                sp_real,
                sg_real,
                self.mask,
                delta=residual,
                residual_mode=self.residual_mode,
                alpha=self.cfg.alpha,
                max_deviation=None,
                guidance_lambda=self.guidance_lambda,
            )
            # Paper-faithful for residual_mode='delta_a' (addition); 'blend' uses convex mixing in act_with_latent_residual
            joint_action = torch.clamp(joint_action, -1.0, 1.0)
            if not self._printed_action_debug:
                print(
                    "[BFM-Downstream][debug] act_with_latent_residual | "
                    f"joint_mean={joint_action.mean().item():.5f} joint_std={joint_action.std(unbiased=False).item():.5f} "
                    f"base_mean={base_action.mean().item():.5f} mu_mean={mu_p.mean().item():.5f}",
                    flush=True,
                )
            # joint_action = residual
        else:
            mu_p = torch.zeros(self.num_envs, self.cfg.latent_dim, device=self.device)
            z = torch.zeros_like(mu_p)
            joint_action = torch.tanh(residual)
            joint_action = torch.clamp(joint_action, -1.0, 1.0)
            if not self._printed_action_debug:
                print(
                    "[BFM-Downstream][debug] student disabled | "
                    f"joint_mean={joint_action.mean().item():.5f} joint_std={joint_action.std(unbiased=False).item():.5f}",
                    flush=True,
                )

        residual_used = torch.tanh(residual)
        residual_norm = residual_used.pow(2).mean(dim=-1, keepdim=True).sqrt()
        residual_l2 = residual_used.pow(2).sum(dim=-1).sqrt()
        residual_penalty = residual_used.pow(2).mean(dim=-1)

        self._last_residual = residual_used.detach()
        self._last_residual_norm = residual_norm.detach()
        self.last_residual_norm = float(residual_norm.mean().item())

        # Step simulator via wrapped vec env
        if not getattr(self, "_printed_step_start", False):
            print("[BFM-Downstream][debug] env_vec.step begin", flush=True)
            self._printed_step_start = True
        obs_in, rew, done, info = self.env_vec.step(joint_action)
        # Update last base obs from wrapped output (TensorDict/dict/tensor safe)
        base_obs_tensor = None
        if TensorDict is not None and isinstance(obs_in, TensorDict):
            try:
                if "policy" in obs_in.keys():
                    base_obs_tensor = obs_in["policy"].to(self.device, dtype=torch.float32)
            except Exception:
                base_obs_tensor = None
        if base_obs_tensor is None:
            if isinstance(obs_in, dict):
                base = obs_in.get("policy", None)
                if base is None:
                    base = next(iter(obs_in.values())) if len(obs_in) > 0 else None
            else:
                base = obs_in
            if isinstance(base, torch.Tensor):
                base_obs_tensor = base.to(self.device, dtype=torch.float32)
        if base_obs_tensor is None:
            base_obs_tensor = torch.zeros(self.num_envs, self.dim_base_obs, device=self.device)
        self._last_base_obs = base_obs_tensor
        if not getattr(self, "_printed_step_done", False):
            print(
                "[BFM-Downstream][debug] env_vec.step done | "
                f"rew_mean={torch.as_tensor(rew).mean().item():.4f}",
                flush=True,
            )
            self._printed_step_done = True
        self.last_action_rms = float(joint_action.pow(2).mean().sqrt().item())
        if not getattr(self, "_printed_step_debug", False):
            try:
                if isinstance(obs_in, dict):
                    keys = list(obs_in.keys())
                else:
                    keys = None
                print(
                    f"[BFM-Downstream][debug] vec.step -> type={type(obs_in)} keys={keys} shape={getattr(obs_in, 'shape', None)}",
                    flush=True,
                )
            except Exception:
                pass
            self._printed_step_debug = True

        # Update history builder with new features
        self.real_builder.step(self.base_env)

        # Reset latent history and masks for any environments that terminated this step
        done_mask = None
        if isinstance(done, torch.Tensor):
            done_mask = done.to(self.device, dtype=torch.bool)
        elif isinstance(done, np.ndarray):
            done_mask = torch.from_numpy(done.astype(np.bool_)).to(self.device)
        if done_mask is not None and done_mask.any():
            done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                # Real-state history should start fresh from the post-reset features
                self.real_builder.reset(self.base_env, env_ids=done_ids)
                # Re-sample control mask so new episodes respect the requested mode
                new_mask = self._sample_control_mask(batch_size=int(done_ids.numel()))
                self.mask[done_ids] = new_mask
                self._last_residual[done_ids] = 0.0
                self._last_residual_norm[done_ids] = 0.0

        # Compute and accumulate debug metrics
        try:
            from whole_body_tracking.tasks.tracking import mdp as mdp_funcs
            v_b = mdp_funcs.base_lin_vel(self.base_env)
            w_b = mdp_funcs.base_ang_vel(self.base_env)
        except Exception:
            v_b = torch.zeros(self.num_envs, 3, device=self.device)
            w_b = torch.zeros(self.num_envs, 3, device=self.device)
        v_forward = v_b[:, 0] * dir_b[:, 0] + v_b[:, 1] * dir_b[:, 1]
        v_lateral = v_b[:, 0] * (-dir_b[:, 1]) + v_b[:, 1] * dir_b[:, 0]
        speed_err_abs = (v_forward - speed).abs()
        yaw_z_abs = w_b[:, 2].abs()
        # Upright cosine via gravity projection
        try:
            robot = self.base_env.scene["robot"]
            quat = robot.data.body_quat_w[:, 0]
            g_w = robot.data.GRAVITY_VEC_W
            if g_w.ndim == 1:
                g = g_w.to(self.device)
                denom = torch.norm(g) + 1e-6
                g_b = self.base_env.device_tensor_api.quat_apply_inverse(quat, g) if hasattr(self.base_env, 'device_tensor_api') else None
                if g_b is None:
                    # fallback compute via mdp if available, else approximate 1.0
                    upright_cos = torch.ones(self.num_envs, device=self.device) - 0.0
                else:
                    upright_cos = -g_b[:, 2] / denom
            else:
                denom = torch.norm(g_w, dim=-1) + 1e-6
                from isaaclab.utils import math as il_math  # type: ignore
                g_b = il_math.quat_apply_inverse(quat, g_w)
                upright_cos = -g_b[:, 2] / denom
        except Exception:
            upright_cos = torch.zeros(self.num_envs, device=self.device)

        delta_residual_l2 = residual_l2
        z_dev_l2 = residual_penalty.sqrt()
        action_rms = joint_action.pow(2).mean(dim=-1).sqrt()
        if not self._printed_action_debug:
            print(
                "[BFM-Downstream][debug] joint_action stats | "
                f"mean={joint_action.mean().item():.4f} std={joint_action.std(unbiased=False).item():.4f} "
                f"residual_mean={residual.mean().item():.4f}",
                flush=True,
            )
            if self.goal_source == "kick":
                try:
                    cmd = self.base_env.command_manager.get_term("motion")
                    metrics = getattr(cmd, "metrics", {})
                    gate = metrics.get("kick/contact_gate")
                    err0 = metrics.get("kick/contact_error_0")
                    err1 = metrics.get("kick/contact_error_1")
                    if torch.is_tensor(gate):
                        print(
                            "[BFM-Downstream][debug] kick metrics | "
                            f"gate={gate.mean().item():.4f} err0={err0.mean().item():.4f} "
                            f"err1={(err1.mean().item() if torch.is_tensor(err1) else float('nan')):.4f}",
                            flush=True,
                        )
                except Exception:
                    pass
            elif self.goal_source == "soccer":
                try:
                    cmd = self.base_env.command_manager.get_term("soccer")
                    metrics = getattr(cmd, "metrics", {})
                    gate = metrics.get("soccer/contact_gate")
                    err = metrics.get("soccer/contact_error")
                    speed_proj = metrics.get("soccer/ball_speed_proj")
                    progress = metrics.get("soccer/ball_progress")
                    stage = metrics.get("soccer/curriculum_stage")
                    gate_width_metric = metrics.get("soccer/gate_width")
                    gate_success_metric = metrics.get("soccer/gate_success")
                    if torch.is_tensor(gate):
                        print(
                            "[BFM-Downstream][debug] soccer metrics | "
                            f"gate={gate.mean().item():.4f} err={(err.mean().item() if torch.is_tensor(err) else float('nan')):.4f} "
                            f"speed={speed_proj.mean().item():.4f} progress={progress.mean().item():.4f} "
                            f"stage={(stage.mean().item() if torch.is_tensor(stage) else float('nan')):.2f} "
                            f"gate_w={(gate_width_metric.mean().item() if torch.is_tensor(gate_width_metric) else float('nan')):.3f} "
                            f"success={(gate_success_metric.mean().item() if torch.is_tensor(gate_success_metric) else float('nan')):.3f}",
                            flush=True,
                        )
                except Exception:
                    pass
            self._printed_action_debug = True

        # Termination proxies
        try:
            tilt_term = bad_body_tilt(self.base_env)
        except Exception:
            tilt_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            low_term = bad_body_low_height(self.base_env)
        except Exception:
            low_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Accumulate
        m = self._metrics
        m["speed_err_abs_sum"] += float(speed_err_abs.mean().item())
        m["speed_forward_sum"] += float(v_forward.mean().item())
        m["speed_target_sum"] += float(speed.mean().item())
        m["lat_speed_abs_sum"] += float(v_lateral.abs().mean().item())
        m["yaw_z_abs_sum"] += float(yaw_z_abs.mean().item())
        m["yaw_cmd_sum"] += float(yaw_rate.mean().item())
        m["upright_cos_sum"] += float(upright_cos.mean().item())
        m["delta_residual_l2_sum"] += float(delta_residual_l2.mean().item())
        m["z_dev_l2_sum"] += float(z_dev_l2.mean().item())
        m["action_rms_sum"] += float(action_rms.mean().item())
        m["tilt_terms"] += float(tilt_term.float().mean().item())
        m["low_terms"] += float(low_term.float().mean().item())
        # done is vector of bools
        if done_mask is not None:
            m["timeouts"] += float(done_mask.float().mean().item())
        m["count"] += 1.0

        if self.goal_source == "soccer":
            try:
                cmd_soccer = self.base_env.command_manager.get_term("soccer")
                metrics = getattr(cmd_soccer, "metrics", {})
                gate = metrics.get("soccer/contact_gate")
                err = metrics.get("soccer/contact_error")
                speed_proj = metrics.get("soccer/ball_speed_proj")
                progress = metrics.get("soccer/ball_progress")
                active = metrics.get("soccer/active_foot")
                stage = metrics.get("soccer/curriculum_stage")
                gate_distance = metrics.get("soccer/gate_distance")
                gate_width = metrics.get("soccer/gate_width")
                gate_height = metrics.get("soccer/gate_height")
                gate_yaw = metrics.get("soccer/gate_yaw")
                gate_lateral = metrics.get("soccer/gate_lateral")
                gate_success = metrics.get("soccer/gate_success")
                gate_forward = metrics.get("soccer/ball_gate_forward")
                gate_lat_abs = metrics.get("soccer/ball_gate_lateral_abs")
                if torch.is_tensor(gate):
                    m["soccer_contact_gate_sum"] += float(gate.mean().item())
                if torch.is_tensor(err):
                    m["soccer_contact_error_sum"] += float(err.mean().item())
                if torch.is_tensor(speed_proj):
                    m["soccer_ball_speed_proj_sum"] += float(speed_proj.mean().item())
                if torch.is_tensor(progress):
                    m["soccer_ball_progress_sum"] += float(progress.mean().item())
                if torch.is_tensor(active):
                    m["soccer_active_foot_sum"] += float(active.mean().item())
                if torch.is_tensor(stage):
                    m["soccer_stage_sum"] += float(stage.mean().item())
                if torch.is_tensor(gate_distance):
                    m["soccer_gate_distance_sum"] += float(gate_distance.mean().item())
                if torch.is_tensor(gate_width):
                    m["soccer_gate_width_sum"] += float(gate_width.mean().item())
                if torch.is_tensor(gate_height):
                    m["soccer_gate_height_sum"] += float(gate_height.mean().item())
                if torch.is_tensor(gate_yaw):
                    m["soccer_gate_yaw_sum"] += float(gate_yaw.mean().item())
                if torch.is_tensor(gate_lateral):
                    m["soccer_gate_lateral_sum"] += float(gate_lateral.mean().item())
                if torch.is_tensor(gate_success):
                    m["soccer_gate_success_sum"] += float(gate_success.mean().item())
                if torch.is_tensor(gate_forward):
                    m["soccer_ball_gate_forward_sum"] += float(gate_forward.mean().item())
                if torch.is_tensor(gate_lat_abs):
                    m["soccer_ball_gate_lat_abs_sum"] += float(gate_lat_abs.mean().item())
            except Exception:
                pass

        # Periodic print & wandb log
        self._global_steps += 1
        if self._global_steps % self._log_every == 0:
            count = max(1.0, m["count"])
            payload = {
                "dbg/speed_err_abs": m["speed_err_abs_sum"] / count,
                "dbg/v_forward": m["speed_forward_sum"] / count,
                "dbg/goal_speed": m["speed_target_sum"] / count,
                "dbg/lat_speed_abs": m["lat_speed_abs_sum"] / count,
                "dbg/yaw_z_abs": m["yaw_z_abs_sum"] / count,
                "dbg/yaw_cmd": m["yaw_cmd_sum"] / count,
                "dbg/upright_cos": m["upright_cos_sum"] / count,
                "dbg/delta_residual_l2": m["delta_residual_l2_sum"] / count,
                "dbg/z_dev_l2": m["z_dev_l2_sum"] / count,
                "dbg/action_rms": m["action_rms_sum"] / count,
                "term/tilt_mean": m["tilt_terms"] / count,
                "term/low_mean": m["low_terms"] / count,
                "term/timeout_mean": m["timeouts"] / count,
                "step": self._global_steps,
            }
            if self.goal_source == "push":
                payload.update(
                    {
                        "cmd/push/box_speed_proj": m["push_box_speed_proj_sum"] / count,
                        "cmd/push/goal_speed": m["push_goal_speed_sum"] / count,
                        "cmd/push/progress_gate": m["push_contact_gate_sum"] / count,
                        "cmd/push/progress_raw": m["push_progress_raw_sum"] / count,
                        "cmd/push/wrist_close_frac": m["push_wrist_close_sum"] / count,
                        "cmd/push/wrist_contact_gate": m["push_wrist_contact_gate_sum"] / count,
                        "cmd/push/wrist_align_reward": m["push_wrist_align_sum"] / count,
                        "cmd/push/box_yaw_err": m["push_box_yaw_err_sum"] / count,
                    }
                )
            elif self.goal_source == "motion":
                payload.update(
                    {
                        "cmd/motion/error_anchor_pos": m["motion_error_anchor_pos_sum"] / count,
                        "cmd/motion/error_anchor_rot": m["motion_error_anchor_rot_sum"] / count,
                        "cmd/motion/error_anchor_lin_vel": m["motion_error_anchor_lin_vel_sum"] / count,
                        "cmd/motion/error_anchor_ang_vel": m["motion_error_anchor_ang_vel_sum"] / count,
                        "cmd/motion/error_body_pos": m["motion_error_body_pos_sum"] / count,
                        "cmd/motion/error_body_rot": m["motion_error_body_rot_sum"] / count,
                        "cmd/motion/error_joint_pos": m["motion_error_joint_pos_sum"] / count,
                        "cmd/motion/error_joint_vel": m["motion_error_joint_vel_sum"] / count,
                    }
                )
            elif self.goal_source == "kick":
                payload.update(
                    {
                        "cmd/kick/contact_gate": m["kick_contact_gate_sum"] / count,
                        "cmd/kick/contact_error": m["kick_contact_error_sum"] / count,
                        "cmd/kick/box_progress": m["kick_box_progress_sum"] / count,
                    }
                )
            elif self.goal_source == "soccer":
                payload.update(
                    {
                        "cmd/soccer/contact_gate": m["soccer_contact_gate_sum"] / count,
                        "cmd/soccer/contact_error": m["soccer_contact_error_sum"] / count,
                        "cmd/soccer/ball_speed_proj": m["soccer_ball_speed_proj_sum"] / count,
                        "cmd/soccer/ball_progress": m["soccer_ball_progress_sum"] / count,
                        "cmd/soccer/active_foot": m["soccer_active_foot_sum"] / count,
                        "cmd/soccer/curriculum_stage": m["soccer_stage_sum"] / count,
                        "cmd/soccer/gate_distance": m["soccer_gate_distance_sum"] / count,
                        "cmd/soccer/gate_width": m["soccer_gate_width_sum"] / count,
                        "cmd/soccer/gate_height": m["soccer_gate_height_sum"] / count,
                        "cmd/soccer/gate_yaw": m["soccer_gate_yaw_sum"] / count,
                        "cmd/soccer/gate_lateral": m["soccer_gate_lateral_sum"] / count,
                        "cmd/soccer/gate_success": m["soccer_gate_success_sum"] / count,
                        "cmd/soccer/ball_gate_forward": m["soccer_ball_gate_forward_sum"] / count,
                        "cmd/soccer/ball_gate_lat_abs": m["soccer_ball_gate_lat_abs_sum"] / count,
                    }
                )
            msg = (
                f"[BFM-Downstream] step={self._global_steps} | "
                f"speed_err={payload['dbg/speed_err_abs']:.3f} v_fwd={payload['dbg/v_forward']:.3f} "
                f"lat={payload['dbg/lat_speed_abs']:.3f} yawz={payload['dbg/yaw_z_abs']:.3f} "
                f"yaw_cmd={payload['dbg/yaw_cmd']:.3f} "
                f"upright={payload['dbg/upright_cos']:.3f} | res={payload['dbg/delta_residual_l2']:.3f} "
                f"zdev={payload['dbg/z_dev_l2']:.3f} act_rms={payload['dbg/action_rms']:.3f} "
                f"term(tilt/low/timeout)={(payload['term/tilt_mean']):.3f}/{(payload['term/low_mean']):.3f}/{(payload['term/timeout_mean']):.3f}"
            )
            if self.goal_source == "push":
                msg += (
                    f" | box_v={payload['cmd/push/box_speed_proj']:.3f} gate={payload['cmd/push/progress_gate']:.3f} "
                    f"wrist_close={payload['cmd/push/wrist_close_frac']:.3f} yaw_err={payload['cmd/push/box_yaw_err']:.3f}"
                )
            elif self.goal_source == "motion":
                msg += (
                    f" | anch_pos={payload['cmd/motion/error_anchor_pos']:.3f} "
                    f"anch_rot={payload['cmd/motion/error_anchor_rot']:.3f} "
                    f"body_pos={payload['cmd/motion/error_body_pos']:.3f} "
                    f"joint={payload['cmd/motion/error_joint_pos']:.3f}"
                )
            elif self.goal_source == "kick":
                msg += (
                    f" | contact_gate={payload['cmd/kick/contact_gate']:.3f} "
                    f"contact_err={payload['cmd/kick/contact_error']:.3f} "
                    f"box_prog={payload['cmd/kick/box_progress']:.3f}"
                )
            elif self.goal_source == "soccer":
                msg += (
                    f" | contact_gate={payload['cmd/soccer/contact_gate']:.3f} "
                    f"contact_err={payload['cmd/soccer/contact_error']:.3f} "
                    f"ball_speed={payload['cmd/soccer/ball_speed_proj']:.3f} "
                    f"ball_progress={payload['cmd/soccer/ball_progress']:.3f} "
                    f"stage={payload['cmd/soccer/curriculum_stage']:.2f} "
                    f"gate_w={payload['cmd/soccer/gate_width']:.3f} "
                    f"gate_success={payload['cmd/soccer/gate_success']:.3f} "
                    f"forward={payload['cmd/soccer/ball_gate_forward']:.3f} "
                    f"lat_abs={payload['cmd/soccer/ball_gate_lat_abs']:.3f}"
                )
            print(msg)
            if self._wandb is not None:
                try:
                    self._wandb.log(payload)
                except Exception:
                    pass
            # reset accumulators
            for k in list(m.keys()):
                m[k] = 0.0

        # Residual penalty (optional, shaping only)
        if self.cfg.residual_l2_weight > 0.0:
            penalty = self.cfg.residual_l2_weight * residual_penalty
            if torch.is_tensor(rew):
                rew = rew - penalty
            else:
                rew = np.asarray(rew) - penalty.detach().cpu().numpy()

        # Build downstream obs after handling per-env resets so histories are consistent
        obs = self._build_obs()

        # Vectorized API expects numpy arrays
        # Enrich info dict with a few scalars for upstream logging (means)
        if not isinstance(info, dict):
            info = {}
        else:
            info = dict(info)
        info.update({
            "dbg/speed_err_abs": float(speed_err_abs.mean().item()),
            "dbg/v_forward": float(v_forward.mean().item()),
            "dbg/goal_speed": float(speed.mean().item()),
            "dbg/lat_speed_abs": float(v_lateral.abs().mean().item()),
            "dbg/yaw_z_abs": float(yaw_z_abs.mean().item()),
            "dbg/yaw_cmd": float(yaw_rate.mean().item()),
            "dbg/upright_cos": float(upright_cos.mean().item()),
            "dbg/delta_residual_l2": float(delta_residual_l2.mean().item()),
            "dbg/z_dev_l2": float(z_dev_l2.mean().item()),
            "dbg/action_rms": float(action_rms.mean().item()),
        })
        if self.goal_source == "push":
            term_name = "push"
        elif self.goal_source in {"kick", "motion"}:
            term_name = "motion"
        elif self.goal_source == "soccer":
            term_name = "soccer"
        else:
            term_name = "speed"
        try:
            cmd_term = self.base_env.command_manager.get_term(term_name)
            cmd_metrics = getattr(cmd_term, "metrics", None)
        except Exception:
            cmd_metrics = None
        if isinstance(cmd_metrics, dict):
            if term_name == "push":
                metric_specs = [
                    ("push/box_speed_proj", "cmd/push/box_speed_proj", "push_box_speed_proj_sum", False),
                    ("push/goal_speed", "cmd/push/goal_speed", "push_goal_speed_sum", False),
                    ("push/progress_gate", "cmd/push/progress_gate", "push_contact_gate_sum", False),
                    ("push/progress_raw", "cmd/push/progress_raw", "push_progress_raw_sum", False),
                    ("push/wrist_close_frac", "cmd/push/wrist_close_frac", "push_wrist_close_sum", False),
                    ("push/wrist_contact_gate", "cmd/push/wrist_contact_gate", "push_wrist_contact_gate_sum", False),
                    ("push/wrist_align_reward", "cmd/push/wrist_align_reward", "push_wrist_align_sum", False),
                    ("push/box_yaw_err", "cmd/push/box_yaw_err", "push_box_yaw_err_sum", True),
                ]
            elif self.goal_source == "kick":
                metric_specs = [
                    ("kick/contact_gate", "cmd/kick/contact_gate", "kick_contact_gate_sum", False),
                    ("kick/contact_error", "cmd/kick/contact_error", "kick_contact_error_sum", False),
                    ("kick/box_progress", "cmd/kick/box_progress", "kick_box_progress_sum", False),
                ]
            elif self.goal_source == "soccer":
                metric_specs = [
                    ("soccer/contact_gate", "cmd/soccer/contact_gate", "soccer_contact_gate_sum", False),
                    ("soccer/contact_error", "cmd/soccer/contact_error", "soccer_contact_error_sum", False),
                    ("soccer/ball_speed_proj", "cmd/soccer/ball_speed_proj", "soccer_ball_speed_proj_sum", False),
                    ("soccer/ball_progress", "cmd/soccer/ball_progress", "soccer_ball_progress_sum", False),
                    ("soccer/active_foot", "cmd/soccer/active_foot", "soccer_active_foot_sum", False),
                    ("soccer/curriculum_stage", "cmd/soccer/curriculum_stage", "soccer_stage_sum", False),
                    ("soccer/gate_distance", "cmd/soccer/gate_distance", "soccer_gate_distance_sum", False),
                    ("soccer/gate_width", "cmd/soccer/gate_width", "soccer_gate_width_sum", False),
                    ("soccer/gate_height", "cmd/soccer/gate_height", "soccer_gate_height_sum", False),
                    ("soccer/gate_yaw", "cmd/soccer/gate_yaw", "soccer_gate_yaw_sum", False),
                    ("soccer/gate_lateral", "cmd/soccer/gate_lateral", "soccer_gate_lateral_sum", False),
                    ("soccer/gate_success", "cmd/soccer/gate_success", "soccer_gate_success_sum", False),
                    ("soccer/ball_gate_forward", "cmd/soccer/ball_gate_forward", "soccer_ball_gate_forward_sum", False),
                    ("soccer/ball_gate_lateral_abs", "cmd/soccer/ball_gate_lat_abs", "soccer_ball_gate_lat_abs_sum", False),
                ]
            elif term_name == "motion":
                metric_specs = [
                    ("error_anchor_pos", "cmd/motion/error_anchor_pos", "motion_error_anchor_pos_sum", False),
                    ("error_anchor_rot", "cmd/motion/error_anchor_rot", "motion_error_anchor_rot_sum", False),
                    ("error_anchor_lin_vel", "cmd/motion/error_anchor_lin_vel", "motion_error_anchor_lin_vel_sum", False),
                    ("error_anchor_ang_vel", "cmd/motion/error_anchor_ang_vel", "motion_error_anchor_ang_vel_sum", False),
                    ("error_body_pos", "cmd/motion/error_body_pos", "motion_error_body_pos_sum", False),
                    ("error_body_rot", "cmd/motion/error_body_rot", "motion_error_body_rot_sum", False),
                    ("error_joint_pos", "cmd/motion/error_joint_pos", "motion_error_joint_pos_sum", False),
                    ("error_joint_vel", "cmd/motion/error_joint_vel", "motion_error_joint_vel_sum", False),
                ]
            else:
                metric_specs = [
                    ("speed/r_track", "cmd/speed/r_track", None, False),
                    ("speed/r_stand", "cmd/speed/r_stand", None, False),
                    ("speed/forward_speed", "cmd/speed/forward_speed", None, False),
                    ("speed/target_speed", "cmd/speed/target_speed", None, False),
                    ("speed/lateral_speed", "cmd/speed/lateral_speed", None, False),
                    ("speed/yaw_err", "cmd/speed/yaw_err", None, True),
                ]
            for metric_name, info_key, agg_key, use_abs in metric_specs:
                val = cmd_metrics.get(metric_name)
                if torch.is_tensor(val):
                    tensor_val = val.abs() if use_abs else val
                    mean_val = float(tensor_val.mean().item())
                    info[info_key] = mean_val
                    if agg_key is not None:
                        m[agg_key] += mean_val
        # Attach observations payload into info under a consistent key
        obs_payload = {"actor": obs, "critic": obs}
        observations_entry = info.get("observations")
        if isinstance(observations_entry, dict):
            observations_entry.update(obs_payload)
            info["observations"] = observations_entry
        else:
            info["observations"] = obs_payload
        self._last_obs = obs
        if self.obs_api == "single":
            return self._wrap_obs_tensor(obs), rew, done, info
        return obs, rew, done, info

    def close(self):
        if hasattr(self.env_vec, "close"):
            try:
                self.env_vec.close()
            except Exception:
                pass
        if hasattr(self.base_env, "close"):
            try:
                self.base_env.close()
            except Exception:
                pass
