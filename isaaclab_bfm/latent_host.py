"""Utilities for reconstructing BFM residual observations during deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .control_interface import BFMControlInterface
from .inference import build_velocity_only_goal, build_velocity_targets_body, mask_velocity_only


@dataclass
class HistoryConfig:
    """Configuration describing the proprioception history layout."""

    history_length: int
    dim_joint_pos: int
    dim_joint_vel: int
    dim_root_ang_vel: int
    dim_gravity: int
    dim_last_action: int

    @property
    def feature_dim(self) -> int:
        return self.dim_joint_pos + self.dim_joint_vel + self.dim_root_ang_vel + self.dim_gravity

    @classmethod
    def from_metadata(cls, meta: Mapping[str, int]) -> "HistoryConfig":
        return cls(
            history_length=int(meta["history_length"]),
            dim_joint_pos=int(meta["dim_joint_pos"]),
            dim_joint_vel=int(meta["dim_joint_vel"]),
            dim_root_ang_vel=int(meta["dim_root_ang_vel"]),
            dim_gravity=int(meta["dim_gravity"]),
            dim_last_action=int(meta["dim_last_action"]),
        )


class ProprioHistory:
    """Ring buffer mirroring :class:`RealStateBuilder` without requiring Isaac Lab."""

    def __init__(self, cfg: HistoryConfig, *, batch_size: int = 1, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._alloc()

    def _alloc(self) -> None:
        H, B, F, A = self.cfg.history_length, self.batch_size, self.cfg.feature_dim, self.cfg.dim_last_action
        self.features = torch.zeros(B, H, F, device=self.device, dtype=self.dtype)
        self.last_actions = torch.zeros(B, H, A, device=self.device, dtype=self.dtype)

    def reset(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        gravity: torch.Tensor,
        last_action: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Bootstrap the history buffer with the current snapshot.

        Args:
            joint_pos: Joint positions [B, A]
            joint_vel: Joint velocities [B, A]
            root_ang_vel: Root angular velocity [B, 3]
            gravity: Gravity vector in body frame [B, 3]
            last_action: Previous joint targets [B, A]
            env_ids: Optional indices for selective reset. When ``None`` all envs reset.
        """
        stacked = self._stack_features(joint_pos, joint_vel, root_ang_vel, gravity)
        if env_ids is None:
            self.features[:] = stacked.unsqueeze(1).expand_as(self.features)
            self.last_actions[:] = last_action.unsqueeze(1).expand_as(self.last_actions)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)
            if ids.numel() == 0:
                return
            stacked_sel = stacked[ids]
            last_sel = last_action[ids]
            self.features[ids] = stacked_sel.unsqueeze(1).expand(-1, self.features.shape[1], -1)
            self.last_actions[ids] = last_sel.unsqueeze(1).expand(-1, self.last_actions.shape[1], -1)

    def update(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        gravity: torch.Tensor,
        last_action: torch.Tensor,
    ) -> None:
        feat = self._stack_features(joint_pos, joint_vel, root_ang_vel, gravity)
        self.features = torch.roll(self.features, shifts=-1, dims=1)
        self.features[:, -1] = feat
        self.last_actions = torch.roll(self.last_actions, shifts=-1, dims=1)
        self.last_actions[:, -1] = last_action

    def get_sp_real(self) -> torch.Tensor:
        history_flat = self.features.flatten(start_dim=1)
        last = self.last_actions[:, -1]
        return torch.cat([history_flat, last], dim=-1)

    def _stack_features(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        gravity: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([joint_pos, joint_vel, root_ang_vel, gravity], dim=-1)


@dataclass
class ResidualConfig:
    """Deployment-time configuration for residual blending."""

    residual_mode: str = "blend"
    alpha: float = 0.1

    def combine(self, base_action: torch.Tensor, residual_action: torch.Tensor) -> torch.Tensor:
        """Combine BFM base actions with residual outputs according to the configured mode."""

        residual_mode = self.residual_mode.lower()
        if residual_mode == "blend":
            return torch.clamp(
                (1.0 - float(self.alpha)) * base_action + float(self.alpha) * residual_action,
                min=-1.0,
                max=1.0,
            )
        if residual_mode == "delta_a":
            return torch.clamp(base_action + float(self.alpha) * residual_action, min=-1.0, max=1.0)
        raise ValueError(f"Unsupported residual_mode '{self.residual_mode}'")


class LatentEnvHost:
    """Deployment helper mirroring observation packing & history management."""

    def __init__(
        self,
        component_slices: Mapping[str, Sequence[int]],
        *,
        obs_mean: np.ndarray | torch.Tensor | None = None,
        obs_var: np.ndarray | torch.Tensor | None = None,
        normalize_inputs: bool = False,
        residual_cfg: ResidualConfig | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        history_cfg: HistoryConfig | None = None,
        control_mode: str = "velocity_only",
        action_dim: int = 23,
        batch_size: int = 1,
        residual_clip: float = 1.0,
    ):
        self.component_slices = {str(k): (int(v[0]), int(v[1])) for k, v in component_slices.items()}
        self.dim_obs = max((end for _, end in self.component_slices.values()), default=0)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.batch_size = int(max(1, batch_size))
        self._residual_cfg = residual_cfg or ResidualConfig()

        self._normalize_inputs = bool(normalize_inputs)
        if self._normalize_inputs and obs_mean is not None and obs_var is not None:
            self._obs_mean = self._to_tensor(obs_mean)
            self._obs_var = self._to_tensor(obs_var)
        else:
            self._obs_mean = None
            self._obs_var = None

        self.control_mode = control_mode
        self.control = BFMControlInterface(action_dim=action_dim, device=self.device)
        mask_template = mask_velocity_only(self.control, batch_size=1)
        self._mask_template = mask_template
        self.residual_clip = float(residual_clip)

        if history_cfg is not None:
            self.history = ProprioHistory(
                history_cfg,
                batch_size=self.batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.history = None

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
        *,
        obs_mean: np.ndarray | torch.Tensor | None = None,
        obs_var: np.ndarray | torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
    ) -> "LatentEnvHost":
        component_slices = metadata.get("component_slices")
        if component_slices is None:
            raise ValueError("Metadata missing 'component_slices'.")
        residual_cfg = ResidualConfig(
            residual_mode=str(metadata.get("residual_mode", "blend")),
            alpha=float(metadata.get("alpha", 0.1)),
        )
        history_cfg = None
        state_specs = metadata.get("state_specs")
        if state_specs is not None:
            history_cfg = HistoryConfig.from_metadata(state_specs)  # type: ignore[arg-type]
        return cls(
            component_slices=component_slices,  # type: ignore[arg-type]
            obs_mean=obs_mean,
            obs_var=obs_var,
            normalize_inputs=bool(metadata.get("host_normalize_obs", False)),
            residual_cfg=residual_cfg,
            device=device,
            dtype=dtype,
            history_cfg=history_cfg,
            control_mode=str(metadata.get("control_mode", "velocity_only")),
            action_dim=int(metadata.get("action_dim", 23)),
            batch_size=int(max(1, batch_size)),
            residual_clip=float(metadata.get("residual_clip", 1.0)),
        )

    def _to_tensor(self, array: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array.to(self.device, dtype=self.dtype)
        return torch.as_tensor(array, device=self.device, dtype=self.dtype)

    def assemble_observation(self, components: Mapping[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        """Pack component tensors into the residual observation layout and normalize if requested."""

        prepared: dict[str, torch.Tensor] = {}
        batch = None
        for name, value in components.items():
            tensor = self._to_tensor(value)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if batch is None:
                batch = tensor.shape[0]
            elif tensor.shape[0] != batch:
                raise ValueError(
                    f"Component '{name}' batch size mismatch ({tensor.shape[0]} vs {batch})."
                )
            prepared[name] = tensor

        batch = batch or self.batch_size
        obs = torch.zeros(batch, self.dim_obs, device=self.device, dtype=self.dtype)
        for name, (start, end) in self.component_slices.items():
            value = prepared.get(name)
            if value is None:
                continue
            expected = end - start
            if value.shape[1] != expected:
                raise ValueError(f"Component '{name}' expected {expected} cols, got {value.shape[1]}")
            obs[:, start:end] = value
        return self._normalize(obs)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._normalize_inputs or self._obs_mean is None or self._obs_var is None:
            return obs
        mean = self._obs_mean
        var = self._obs_var
        if mean.ndim == 1:
            mean = mean.unsqueeze(0)
        if var.ndim == 1:
            var = var.unsqueeze(0)
        return (obs - mean) / torch.sqrt(var + 1e-6)

    def combine_actions(
        self,
        base_action: np.ndarray | torch.Tensor,
        residual_action: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Blend residual actions with the base decoder output."""

        base = self._to_tensor(base_action)
        residual = self._to_tensor(residual_action)
        squeeze = False
        if base.ndim == 1:
            base = base.unsqueeze(0)
            squeeze = True
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)
            squeeze = True and residual.shape[0] == 1
        if base.shape != residual.shape:
            raise ValueError(f"Base and residual shapes must match (got {base.shape} vs {residual.shape}).")
        combined = self._residual_cfg.combine(base, residual)
        return combined.squeeze(0) if squeeze else combined

    @property
    def residual_mode(self) -> str:
        return self._residual_cfg.residual_mode

    @property
    def alpha(self) -> float:
        return self._residual_cfg.alpha

    def reset_history(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        gravity: torch.Tensor,
        last_action: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if self.history is None:
            raise RuntimeError("History management not configured for this host.")
        tensors = [joint_pos, joint_vel, root_ang_vel, gravity, last_action]
        tensors = [self._ensure_tensor(t) for t in tensors]
        self.history.reset(*tensors, env_ids=env_ids)

    def update_history(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        gravity: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        if self.history is None:
            raise RuntimeError("History management not configured for this host.")
        tensors = [joint_pos, joint_vel, root_ang_vel, gravity, last_action]
        tensors = [self._ensure_tensor(t) for t in tensors]
        self.history.update(*tensors[:-1], tensors[-1])
        return self.history.get_sp_real()

    def build_velocity_goal(
        self,
        dir_body: torch.Tensor,
        speed: torch.Tensor,
        yaw_rate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dir_body_tensor = self._ensure_tensor(dir_body)
        if dir_body_tensor.ndim == 1:
            dir_body_tensor = dir_body_tensor.unsqueeze(0)
        speed_tensor = self._ensure_tensor(speed)
        if speed_tensor.ndim == 0:
            speed_tensor = speed_tensor.unsqueeze(0).expand(dir_body_tensor.shape[0])
        if speed_tensor.ndim == 1:
            speed_tensor = speed_tensor.unsqueeze(1)
        yaw_tensor: torch.Tensor
        if yaw_rate is None:
            yaw_tensor = torch.zeros(dir_body_tensor.shape[0], 1, device=self.device, dtype=self.dtype)
        else:
            yaw_tensor = self._ensure_tensor(yaw_rate)
            if yaw_tensor.ndim == 0:
                yaw_tensor = yaw_tensor.unsqueeze(0).unsqueeze(1).expand(dir_body_tensor.shape[0], 1)
            elif yaw_tensor.ndim == 1:
                yaw_tensor = yaw_tensor.unsqueeze(1)

        v_cmd, w_cmd = build_velocity_targets_body(
            dir_body_tensor,
            speed_tensor.squeeze(-1),
            yaw_tensor.squeeze(-1),
        )
        sg_real = build_velocity_only_goal(self.control, v_cmd, w_cmd)
        mask = mask_velocity_only(self.control, batch_size=sg_real.shape[0]).to(self.device)
        return sg_real, mask

    def get_mask(self, batch_size: int | None = None) -> torch.Tensor:
        batch = batch_size or self.batch_size
        if batch <= 0:
            raise ValueError("batch_size must be positive.")
        if self._mask_template.shape[0] == batch:
            return self._mask_template.clone().to(self.device)
        return self._mask_template.expand(batch, -1).clone().to(self.device)

    def get_sp_real(self) -> torch.Tensor:
        if self.history is None:
            raise RuntimeError("History management not configured for this host.")
        return self.history.get_sp_real()

    def _ensure_tensor(self, value: torch.Tensor | np.ndarray | None) -> torch.Tensor:
        if value is None:
            raise ValueError("Tensor value cannot be None")
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=self.dtype)
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)
