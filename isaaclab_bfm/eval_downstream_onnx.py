#!/usr/bin/env python
"""Evaluate a downstream speed controller using exported BFM base and residual ONNX modules.

This script mirrors :mod:`eval_downstream.py` but replaces the hybrid PyTorch latent wrapper
with :class:`behaviorfoundation.bfm.latent_host.LatentEnvHost`. Both the BFM prior+decoder and
the residual policy are served through ONNX Runtime, allowing validation of the deployment pair
that will run inside the speed controller.

Example
-------

```bash
source ${ISAACLAB_PATH}/python_env.sh
python scripts/behaviorfoundation/eval_downstream_onnx.py \
    --bfm_base deploy/speed_controller/BFMBase.onnx \
    --residual deploy/speed_controller/Residual.onnx \
    --metadata logs/rsl_rl/speed_run/deploy_meta.json \
    --obs_norm logs/rsl_rl/speed_run/deploy_obs_norm.npz \
    --task Downstream-Speed-G1-v0 --num_envs 1024 --eval_steps 102400 --headless
```
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from isaaclab.app import AppLauncher


def _load_train_downstream_module():
    """Import the train-downstream script despite the hyphenated filename."""

    existing = sys.modules.get("train_downstream_module")
    if isinstance(existing, ModuleType):
        return existing

    scripts_root = Path(__file__).resolve().parent.parent
    module_path = scripts_root / "train-downstream.py"
    if not module_path.is_file():
        raise ImportError(f"train-downstream.py not found at {module_path}")

    spec = importlib.util.spec_from_file_location("train_downstream_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to build import spec for train-downstream.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("train_downstream_module", module)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_train_downstream = _load_train_downstream_module()
try:
    _create_env = getattr(_train_downstream, "_create_env")  # type: ignore[attr-defined]
except AttributeError as err:
    raise ImportError("train-downstream.py is missing the _create_env helper") from err


def _add_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a downstream residual ONNX policy inside Isaac Lab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bfm_base", required=True, type=str, help="Path to the exported BFM prior+decoder ONNX file")
    parser.add_argument("--residual", required=True, type=str, help="Path to the exported residual ONNX policy")
    parser.add_argument("--metadata", required=True, type=str, help="deploy_meta.json produced during training")
    parser.add_argument("--obs_norm", type=str, default=None, help="Optional deploy_obs_norm.npz for diagnostics")
    parser.add_argument("--task", type=str, default="Downstream-Speed-G1-v0")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--eval_steps", type=int, default=20480, help="Total simulator steps (across all envs)")
    parser.add_argument("--log_interval", type=int, default=512, help="Number of simulation ticks between prints")
    parser.add_argument("--seed", type=int, default=None, help="Optional environment seed")

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_metadata(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_obs_norm(path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None or not path.is_file():
        return None, None
    stats = np.load(path)
    return stats.get("mean"), stats.get("var")


def _apply_speed_metadata(args: argparse.Namespace, deploy_meta: Mapping[str, Any]) -> None:
    """Back-fill CLI arguments with recorded speed command config."""

    speed_meta = deploy_meta.get("speed_command")
    if not speed_meta:
        return

    scalar_attrs = [
        "speed_max_start",
        "speed_max_end",
        "speed_warmup_iters",
        "grace_steps",
        "speed_yaw_std",
        "speed_yaw_limit",
    ]
    bool_attrs = ["enable_yaw_command", "debug_vis", "align_heading"]
    str_attrs = ["direction_mode"]
    range_attrs = ["speed_range", "yaw_rate_range"]

    for attr in scalar_attrs:
        if attr in speed_meta:
            setattr(args, attr, float(speed_meta[attr]))
    for attr in bool_attrs:
        if attr in speed_meta:
            setattr(args, attr, bool(speed_meta[attr]))
    for attr in str_attrs:
        if attr in speed_meta:
            setattr(args, attr, str(speed_meta[attr]))
    for attr in range_attrs:
        if attr in speed_meta:
            rng = speed_meta[attr]
            if isinstance(rng, Sequence):
                setattr(args, attr, [float(v) for v in rng])


def _apply_speed_term_state(base_env, deploy_meta: Mapping[str, Any]) -> None:
    """Restore the stored global_step for the speed command term."""

    speed_meta = deploy_meta.get("speed_command")
    if not speed_meta:
        return
    manager = getattr(base_env, "command_manager", None)
    if manager is None:
        manager = getattr(getattr(base_env, "unwrapped", None), "command_manager", None)
    if manager is None:
        return
    try:
        term = manager.get_term("speed")
    except Exception:
        return
    if term is None:
        return
    try:
        if "global_step" in speed_meta:
            term._global_step = int(speed_meta["global_step"])
    except Exception:
        pass


def _extract_policy_obs(obs: Any) -> torch.Tensor:
    """Extract policy observation tensor from various observation formats."""
    # Handle tuple: (obs, extras) or (obs,)
    if isinstance(obs, tuple) and obs:
        obs = obs[0]

    # Handle direct tensor
    if isinstance(obs, torch.Tensor):
        return obs

    # Handle dict-like objects (dict or TensorDict)
    if hasattr(obs, "get"):
        tensor = obs.get("policy", None)
        if isinstance(tensor, torch.Tensor):
            return tensor

    if hasattr(obs, "keys"):
        try:
            if "policy" in obs:
                tensor = obs["policy"]
                if isinstance(tensor, torch.Tensor):
                    return tensor
        except Exception:
            pass

    raise TypeError(f"Unable to extract policy observation from {type(obs)}")


def _ensure_numpy_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected a rank-2 array, got shape {array.shape}")
    return array


def _make_session(path: Path, providers: Sequence[str] | None = None) -> ort.InferenceSession:
    providers = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    try:
        return ort.InferenceSession(path.as_posix(), providers=providers)
    except Exception:
        return ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])


def _gather_robot_state(base_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from whole_body_tracking.tasks.tracking import mdp as mdp_funcs  # type: ignore
    except ImportError as err:
        raise ImportError("whole_body_tracking.tasks.tracking.mdp not found. Did you install the package?") from err

    joint_pos = mdp_funcs.joint_pos_rel(base_env)
    joint_vel = mdp_funcs.joint_vel_rel(base_env)
    root_ang_vel = mdp_funcs.base_ang_vel(base_env)
    last_action = mdp_funcs.last_action(base_env)

    robot = base_env.scene["robot"]
    root_quat = robot.data.root_quat_w

    gravity_world = None
    for attr in ("GRAVITY_VEC_W", "gravity_w"):
        gw = getattr(robot.data, attr, None)
        if gw is not None:
            gravity_world = gw
            break
    if gravity_world is None:
        gravity_world = torch.tensor([0.0, 0.0, -9.81], device=root_quat.device)

    if gravity_world.ndim == 1:
        gravity_world = gravity_world.unsqueeze(0).expand(root_quat.shape[0], -1)

    try:
        import isaaclab.utils.math as il_math  # type: ignore
    except ImportError as err:
        raise ImportError("Failed to import isaaclab.utils.math. Ensure Isaac Lab is sourced.") from err

    gravity_body = il_math.quat_apply_inverse(root_quat, gravity_world)
    gravity_body = F.normalize(gravity_body, dim=-1)

    return joint_pos, joint_vel, root_ang_vel, gravity_body, last_action


def _gather_base_velocities(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from whole_body_tracking.tasks.tracking import mdp as mdp_funcs  # type: ignore
    except ImportError as err:
        raise ImportError("whole_body_tracking.tasks.tracking.mdp not found. Did you install the package?") from err

    base_lin = mdp_funcs.base_lin_vel(base_env)
    base_ang = mdp_funcs.base_ang_vel(base_env)
    return base_lin, base_ang


def _get_speed_command(base_env):
    manager = getattr(base_env, "command_manager", None)
    if manager is None:
        manager = getattr(getattr(base_env, "unwrapped", None), "command_manager", None)
    if manager is None:
        raise RuntimeError("Environment is missing a command_manager")
    try:
        return manager.get_term("speed")
    except Exception as err:
        raise RuntimeError("Command term 'speed' not found; did you load a speed task?") from err


def _run_eval(args: argparse.Namespace) -> None:
    module_root = Path(__file__).resolve().parent.parent
    module_root_str = str(module_root)
    if module_root_str not in sys.path:
        sys.path.append(module_root_str)
    from behaviorfoundation.bfm.latent_host import LatentEnvHost  # type: ignore

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    metadata = _load_metadata(Path(args.metadata).expanduser().resolve())
    _apply_speed_metadata(args, metadata)

    app = AppLauncher(args)
    simulation_app = app.app

    env, vec_env = _create_env(args)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    _apply_speed_term_state(base_env, metadata)

    num_envs = int(getattr(base_env, "num_envs", args.num_envs))
    env_device = torch.device(getattr(base_env, "device", args.device))
    host_device = env_device  # CRITICAL: Use same device as env to avoid CPU/GPU mismatch

    obs_mean, obs_var = _load_obs_norm(Path(args.obs_norm).expanduser().resolve() if args.obs_norm else None)
    host = LatentEnvHost.from_metadata(
        metadata,
        obs_mean=obs_mean,
        obs_var=obs_var,
        device=host_device,
        batch_size=num_envs,
    )

    base_session = _make_session(Path(args.bfm_base).expanduser().resolve())
    residual_session = _make_session(Path(args.residual).expanduser().resolve())

    base_input_names = [tensor.name for tensor in base_session.get_inputs()]
    base_output_names = [tensor.name for tensor in base_session.get_outputs()]
    residual_input_names = [tensor.name for tensor in residual_session.get_inputs()]
    residual_output_names = [tensor.name for tensor in residual_session.get_outputs()]
    if not base_input_names or not residual_input_names:
        raise RuntimeError("ONNX models are missing graph inputs; check the exported artefacts.")

    print(
        "[Eval-ONNX] Sessions ready | base_inputs=%s base_outputs=%s residual_inputs=%s residual_outputs=%s"
        % (base_input_names, base_output_names, residual_input_names, residual_output_names),
        flush=True,
    )
    print(
        "[Eval-ONNX] Action dimensions | expected_dim=%d (from metadata) bfm_will_pad_to=%d"
        % (int(metadata.get("action_dim", 29)), int(metadata.get("action_dim", 29))),
        flush=True,
    )

    if obs_mean is not None and obs_var is not None:
        print(
            "[Eval-ONNX] deploy_obs_norm stats | mean_abs={:.6f} var_mean={:.6f}".format(
                float(np.mean(np.abs(obs_mean))), float(np.mean(obs_var))
            ),
            flush=True,
        )

    reset_out = vec_env.reset()
    base_obs_tensor = _extract_policy_obs(reset_out)
    base_obs_host = base_obs_tensor.detach().to(host_device, dtype=host.dtype)

    joint_pos, joint_vel, root_ang_vel, gravity_body, last_action = _gather_robot_state(base_env)
    host.reset_history(
        joint_pos.to(host_device, host.dtype),
        joint_vel.to(host_device, host.dtype),
        root_ang_vel.to(host_device, host.dtype),
        gravity_body.to(host_device, host.dtype),
        last_action.to(host_device, host.dtype),
    )

    speed_term = _get_speed_command(base_env)
    warmup_grace = int(metadata.get("speed_command", {}).get("grace_steps", 0))

    last_residual = torch.zeros(num_envs, int(metadata.get("action_dim", last_action.shape[-1])), device=host_device)
    last_residual_norm = torch.zeros(num_envs, 1, device=host_device)

    per_env_returns = torch.zeros(num_envs, device=host_device)
    per_env_lengths = torch.zeros(num_envs, device=host_device)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    total_steps = 0
    sim_ticks = 0
    start_time = time.time()
    residual_clip = float(getattr(host, "residual_clip", metadata.get("residual_clip", 1.0)))

    while total_steps < args.eval_steps:
        joint_pos, joint_vel, root_ang_vel, gravity_body, last_action = _gather_robot_state(base_env)
        sp_real = host.update_history(
            joint_pos.to(host_device, host.dtype),
            joint_vel.to(host_device, host.dtype),
            root_ang_vel.to(host_device, host.dtype),
            gravity_body.to(host_device, host.dtype),
            last_action.to(host_device, host.dtype),
        )

        dir_body = speed_term.goal_dir_b.to(host_device, host.dtype)
        speed_scalar = speed_term.goal_speed[:, 0].to(host_device, host.dtype)
        yaw_attr = getattr(speed_term, "goal_yaw_rate", None)
        if yaw_attr is not None:
            yaw_cmd = yaw_attr[:, 0].to(host_device, host.dtype)
        else:
            yaw_cmd = torch.zeros_like(speed_scalar)

        sg_real, mask = host.build_velocity_goal(dir_body, speed_scalar, yaw_cmd)
        sg_real_masked = (sg_real * mask).to(host_device, host.dtype)

        base_inputs = {
            base_input_names[0]: _ensure_numpy_2d(sp_real.cpu().numpy()),
            base_input_names[1]: _ensure_numpy_2d(sg_real_masked.cpu().numpy()),
        }
        base_results = base_session.run(None, base_inputs)
        base_outputs = {base_output_names[idx]: base_results[idx] for idx in range(len(base_results))}

        base_action_np = base_outputs.get("base_action", base_results[0])
        mu_p_np = base_outputs.get("mu_p", base_results[-1])

        base_action = torch.from_numpy(np.asarray(base_action_np, dtype=np.float32)).to(host_device)
        mu_p = torch.from_numpy(np.asarray(mu_p_np, dtype=np.float32)).to(host_device)

        base_lin_vel, base_ang_vel = _gather_base_velocities(base_env)
        base_lin_vel = base_lin_vel.to(host_device, host.dtype)
        base_ang_vel = base_ang_vel.to(host_device, host.dtype)

        components: dict[str, torch.Tensor] = {
            "base_obs": base_obs_host,
            "sp_real": sp_real,
            "sg_real_masked": sg_real_masked,
            "mu_p": mu_p,
            "base_action": base_action,
            "dir_b": dir_body,
            "speed": speed_scalar.unsqueeze(-1),
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "residual": last_residual,
            "residual_norm": last_residual_norm,
        }
        if "yaw_rate" in host.component_slices:
            components["yaw_rate"] = yaw_cmd.unsqueeze(-1)

        residual_obs = host.assemble_observation(components)
        residual_inputs = {
            residual_input_names[0]: _ensure_numpy_2d(residual_obs.cpu().numpy()),
        }
        residual_results = residual_session.run(None, residual_inputs)
        residual_outputs = {residual_output_names[idx]: residual_results[idx] for idx in range(len(residual_results))}
        residual_np = residual_outputs.get(residual_output_names[0], residual_results[0])

        residual = torch.from_numpy(np.asarray(residual_np, dtype=np.float32)).to(host_device, host.dtype)
        residual_clamped = residual.clamp(-residual_clip, residual_clip)

        if warmup_grace > 0:
            enable_mask = (speed_term.steps_since_reset.to(host_device) > warmup_grace).unsqueeze(-1)
            residual_clamped = torch.where(enable_mask, residual_clamped, torch.zeros_like(residual_clamped))

        residual_for_obs = torch.tanh(residual_clamped)
        residual_norm_obs = residual_for_obs.pow(2).mean(dim=-1, keepdim=True).sqrt()

        blended = host.combine_actions(base_action, residual_clamped)
        blended = blended.clamp(-1.0, 1.0)
        action_env = blended.to(env_device)

        obs_next, reward, done, info = vec_env.step(action_env)
        base_obs_host = _extract_policy_obs(obs_next).detach().to(host_device, host.dtype)

        reward_tensor = torch.as_tensor(reward, device=host_device, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, device=host_device, dtype=torch.bool)

        per_env_returns += reward_tensor
        per_env_lengths += 1

        if done_tensor.any():
            finished_ids = torch.nonzero(done_tensor, as_tuple=False).flatten()
            episode_returns.extend(per_env_returns[finished_ids].cpu().tolist())
            episode_lengths.extend(per_env_lengths[finished_ids].cpu().tolist())
            per_env_returns[finished_ids] = 0.0
            per_env_lengths[finished_ids] = 0.0

            joint_pos, joint_vel, root_ang_vel, gravity_body, last_action = _gather_robot_state(base_env)
            host.reset_history(
                joint_pos.to(host_device, host.dtype),
                joint_vel.to(host_device, host.dtype),
                root_ang_vel.to(host_device, host.dtype),
                gravity_body.to(host_device, host.dtype),
                last_action.to(host_device, host.dtype),
                env_ids=finished_ids.to(host_device),
            )
            last_residual[finished_ids] = 0.0
            last_residual_norm[finished_ids] = 0.0

        last_residual = residual_for_obs.detach()
        last_residual_norm = residual_norm_obs.detach()

        total_steps += num_envs
        sim_ticks += 1

        if args.log_interval > 0 and sim_ticks % args.log_interval == 0:
            mean_reward = float(reward_tensor.mean().cpu().item())
            base_act_norm = float(base_action.norm(dim=-1).mean().item())
            residual_norm = float(residual_norm_obs.mean().item())
            blended_norm = float(blended.norm(dim=-1).mean().item())
            print(
                f"[Eval-ONNX] ticks={sim_ticks:06d} steps={total_steps:08d} reward_mean={mean_reward:.4f} "
                f"episodes={len(episode_returns)} | base_norm={base_act_norm:.3f} res_norm={residual_norm:.3f} blend_norm={blended_norm:.3f}",
                flush=True,
            )

    elapsed = max(1e-6, time.time() - start_time)
    sim_fps = total_steps / elapsed

    def _safe_mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    print("\n========== ONNX Evaluation Summary ==========")
    print(f"Total steps     : {total_steps}")
    print(f"Elapsed (s)     : {elapsed:.2f}  |  Sim FPS: {sim_fps:.1f}")
    print(f"Episodes closed : {len(episode_returns)}")
    print(f"Return mean     : {_safe_mean(episode_returns):.4f}")
    print(f"Episode length  : {_safe_mean(episode_lengths):.2f}")
    print("=============================================\n", flush=True)

    simulation_app.close()


def main() -> int:
    parser = _add_parser()
    args = parser.parse_args()
    try:
        _run_eval(args)
    except KeyboardInterrupt:
        print("\n[Eval-ONNX] Interrupted by user.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
