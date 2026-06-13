from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.path_utils import (
    DEFAULT_ZIGZAG_WAYPOINTS,
    project_points_to_path,
    root_yaw_from_quat_wxyz,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _base_xy_env_local(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]


def path_centerline_l2(
    env: "ManagerBasedRlEnv",
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize signed lateral deviation from path centerline."""
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)
    return torch.square(path_info["lateral_error"])


def path_edge_penalty(
    env: "ManagerBasedRlEnv",
    bridge_half_width: float,
    edge_margin: float = 0.03,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize getting too close to or beyond bridge edge.

    safe_half_width = bridge_half_width - edge_margin.
    """
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    safe_half_width = max(bridge_half_width - edge_margin, 0.0)
    excess = torch.abs(path_info["lateral_error"]) - safe_half_width
    return torch.square(torch.clamp(excess, min=0.0))


def path_heading_alignment(
    env: "ManagerBasedRlEnv",
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw error between robot heading and path tangent."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    base_yaw = root_yaw_from_quat_wxyz(asset.data.root_link_quat_w)
    heading_error = wrap_to_pi(base_yaw - path_info["tangent_yaw"])

    return torch.square(heading_error)


def path_forward_velocity(
    env: "ManagerBasedRlEnv",
    max_vel: float = 1.0,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward velocity projected along current path tangent."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    tangent_yaw = path_info["tangent_yaw"]
    tangent = torch.stack(
        [
            torch.cos(tangent_yaw),
            torch.sin(tangent_yaw),
        ],
        dim=-1,
    )

    vel_xy_w = asset.data.root_link_lin_vel_w[:, :2]
    forward_vel = torch.sum(vel_xy_w * tangent, dim=-1)

    return torch.clamp(forward_vel, min=0.0, max=max_vel)


def path_completion(
    env: "ManagerBasedRlEnv",
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward continuous path completion ratio in [0, 1]."""
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)
    return path_info["progress"]

def path_max_completion(
    env: "ManagerBasedRlEnv",
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Max achieved path completion ratio in current episode."""
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    path_s = path_info["path_s"].detach()
    path_length = path_info["path_length"]

    if (
        not hasattr(env, "_max_path_s")
        or env._max_path_s.shape != path_s.shape
        or env._max_path_s.device != path_s.device
    ):
        env._max_path_s = path_s.clone()

    max_completion = env._max_path_s / path_length.clamp_min(1e-6)
    return torch.clamp(max_completion, 0.0, 1.0)


def path_success(
    env: "ManagerBasedRlEnv",
    success_progress: float = 0.98,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward reaching the end of the reference path."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    path_info = project_points_to_path(base_xy, waypoints)

    progress = path_info["progress"]
    return (progress >= success_progress).float()


def is_terminated_no_path_goal(
    env: "ManagerBasedRlEnv",
    success_progress: float = 0.98,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize terminations except successful path-goal termination."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    path_info = project_points_to_path(base_xy, waypoints)

    progress = path_info["progress"]
    reached_goal = progress >= success_progress

    return env.termination_manager.terminated & (~reached_goal)

def track_path_speed(
    env: "ManagerBasedRlEnv",
    desired_speed: float = 0.4,
    std: float = 0.25,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track desired speed along current path tangent."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    tangent_yaw = path_info["tangent_yaw"]
    tangent = torch.stack(
        [torch.cos(tangent_yaw), torch.sin(tangent_yaw)],
        dim=-1,
    )

    vel_xy_w = asset.data.root_link_lin_vel_w[:, :2]
    path_vel = torch.sum(vel_xy_w * tangent, dim=-1)

    error = torch.square(path_vel - desired_speed)
    return torch.exp(-error / (std * std))

def path_progress_reward(
    env: "ManagerBasedRlEnv",
    max_delta_s: float = 0.05,
    reset_jump_threshold: float = 0.30,
    progress_scale: float = 1.0,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward monotonic forward progress along the reference path.

    Uses per-env max achieved path_s instead of raw path_s difference.
    This avoids reward hacking near corners where nearest-segment projection can jump.
    """
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)
    path_s = path_info["path_s"].detach()

    if (
        not hasattr(env, "_max_path_s")
        or env._max_path_s.shape != path_s.shape
        or env._max_path_s.device != path_s.device
    ):
        env._max_path_s = path_s.clone()
        return torch.zeros_like(path_s)

    old_max_path_s = env._max_path_s

    # Detect reset / teleport: if current path_s is far behind previous max,
    # treat it as new episode and reinitialize memory for those envs.
    reset_like = path_s < (old_max_path_s - reset_jump_threshold)

    old_max_path_s = torch.where(reset_like, path_s, old_max_path_s)

    new_max_path_s = torch.maximum(old_max_path_s, path_s)
    delta_s = new_max_path_s - old_max_path_s

    delta_s = torch.clamp(delta_s, min=0.0, max=max_delta_s)

    env._max_path_s = new_max_path_s.clone()

    return delta_s * progress_scale