from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import math
import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.path_utils import (
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    DEFAULT_ZIGZAG_TIGHT_PATH_WAYPOINTS,
    DEFAULT_ZIGZAG_WAYPOINTS,
    project_points_to_path,
    sample_path_by_s,
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


def _base_forward_yaw(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    forward_b = torch.zeros(env.num_envs, 3, device=env.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(asset.data.root_link_quat_w, forward_b)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0])


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

    base_yaw = _base_forward_yaw(env, asset)
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
    reset_jump_threshold: float = 0.30,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Max achieved path completion ratio in current episode.

    Uses a metric-specific max state so it does not interact with progress reward
    state or depend on reward term evaluation order.
    """
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    path_s = path_info["path_s"].detach()
    path_length = path_info["path_length"]

    if (
        not hasattr(env, "_metric_max_path_s")
        or env._metric_max_path_s.shape != path_s.shape
        or env._metric_max_path_s.device != path_s.device
    ):
        env._metric_max_path_s = path_s.clone()

    reset_like = path_s < (env._metric_max_path_s - reset_jump_threshold)
    env._metric_max_path_s = torch.where(
        reset_like,
        path_s,
        torch.maximum(env._metric_max_path_s, path_s),
    ).clone()

    max_completion = env._metric_max_path_s / path_length.clamp_min(1e-6)
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
    max_forward_jump: float = 0.15,
    progress_scale: float = 1.0,
    heading_gate_std: float | None = None,
    heading_gate_lookahead: float = 0.5,
    centerline_gate_std: float | None = None,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward new monotonic forward progress along the reference path.

    Nearest-segment projection can jump around at corners, so this rewards only
    progress beyond the best previously accepted path_s. Large forward projection
    jumps are ignored and are not committed to the max state.

    If heading_gate_std is set, progress is multiplied by a lookahead heading
    gate, so moving forward only pays fully when the robot faces the path ahead.
    """
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)
    path_s = path_info["path_s"].detach()

    if (
        not hasattr(env, "_reward_max_path_s")
        or env._reward_max_path_s.shape != path_s.shape
        or env._reward_max_path_s.device != path_s.device
    ):
        env._reward_max_path_s = path_s.clone()
        return torch.zeros_like(path_s)

    old_max_s = env._reward_max_path_s

    raw_forward = path_s - old_max_s
    valid_forward = (raw_forward >= 0.0) & (raw_forward <= max_forward_jump)
    backward_projection_jump = path_s < (old_max_s - reset_jump_threshold)

    # Backward path_s jumps are usually nearest-projection artifacts near corners.
    # Do not treat them as episode resets; otherwise the policy can get rewarded
    # again by jumping backward then re-advancing along the path.
    delta_s = torch.where(valid_forward, raw_forward, torch.zeros_like(raw_forward))
    delta_s = torch.clamp(delta_s, min=0.0, max=max_delta_s)

    new_max_s = torch.where(valid_forward, torch.maximum(old_max_s, path_s), old_max_s)
    env._reward_max_path_s = new_max_s.clone()
    env.extras["log"]["Metrics/path_progress_backward_projection_jump"] = torch.mean(
        backward_projection_jump.float()
    )

    progress_reward = delta_s * progress_scale

    env.extras["log"]["Metrics/path_progress_valid_forward"] = torch.mean(
        valid_forward.float()
    )

    if centerline_gate_std is not None:
        centerline_gate = torch.exp(
            -torch.square(path_info["lateral_error"]) / (centerline_gate_std * centerline_gate_std)
        )
        progress_reward = progress_reward * centerline_gate
        env.extras["log"]["Metrics/path_progress_centerline_gate"] = torch.mean(
            centerline_gate
        )

    if heading_gate_std is None:
        return progress_reward

    target_s = path_s + heading_gate_lookahead
    target = sample_path_by_s(target_s, waypoints=waypoints)
    target_yaw = target["tangent_yaw"]

    base_yaw = _base_forward_yaw(env, asset)
    heading_error = wrap_to_pi(base_yaw - target_yaw)
    heading_gate = torch.exp(-torch.square(heading_error) / (heading_gate_std * heading_gate_std))

    env.extras["log"]["Metrics/path_progress_heading_gate"] = torch.mean(heading_gate)

    return progress_reward * heading_gate


def path_forward_velocity_exp(
    env: "ManagerBasedRlEnv",
    desired_speed: float = 0.4,
    std: float = 0.25,
    backward_speed_tolerance: float = 0.02,
    heading_gate_std: float | None = None,
    heading_gate_lookahead: float = 0.5,
    centerline_gate_std: float | None = None,
    body_forward_gate: bool = False,
    body_lateral_gate_std: float | None = None,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward velocity along path with zero reward when nearly stopped.

    Optional gates discount path-speed reward if the robot faces the wrong path,
    walks backward in its body frame, or side-steps to achieve path velocity.
    """
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
    reward = torch.exp(-error / (std * std))
    moving_forward = path_vel > backward_speed_tolerance
    reward = reward * moving_forward.float()

    if centerline_gate_std is not None:
        centerline_gate = torch.exp(
            -torch.square(path_info["lateral_error"]) / (centerline_gate_std * centerline_gate_std)
        )
        reward = reward * centerline_gate
        env.extras["log"]["Metrics/path_speed_centerline_gate"] = torch.mean(centerline_gate)

    if heading_gate_std is not None:
        target_s = path_info["path_s"] + heading_gate_lookahead
        target = sample_path_by_s(target_s, waypoints=waypoints)
        target_yaw = target["tangent_yaw"]

        base_yaw = _base_forward_yaw(env, asset)
        heading_error = wrap_to_pi(base_yaw - target_yaw)
        heading_gate = torch.exp(-torch.square(heading_error) / (heading_gate_std * heading_gate_std))
        reward = reward * heading_gate
        env.extras["log"]["Metrics/path_speed_heading_gate"] = torch.mean(heading_gate)

    if body_forward_gate:
        body_forward_vel = asset.data.root_link_lin_vel_b[:, 0]
        forward_body_gate = (body_forward_vel > backward_speed_tolerance).float()
        reward = reward * forward_body_gate
        env.extras["log"]["Metrics/path_speed_body_forward_gate"] = torch.mean(
            forward_body_gate
        )

    if body_lateral_gate_std is not None:
        body_lateral_vel = asset.data.root_link_lin_vel_b[:, 1]
        lateral_gate = torch.exp(
            -torch.square(body_lateral_vel) / (body_lateral_gate_std * body_lateral_gate_std)
        )
        reward = reward * lateral_gate
        env.extras["log"]["Metrics/path_speed_body_lateral_gate"] = torch.mean(lateral_gate)

    return reward


def path_no_progress_penalty(
    env: "ManagerBasedRlEnv",
    target_delta_s: float = 0.01,
    reset_jump_threshold: float = 0.30,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize timesteps that fail to make enough forward path progress."""
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)
    path_s = path_info["path_s"].detach()

    if (
        not hasattr(env, "_prev_path_s_no_progress")
        or env._prev_path_s_no_progress.shape != path_s.shape
        or env._prev_path_s_no_progress.device != path_s.device
    ):
        env._prev_path_s_no_progress = path_s.clone()
        return torch.zeros_like(path_s)

    old_path_s = env._prev_path_s_no_progress
    reset_like = path_s < (old_path_s - reset_jump_threshold)
    delta_s = torch.where(reset_like, torch.zeros_like(path_s), path_s - old_path_s)
    env._prev_path_s_no_progress = path_s.clone()

    return (delta_s < target_delta_s).float()


def path_stay_alive_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Per-step cost so timeout while standing still is not attractive."""
    return torch.ones(env.num_envs, device=env.device)


def path_heading_reward(
    env: "ManagerBasedRlEnv",
    std: float = 0.5,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward facing the current path tangent."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    base_yaw = _base_forward_yaw(env, asset)
    heading_error = wrap_to_pi(base_yaw - path_info["tangent_yaw"])

    return torch.exp(-torch.square(heading_error) / (std * std))


def _lookahead_heading_error(
    env: "ManagerBasedRlEnv",
    lookahead_distance: float,
    waypoints: Tuple[Tuple[float, float, float], ...],
    asset_cfg: SceneEntityCfg,
    heading_waypoints: Tuple[Tuple[float, float, float], ...] | None = None,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    heading_waypoints = heading_waypoints or waypoints
    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, heading_waypoints)

    target_s = path_info["path_s"] + lookahead_distance
    target = sample_path_by_s(target_s, waypoints=heading_waypoints)
    target_yaw = target["tangent_yaw"]

    base_yaw = _base_forward_yaw(env, asset)
    return wrap_to_pi(base_yaw - target_yaw)


def path_lookahead_heading_reward(
    env: "ManagerBasedRlEnv",
    lookahead_distance: float = 0.15,
    std: float = 0.5,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    heading_waypoints: Tuple[Tuple[float, float, float], ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward facing the path tangent a short distance ahead."""
    heading_error = _lookahead_heading_error(
        env,
        lookahead_distance=lookahead_distance,
        waypoints=waypoints,
        heading_waypoints=heading_waypoints,
        asset_cfg=asset_cfg,
    )
    env.extras["log"]["Metrics/path_lookahead_heading_error_abs"] = torch.mean(
        torch.abs(heading_error)
    )
    return torch.exp(-torch.square(heading_error) / (std * std))


def path_lookahead_heading_l2(
    env: "ManagerBasedRlEnv",
    lookahead_distance: float = 0.0,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    heading_waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw error to the heading path tangent.

    Defaults to the sharp terrain control path so heading does not anticipate the
    filleted reward path before the robot reaches a physical corner.
    """
    heading_error = _lookahead_heading_error(
        env,
        lookahead_distance=lookahead_distance,
        waypoints=waypoints,
        heading_waypoints=heading_waypoints,
        asset_cfg=asset_cfg,
    )
    return torch.square(heading_error)


def path_blended_heading_l2(
    env: "ManagerBasedRlEnv",
    turn_blend_distance: float = 0.15,
    heading_waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw error to a short blended control-path heading.

    Far from a corner, target heading is the current sharp segment tangent. Within
    turn_blend_distance of the next control waypoint, target heading smoothly
    blends toward the next segment tangent. This avoids both early fillet turning
    and hard heading jumps exactly at the corner.
    """
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, heading_waypoints)

    current_yaw = path_info["tangent_yaw"]
    next_turn_angle = path_info["next_turn_angle"]
    distance_to_turn = path_info["distance_to_next_turn"]

    blend_distance = max(float(turn_blend_distance), 1e-6)
    blend = torch.clamp((blend_distance - distance_to_turn) / blend_distance, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)

    target_yaw = wrap_to_pi(current_yaw + blend * next_turn_angle)

    base_yaw = _base_forward_yaw(env, asset)
    heading_error = wrap_to_pi(base_yaw - target_yaw)

    env.extras["log"]["Metrics/path_blended_heading_error_abs"] = torch.mean(
        torch.abs(heading_error)
    )
    env.extras["log"]["Metrics/path_blended_heading_blend"] = torch.mean(blend)

    return torch.square(heading_error)


def path_turn_heading_l2(
    env: "ManagerBasedRlEnv",
    turn_gate_distance: float = 0.45,
    turn_blend_distance: float = 0.30,
    heading_waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw error only near upcoming control-path turns.

    This keeps straight segments free from heading shaping, but gives the policy a
    soft cue to rotate once it is close enough to a corner.
    """
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, heading_waypoints)

    current_yaw = path_info["tangent_yaw"]
    next_turn_angle = path_info["next_turn_angle"]
    distance_to_turn = path_info["distance_to_next_turn"]

    gate_distance = max(float(turn_gate_distance), 1e-6)
    blend_distance = max(float(turn_blend_distance), 1e-6)

    has_turn = torch.abs(next_turn_angle) > 1e-4
    gate = torch.clamp((gate_distance - distance_to_turn) / gate_distance, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    gate = gate * has_turn.float()

    blend = torch.clamp((blend_distance - distance_to_turn) / blend_distance, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)

    target_yaw = wrap_to_pi(current_yaw + blend * next_turn_angle)
    base_yaw = _base_forward_yaw(env, asset)
    heading_error = wrap_to_pi(base_yaw - target_yaw)

    env.extras["log"]["Metrics/path_turn_heading_gate"] = torch.mean(gate)
    env.extras["log"]["Metrics/path_turn_heading_error_abs"] = torch.mean(
        torch.abs(heading_error)
    )

    return gate * torch.square(heading_error)


def path_lateral_velocity_l2(
    env: "ManagerBasedRlEnv",
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize velocity perpendicular to current path tangent."""
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = _base_xy_env_local(env, asset_cfg)
    path_info = project_points_to_path(base_xy, waypoints)

    tangent_yaw = path_info["tangent_yaw"]
    normal = torch.stack(
        [-torch.sin(tangent_yaw), torch.cos(tangent_yaw)],
        dim=-1,
    )

    vel_xy_w = asset.data.root_link_lin_vel_w[:, :2]
    lateral_vel = torch.sum(vel_xy_w * normal, dim=-1)

    return torch.square(lateral_vel)


def body_lateral_velocity_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize moving sideways in the robot body frame."""
    asset: Entity = env.scene[asset_cfg.name]
    lateral_vel = asset.data.root_link_lin_vel_b[:, 1]
    env.extras["log"]["Metrics/body_lateral_velocity_abs"] = torch.mean(torch.abs(lateral_vel))
    return torch.square(lateral_vel)