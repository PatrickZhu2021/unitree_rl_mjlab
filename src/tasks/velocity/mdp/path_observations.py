from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.path_utils import (
    DEFAULT_ZIGZAG_WAYPOINTS,
    project_points_to_path,
    root_yaw_from_quat_wxyz,
    sample_lookahead_path_points,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def zigzag_path_prior(
    env: "ManagerBasedRlEnv",
    bridge_half_width: float,
    turn_dist_scale: float = 2.0,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Privileged path-local prior for zigzag bridge navigation.

    Output:
      [
        lateral_error / bridge_half_width,
        heading_error / pi,
        left_edge_dist / bridge_half_width,
        right_edge_dist / bridge_half_width,
        progress,
        distance_to_next_turn / turn_dist_scale,
        next_turn_angle / pi,
      ]
    """
    asset: Entity = env.scene[asset_cfg.name]

    bridge_half_width = max(float(bridge_half_width), 1e-6)
    turn_dist_scale = max(float(turn_dist_scale), 1e-6)

    base_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

    path_info = project_points_to_path(base_xy, waypoints=waypoints)

    lateral_error = path_info["lateral_error"]
    tangent_yaw = path_info["tangent_yaw"]
    progress = path_info["progress"]
    distance_to_next_turn = path_info["distance_to_next_turn"]
    next_turn_angle = path_info["next_turn_angle"]

    base_yaw = root_yaw_from_quat_wxyz(asset.data.root_link_quat_w)
    heading_error = wrap_to_pi(base_yaw - tangent_yaw)

    left_edge_dist = bridge_half_width - lateral_error
    right_edge_dist = bridge_half_width + lateral_error

    obs = torch.stack(
        [
            lateral_error / bridge_half_width,
            heading_error / math.pi,
            left_edge_dist / bridge_half_width,
            right_edge_dist / bridge_half_width,
            progress,
            distance_to_next_turn / turn_dist_scale,
            next_turn_angle / math.pi,
        ],
        dim=-1,
    )

    return torch.clamp(obs, -5.0, 5.0)

def zigzag_lookahead_path_prior(
    env: "ManagerBasedRlEnv",
    bridge_half_width: float,
    turn_dist_scale: float = 2.0,
    lookahead_distances: Tuple[float, ...] = (0.3, 0.6, 0.9, 1.2),
    xy_scale: float = 1.0,
    z_scale: float = 0.3,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Privileged path-local prior with egocentric lookahead trajectory.

    Output:
      base prior:
      [
        lateral_error / bridge_half_width,
        heading_error / pi,
        left_edge_dist / bridge_half_width,
        right_edge_dist / bridge_half_width,
        progress,
        distance_to_next_turn / turn_dist_scale,
        next_turn_angle / pi,
      ]

      lookahead prior for each distance d:
      [
        dx_body / xy_scale,
        dy_body / xy_scale,
        dz / z_scale,
        heading_error_to_lookahead_tangent / pi,
      ]
    """
    asset: Entity = env.scene[asset_cfg.name]

    bridge_half_width = max(float(bridge_half_width), 1e-6)
    turn_dist_scale = max(float(turn_dist_scale), 1e-6)
    xy_scale = max(float(xy_scale), 1e-6)
    z_scale = max(float(z_scale), 1e-6)

    # Base pose in env-local frame.
    base_pos_w = asset.data.root_link_pos_w
    base_xy = base_pos_w[:, :2] - env.scene.env_origins[:, :2]
    base_z = base_pos_w[:, 2] - env.scene.env_origins[:, 2]

    base_yaw = root_yaw_from_quat_wxyz(asset.data.root_link_quat_w)

    path_info = project_points_to_path(base_xy, waypoints=waypoints)

    lateral_error = path_info["lateral_error"]
    tangent_yaw = path_info["tangent_yaw"]
    progress = path_info["progress"]
    path_s = path_info["path_s"]
    distance_to_next_turn = path_info["distance_to_next_turn"]
    next_turn_angle = path_info["next_turn_angle"]

    heading_error = wrap_to_pi(base_yaw - tangent_yaw)

    left_edge_dist = bridge_half_width - lateral_error
    right_edge_dist = bridge_half_width + lateral_error

    base_prior = torch.stack(
        [
            lateral_error / bridge_half_width,
            heading_error / math.pi,
            left_edge_dist / bridge_half_width,
            right_edge_dist / bridge_half_width,
            progress,
            distance_to_next_turn / turn_dist_scale,
            next_turn_angle / math.pi,
        ],
        dim=-1,
    )

    # Lookahead points along path_s.
    sampled = sample_lookahead_path_points(
        current_path_s=path_s,
        lookahead_distances=lookahead_distances,
        waypoints=waypoints,
    )

    # [N, K, 2]
    target_xy = sampled["point_xy"]
    target_z = sampled["point_z"]
    target_tangent_yaw = sampled["tangent_yaw"]

    # Convert lookahead target position into body frame.
    delta_xy = target_xy - base_xy[:, None, :]

    cos_yaw = torch.cos(base_yaw)
    sin_yaw = torch.sin(base_yaw)

    dx_world = delta_xy[:, :, 0]
    dy_world = delta_xy[:, :, 1]

    # Rotate world/env-local delta by -base_yaw.
    dx_body = cos_yaw[:, None] * dx_world + sin_yaw[:, None] * dy_world
    dy_body = -sin_yaw[:, None] * dx_world + cos_yaw[:, None] * dy_world

    dz = target_z - base_z[:, None]

    lookahead_heading_error = wrap_to_pi(target_tangent_yaw - base_yaw[:, None])

    lookahead_prior = torch.stack(
        [
            dx_body / xy_scale,
            dy_body / xy_scale,
            dz / z_scale,
            lookahead_heading_error / math.pi,
        ],
        dim=-1,
    )

    lookahead_prior = lookahead_prior.reshape(base_xy.shape[0], -1)

    obs = torch.cat([base_prior, lookahead_prior], dim=-1)

    return torch.clamp(obs, -5.0, 5.0)