from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp.path_utils import root_yaw_from_quat_wxyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_HEIGHT_EPS = 1e-6
_EDGE_SENTINEL = 1e6
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _log_scalar(env: "ManagerBasedRlEnv", name: str, value: torch.Tensor) -> None:
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})[name] = torch.mean(value)


def _grid_axes(sensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return local x/y axes for a GridPattern raycast sensor.

    GridPatternCfg flattens rays from meshgrid(indexing="xy"), so x varies
    fastest. Sensor data can therefore be reshaped as [B, ny, nx] and then
    transposed to [B, nx, ny].
    """
    pattern = sensor.cfg.pattern
    if not hasattr(pattern, "size") or not hasattr(pattern, "resolution"):
        raise ValueError(
            "heightmap rewards require a GridPatternCfg raycast sensor; "
            f"got {type(pattern).__name__}."
        )

    size_x, size_y = pattern.size
    resolution = pattern.resolution
    nx = int(round(float(size_x) / float(resolution))) + 1
    ny = int(round(float(size_y) / float(resolution))) + 1

    if nx * ny != sensor.num_rays:
        raise ValueError(
            "terrain_scan grid shape does not match ray count: "
            f"nx={nx}, ny={ny}, nx*ny={nx * ny}, num_rays={sensor.num_rays}"
        )

    device = sensor.data.distances.device
    xs = torch.linspace(-float(size_x) / 2.0, float(size_x) / 2.0, nx, device=device)
    ys = torch.linspace(-float(size_y) / 2.0, float(size_y) / 2.0, ny, device=device)
    return xs, ys, nx, ny


def _heightmap_walkable_grid(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer walkable cells from a local height scan.

    The returned mask is derived only from the raycast sensor, not from a
    reference path. A cell is walkable when the downward ray hits a mostly
    horizontal surface within the configured height band. If
    relative_height_margin is set, the mask is further restricted to surfaces
    close to the highest visible local terrain, which rejects lower ground below
    raised bridge decks when the raycast can see it.

    Returns:
        walkable: bool tensor [num_envs, nx, ny]
        xs: local scan x coordinates [nx]
        ys: local scan y coordinates [ny]
    """
    sensor = env.scene[sensor_name]
    xs, ys, nx, ny = _grid_axes(sensor)

    hit = sensor.data.distances >= 0.0
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[..., 2]
    normal_z = sensor.data.normals_w[..., 2]

    walkable = (
        hit
        & (heights >= min_height)
        & (heights <= max_height)
        & (normal_z >= min_normal_z)
    )

    if relative_height_margin is not None:
        masked_heights = torch.where(hit, heights, torch.full_like(heights, _EDGE_SENTINEL))
        nearest_height = torch.min(masked_heights, dim=1).values
        has_hit = torch.any(hit, dim=1)
        relative_ok = heights <= (nearest_height.unsqueeze(1) + relative_height_margin)
        walkable = walkable & relative_ok & has_hit.unsqueeze(1)

    # GridPattern flatten order is [ny, nx] with x fastest.
    walkable = walkable.view(env.num_envs, ny, nx).transpose(1, 2)
    return walkable, xs, ys


def _select_lookahead_rows(
    values: torch.Tensor,
    xs: torch.Tensor,
    lookahead_x_min: float,
    lookahead_x_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_mask = (xs >= lookahead_x_min) & (xs <= lookahead_x_max)
    if not bool(torch.any(row_mask).item()):
        raise ValueError(
            "heightmap reward lookahead window selected no scan rows: "
            f"lookahead_x_min={lookahead_x_min}, lookahead_x_max={lookahead_x_max}"
        )
    return values[:, row_mask, :], xs[row_mask]


def _walkable_geometry(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "terrain_scan",
    min_valid_cols: int = 3,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
) -> dict[str, torch.Tensor]:
    """Return per-row local walkable band geometry inferred from heightmap."""
    walkable, xs, ys = _heightmap_walkable_grid(
        env,
        sensor_name=sensor_name,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )
    left_y, right_y, row_valid, valid_count = _row_edges(
        walkable, ys, min_valid_cols=min_valid_cols
    )
    center_y = 0.5 * (left_y + right_y)
    width = torch.clamp(right_y - left_y, min=0.0)
    return {
        "walkable": walkable,
        "xs": xs,
        "ys": ys,
        "left_y": left_y,
        "right_y": right_y,
        "center_y": center_y,
        "width": width,
        "row_valid": row_valid,
        "valid_count": valid_count,
    }


def _sample_rows_by_x(
    row_values: torch.Tensor,
    row_valid: torch.Tensor,
    xs: torch.Tensor,
    query_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample [B, nx] row values at per-env local x queries with linear interpolation."""
    bsz, nx = row_values.shape
    x_min = xs[0]
    x_max = xs[-1]
    dx = (xs[1] - xs[0]).clamp_min(_HEIGHT_EPS)

    q = torch.clamp(query_x, min=x_min, max=x_max)
    fidx = (q - x_min) / dx
    idx0 = torch.floor(fidx).long().clamp(0, nx - 1)
    idx1 = torch.clamp(idx0 + 1, max=nx - 1)
    frac = (fidx - idx0.float()).clamp(0.0, 1.0)

    batch = torch.arange(bsz, device=row_values.device)
    v0 = row_values[batch, idx0]
    v1 = row_values[batch, idx1]
    valid0 = row_valid[batch, idx0]
    valid1 = row_valid[batch, idx1]

    both_valid = valid0 & valid1
    either_valid = valid0 | valid1
    interp = (1.0 - frac) * v0 + frac * v1
    fallback = torch.where(valid0, v0, v1)
    values = torch.where(both_valid, interp, fallback)
    values = torch.where(either_valid, values, torch.zeros_like(values))
    return values, either_valid


def _heightmap_forward_direction(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "terrain_scan",
    lookahead_x: float = 0.40,
    min_valid_cols: int = 3,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer a local walkable direction from the heightmap.

    The first version sampled the centerline at a fixed positive x. At a sharp
    zigzag corner, the next bridge segment may be mostly to the robot's side,
    so that fixed-x row is invalid and the reward falls back to +x exactly when
    the policy needs a turning cue. This version uses the centroid of visible
    walkable cells in a short forward/side window, which yields a diagonal or
    lateral target near corners while still producing +x on straight segments.
    """
    geom = _walkable_geometry(
        env,
        sensor_name=sensor_name,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )
    walkable = geom["walkable"]
    xs = geom["xs"]
    ys = geom["ys"]
    center_y = geom["center_y"]
    row_valid = geom["row_valid"]

    device = center_y.device
    current_x = torch.zeros(env.num_envs, device=device)
    current_y, current_valid = _sample_rows_by_x(center_y, row_valid, xs, current_x)

    # Include a little area behind the base so a side branch at a corner is still
    # visible, but prefer cells with positive x when they exist.
    candidate_x_min = -0.10
    candidate_x_max = max(float(lookahead_x), 0.10)
    x_grid = xs.view(1, -1, 1)
    y_grid = ys.view(1, 1, -1)
    candidate = walkable & (x_grid >= candidate_x_min) & (x_grid <= candidate_x_max)

    forward_bias = 1.0 + torch.clamp(x_grid, min=0.0) / max(candidate_x_max, _HEIGHT_EPS)
    weights = candidate.float() * forward_bias
    weight_sum = weights.sum(dim=(1, 2))
    enough_cells = weight_sum >= float(min_valid_cols)

    target_x = (weights * x_grid).sum(dim=(1, 2)) / weight_sum.clamp_min(1.0)
    target_y = (weights * y_grid).sum(dim=(1, 2)) / weight_sum.clamp_min(1.0)

    tangent = torch.stack(
        [target_x - current_x, target_y - current_y],
        dim=-1,
    )
    norm = torch.linalg.norm(tangent, dim=-1)
    valid = current_valid & enough_cells & (norm > 0.03)
    tangent = tangent / norm.clamp_min(_HEIGHT_EPS).unsqueeze(-1)

    # If heightmap cannot infer a direction, fall back to local +x so reward is
    # well-defined but not reference-path informed.
    fallback = torch.zeros_like(tangent)
    fallback[:, 0] = 1.0
    tangent = torch.where(valid.unsqueeze(-1), tangent, fallback)
    tangent_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])

    _log_scalar(env, "Metrics/heightmap_target_x", target_x)
    _log_scalar(env, "Metrics/heightmap_target_y", target_y)
    _log_scalar(env, "Metrics/heightmap_direction_valid", valid.float())
    return tangent, tangent_yaw, valid


def _row_edges(
    local_walkable: torch.Tensor,
    ys: torch.Tensor,
    min_valid_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute left/right y edge per selected x row.

    Args:
        local_walkable: bool tensor [B, R, ny]
        ys: y coordinates [ny]

    Returns:
        left_y: smallest walkable y per row [B, R]
        right_y: largest walkable y per row [B, R]
        row_valid: rows with enough walkable cells [B, R]
        valid_count: number of walkable y cells per row [B, R]
    """
    bsz, rows, _ = local_walkable.shape
    y_values = ys.view(1, 1, -1).expand(bsz, rows, -1)

    valid_count = local_walkable.sum(dim=-1)
    row_valid = valid_count >= min_valid_cols

    large = torch.full_like(y_values, _EDGE_SENTINEL)
    small = torch.full_like(y_values, -_EDGE_SENTINEL)
    left_y = torch.min(torch.where(local_walkable, y_values, large), dim=-1).values
    right_y = torch.max(torch.where(local_walkable, y_values, small), dim=-1).values

    zeros = torch.zeros_like(left_y)
    left_y = torch.where(row_valid, left_y, zeros)
    right_y = torch.where(row_valid, right_y, zeros)
    return left_y, right_y, row_valid, valid_count


def _goal_xy_tensor(
    env: "ManagerBasedRlEnv",
    goal_xy: tuple[float, float] = (6.4, 0.0),
) -> torch.Tensor:
    return torch.tensor(goal_xy, device=env.device, dtype=torch.float32).unsqueeze(0)


def heightmap_goal_success(
    env: "ManagerBasedRlEnv",
    goal_xy: tuple[float, float] = (6.4, 0.0),
    success_radius: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward reaching a fixed env-local goal region without reference path_s."""
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    goal = _goal_xy_tensor(env, goal_xy)
    distance = torch.linalg.norm(base_xy - goal, dim=-1)
    success = distance <= success_radius
    _log_scalar(env, "Metrics/heightmap_goal_distance", distance)
    _log_scalar(env, "Metrics/heightmap_goal_success", success.float())
    return success.float()


def is_terminated_no_heightmap_goal(
    env: "ManagerBasedRlEnv",
    goal_xy: tuple[float, float] = (6.4, 0.0),
    success_radius: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize terminations except successful env-local goal termination."""
    success = heightmap_goal_success(
        env,
        goal_xy=goal_xy,
        success_radius=success_radius,
        asset_cfg=asset_cfg,
    ).bool()
    return env.termination_manager.terminated & (~success)


def heightmap_lookahead_prior(
    env: "ManagerBasedRlEnv",
    bridge_half_width: float,
    turn_dist_scale: float = 2.0,
    lookahead_distances: Tuple[float, ...] = (0.20, 0.40, 0.60, 0.80),
    xy_scale: float = 1.0,
    z_scale: float = 0.3,
    include_heading_features: bool = False,
    sensor_name: str = "terrain_scan",
    min_valid_cols: int = 3,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Heightmap-derived replacement for zigzag_lookahead_path_prior.

    This observation has the same output size pattern as the path prior, but it
    estimates local center/edges/direction from terrain_scan instead of using
    reference waypoints. The fifth base feature is visible-row ratio rather than
    global path progress, because a local heightmap alone cannot know path_s.
    """
    asset: Entity = env.scene[asset_cfg.name]
    del asset  # The scan is already base-yaw aligned; keep signature consistent.

    bridge_half_width = max(float(bridge_half_width), _HEIGHT_EPS)
    turn_dist_scale = max(float(turn_dist_scale), _HEIGHT_EPS)
    xy_scale = max(float(xy_scale), _HEIGHT_EPS)
    z_scale = max(float(z_scale), _HEIGHT_EPS)

    geom = _walkable_geometry(
        env,
        sensor_name=sensor_name,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )
    xs = geom["xs"]
    center_y_rows = geom["center_y"]
    left_y_rows = geom["left_y"]
    right_y_rows = geom["right_y"]
    row_valid = geom["row_valid"]

    device = center_y_rows.device
    bsz = env.num_envs
    zero_x = torch.zeros(bsz, device=device)
    center_y, center_valid = _sample_rows_by_x(center_y_rows, row_valid, xs, zero_x)
    left_y, left_valid = _sample_rows_by_x(left_y_rows, row_valid, xs, zero_x)
    right_y, right_valid = _sample_rows_by_x(right_y_rows, row_valid, xs, zero_x)

    tangent, tangent_yaw, tangent_valid = _heightmap_forward_direction(
        env,
        sensor_name=sensor_name,
        lookahead_x=float(lookahead_distances[0]) if lookahead_distances else 0.40,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )
    del tangent

    heading_error = -tangent_yaw
    if not include_heading_features:
        heading_error = torch.zeros_like(heading_error)

    # local y=0 is the robot center. Positive center_y means the locally visible
    # bridge center is to the robot's left in sensor coordinates.
    lateral_error = -center_y
    left_edge_dist = -left_y
    right_edge_dist = right_y
    edge_valid = center_valid & left_valid & right_valid
    left_edge_dist = torch.where(edge_valid, left_edge_dist, torch.zeros_like(left_edge_dist))
    right_edge_dist = torch.where(edge_valid, right_edge_dist, torch.zeros_like(right_edge_dist))

    visible_ratio = row_valid.float().mean(dim=1)
    visible_forward = torch.where(row_valid, xs.view(1, -1), torch.full_like(center_y_rows, -_EDGE_SENTINEL))
    visible_forward = torch.max(visible_forward, dim=1).values.clamp_min(0.0)
    center_slope = torch.atan2(center_y, torch.ones_like(center_y).clamp_min(_HEIGHT_EPS))
    if not include_heading_features:
        center_slope = torch.zeros_like(center_slope)

    base_prior = torch.stack(
        [
            lateral_error / bridge_half_width,
            heading_error / math.pi,
            left_edge_dist / bridge_half_width,
            right_edge_dist / bridge_half_width,
            visible_ratio,
            visible_forward / turn_dist_scale,
            center_slope / math.pi,
        ],
        dim=-1,
    )

    lookahead_features = []
    for distance in lookahead_distances:
        qx = torch.full((bsz,), float(distance), device=device)
        qy, qvalid = _sample_rows_by_x(center_y_rows, row_valid, xs, qx)
        qy = torch.where(qvalid, qy, torch.zeros_like(qy))

        # Estimate local tangent at this lookahead from a small forward sample.
        qx2 = torch.full((bsz,), float(distance) + 0.20, device=device)
        qy2, qvalid2 = _sample_rows_by_x(center_y_rows, row_valid, xs, qx2)
        local_heading = torch.atan2(qy2 - qy, qx2 - qx)
        local_heading = torch.where(qvalid & qvalid2, local_heading, torch.zeros_like(local_heading))
        if not include_heading_features:
            local_heading = torch.zeros_like(local_heading)

        dz = torch.zeros_like(qy)
        lookahead_features.append(
            torch.stack(
                [
                    qx / xy_scale,
                    qy / xy_scale,
                    dz / z_scale,
                    local_heading / math.pi,
                ],
                dim=-1,
            )
        )

    if lookahead_features:
        lookahead_prior = torch.stack(lookahead_features, dim=1).reshape(bsz, -1)
        obs = torch.cat([base_prior, lookahead_prior], dim=-1)
    else:
        obs = base_prior

    _log_scalar(env, "Metrics/heightmap_prior_visible_ratio", visible_ratio)
    _log_scalar(env, "Metrics/heightmap_prior_tangent_valid", tangent_valid.float())
    return torch.clamp(obs, -5.0, 5.0)


def heightmap_centerline_l2(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "terrain_scan",
    lookahead_x_min: float = 0.0,
    lookahead_x_max: float = 0.50,
    min_valid_cols: int = 3,
    min_valid_rows: int = 1,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
    missing_penalty: float = 0.0,
) -> torch.Tensor:
    """Penalize offset from the center of the locally visible walkable area.

    Unlike path_centerline_l2, this does not use waypoints. It infers the local
    center from the terrain_scan walkable mask. The robot base is at local y=0;
    if the walkable bridge center is left/right of the base, the penalty grows.
    """
    walkable, xs, ys = _heightmap_walkable_grid(
        env,
        sensor_name=sensor_name,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )
    local_walkable, _ = _select_lookahead_rows(
        walkable, xs, lookahead_x_min, lookahead_x_max
    )
    left_y, right_y, row_valid, valid_count = _row_edges(
        local_walkable, ys, min_valid_cols=min_valid_cols
    )

    row_center_y = 0.5 * (left_y + right_y)
    valid_weight = row_valid.float()
    valid_rows = valid_weight.sum(dim=1)
    enough_rows = valid_rows >= min_valid_rows

    center_y = torch.sum(row_center_y * valid_weight, dim=1) / valid_rows.clamp_min(1.0)
    penalty = torch.square(center_y)
    penalty = torch.where(
        enough_rows,
        penalty,
        torch.full_like(penalty, float(missing_penalty)),
    )

    _log_scalar(env, "Metrics/heightmap_center_y_abs", torch.abs(center_y))
    _log_scalar(env, "Metrics/heightmap_center_valid_rows", valid_rows)
    _log_scalar(env, "Metrics/heightmap_center_valid_cols", valid_count.float())
    return penalty


def heightmap_edge_penalty(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "terrain_scan",
    lookahead_x_min: float = -0.10,
    lookahead_x_max: float = 0.40,
    required_clearance: float = 0.15,
    min_valid_cols: int = 3,
    min_valid_rows: int = 1,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
    missing_penalty: float | None = None,
) -> torch.Tensor:
    """Penalize being close to heightmap-inferred walkable edges.

    The nearest left/right edge is inferred per lookahead scan row. The robot is
    considered centered at local y=0. If y=0 is outside the visible walkable
    band, the signed clearance becomes negative and the penalty increases.
    """
    if missing_penalty is None:
        missing_penalty = required_clearance * required_clearance

    walkable, xs, ys = _heightmap_walkable_grid(
        env,
        sensor_name=sensor_name,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )
    local_walkable, _ = _select_lookahead_rows(
        walkable, xs, lookahead_x_min, lookahead_x_max
    )
    left_y, right_y, row_valid, valid_count = _row_edges(
        local_walkable, ys, min_valid_cols=min_valid_cols
    )

    # Signed clearance from local y=0 to the inferred left/right edges. For a
    # normal bridge row left_y < 0 < right_y, both terms are positive. If the
    # base is already outside the visible walkable band, one term is negative.
    left_clearance = -left_y
    right_clearance = right_y
    clearance = torch.minimum(left_clearance, right_clearance)

    row_penalty = torch.square(torch.clamp(required_clearance - clearance, min=0.0))
    valid_weight = row_valid.float()
    valid_rows = valid_weight.sum(dim=1)
    enough_rows = valid_rows >= min_valid_rows

    penalty = torch.sum(row_penalty * valid_weight, dim=1) / valid_rows.clamp_min(1.0)
    penalty = torch.where(
        enough_rows,
        penalty,
        torch.full_like(penalty, float(missing_penalty)),
    )

    valid_clearance = torch.where(row_valid, clearance, torch.full_like(clearance, _EDGE_SENTINEL))
    min_clearance = torch.min(valid_clearance, dim=1).values
    min_clearance = torch.where(enough_rows, min_clearance, torch.zeros_like(min_clearance))

    _log_scalar(env, "Metrics/heightmap_edge_min_clearance", min_clearance)
    _log_scalar(env, "Metrics/heightmap_edge_valid_rows", valid_rows)
    _log_scalar(env, "Metrics/heightmap_edge_valid_cols", valid_count.float())
    return penalty


def _heightmap_tangent_world(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    lookahead_x: float,
    min_valid_cols: int,
    min_height: float,
    max_height: float,
    min_normal_z: float,
    relative_height_margin: float | None,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return heightmap-inferred tangent in world frame and local tangent yaw."""
    asset: Entity = env.scene[asset_cfg.name]
    tangent_local, tangent_yaw_local, valid = _heightmap_forward_direction(
        env,
        sensor_name=sensor_name,
        lookahead_x=lookahead_x,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
    )

    base_yaw = root_yaw_from_quat_wxyz(asset.data.root_link_quat_w)
    cos_yaw = torch.cos(base_yaw)
    sin_yaw = torch.sin(base_yaw)

    tangent_w = torch.stack(
        [
            cos_yaw * tangent_local[:, 0] - sin_yaw * tangent_local[:, 1],
            sin_yaw * tangent_local[:, 0] + cos_yaw * tangent_local[:, 1],
        ],
        dim=-1,
    )
    tangent_w = tangent_w / torch.linalg.norm(tangent_w, dim=-1).clamp_min(_HEIGHT_EPS).unsqueeze(-1)
    normal_w = torch.stack([-tangent_w[:, 1], tangent_w[:, 0]], dim=-1)
    return tangent_w, normal_w, tangent_yaw_local, valid


def body_forward_velocity(
    env: "ManagerBasedRlEnv",
    max_vel: float = 0.6,
    min_vel: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward moving in the robot body's forward direction.

    This is a bootstrap movement reward: it does not use reference path or
    heightmap geometry, so it only says "walk forward" and lets heightmap
    center/edge rewards decide where safe forward motion is possible.
    """
    asset: Entity = env.scene[asset_cfg.name]
    forward_vel = asset.data.root_link_lin_vel_b[:, 0]
    reward = torch.clamp(forward_vel, min=min_vel, max=max_vel)
    _log_scalar(env, "Metrics/body_forward_velocity", forward_vel)
    return reward


def heightmap_forward_velocity_exp(
    env: "ManagerBasedRlEnv",
    desired_speed: float = 0.4,
    std: float = 0.25,
    backward_speed_tolerance: float = 0.02,
    lookahead_x: float = 0.40,
    heading_gate_std: float | None = None,
    centerline_gate_std: float | None = None,
    body_forward_gate: bool = False,
    body_lateral_gate_std: float | None = None,
    sensor_name: str = "terrain_scan",
    min_valid_cols: int = 3,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward speed along the heightmap-inferred local walkable direction.

    This is the heightmap-derived replacement for path_forward_velocity_exp. It
    projects base velocity onto the local bridge/terrain center direction inferred
    from terrain_scan, not onto reference waypoints.
    """
    asset: Entity = env.scene[asset_cfg.name]
    tangent_w, _, tangent_yaw_local, valid = _heightmap_tangent_world(
        env,
        sensor_name=sensor_name,
        lookahead_x=lookahead_x,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
        asset_cfg=asset_cfg,
    )

    vel_xy_w = asset.data.root_link_lin_vel_w[:, :2]
    terrain_vel = torch.sum(vel_xy_w * tangent_w, dim=-1)
    error = torch.square(terrain_vel - desired_speed)
    reward = torch.exp(-error / (std * std))
    reward = reward * (terrain_vel > backward_speed_tolerance).float()
    reward = reward * valid.float()

    if centerline_gate_std is not None:
        geom = _walkable_geometry(
            env,
            sensor_name=sensor_name,
            min_valid_cols=min_valid_cols,
            min_height=min_height,
            max_height=max_height,
            min_normal_z=min_normal_z,
            relative_height_margin=relative_height_margin,
        )
        center_y, center_valid = _sample_rows_by_x(
            geom["center_y"],
            geom["row_valid"],
            geom["xs"],
            torch.zeros(env.num_envs, device=vel_xy_w.device),
        )
        centerline_gate = torch.exp(-torch.square(center_y) / (centerline_gate_std * centerline_gate_std))
        reward = reward * torch.where(center_valid, centerline_gate, torch.zeros_like(centerline_gate))
        _log_scalar(env, "Metrics/heightmap_speed_centerline_gate", centerline_gate)

    if heading_gate_std is not None:
        heading_gate = torch.exp(-torch.square(tangent_yaw_local) / (heading_gate_std * heading_gate_std))
        reward = reward * heading_gate
        _log_scalar(env, "Metrics/heightmap_speed_heading_gate", heading_gate)

    if body_forward_gate:
        body_forward_vel = asset.data.root_link_lin_vel_b[:, 0]
        forward_body_gate = (body_forward_vel > backward_speed_tolerance).float()
        reward = reward * forward_body_gate
        _log_scalar(env, "Metrics/heightmap_speed_body_forward_gate", forward_body_gate)

    if body_lateral_gate_std is not None:
        body_lateral_vel = asset.data.root_link_lin_vel_b[:, 1]
        lateral_gate = torch.exp(
            -torch.square(body_lateral_vel) / (body_lateral_gate_std * body_lateral_gate_std)
        )
        reward = reward * lateral_gate
        _log_scalar(env, "Metrics/heightmap_speed_body_lateral_gate", lateral_gate)

    _log_scalar(env, "Metrics/heightmap_forward_velocity", terrain_vel)
    _log_scalar(env, "Metrics/heightmap_forward_valid", valid.float())
    return reward


def heightmap_lateral_velocity_l2(
    env: "ManagerBasedRlEnv",
    lookahead_x: float = 0.40,
    sensor_name: str = "terrain_scan",
    min_valid_cols: int = 3,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
    invalid_penalty: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize velocity perpendicular to the heightmap-inferred direction."""
    asset: Entity = env.scene[asset_cfg.name]
    _, normal_w, _, valid = _heightmap_tangent_world(
        env,
        sensor_name=sensor_name,
        lookahead_x=lookahead_x,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
        asset_cfg=asset_cfg,
    )

    lateral_vel = torch.sum(asset.data.root_link_lin_vel_w[:, :2] * normal_w, dim=-1)
    penalty = torch.square(lateral_vel)
    penalty = torch.where(valid, penalty, torch.full_like(penalty, float(invalid_penalty)))
    _log_scalar(env, "Metrics/heightmap_lateral_velocity_abs", torch.abs(lateral_vel))
    return penalty


def heightmap_progress_reward(
    env: "ManagerBasedRlEnv",
    max_delta_s: float = 0.05,
    reset_jump_threshold: float = 0.50,
    max_forward_jump: float = 0.20,
    progress_scale: float = 1.0,
    lookahead_x: float = 0.40,
    heading_gate_std: float | None = None,
    centerline_gate_std: float | None = None,
    sensor_name: str = "terrain_scan",
    min_valid_cols: int = 3,
    min_height: float = 0.0,
    max_height: float = 0.90,
    min_normal_z: float = 0.5,
    relative_height_margin: float | None = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward displacement along the heightmap-inferred local walkable direction.

    A local heightmap cannot provide global path_s. This reward is therefore a
    reference-free progress analogue: it stores the previous base position and
    rewards positive clipped displacement projected onto the current
    heightmap-inferred tangent.
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_xy_w = asset.data.root_link_pos_w[:, :2]
    tangent_w, _, tangent_yaw_local, valid = _heightmap_tangent_world(
        env,
        sensor_name=sensor_name,
        lookahead_x=lookahead_x,
        min_valid_cols=min_valid_cols,
        min_height=min_height,
        max_height=max_height,
        min_normal_z=min_normal_z,
        relative_height_margin=relative_height_margin,
        asset_cfg=asset_cfg,
    )

    if (
        not hasattr(env, "_reward_prev_heightmap_progress_xy")
        or env._reward_prev_heightmap_progress_xy.shape != base_xy_w.shape
        or env._reward_prev_heightmap_progress_xy.device != base_xy_w.device
    ):
        env._reward_prev_heightmap_progress_xy = base_xy_w.clone()
        return torch.zeros(env.num_envs, device=base_xy_w.device)

    prev_xy_w = env._reward_prev_heightmap_progress_xy
    delta_xy_w = base_xy_w - prev_xy_w
    env._reward_prev_heightmap_progress_xy = base_xy_w.clone()

    delta_norm = torch.linalg.norm(delta_xy_w, dim=-1)
    raw_forward = torch.sum(delta_xy_w * tangent_w, dim=-1)
    valid_forward = (
        valid
        & (delta_norm <= reset_jump_threshold)
        & (raw_forward >= 0.0)
        & (raw_forward <= max_forward_jump)
    )

    progress_reward = torch.where(valid_forward, raw_forward, torch.zeros_like(raw_forward))
    progress_reward = torch.clamp(progress_reward, min=0.0, max=max_delta_s) * progress_scale

    if centerline_gate_std is not None:
        geom = _walkable_geometry(
            env,
            sensor_name=sensor_name,
            min_valid_cols=min_valid_cols,
            min_height=min_height,
            max_height=max_height,
            min_normal_z=min_normal_z,
            relative_height_margin=relative_height_margin,
        )
        center_y, center_valid = _sample_rows_by_x(
            geom["center_y"],
            geom["row_valid"],
            geom["xs"],
            torch.zeros(env.num_envs, device=base_xy_w.device),
        )
        centerline_gate = torch.exp(-torch.square(center_y) / (centerline_gate_std * centerline_gate_std))
        progress_reward = progress_reward * torch.where(
            center_valid,
            centerline_gate,
            torch.zeros_like(centerline_gate),
        )
        _log_scalar(env, "Metrics/heightmap_progress_centerline_gate", centerline_gate)

    if heading_gate_std is not None:
        heading_gate = torch.exp(-torch.square(tangent_yaw_local) / (heading_gate_std * heading_gate_std))
        progress_reward = progress_reward * heading_gate
        _log_scalar(env, "Metrics/heightmap_progress_heading_gate", heading_gate)

    _log_scalar(env, "Metrics/heightmap_progress_valid_forward", valid_forward.float())
    _log_scalar(env, "Metrics/heightmap_progress_raw_forward", raw_forward)
    return progress_reward
