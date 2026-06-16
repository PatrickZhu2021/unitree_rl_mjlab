from __future__ import annotations

import math
from typing import Tuple

import torch


# This is the sharp polyline used to generate terrain geometry.
DEFAULT_ZIGZAG_CONTROL_WAYPOINTS = (
    (-1.0, 0.0, 0.50),
    (1.0, 0.0, 0.50),
    (1.0, 1.0, 0.50),
    (3.0, 1.0, 0.50),
    (3.0, -1.0, 0.50),
    (5.0, -1.0, 0.50),
    (5.0, 0.0, 0.50),
)


def _normalize_2d(v: tuple[float, float]) -> tuple[tuple[float, float], float] | None:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1])
    if norm < 1e-8:
        return None
    return (v[0] / norm, v[1] / norm), norm


def _left_normal_2d(v: tuple[float, float]) -> tuple[float, float]:
    return (-v[1], v[0])


def _dist_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def make_filleted_path_waypoints(
    control_waypoints: Tuple[Tuple[float, float, float], ...],
    radius: float = 0.35,
    arc_points: int = 8,
) -> Tuple[Tuple[float, float, float], ...]:
    """Convert sharp polyline control waypoints into dense filleted path waypoints.

    Terrain can still use the sharp control waypoints.
    Observation/reward/termination should use this filleted path.
    """
    if len(control_waypoints) <= 2:
        return control_waypoints

    pts = [tuple(map(float, p)) for p in control_waypoints]
    out: list[tuple[float, float, float]] = [pts[0]]

    for i in range(1, len(pts) - 1):
        a = pts[i - 1]
        b = pts[i]
        c = pts[i + 1]

        v_in_raw = (b[0] - a[0], b[1] - a[1])
        v_out_raw = (c[0] - b[0], c[1] - b[1])

        norm_in = _normalize_2d(v_in_raw)
        norm_out = _normalize_2d(v_out_raw)

        if norm_in is None or norm_out is None:
            out.append(b)
            continue

        v_in, len_in = norm_in
        v_out, len_out = norm_out

        dot = max(-1.0, min(1.0, v_in[0] * v_out[0] + v_in[1] * v_out[1]))
        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]

        # Nearly straight or near U-turn: keep original corner.
        if abs(cross) < 1e-5 or dot < -0.95:
            out.append(b)
            continue

        theta = math.acos(dot)

        # Distance from corner to tangent point.
        desired_tangent_dist = radius / max(math.tan(theta * 0.5), 1e-6)

        # Avoid consuming too much of neighboring straight segments.
        max_tangent_dist = 0.45 * min(len_in, len_out)
        tangent_dist = min(desired_tangent_dist, max_tangent_dist)

        if tangent_dist < 1e-4:
            out.append(b)
            continue

        actual_radius = tangent_dist * math.tan(theta * 0.5)

        # Tangent point on incoming segment.
        p_in = (
            b[0] - v_in[0] * tangent_dist,
            b[1] - v_in[1] * tangent_dist,
            b[2] - (b[2] - a[2]) * (tangent_dist / max(len_in, 1e-6)),
        )

        # Tangent point on outgoing segment.
        p_out = (
            b[0] + v_out[0] * tangent_dist,
            b[1] + v_out[1] * tangent_dist,
            b[2] + (c[2] - b[2]) * (tangent_dist / max(len_out, 1e-6)),
        )

        turn_sign = 1.0 if cross > 0.0 else -1.0
        left_n = _left_normal_2d(v_in)

        center = (
            p_in[0] + turn_sign * left_n[0] * actual_radius,
            p_in[1] + turn_sign * left_n[1] * actual_radius,
        )

        # Add the straight endpoint before the arc.
        if _dist_xy(out[-1], p_in) > 1e-5:
            out.append(p_in)

        a0 = math.atan2(p_in[1] - center[1], p_in[0] - center[0])
        a1 = math.atan2(p_out[1] - center[1], p_out[0] - center[0])

        # Ensure angular interpolation follows the correct turn direction.
        if turn_sign > 0.0:
            while a1 <= a0:
                a1 += 2.0 * math.pi
        else:
            while a1 >= a0:
                a1 -= 2.0 * math.pi

        for k in range(1, arc_points + 1):
            u = k / arc_points
            ang = (1.0 - u) * a0 + u * a1
            z = (1.0 - u) * p_in[2] + u * p_out[2]

            out.append(
                (
                    center[0] + actual_radius * math.cos(ang),
                    center[1] + actual_radius * math.sin(ang),
                    z,
                )
            )

    out.append(pts[-1])
    return tuple(out)


# This is the actual learning/reference path used by obs/reward/termination.
DEFAULT_ZIGZAG_PATH_WAYPOINTS = make_filleted_path_waypoints(
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    radius=0.35,
    arc_points=8,
)

# Tighter fillet for zigzag bridge training: avoids the sharp path cutting into
# bridge edges while preserving a visibly zigzag route with late turns.
DEFAULT_ZIGZAG_TIGHT_PATH_WAYPOINTS = make_filleted_path_waypoints(
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    radius=0.30,
    arc_points=10,
)

def make_env_local_waypoints(
    waypoints: Tuple[Tuple[float, float, float], ...],
    origin_xy: tuple[float, float],
) -> Tuple[Tuple[float, float, float], ...]:
    """Convert terrain-local waypoints into env-local waypoints."""
    origin_x, origin_y = origin_xy
    return tuple((x - origin_x, y - origin_y, z) for x, y, z in waypoints)


# ZigZagRiskyBridgeTerrainCfg returns origin = [start_x - 0.4, start_y, ...].
# Path rewards/observations use robot position after subtracting env.scene.env_origins,
# so their waypoints need to be shifted into the same env-local frame.
DEFAULT_ZIGZAG_ENV_ORIGIN_XY = (
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS[0][0] - 0.4,
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS[0][1],
)
DEFAULT_ZIGZAG_ENV_LOCAL_CONTROL_WAYPOINTS = make_env_local_waypoints(
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    DEFAULT_ZIGZAG_ENV_ORIGIN_XY,
)
DEFAULT_ZIGZAG_ENV_LOCAL_PATH_WAYPOINTS = make_env_local_waypoints(
    DEFAULT_ZIGZAG_PATH_WAYPOINTS,
    DEFAULT_ZIGZAG_ENV_ORIGIN_XY,
)
DEFAULT_ZIGZAG_ENV_LOCAL_TIGHT_PATH_WAYPOINTS = make_env_local_waypoints(
    DEFAULT_ZIGZAG_TIGHT_PATH_WAYPOINTS,
    DEFAULT_ZIGZAG_ENV_ORIGIN_XY,
)

# Backward-compatible name for reward/observation helpers that work in env-local coordinates.
DEFAULT_ZIGZAG_WAYPOINTS = DEFAULT_ZIGZAG_ENV_LOCAL_PATH_WAYPOINTS


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def get_path_tensors(
    device: torch.device,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return path segment tensors.

    Returns:
        p0_xy: [num_segments, 2]
        p1_xy: [num_segments, 2]
        seg_vec: [num_segments, 2]
        seg_len: [num_segments]
    """
    pts = torch.tensor(waypoints, dtype=torch.float32, device=device)
    pts_xy = pts[:, :2]

    p0_xy = pts_xy[:-1]
    p1_xy = pts_xy[1:]
    seg_vec = p1_xy - p0_xy
    seg_len = torch.norm(seg_vec, dim=-1).clamp_min(1e-6)

    return p0_xy, p1_xy, seg_vec, seg_len


def project_points_to_path(
    points_xy: torch.Tensor,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
) -> dict[str, torch.Tensor]:
    """Project env-local base xy positions to the nearest polyline segment.

    Args:
        points_xy: [num_envs, 2], env-local xy position.

    Returns:
        dictionary containing:
            seg_idx: [num_envs]
            projection_xy: [num_envs, 2]
            lateral_error: [num_envs]
            tangent_yaw: [num_envs]
            path_s: [num_envs]
            path_length: scalar tensor
            progress: [num_envs]
            distance_to_next_turn: [num_envs]
            next_turn_angle: [num_envs]
    """
    device = points_xy.device
    p0_xy, p1_xy, seg_vec, seg_len = get_path_tensors(device, waypoints)

    num_segments = p0_xy.shape[0]

    # [N, S, 2]
    rel = points_xy[:, None, :] - p0_xy[None, :, :]
    seg_vec_b = seg_vec[None, :, :]
    seg_len_sq = (seg_len ** 2)[None, :]

    # Projection ratio on each segment.
    t = torch.sum(rel * seg_vec_b, dim=-1) / seg_len_sq
    t_clamped = torch.clamp(t, 0.0, 1.0)

    # Projection point on each segment.
    proj = p0_xy[None, :, :] + t_clamped[:, :, None] * seg_vec_b

    # Distance to each projected point.
    diff = points_xy[:, None, :] - proj
    dist_sq = torch.sum(diff * diff, dim=-1)

    # Nearest segment.
    seg_idx = torch.argmin(dist_sq, dim=-1)

    batch_idx = torch.arange(points_xy.shape[0], device=device)
    nearest_proj = proj[batch_idx, seg_idx]
    nearest_t = t_clamped[batch_idx, seg_idx]

    nearest_seg_vec = seg_vec[seg_idx]
    nearest_seg_len = seg_len[seg_idx]

    # Tangent and normal.
    tangent = nearest_seg_vec / nearest_seg_len[:, None]
    normal_left = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)

    # Signed lateral error: positive means left side of path tangent.
    lateral_vec = points_xy - nearest_proj
    lateral_error = torch.sum(lateral_vec * normal_left, dim=-1)

    tangent_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])

    # Path progress s.
    cumulative_len = torch.cat(
        [
            torch.zeros(1, device=device),
            torch.cumsum(seg_len, dim=0),
        ],
        dim=0,
    )

    path_s = cumulative_len[seg_idx] + nearest_t * nearest_seg_len
    path_length = cumulative_len[-1]
    progress = torch.clamp(path_s / path_length.clamp_min(1e-6), 0.0, 1.0)

    # Distance to next turn.
    next_turn_s = cumulative_len[seg_idx + 1]
    distance_to_next_turn = torch.clamp(next_turn_s - path_s, min=0.0)

    # Next turn angle: angle from current segment tangent to next segment tangent.
    next_turn_angle = torch.zeros_like(path_s)

    has_next = seg_idx < (num_segments - 1)
    if torch.any(has_next):
        current_yaw = torch.atan2(
            seg_vec[seg_idx[has_next], 1],
            seg_vec[seg_idx[has_next], 0],
        )
        next_yaw = torch.atan2(
            seg_vec[seg_idx[has_next] + 1, 1],
            seg_vec[seg_idx[has_next] + 1, 0],
        )
        next_turn_angle[has_next] = wrap_to_pi(next_yaw - current_yaw)

    return {
        "seg_idx": seg_idx,
        "projection_xy": nearest_proj,
        "lateral_error": lateral_error,
        "tangent_yaw": tangent_yaw,
        "path_s": path_s,
        "path_length": path_length,
        "progress": progress,
        "distance_to_next_turn": distance_to_next_turn,
        "next_turn_angle": next_turn_angle,
    }


def root_yaw_from_quat_wxyz(quat_w: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def sample_path_by_s(
    path_s: torch.Tensor,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
) -> dict[str, torch.Tensor]:
    """Sample polyline path by arc-length coordinate s.

    Args:
        path_s: [num_envs] or [num_envs, num_samples], arc-length along path.

    Returns:
        dictionary containing:
            point_xyz: [..., 3]
            point_xy: [..., 2]
            point_z: [...]
            tangent_yaw: [...]
            seg_idx: [...]
            path_length: scalar tensor
    """
    device = path_s.device
    original_shape = path_s.shape
    flat_s = path_s.reshape(-1)

    pts = torch.tensor(waypoints, dtype=torch.float32, device=device)
    pts_xyz = pts[:, :3]
    pts_xy = pts[:, :2]

    p0_xyz = pts_xyz[:-1]
    p1_xyz = pts_xyz[1:]

    p0_xy = pts_xy[:-1]
    p1_xy = pts_xy[1:]

    seg_vec_xyz = p1_xyz - p0_xyz
    seg_vec_xy = p1_xy - p0_xy

    seg_len = torch.norm(seg_vec_xy, dim=-1).clamp_min(1e-6)

    cumulative_len = torch.cat(
        [
            torch.zeros(1, device=device),
            torch.cumsum(seg_len, dim=0),
        ],
        dim=0,
    )

    path_length = cumulative_len[-1]
    flat_s = torch.clamp(flat_s, 0.0, path_length)

    # Find segment index such that:
    # cumulative_len[seg_idx] <= s <= cumulative_len[seg_idx + 1]
    seg_idx = torch.searchsorted(cumulative_len[1:], flat_s, right=False)
    seg_idx = torch.clamp(seg_idx, 0, seg_len.shape[0] - 1)

    seg_start_s = cumulative_len[seg_idx]
    local_s = flat_s - seg_start_s
    t = torch.clamp(local_s / seg_len[seg_idx], 0.0, 1.0)

    point_xyz = p0_xyz[seg_idx] + t[:, None] * seg_vec_xyz[seg_idx]
    point_xy = point_xyz[:, :2]
    point_z = point_xyz[:, 2]

    tangent_xy = seg_vec_xy[seg_idx] / seg_len[seg_idx][:, None]
    tangent_yaw = torch.atan2(tangent_xy[:, 1], tangent_xy[:, 0])

    return {
        "point_xyz": point_xyz.reshape(*original_shape, 3),
        "point_xy": point_xy.reshape(*original_shape, 2),
        "point_z": point_z.reshape(*original_shape),
        "tangent_yaw": tangent_yaw.reshape(*original_shape),
        "seg_idx": seg_idx.reshape(*original_shape),
        "path_length": path_length,
    }


def sample_lookahead_path_points(
    current_path_s: torch.Tensor,
    lookahead_distances: Tuple[float, ...] = (0.3, 0.6, 0.9, 1.2),
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
) -> dict[str, torch.Tensor]:
    """Sample multiple lookahead points along the path.

    Args:
        current_path_s: [num_envs]
        lookahead_distances: distances ahead along path.

    Returns:
        dictionary containing:
            lookahead_s: [num_envs, num_samples]
            point_xyz: [num_envs, num_samples, 3]
            point_xy: [num_envs, num_samples, 2]
            point_z: [num_envs, num_samples]
            tangent_yaw: [num_envs, num_samples]
            seg_idx: [num_envs, num_samples]
            path_length: scalar tensor
    """
    device = current_path_s.device

    lookahead = torch.tensor(
        lookahead_distances,
        dtype=torch.float32,
        device=device,
    )

    lookahead_s = current_path_s[:, None] + lookahead[None, :]

    sampled = sample_path_by_s(
        lookahead_s,
        waypoints=waypoints,
    )

    sampled["lookahead_s"] = lookahead_s

    return sampled