from dataclasses import dataclass

import mujoco
import numpy as np
from typing import Tuple

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainOutput,
  TerrainGeometry,
)
from src.tasks.velocity.mdp.path_utils import (
    DEFAULT_ZIGZAG_CONTROL_WAYPOINTS,
    DEFAULT_ZIGZAG_PATH_WAYPOINTS,
)
@dataclass
class BridgeTerrainCfg(SubTerrainCfg):
  """Bridge terrain: start platform + narrow bridge + end platform."""

  # easiest level bridge half width
  bridge_half_width: float = 0.6
  # hardest level bridge half width
  min_bridge_half_width: float = 0.2

  platform_half_width: float = 1.5
  height: float = 0.5

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del rng

    # difficulty: 0.0 -> easiest, 1.0 -> hardest
    bridge_half_width = (
      self.bridge_half_width
      - difficulty * (self.bridge_half_width - self.min_bridge_half_width)
    )

    # safety clamp
    bridge_half_width = max(bridge_half_width, self.min_bridge_half_width)

    body = spec.body("terrain")

    platform_a = body.add_geom(
      name=f"platform_a_{difficulty}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(1.0, self.platform_half_width, 0.05),
      pos=(-1.0, 0.0, self.height),
      rgba=(0.3, 0.3, 0.3, 1.0),
    )

    bridge = body.add_geom(
      name=f"bridge_{difficulty}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(2.0, bridge_half_width, 0.05),
      pos=(2.0, 0.0, self.height),
      rgba=(0.5, 0.4, 0.3, 1.0),
    )

    platform_b = body.add_geom(
      name=f"platform_b_{difficulty}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(1.5, self.platform_half_width, 0.05),
      pos=(5.5, 0.0, self.height),
      rgba=(0.3, 0.3, 0.3, 1.0),
    )

    return TerrainOutput(
      origin=np.array([-1.5, 0.0, self.height + 0.5]),
      geometries=[
        TerrainGeometry(geom=platform_a),
        TerrainGeometry(geom=bridge),
        TerrainGeometry(geom=platform_b),
      ],
    )
    
    from typing import Tuple




def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
  """MuJoCo quaternion [w, x, y, z] for yaw rotation."""
  half_yaw = 0.5 * yaw
  return (
    float(np.cos(half_yaw)),
    0.0,
    0.0,
    float(np.sin(half_yaw)),
  )


# path helper
    
def _add_visual_path_segment(
    body,
    p0: np.ndarray,
    p1: np.ndarray,
    z_lift: float = 0.06,
    half_width: float = 0.02,
    rgba=(1.0, 0.1, 0.1, 1.0),
):
    vec = p1 - p0
    dx, dy = float(vec[0]), float(vec[1])
    seg_len_xy = float(np.linalg.norm([dx, dy]))
    if seg_len_xy < 1e-6:
        return None

    yaw = float(np.arctan2(dy, dx))
    center = 0.5 * (p0 + p1)

    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(seg_len_xy * 0.5, half_width, 0.01),
        pos=(float(center[0]), float(center[1]), float(center[2] + z_lift)),
        quat=_yaw_quat(yaw),
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )
    return geom
  
def _add_visual_waypoint(
    body,
    p: np.ndarray,
    radius: float = 0.05,
    z_lift: float = 0.10,
    rgba=(0.1, 0.8, 1.0, 1.0),
):
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius,),
        pos=(float(p[0]), float(p[1]), float(p[2] + z_lift)),
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )
    return geom




@dataclass
class ZigZagRiskyBridgeTerrainCfg(SubTerrainCfg):
  """Zigzag bridge terrain with optional gaps and height steps.

  This class does not modify BridgeTerrainCfg.

  difficulty:
    0.0 -> widest / easiest
    1.0 -> narrowest / hardest
  """

  # Width curriculum
  bridge_half_width: float = 0.50
  min_bridge_half_width: float = 0.25

  # Geometry
  platform_half_width: float = 1.5
  thickness_half: float = 0.05
  piece_length: float = 0.35

  # Whether to use waypoint z values.
  # If False, all waypoints are flattened to the first waypoint height.
  enable_slope: bool = False

  # Terrain control waypoints: used to generate bridge geometry.
  waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_CONTROL_WAYPOINTS

  # Learning/reference path waypoints: used only for visualization here.
  # Observation/reward/termination also use this path from path_utils.py.
  path_waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_PATH_WAYPOINTS

  # Whether to draw reference path.
  visualize_path: bool = True

  # Optional gap difficulty
  enable_gap: bool = False
  gap_probability: float = 0.0
  min_gap_length: float = 0.08
  max_gap_length: float = 0.20

  # Optional sudden height step difficulty
  enable_height_step: bool = False
  step_probability: float = 0.0
  min_step_height: float = 0.03
  max_step_height: float = 0.10

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # Width curriculum
    bridge_half_width = (
      self.bridge_half_width
      - difficulty * (self.bridge_half_width - self.min_bridge_half_width)
    )
    bridge_half_width = max(bridge_half_width, self.min_bridge_half_width)

    # Gap curriculum
    gap_probability = self.gap_probability * difficulty
    gap_length = self.min_gap_length + difficulty * (
      self.max_gap_length - self.min_gap_length
    )

    # Step curriculum
    step_probability = self.step_probability * difficulty
    step_height = self.min_step_height + difficulty * (
      self.max_step_height - self.min_step_height
    )

    body = spec.body("terrain")
    geometries = []

    # These points generate the terrain geometry.
    points = [np.array(p, dtype=np.float64) for p in self.waypoints]

    if not self.enable_slope:
      base_z = float(points[0][2])
      for p in points:
        p[2] = base_z

    # Start platform
    start = points[0]
    platform_a = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.8, self.platform_half_width, self.thickness_half),
      pos=(float(start[0] - 0.4), float(start[1]), float(start[2])),
      rgba=(0.3, 0.3, 0.3, 1.0),
    )
    geometries.append(TerrainGeometry(geom=platform_a))

    height_offset = 0.0

    # Bridge pieces
    for seg_idx in range(len(points) - 1):
      p0 = points[seg_idx].copy()
      p1 = points[seg_idx + 1].copy()

      vec = p1 - p0
      seg_len_xy = np.linalg.norm(vec[:2])

      if seg_len_xy < 1e-6:
        continue

      yaw = float(np.arctan2(vec[1], vec[0]))

      num_pieces = max(1, int(np.ceil(seg_len_xy / self.piece_length)))
      actual_piece_len = seg_len_xy / num_pieces

      # Each major segment has at most one gap.
      gap_piece_idx = None
      gap_piece_count = 0

      if self.enable_gap and num_pieces >= 4 and rng.random() < gap_probability:
        gap_piece_idx = int(rng.integers(1, num_pieces - 1))
        gap_piece_count = max(1, int(np.ceil(gap_length / actual_piece_len)))

      for piece_idx in range(num_pieces):
        # Optional gap: skip this piece.
        if gap_piece_idx is not None:
          if gap_piece_idx <= piece_idx < gap_piece_idx + gap_piece_count:
            continue

        t0 = piece_idx / num_pieces
        t1 = (piece_idx + 1) / num_pieces
        tm = 0.5 * (t0 + t1)

        center = (1.0 - tm) * p0 + tm * p1

        # Optional sudden height step.
        if (
          self.enable_height_step
          and piece_idx > 0
          and piece_idx < num_pieces - 1
          and rng.random() < step_probability
        ):
          height_offset += float(rng.choice([-1.0, 1.0])) * step_height

        center[2] += height_offset

        bridge_piece = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(
            float(actual_piece_len * 0.5),
            float(bridge_half_width),
            float(self.thickness_half),
          ),
          pos=(float(center[0]), float(center[1]), float(center[2])),
          quat=_yaw_quat(yaw),
          rgba=(0.5, 0.4, 0.3, 1.0),
        )
        geometries.append(TerrainGeometry(geom=bridge_piece))

    # End platform
    end = points[-1]
    platform_b = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(1.0, self.platform_half_width, self.thickness_half),
      pos=(float(end[0] + 0.5), float(end[1]), float(end[2] + height_offset)),
      rgba=(0.3, 0.3, 0.3, 1.0),
    )
    geometries.append(TerrainGeometry(geom=platform_b))

    # Visualize learning/reference path, not terrain control waypoints.
    if self.visualize_path:
      path_points = [np.array(p, dtype=np.float64) for p in self.path_waypoints]

      if not self.enable_slope:
        base_z = float(points[0][2])
        for p in path_points:
          p[2] = base_z

      for i in range(len(path_points)):
        visual_waypoint = _add_visual_waypoint(
          body,
          path_points[i],
          radius=0.035,
          z_lift=0.08,
          rgba=(0.1, 0.8, 1.0, 1.0),
        )
        if visual_waypoint is not None:
          geometries.append(TerrainGeometry(geom=visual_waypoint))

      for i in range(len(path_points) - 1):
        visual_segment = _add_visual_path_segment(
          body,
          path_points[i],
          path_points[i + 1],
          z_lift=0.06,
          half_width=0.012,
          rgba=(1.0, 0.1, 0.1, 1.0),
        )
        if visual_segment is not None:
          geometries.append(TerrainGeometry(geom=visual_segment))

    origin = np.array([
      start[0] - 0.4,
      start[1],
      start[2] + 0.5,
    ])

    return TerrainOutput(
      origin=origin,
      geometries=geometries,
    )