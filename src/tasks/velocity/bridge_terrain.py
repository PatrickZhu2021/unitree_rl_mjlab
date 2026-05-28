from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainOutput,
  TerrainGeometry,
)


@dataclass
class BridgeTerrainCfg(SubTerrainCfg):
  """Bridge terrain: start platform + narrow bridge + end platform."""

  bridge_half_width: float = 0.8
  platform_half_width: float = 1.5
  height: float = 0.5

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del difficulty, rng

    body = spec.body("terrain")

    platform_a = body.add_geom(
      name="platform_a",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(1.0, self.platform_half_width, 0.05),
      pos=(-1.0, 0.0, self.height),
      rgba=(0.3, 0.3, 0.3, 1.0),
    )

    bridge = body.add_geom(
      name="bridge",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(2.0, self.bridge_half_width, 0.05),
      pos=(2.0, 0.0, self.height),
      rgba=(0.5, 0.4, 0.3, 1.0),
    )

    platform_b = body.add_geom(
      name="platform_b",
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