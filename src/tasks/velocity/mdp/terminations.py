from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)

def reached_goal_x(env: ManagerBasedRlEnv, goal_x: float = 4.0) -> torch.Tensor:
    """Terminate episode when robot reaches goal_x in env-local x."""
    asset = env.scene["robot"]

    x = asset.data.root_link_pos_w[:, 0]
    if hasattr(env.scene, "env_origins"):
        x = x - env.scene.env_origins[:, 0]

    return x >= goal_x