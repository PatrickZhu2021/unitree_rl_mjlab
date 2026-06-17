from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reached_heightmap_goal(
    env: "ManagerBasedRlEnv",
    goal_xy: tuple[float, float] = (6.4, 0.0),
    success_radius: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the base reaches a fixed env-local goal region.

    This is the goal-region replacement for reached_path_goal. It does not use
    reference path progress or zigzag waypoints; it only checks the robot's
    env-local xy distance to the requested goal.
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    goal = torch.tensor(goal_xy, device=env.device, dtype=torch.float32).unsqueeze(0)
    distance = torch.linalg.norm(base_xy - goal, dim=-1)
    return distance <= success_radius
