from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.path_utils import (
    DEFAULT_ZIGZAG_WAYPOINTS,
    project_points_to_path,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reached_path_goal(
    env: "ManagerBasedRlEnv",
    success_progress: float = 0.98,
    waypoints: Tuple[Tuple[float, float, float], ...] = DEFAULT_ZIGZAG_WAYPOINTS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    asset: Entity = env.scene[asset_cfg.name]

    base_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    path_info = project_points_to_path(base_xy, waypoints)

    return path_info["progress"] >= success_progress
