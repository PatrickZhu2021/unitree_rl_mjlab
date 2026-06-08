from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

from .bridge_utils import get_current_bridge_half_width


def navigation_pose_2d(
  env: ManagerBasedRlEnv,
  x_scale: float = 6.0,
  y_scale: float = 0.8,
  target_yaw: float = 0.0,
  use_curriculum_width: bool = False,
  max_bridge_half_width: float = 0.8,
  min_bridge_half_width: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  pos_w = asset.data.root_link_pos_w
  x = pos_w[:, 0]
  y = pos_w[:, 1]

  if hasattr(env.scene, "env_origins"):
    x = x - env.scene.env_origins[:, 0]
    y = y - env.scene.env_origins[:, 1]

  if use_curriculum_width:
    y_scale_tensor = get_current_bridge_half_width(
      env,
      max_bridge_half_width=max_bridge_half_width,
      min_bridge_half_width=min_bridge_half_width,
    )
  else:
    y_scale_tensor = torch.full_like(y, y_scale)

  quat_w = asset.data.root_link_quat_w

  forward_b = torch.zeros(env.num_envs, 3, device=env.device)
  forward_b[:, 0] = 1.0
  forward_w = quat_apply(quat_w, forward_b)

  yaw_world = torch.atan2(forward_w[:, 1], forward_w[:, 0])

  if hasattr(env.scene, "env_origins_yaw"):
    env_yaw = env.scene.env_origins_yaw
  else:
    env_yaw = torch.zeros_like(yaw_world)

  yaw_error = yaw_world - env_yaw - target_yaw
  yaw_error = (yaw_error + torch.pi) % (2.0 * torch.pi) - torch.pi

  return torch.stack(
    [
      x / x_scale,
      y / torch.clamp(y_scale_tensor, min=1e-4),
      yaw_error / torch.pi,
    ],
    dim=-1,
  )