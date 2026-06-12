from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_bridge_env_cfg,
  unitree_go2_bridge_nav_env_cfg,
  unitree_go2_flat_nav_env_cfg,
  unitree_go2_flat_zigzag_nav_env_cfg,
  unitree_go2_zigzag_bridge_debug_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Bridge",
  env_cfg=unitree_go2_bridge_env_cfg(),
  play_env_cfg=unitree_go2_bridge_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Bridge-nav",
  env_cfg=unitree_go2_bridge_nav_env_cfg(),
  play_env_cfg=unitree_go2_bridge_nav_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-nav",
  env_cfg=unitree_go2_flat_nav_env_cfg(),
  play_env_cfg=unitree_go2_flat_nav_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-Zigzag-nav",
  env_cfg=unitree_go2_flat_zigzag_nav_env_cfg(),
  play_env_cfg=unitree_go2_flat_zigzag_nav_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-ZigZagBridge-Debug",
  env_cfg=unitree_go2_zigzag_bridge_debug_env_cfg(),
  play_env_cfg=unitree_go2_zigzag_bridge_debug_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)