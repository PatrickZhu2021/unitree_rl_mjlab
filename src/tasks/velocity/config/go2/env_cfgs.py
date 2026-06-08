"""Unitree Go2 velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from src.tasks.velocity.bridge_terrain import BridgeTerrainCfg
import src.tasks.velocity.mdp.terminations as bridge_terminations

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import src.tasks.velocity.mdp.rewards as bridge_rewards
from src.tasks.velocity.mdp.velocity_command import BridgeVelocityCommandCfg

import src.tasks.velocity.mdp.observations as bridge_observations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
import src.tasks.velocity.mdp.curriculums as bridge_curriculums

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )


  
  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  #####################################################
  # Pat tune: encourage actual forward gait instead of standing wobble.
  #cfg.rewards["pose"].weight = 0.2
  #cfg.rewards["foot_gait"].weight = 2.0
  #cfg.rewards["foot_clearance"].weight = -0.2
  #cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["track_angular_velocity"].params["std"] = 0.5  # 0.7071067811865476
  
  if not play:
    cfg.scene.num_envs = 512
    cfg.events.pop("push_robot", None)

  # cfg.terminations["illegal_contact"].params["force_threshold"] = 10.0

  # twist_cmd = cfg.commands["twist"]
  # assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
  # twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  # twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  #####################################################
  
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg

def unitree_go2_flat_nav_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  
  goal_x = 5.0
  bridge_half_width = 0.8
    
  cfg.observations["actor"].terms["navigation_pose_2d"] = ObservationTermCfg(
      func=bridge_observations.navigation_pose_2d,
      params={
          "x_scale": goal_x,
          "y_scale": 0.8,
          "target_yaw": 0.0,
      },
   )

  cfg.observations["critic"].terms["navigation_pose_2d"] = ObservationTermCfg(
      func=bridge_observations.navigation_pose_2d,
      params={
          "x_scale": goal_x,
          "y_scale": 0.8,
          "target_yaw": 0.0,
      },
  )
  
  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  #####################################################
  # Pat tune: encourage actual forward gait instead of standing wobble.
  #cfg.rewards["pose"].weight = 0.2
  #cfg.rewards["foot_gait"].weight = 2.0
  #cfg.rewards["foot_clearance"].weight = -0.2
  #cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["track_angular_velocity"].params["std"] = 0.5  # 0.7071067811865476
  
  if not play:
    cfg.scene.num_envs = 512
    cfg.events.pop("push_robot", None)

  # cfg.terminations["illegal_contact"].params["force_threshold"] = 10.0

  # twist_cmd = cfg.commands["twist"]
  # assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
  # twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  # twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  #####################################################
  
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg

def unitree_go2_bridge_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 bridge crossing environment."""
  training_bridge_half_width=0.2
  goal_x= 6.0

  
  cfg = unitree_go2_flat_env_cfg(play=play)

  # BridgeTerrainCfg in the current cloud code uses:
  # bridge_half_width, platform_half_width, height
  # It does NOT use surface_z or thickness.
  bridge_terrain_cfg = TerrainGeneratorCfg(
    seed=0,
    curriculum=False,
    size=(5.0, 1.6),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    color_scheme="none",
    sub_terrains={
      "bridge": BridgeTerrainCfg(
        proportion=1.0,
        size=(5.0, 1.6),
        bridge_half_width=training_bridge_half_width,
        platform_half_width=1.5,
        height=0.5,
      )
    },
  )

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"

  # Important:
  # Assign the TerrainGeneratorCfg directly.
  # Do NOT write TerrainGenerator(bridge_terrain_cfg).
  cfg.scene.terrain.terrain_generator = bridge_terrain_cfg
  cfg.scene.terrain.max_init_terrain_level = 0

  # No terrain curriculum/random terrain for bridge.
  cfg.curriculum.pop("terrain_levels", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.curriculum.pop("command_vel", None)
  
  # The current BridgeTerrainCfg returns origin = [-1.5, 0.0, height + 0.5].
  # Therefore reset x around 0.5 places the robot near world x = -1.0,
  # which is the center of platform_a.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.1, 0.1),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "yaw": (-0.0, 0.0),
  }
  
  # Bridge-specific rewards.
  # These functions must exist in src/tasks/velocity/mdp/rewards.py.
  
  # cfg.rewards["lateral_velocity_l2"] = RewardTermCfg(
  #     func=bridge_rewards.lateral_velocity_l2,
  #     weight=-0.05,
  #   )

  # cfg.rewards["bridge_centerline_l2"] = RewardTermCfg(
  #   func=bridge_rewards.bridge_centerline_l2,
  #   weight=-0.1,
  #   params={"target_y": 0.0},    )

  # cfg.rewards["distance_to_goal"] = RewardTermCfg(
  #   func=bridge_rewards.distance_to_goal_reward,
  #   weight=10.0,
  #   params={"goal_x": 4.0},
  # )
    
  # cfg.rewards["bridge_success"] = RewardTermCfg(
  #   func=bridge_rewards.bridge_success,
  #   weight=15.0,
  #   params={
  #     "goal_x": 4.0,
  #   },
  # )
    
  # cfg.rewards["forward_velocity_x"] = RewardTermCfg(
  #   func=bridge_rewards.forward_velocity_x,
  #   weight=5.0,
  # )
    
  # cfg.rewards["heading_alignment_x_env_local"] = RewardTermCfg(
  #   func=bridge_rewards.heading_alignment_x_env_local,
  #   weight=-0.2,
  # )

  # cfg.rewards["stop_after_goal"] = RewardTermCfg(
  #   func=bridge_rewards.stop_after_goal,
  #   weight=-2.0,  # 可根据训练阶段加大惩罚
  #   params={"goal_x": 4.0},
  # )  
  cfg.terminations["reached_goal"] = TerminationTermCfg(
    func=bridge_terminations.reached_goal_x,
    params={"goal_x": goal_x},
  )
  cfg.rewards["is_terminated"] = RewardTermCfg(
    func=bridge_rewards.is_terminated_no_goal,
    weight=-200.0,
    params={"goal_x": goal_x},
  )
  
  cfg.rewards["bridge_success"] = RewardTermCfg(
    func=bridge_rewards.bridge_success,
    weight=20.0,
    params={"goal_x": goal_x},
  )
  cfg.rewards["track_linear_velocity"].weight = 5.0 # 3.0
  cfg.rewards["track_angular_velocity"].weight = 2.0  # 1.0
  cfg.rewards["pose"].weight = 0.5
  # cfg.rewards["foot_gait"].weight = 0.5
  
  cfg.commands["twist"] = BridgeVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3, 8),

    heading_command=False,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    heading_control_stiffness=0.0,

    ranges=BridgeVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 2.0),
      lin_vel_y=(-0.1, 0.1),
      ang_vel_z=(-0.1, 0.1),
      heading=None,
    ),

    bridge_goal_x = goal_x,
    bridge_forward_speed_range=(1.5, 2.0),

  )
  # if not play:
  #   cfg.commands["twist"] = BridgeVelocityCommandCfg(
  #     entity_name="robot",
  #     resampling_time_range=(1e9, 1e9),

  #     heading_command=False,
  #     rel_standing_envs=0.0,
  #     rel_heading_envs=0.0,
  #     heading_control_stiffness=0.0,

  #     ranges=BridgeVelocityCommandCfg.Ranges(
  #       lin_vel_x=(0.5, 0.5),
  #       lin_vel_y=(0.0, 0.0),
  #       ang_vel_z=(0.0, 0.0),
  #       heading=None,
  #     ),

  #     bridge_goal_x=4.0,
  #     bridge_forward_speed=0.5,
  #   )
    
  #   ######################################################
  # if play:
  #   twist_cmd = cfg.commands["twist"]
  #   assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  #   twist_cmd.ranges.lin_vel_x = (-0.2, 2.0)
  #   twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  #   twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
    
  return cfg


def unitree_go2_bridge_nav_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Bridge navigation only: Stage 1 navigation task."""
    # important parameters
    training_goal_x = 6.0
    training_bridge_half_width = 0.20
    min_training_bridge_half_width = 0.10
    CurriculumState= True
    
    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.episode_length_s = int(30)  # 1e9
    # Bridge terrain
    bridge_terrain_cfg = TerrainGeneratorCfg(
        seed=0,
        curriculum=CurriculumState,
        size=(9.0, 3.0),
        border_width=0.0,
        num_rows=10,
        num_cols=1,
        color_scheme="none",
        sub_terrains={
            "bridge": BridgeTerrainCfg(
                bridge_half_width=training_bridge_half_width,
                min_bridge_half_width=min_training_bridge_half_width,
                platform_half_width=1.5,
                height=0.5,
            )
        },
    )
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = bridge_terrain_cfg
    cfg.scene.terrain.max_init_terrain_level = 0
    
    if(CurriculumState == True):
      cfg.curriculum["bridge_narrow"] = CurriculumTermCfg(
      func=bridge_curriculums.bridge_narrow,
      params={
          "goal_x": training_goal_x,
          "success_ratio": 1.0,  # 0.85
          "move_down_ratio": 0.25,
      },
      )
    else:
      cfg.curriculum.pop("bridge_narrow", None)
    
    # Reset base
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.0, 0.0),
        "yaw": (-0.0, 0.0),
    }

    # Command: fixed dummy, don't influence navigation
    
    cfg.observations["actor"].terms["navigation_pose_2d"] = ObservationTermCfg(
        func=bridge_observations.navigation_pose_2d,
        params={
            "x_scale": training_goal_x,
            "y_scale": training_bridge_half_width,
            "target_yaw": 0.0,
            "use_curriculum_width": CurriculumState,
            "max_bridge_half_width": training_bridge_half_width,
            "min_bridge_half_width": min_training_bridge_half_width,
        },
    )

    cfg.observations["critic"].terms["navigation_pose_2d"] = ObservationTermCfg(
        func=bridge_observations.navigation_pose_2d,
        params={
            "x_scale": training_goal_x,
            "y_scale": training_bridge_half_width,
            "target_yaw": 0.0,
            "use_curriculum_width": CurriculumState,
            "max_bridge_half_width": training_bridge_half_width,
            "min_bridge_half_width": min_training_bridge_half_width,
        },
    )
    
    cfg.commands["twist"] = BridgeVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3, 8),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_control_stiffness=0.0,
        ranges=BridgeVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.6),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            heading=None,
        ),
        bridge_goal_x=training_goal_x,
        bridge_forward_speed_range=(0.2, 2.0),
    )

    # Rewards for pure navigation
    
    cfg.rewards["distance_to_goal"] = RewardTermCfg(
        func=bridge_rewards.distance_to_goal_reward,
        weight=0.0, # 5.0/2.5
        params={"goal_x": training_goal_x},
    )
    cfg.rewards["forward_velocity_x"] = RewardTermCfg(
      func=bridge_rewards.forward_velocity_x,
      weight=5.0, # 3.0
      params={
        "goal_x": training_goal_x,
        "max_vel": 2.0,
      },
    )
    cfg.rewards["bridge_centerline_l2"] = RewardTermCfg(
        func=bridge_rewards.bridge_centerline_l2,
        weight=-0.5,  # -0.5/-1.0/-2.0
        params={"target_y": 0.0},
    )
    cfg.rewards["bridge_edge_penalty"] = RewardTermCfg(
        func=bridge_rewards.bridge_edge_penalty,
        weight=-10.0, # -5.0/-10.0
        params={"safe_half_width": max(training_bridge_half_width-0.2, 0.0)},
    )
    cfg.rewards["heading_alignment_x_env_local"] = RewardTermCfg(
        func=bridge_rewards.heading_alignment_x_env_local,
        weight=-0.3,  # -0.3/-0.5/-1.0
    )
    cfg.rewards["lateral_velocity_l2"] = RewardTermCfg(
        func=bridge_rewards.lateral_velocity_l2,
        weight=-0.2,
    )
    
    cfg.rewards["bridge_success"] = RewardTermCfg(
        func=bridge_rewards.bridge_success,
        weight=30.0,
        params={"goal_x": training_goal_x},
    )
    cfg.rewards["track_linear_velocity"].weight = 0.0 # 1.0/0.5
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["stand_still"].weight = 0.5 # 1.0/0.5
    cfg.rewards["pose"].weight = 0.5  # 1.0/0.5
    # cfg.rewards["foot_gait"].weight = 0.0
    # cfg.rewards["foot_clearance"].weight = 0.0
    # cfg.rewards["foot_slip"].weight = 0.0
    # cfg.rewards["soft_landing"].weight = 0.0

    # Terminations
    cfg.terminations["reached_goal"] = TerminationTermCfg(
        func=bridge_terminations.reached_goal_x,
        params={"goal_x": training_goal_x},
    )
    cfg.rewards["is_terminated"] = RewardTermCfg(
        func=bridge_rewards.is_terminated_no_goal,
        weight=-200.0,
        params={"goal_x": training_goal_x},
    )


    if play:
      # Visualization only: do not reset immediately at goal.
      cfg.episode_length_s = int(1e9)
      
      # Disable curriculum manager in play.
      cfg.curriculum = {}

      # Show only the hardest/narrowest bridge in play.
      assert cfg.scene.terrain is not None
      cfg.scene.terrain.terrain_type = "generator"
      cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
          seed=0,
          curriculum=False,
          size=(9.0, 3.0),
          border_width=0.0,
          num_rows=1,
          num_cols=1,
          color_scheme="none",
          sub_terrains={
            "bridge": BridgeTerrainCfg(
                proportion=1.0,
                size=(9.0, 3.0),
                # bridge_half_width=0.1,
                # min_bridge_half_width=0.1,
                
                bridge_half_width=min_training_bridge_half_width,
                min_bridge_half_width=min_training_bridge_half_width,
                
                platform_half_width=1.5,
                height=0.5,
            )
          },
       )
      cfg.scene.terrain.max_init_terrain_level = 0
      # Do not reset when reaching goal during visualization.
      cfg.terminations.pop("reached_goal", None)
      
    return cfg