from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.scene import InteractiveSceneCfg

from ball_balance_lab.assets.ball_balancer_cfg import BALL_BALANCER_CFG

@configclass
class BallBalanceSceneCfg(InteractiveSceneCfg):
    # robot
    robot = BALL_BALANCER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ball
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.020, # 40mm diameter ping pong ball

            collision_props=sim_utils.CollisionPropertiesCfg(),

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=6, 
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),

            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027), # 2.7g ping pong ball

            physics_material=RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.02, 0.0, 0.12)),
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot/ball_balancer_system/Table",
        spawn=None,
    )
    
@configclass
class BallBalanceDirectEnvCfg(DirectRLEnvCfg):
    # --- env ---
    decimation = 2 # apply actions every 2 sim steps
    episode_length_s = 20.0
    action_space = 3
    observation_space = 10 # [q(3), qd(3), ball_pos_xy(2), ball_vel_xy(2)]
    state_space = 0
    asymmetric_obs = False
    action_joint_gains = (20.0, 1.0, 5.0)

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            enable_stabilization=True,
            min_position_iteration_count=12,
            max_position_iteration_count=12,
            min_velocity_iteration_count=4,
            max_velocity_iteration_count=4,
            enable_external_forces_every_iteration=True,
        )
    )

    # --- robot ---
    robot_cfg: ArticulationCfg = BALL_BALANCER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    servo_joint_names = [
        "Servo_arm_1_revolute",
        "Servo_arm_2_revolute",
        "Servo_arm_3_revolute",
    ]

    # --- action scaling ---
    action_scale_rad = 0.12 # map [-1,1] -> +/- this many radians around nominal
    action_smoothing = 0.40

    # --- reset randomization ---
    reset_ball_xy_range = 0.015
    reset_ball_height = 0.14
    reset_ball_linvel = 0.03

    #target position tracking, set fixed vertices for square 
    #clockwise ordered, repeatable path
    square_targets_xy = (
        (-0.03, -0.03),
        ( 0.03, -0.03),
        ( 0.03,  0.03),
        (-0.03,  0.03),
    )
    #consider target "reached" when ball is close enough and moving slow enough
    target_radius = 0.015
    target_speed_tolerance = 0.06

    #hold on target for several steps
    target_hold_steps = 4
    target_bonus = 30.0

    #reward shaping
    pos_reward_scale = 25.0
    progress_reward_scale = 6.0
    move_to_target_reward_scale = 3.5

    near_target_radius = 0.020
    settle_reward_scale = 0.25
    settle_speed_scale = 18.0

    previous_target_linger_radius = 0.020
    linger_previous_penalty_scale = 0.75

    action_rate_penalty_scale = 0.06
    joint_vel_penalty_scale = 0.001


    # --- termination ---
    # if ball goes too far from center in table frame
    ball_fail_radius = 0.11

    scene: BallBalanceSceneCfg = BallBalanceSceneCfg(
        num_envs=2048,  # more parallel envs = stable gradient estimates
        env_spacing=1.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

