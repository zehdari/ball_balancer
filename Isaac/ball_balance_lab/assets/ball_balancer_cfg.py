from importlib.resources import files

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

def pkg_asset(*parts: str) -> str:
    return str(files("ball_balance_lab").joinpath(*parts))

BALL_BALANCER_USD = pkg_asset("assets", "ball_balancer.usd")

BALL_BALANCER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BALL_BALANCER_USD,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "Servo_arm_1_revolute": -0.35,
            "Servo_arm_2_revolute": -0.35,
            "Servo_arm_3_revolute": -0.35,
        },
    ),
    actuators={
        "servos": ImplicitActuatorCfg(
            joint_names_expr=[
                "Servo_arm_1_revolute",
                "Servo_arm_2_revolute",
                "Servo_arm_3_revolute",
            ],
            stiffness=140.0,
            damping=12.0,
            effort_limit_sim=4.0,
            velocity_limit_sim=6.0,
        )
    },
)