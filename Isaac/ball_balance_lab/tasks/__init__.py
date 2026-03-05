import gymnasium as gym

from ball_balance_lab.envs.ball_balance_direct_env import BallBalanceDirectEnv
from ball_balance_lab.envs.ball_balance_direct_env_cfg import BallBalanceDirectEnvCfg

gym.register(
    id="Isaac-BallBalance-BallBalancer-Direct",
    entry_point=BallBalanceDirectEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallBalanceDirectEnvCfg,
        "sb3_cfg_entry_point": "ball_balance_lab.tasks:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": "ball_balance_lab.tasks:rl_games_ppo_cfg.yaml",
    },
)