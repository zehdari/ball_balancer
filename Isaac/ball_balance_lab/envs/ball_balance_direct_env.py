import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply_inverse

from ball_balance_lab.envs.ball_balance_direct_env_cfg import BallBalanceDirectEnvCfg


class BallBalanceDirectEnv(DirectRLEnv):
    cfg: BallBalanceDirectEnvCfg

    def __init__(self, cfg: BallBalanceDirectEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.robot = self.scene["robot"]
        self.ball = self.scene["ball"]
        self.table = self.scene["table"]

        self.servo_ids = self.robot.find_joints(self.cfg.servo_joint_names)[0]

        # nominal servo joint position (servos at rest)
        self._servo_joint_pos_nominal_rad = torch.full(
            (self.num_envs, 3), -0.35, device=self.device
        )
        self._servo_joint_pos_target_rad = (
            self._servo_joint_pos_nominal_rad.clone()
        )
        self._prev_servo_actions = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

    # Reset
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self._servo_joint_pos_target_rad[env_ids] = \
            self._servo_joint_pos_nominal_rad[env_ids]
        self._prev_servo_actions[env_ids] = 0.0

        # robot joints
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self.servo_ids] = \
            self._servo_joint_pos_nominal_rad[env_ids]
        joint_vel[:, self.servo_ids] = 0.0

        self.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids
        )
        self.robot.set_joint_position_target(
            self._servo_joint_pos_target_rad,
            joint_ids=self.servo_ids,
        )

        # ball: random XY position + small random velocity
        n = len(env_ids)
        ball_state = self.ball.data.default_root_state[env_ids].clone()

        ball_state[:, 0:2] = (
            (torch.rand((n, 2), device=self.device) * 2 - 1)
            * self.cfg.reset_ball_xy_range
        )
        ball_state[:, 2] = self.cfg.reset_ball_height
        ball_state[:, 3] = 1.0
        ball_state[:, 4:7] = 0.0
        ball_state[:, 7:] = 0.0
        ball_state[:, 7:9] = (
            (torch.rand((n, 2), device=self.device) * 2 - 1)
            * self.cfg.reset_ball_linvel
        )

        if hasattr(self.scene, "env_origins"):
            ball_state[:, 0:3] += self.scene.env_origins[env_ids]

        self.ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)
        self.ball.reset(env_ids=env_ids)

        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step()
        self.scene.update(dt=self.cfg.sim.dt)

    # Stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_servo_actions[:] = (
            self.actions.clone() if hasattr(self, "actions")
            else torch.zeros_like(actions)
        )
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        # actions are per-servo offsets from nominal, scaled to radians
        servo_joint_pos_desired_rad = (
            self._servo_joint_pos_nominal_rad
            + self.actions * self.cfg.action_scale_rad
        )

        # optional smoothing
        alpha = float(getattr(self.cfg, "action_smoothing", 0.0))
        self._servo_joint_pos_target_rad = (
            (1.0 - alpha) * servo_joint_pos_desired_rad
            + alpha * self._servo_joint_pos_target_rad
        )

        self.robot.set_joint_position_target(
            self._servo_joint_pos_target_rad,
            joint_ids=self.servo_ids,
        )

    # Observations:
    # servo_joint_pos_rad(3),
    # servo_joint_vel_rad_s(3),
    # ball_pos_xy_base_m(2),
    # ball_vel_xy_base_m_s(2) -> total = 10
    def _get_observations(self) -> dict:
        servo_joint_pos_rad = self.robot.data.joint_pos[:, self.servo_ids]
        servo_joint_vel_rad_s = self.robot.data.joint_vel[:, self.servo_ids]

        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        ball_pos_w = self.ball.data.root_pos_w
        ball_vel_w = self.ball.data.root_lin_vel_w

        ball_pos_base = quat_apply_inverse(
            base_quat_w, ball_pos_w - base_pos_w
        )
        ball_vel_base = quat_apply_inverse(
            base_quat_w, ball_vel_w
        )

        ball_pos_xy_base_m = ball_pos_base[:, 0:2]
        ball_vel_xy_base_m_s = ball_vel_base[:, 0:2]

        obs = torch.cat(
            [
                servo_joint_pos_rad,
                servo_joint_vel_rad_s,
                ball_pos_xy_base_m,
                ball_vel_xy_base_m_s,
            ],
            dim=1,
        )
        return {"policy": obs}

    # Rewards (computed in base_link frame for sim-to-real consistency)
    def _get_rewards(self) -> torch.Tensor:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        ball_pos_w = self.ball.data.root_pos_w
        ball_vel_w = self.ball.data.root_lin_vel_w

        ball_pos_base = quat_apply_inverse(
            base_quat_w, ball_pos_w - base_pos_w
        )
        ball_vel_base = quat_apply_inverse(
            base_quat_w, ball_vel_w
        )

        ball_xy_base_m = ball_pos_base[:, 0:2]
        ball_vxy_base_m_s = ball_vel_base[:, 0:2]

        dist_m = torch.norm(ball_xy_base_m, dim=-1)
        speed_m_s = torch.norm(ball_vxy_base_m_s, dim=-1)

        pos_reward = 1.0 / (1.0 + dist_m)
        speed_reward = 1.0 / (1.0 + speed_m_s)

        total_reward = pos_reward * speed_reward

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["dist_m_mean"] = dist_m.mean()
        self.extras["log"]["speed_m_s_mean"] = speed_m_s.mean()
        self.extras["log"]["pos_reward_mean"] = pos_reward.mean()
        self.extras["log"]["speed_reward_mean"] = speed_reward.mean()
        self.extras["log"]["total_reward_mean"] = total_reward.mean()

        return total_reward

    # Dones (computed in base_link frame)
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        ball_pos_w = self.ball.data.root_pos_w

        ball_pos_base = quat_apply_inverse(
            base_quat_w, ball_pos_w - base_pos_w
        )

        fail = torch.norm(ball_pos_base[:, 0:2], dim=-1) \
            > self.cfg.ball_fail_radius

        time_out = self.episode_length_buf >= int(
            self.cfg.episode_length_s
            / (self.cfg.sim.dt * self.cfg.decimation)
        )

        return fail, time_out