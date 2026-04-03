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

        self._targets_xy = torch.tensor(
            self.cfg.square_targets_xy, 
            dtype=torch.float32,
            device=self.device,
        )

        self._num_targets = self._targets_xy.shape[0]

        #target progress state
        self._current_target_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._target_hold_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        #remeber if the ball is closer or farther from target compared to last.
        self._prev_dist_to_target = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._previous_target_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    # Reset
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self._servo_joint_pos_target_rad[env_ids] = \
            self._servo_joint_pos_nominal_rad[env_ids]
        self._prev_servo_actions[env_ids] = 0.0

        #reset to first target

        self._current_target_idx[env_ids] = torch.randint(
            0, self._num_targets, (len(env_ids),), device=self.device
        )
        self._previous_target_idx[env_ids] = self._current_target_idx[env_ids]
        self._target_hold_counter[env_ids] = 0
        self._prev_dist_to_target[env_ids] = 0.0

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
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step()
        self.scene.update(dt=self.cfg.sim.dt)

        #initialize previous distance to target after reset
        base_pos_w = self.robot.data.root_pos_w[env_ids]
        base_quat_w = self.robot.data.root_quat_w[env_ids]
        ball_pos_w = self.ball.data.root_pos_w[env_ids]

        ball_pos_base = quat_apply_inverse(
            base_quat_w, ball_pos_w - base_pos_w
        )
        ball_xy_base_m = ball_pos_base[:, 0:2]
        current_targets_xy = self._targets_xy[self._current_target_idx[env_ids]]
        target_error_xy_m = ball_xy_base_m - current_targets_xy
        self._prev_dist_to_target[env_ids] = torch.norm(target_error_xy_m, dim=-1)

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
    # ball_pos_xy_error_to_target_m(2),
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

        current_targets_xy = self._targets_xy[self._current_target_idx]
        target_error_xy_m = ball_pos_xy_base_m - current_targets_xy

        obs = torch.cat(
            [
                servo_joint_pos_rad,
                servo_joint_vel_rad_s,
                target_error_xy_m,
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

        current_targets_xy = self._targets_xy[self._current_target_idx]
        target_error_xy_m = ball_xy_base_m - current_targets_xy

        dist_to_target_m = torch.norm(target_error_xy_m, dim=-1)
        speed_m_s = torch.norm(ball_vxy_base_m_s, dim=-1)

        previous_targets_xy = self._targets_xy[self._previous_target_idx]
        previous_target_error_xy_m = ball_xy_base_m - previous_targets_xy
        dist_to_previous_target_m = torch.norm(previous_target_error_xy_m, dim=-1)

        #Only apply penalty for being close to previous target AFTER the target switches
        previous_is_different = (
            self._current_target_idx != self._previous_target_idx
        ).float()

        linger_previous_penalty = self.cfg.linger_previous_penalty_scale * (
            dist_to_previous_target_m < self.cfg.previous_target_linger_radius
        ).float() * (dist_to_target_m > self.cfg.target_radius).float() * previous_is_different


        #Dense position reward, get as close as possible
        pos_reward = 0.15 * torch.exp(-self.cfg.pos_reward_scale * dist_to_target_m)

        #reward reducing distance each step, using old distance
        progress_reward = self.cfg.progress_reward_scale * torch.clamp(
            self._prev_dist_to_target - dist_to_target_m,
            min=-0.002,
            max=0.002,
        )

        #reward moving faster when farther away
        dir_to_target = -target_error_xy_m / torch.clamp(
            dist_to_target_m.unsqueeze(-1), min=1e-6
        )
        velocity_toward_target = torch.sum(ball_vxy_base_m_s * dir_to_target, dim=-1)

        move_to_target_reward = self.cfg.move_to_target_reward_scale * torch.clamp(
            velocity_toward_target, min=0.0
        ) * (dist_to_target_m > self.cfg.target_radius).float()

        #slow down when close to target, higher reward for lower speed, when close.
        settle_reward = self.cfg.settle_reward_scale * torch.exp(
            -self.cfg.settle_speed_scale * speed_m_s
        ) * (dist_to_target_m <= self.cfg.near_target_radius).float()
        

        #penalize unnecesarry / random jittering.
        action_rate_penalty = self.cfg.action_rate_penalty_scale * torch.sum(
            (self.actions - self._prev_servo_actions) ** 2, dim=-1
        )

        joint_vel_penalty = self.cfg.joint_vel_penalty_scale * torch.sum(
            self.robot.data.joint_vel[:, self.servo_ids] ** 2, dim=-1
        )

        # target completion
        target_reached = (
            (dist_to_target_m <= self.cfg.target_radius)
            & (speed_m_s <= self.cfg.target_speed_tolerance)
        )

        self._target_hold_counter = torch.where(
            target_reached,
            self._target_hold_counter + 1,
            torch.zeros_like(self._target_hold_counter),
        )

        target_completed = self._target_hold_counter >= self.cfg.target_hold_steps
        completion_bonus = target_completed.float() * self.cfg.target_bonus

      

        completed_env_ids = torch.nonzero(target_completed, as_tuple=False).squeeze(-1)

        if completed_env_ids.numel() > 0:
            self._previous_target_idx[completed_env_ids] = self._current_target_idx[completed_env_ids]
            self._current_target_idx[completed_env_ids] = (
                self._current_target_idx[completed_env_ids] + 1
            ) % self._num_targets

            self._target_hold_counter[completed_env_ids] = 0

            # refresh previous-distance memory against the new target
            new_targets_xy = self._targets_xy[self._current_target_idx[completed_env_ids]]
            new_target_error_xy_m = ball_xy_base_m[completed_env_ids] - new_targets_xy
            self._prev_dist_to_target[completed_env_ids] = torch.norm(
                new_target_error_xy_m, dim=-1
            )

        total_reward = (
            pos_reward
            + progress_reward
            + move_to_target_reward
            + settle_reward
            + completion_bonus
            - action_rate_penalty
            - joint_vel_penalty
            -linger_previous_penalty
        )

        # store previous distance for next step
        non_completed_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        non_completed_mask[completed_env_ids] = False
        self._prev_dist_to_target[non_completed_mask] = dist_to_target_m[non_completed_mask]

        

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["dist_to_target_m_mean"] = dist_to_target_m.mean()
        self.extras["log"]["speed_m_s_mean"] = speed_m_s.mean()
        self.extras["log"]["pos_reward_mean"] = pos_reward.mean()
        self.extras["log"]["progress_reward_mean"] = progress_reward.mean()
        self.extras["log"]["move_to_target_reward_mean"] = move_to_target_reward.mean()
        self.extras["log"]["settle_reward_mean"] = settle_reward.mean()
        self.extras["log"]["completion_bonus_mean"] = completion_bonus.mean()
        self.extras["log"]["action_rate_penalty_mean"] = action_rate_penalty.mean()
        self.extras["log"]["joint_vel_penalty_mean"] = joint_vel_penalty.mean()
        self.extras["log"]["current_target_idx_mean"] = self._current_target_idx.float().mean()
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