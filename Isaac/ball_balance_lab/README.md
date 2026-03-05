# Ball Balance Lab - Training & Playback Guide

RL training for a 3-servo ball-balancing robot in Isaac Lab using PPO via `rl_games`. The robot learns to keep a ping pong ball centered on a tiltable plate.

---

## Repository Layout

```
ball_balancer/
└── Isaac/
    └── ball_balance_lab/
        ├── train.py
        ├── play.py
        ├── assets/
        │   ├── ball_balancer.usd           # Robot 3D asset
        │   └── ball_balancer_cfg.py        # Robot articulation config
        ├── envs/
        │   ├── ball_balance_direct_env.py      # Environment logic
        │   └── ball_balance_direct_env_cfg.py  # Environment parameters
        └── tasks/
            └── rl_games_ppo_cfg.yaml       # PPO / network hyperparameters
```

---

## What Each File Does

### `assets/ball_balancer_cfg.py`
Defines the physical robot as Isaac Lab understands it. It points to the USD file, sets the servos' starting joint angles (-0.35 rad, the nominal rest position), and configures the actuator model - stiffness, damping, and effort/velocity limits. If you change the physical robot (different servos, different joint names, different resting pose), this is where to update it.

### `envs/ball_balance_direct_env_cfg.py`
All the tunable numbers for the simulation and environment in one place. This includes the simulation timestep (120 Hz), how many parallel environments to run, episode length, how much to randomize the ball's starting position and velocity on reset, the action scaling (how many radians a max policy output maps to), and the failure radius (how far the ball can travel before the episode ends).

### `envs/ball_balance_direct_env.py`
The actual environment logic - what happens at each step. It handles four things:
- **Reset:** randomizes the ball's starting XY position and gives it a small random velocity each episode
- **Actions:** takes the policy's 3 outputs, scales them to radians, and sends them as position targets to the 3 servos
- **Observations:** packages 10 values for the policy - servo positions (3), servo velocities (3), ball XY position relative to the robot base (2), and ball XY velocity relative to the base (2)
- **Reward & termination:** rewards the policy for keeping the ball close to center and slow, and ends the episode early if the ball rolls past the fail radius

### `tasks/rl_games_ppo_cfg.yaml`
The PPO algorithm config passed to `rl_games`. Controls the neural network architecture (a 3-layer MLP: 128→64→32 with ELU activations), learning rate and schedule, discount factor, batch sizes, and how often to save checkpoints. Touch this if you want to experiment with network size, learning rate, or training length.

---

## Setup

### Install the package into Isaac Lab's Python environment

From `~/ball_balancer/Isaac/`, run once:

```bash
pip install -e .
```

---

## Training

### 1. Check the path in `train.py`

Open `train.py` and confirm this matches your Isaac Lab installation:

```python
ISAACLAB_TRAIN = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rl_games/train.py")
```

### 2. Configure the FLAGS (optional)

All training options are set at the top of `train.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `NUM_ENVS` | `2048` | Parallel simulation environments |
| `SEED` | `42` | Random seed |
| `MAX_ITERS` | `None` | Cap training at N epochs (overrides YAML); `None` uses YAML default (1000) |
| `CHECKPOINT` | `None` | Path to a `.pth` to resume from |
| `HEADLESS` | `False` | `True` to run without GUI |
| `DEVICE` | `None` | e.g. `"cuda:0"` - `None` uses YAML default |
| `TENSORBOARD` | `True` | Auto-launches TensorBoard at `http://localhost:6006` |

### 3. Run

```bash
./isaaclab.sh -p ~/ball_balancer/Isaac/ball_balance_lab/train.py
```

Checkpoints and logs are saved to `~/IsaacLab/logs/rl_games/ball_balance_rlgames/`.

---

## Playback

### 1. Check the path in `play.py`

```python
ISAACLAB_PLAY = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rl_games/play.py")
```

### 2. Configure the FLAGS (optional)

| Flag | Default | Description |
|------|---------|-------------|
| `CHECKPOINT` | `None` | Path to a `.pth`. If `None`, auto-loads the latest checkpoint from the log directory |
| `NUM_ENVS` | `1` | Environments to visualize |

### 3. Run

```bash
./isaaclab.sh -p ~/ball_balancer/Isaac/ball_balance_lab/play.py
```