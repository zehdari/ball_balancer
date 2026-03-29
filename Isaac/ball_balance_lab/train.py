"""
Ball Balance training script wrapper.
Edit the FLAGS section below, then run with:
    ./isaaclab.sh -p ~/ball_balancer/Isaac/ball_balance_lab/train.py
"""
import os
import sys
import subprocess

# ============================================================
# FLAGS - edit these
# ============================================================
ISAACLAB_TRAIN   = os.path.expanduser("/home/tburk/IsaacLab/scripts/reinforcement_learning/rl_games/train.py")
TASK             = "Isaac-BallBalance-BallBalancer-Direct"
NUM_ENVS         = 2048
SEED             = 42
MAX_ITERS        = None   # int to override max_epochs in yaml, else None
CHECKPOINT       = None   # path to checkpoint to resume from, else None
HEADLESS         = True  # True to run without GUI
DEVICE           = None   # e.g. "cuda:0" or "cpu", None to use yaml default
TENSORBOARD      = True   # True to launch tensorboard alongside training
TENSORBOARD_PORT = 6006
# ============================================================

LOG_DIR = os.path.expanduser("~/IsaacLab/logs/rl_games/ball_balance_rlgames")

if TENSORBOARD:
    os.makedirs(LOG_DIR, exist_ok=True)
    subprocess.Popen(
        ["tensorboard", "--logdir", LOG_DIR, "--port", str(TENSORBOARD_PORT), "--reload_interval", "5"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[INFO] TensorBoard running at http://localhost:{TENSORBOARD_PORT}")

sys.argv = [ISAACLAB_TRAIN, "--task", TASK, "--num_envs", str(NUM_ENVS), "--seed", str(SEED)]

if MAX_ITERS is not None:
    sys.argv += ["--max_iterations", str(MAX_ITERS)]
if CHECKPOINT is not None:
    sys.argv += ["--checkpoint", CHECKPOINT]
if HEADLESS:
    sys.argv += ["--headless"]
if DEVICE is not None:
    sys.argv += ["--device", DEVICE]

# Inject ball_balance_lab import into the train script and exec it
globs = {"__name__": "__main__", "__file__": ISAACLAB_TRAIN}
exec(
    open(ISAACLAB_TRAIN).read().replace(
        "import isaaclab_tasks",
        "import isaaclab_tasks\nimport ball_balance_lab"
    ),
    globs
)