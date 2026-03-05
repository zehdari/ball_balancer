"""
Ball Balance playback script.
Run with:
    ./isaaclab.sh -p ~/ball_balancer/Isaac/ball_balance_lab/play.py
"""
import os
import sys

# ============================================================
# FLAGS
# ============================================================
ISAACLAB_PLAY = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rl_games/play.py")
CHECKPOINT    = None   # path to .pth, or None to use latest best
NUM_ENVS      = 1
# ============================================================

if CHECKPOINT is None:
    log_dir = os.path.expanduser("~/IsaacLab/logs/rl_games/ball_balance_rlgames")
    best = os.path.join(log_dir, "ball_balance_rlgames.pth")
    if not os.path.exists(best):
        candidates = []
        for root, _, files in os.walk(log_dir):
            for f in files:
                if f.endswith(".pth"):
                    candidates.append(os.path.join(root, f))
        CHECKPOINT = max(candidates, key=os.path.getmtime)
    else:
        CHECKPOINT = best

print(f"[INFO] Loading checkpoint: {CHECKPOINT}")

sys.argv = [ISAACLAB_PLAY, "--task", "Isaac-BallBalance-BallBalancer-Direct",
            "--num_envs", str(NUM_ENVS), "--checkpoint", CHECKPOINT]

globs = {"__name__": "__main__", "__file__": ISAACLAB_PLAY}
exec(
    open(ISAACLAB_PLAY).read().replace(
        "import isaaclab_tasks",
        "import isaaclab_tasks\nimport ball_balance_lab"
    ),
    globs
)