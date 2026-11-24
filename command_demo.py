import sys

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPickAndPlace-v4"
model_path = sys.argv[1] if len(sys.argv) > 1 else "best_model.zip"

SIDE = 0.12    # metres left/right/forward/back (training range is +/-0.15)
LIFT = 0.20    # metres above the table for "in the air"


def parse_command(text, center_xy, table_z):
    t = text.lower()
    x, y = center_xy
    z = table_z
    if "left" in t:
        y += SIDE
    if "right" in t:
        y -= SIDE
    if "forward" in t or "front" in t or "far" in t:
        x += SIDE
    if "back" in t or "near" in t or "close" in t:
        x -= SIDE
    if any(w in t for w in ("up", "lift", "high", "air", "raise", "hold")):
        z = table_z + LIFT
    if "very high" in t or "highest" in t:
        z = table_z + 0.35
    return np.array([x, y, z])


env = gym.make(ENV_ID, render_mode="human")
model = SAC.load(
    model_path,
    custom_objects={"replay_buffer_class": None, "replay_buffer_kwargs": {}, "buffer_size": 1},
)

print("Type a command, for example:")
print("  put the block on the table to the left")
print("  lift the block high in the middle")
print("  place it forward and to the right")
print("  hold it up and back")
print("Type 'quit' to exit.\n")

obs, _ = env.reset()
while True:
    text = input("command> ").strip()
    if text.lower() in ("quit", "exit", "q"):
        break
    if not text:
        continue

    obs, _ = env.reset()
    base = env.unwrapped
    table_z = getattr(base, "height_offset", 0.42)
    goal = parse_command(text, base.initial_gripper_xpos[:2], table_z)
    base.goal = goal
    obs = base._get_obs()
    print("target set to", np.round(goal, 3))

    done, info = False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print("success:", bool(info.get("is_success", 0)), "\n")

env.close()
