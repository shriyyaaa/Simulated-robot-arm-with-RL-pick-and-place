"""Stretch goal: give the trained arm a goal in plain language.

    python command_demo.py --model runs/.../best_model.zip --command "put the block up and to the left"

The parser is a simple keyword rule. Swap `parse_command` for an intent/slot
model (e.g. a fine-tuned BERT) or an ASR -> text pipeline to extend it.
Axes assumed for Fetch: +x = away from the robot base, +y = left, +z = up.
"""
import argparse

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

STEP = 0.12  # metres per direction word (target_range is 0.15)


def parse_command(text: str) -> np.ndarray:
    t = text.lower()
    off = np.zeros(3)
    if "left" in t:
        off[1] += STEP
    if "right" in t:
        off[1] -= STEP
    if "forward" in t or "far" in t:
        off[0] += STEP
    if "back" in t or "near" in t:
        off[0] -= STEP
    if "up" in t or "lift" in t or "high" in t:
        off[2] += STEP
    return off


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--command", required=True)
    p.add_argument("--env", default="FetchPickAndPlace-v4")
    args = p.parse_args()

    env = gym.make(args.env, render_mode="human")
    model = SAC.load(args.model)
    obs, _ = env.reset()

    base = env.unwrapped
    goal = base.initial_gripper_xpos[:3] + parse_command(args.command)
    goal[2] = max(goal[2], base.initial_gripper_xpos[2] - 0.05)  # keep above table
    base.goal = goal
    obs = base._get_obs()
    print("Goal set to", np.round(goal, 3))

    done, info = False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print("Success:", bool(info.get("is_success", 0)))


if __name__ == "__main__":
    main()
