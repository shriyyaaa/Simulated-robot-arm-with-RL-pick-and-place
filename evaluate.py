"""Measure success rate and optionally watch / record the trained policy.

    python evaluate.py --model runs/FetchPickAndPlace-v4_seed0/best/best_model.zip
    python evaluate.py --model ... --render          # live window
    python evaluate.py --model ... --video out.mp4   # headless recording
"""
import argparse

import gymnasium as gym
import gymnasium_robotics
import imageio
import numpy as np
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--env", default="FetchPickAndPlace-v4")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--render", action="store_true")
    p.add_argument("--video", default=None)
    args = p.parse_args()

    mode = "human" if args.render else ("rgb_array" if args.video else None)
    env = gym.make(args.env, render_mode=mode)
    model = SAC.load(args.model)

    frames, successes = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, info = False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if args.video and ep < 5:
                frames.append(env.render())
        successes.append(float(info.get("is_success", 0.0)))
        print(f"episode {ep + 1}: success={successes[-1]:.0f}")

    print(f"\nSuccess rate over {args.episodes} episodes: {np.mean(successes):.1%}")
    if args.video and frames:
        imageio.mimsave(args.video, frames, fps=30)
        print(f"Saved {args.video}")


if __name__ == "__main__":
    main()
