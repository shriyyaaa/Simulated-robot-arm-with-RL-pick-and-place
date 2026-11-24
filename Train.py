"""Train SAC + Hindsight Experience Replay on a simulated Fetch arm.

Usage (CPU is fine, just slower):
    python train.py --env FetchReach-v4 --steps 50000          # sanity check, minutes
    python train.py --env FetchPush-v4 --steps 1000000         # harder
    python train.py --env FetchPickAndPlace-v4 --steps 3000000 # the real task
"""
import argparse
import os

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

gym.register_envs(gymnasium_robotics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="FetchPickAndPlace-v4")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    run_dir = os.path.join(args.out, f"{args.env}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(args.env, n_envs=1, seed=args.seed + 100)

    model = SAC(
        "MultiInputPolicy",  # dict observations: observation / achieved_goal / desired_goal
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=512,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=-1,  # one gradient step per env step collected
        learning_starts=2_000,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
        tensorboard_log=os.path.join(run_dir, "tb"),
        seed=args.seed,
        verbose=1,
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best"),
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=max(10_000 // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=os.path.join(run_dir, "ckpt"),
        ),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(run_dir, "final"))


if __name__ == "__main__":
    main()
