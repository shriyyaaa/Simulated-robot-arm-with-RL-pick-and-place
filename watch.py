import sys
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

env_id = sys.argv[1] if len(sys.argv) > 1 else "FetchReach-v4"
model_path = sys.argv[2] if len(sys.argv) > 2 else "best_model.zip"

env = gym.make(env_id, render_mode="human")
model = SAC.load(model_path)

for ep in range(10):
    obs, _ = env.reset()
    done, info = False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"episode {ep + 1}: success={bool(info.get('is_success', 0))}")
env.close()
