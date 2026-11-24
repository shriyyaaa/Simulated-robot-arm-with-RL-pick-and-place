import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4", render_mode="human")
env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())

env.close()
