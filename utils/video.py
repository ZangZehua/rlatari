import gymnasium as gym

from gymnasium.wrappers import RecordVideo, PixelObservationWrapper
from utils.util import TransformEnv

env = gym.make("Acrobot-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, "video")
observation, _ = env.reset(seed=42)
for _ in range(10):
    for _ in range(1000):
        action = env.action_space.sample()  # User-defined policy function
        observation, reward, done, info, _ = env.step(action)
        print(observation)
        if done:
            observation, _ = env.reset()
env.close()
