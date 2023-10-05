import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import numpy as np


class Logger(object):
    def __init__(self, stdout, filename="log.txt"):
        self.terminal = stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()  # 缓冲区的内容及时更新到log文件中

    def flush(self):
        pass

    def close(self):
        self.log.close()


class PixelObsWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, PixelObservationWrapper(env))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.observation_space["pixels"].shape[0],
                self.observation_space["pixels"].shape[1],
                self.observation_space["pixels"].shape[2],
            ),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return obs["pixels"], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs["pixels"], reward, terminated, truncated, info


class StackWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(9, 84, 84),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.transpose(obs, (0, 3, 1, 2)).reshape(9, 84, 84), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.transpose(obs, (0, 3, 1, 2)).reshape(9, 84, 84), reward, terminated, truncated, info
