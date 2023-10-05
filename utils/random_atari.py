import numpy as np
from collections import deque

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


class Agent:
    def __init__(self, eval_env):
        self.eval_env = eval_env

    def eval(self):
        obs, _ = self.eval_env.reset()
        while True:
            action = np.array([self.eval_env.single_action_space.sample() for _ in range(1)])
            obs, _, _, _, infos = self.eval_env.step(action)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if "episode" not in info:
                        continue
                    return info['episode']['r'], info['episode']['l']

    def run(self):
        eval_rewards = deque(maxlen=10)
        eval_steps = deque(maxlen=10)
        for episode in range(10):
            eval_reward, eval_step = self.eval()
            eval_rewards.append(eval_reward)
            eval_steps.append(eval_step)
        return np.mean(eval_rewards), np.mean(eval_steps)


if __name__ == "__main__":
    def make_env(env_name, seed, resize=84):
        def thunk():
            env = gym.make(env_name)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            if len(env.observation_space.shape):  # pixel obs
                env = gym.wrappers.ResizeObservation(env, (resize, resize))
                env = gym.wrappers.GrayScaleObservation(env)
                env = gym.wrappers.FrameStack(env, 4)
            env.action_space.seed(seed)
            return env

        return thunk


    env_names = [
        "Alien-v5", "Amidar-v5", "Assault-v5", "Asterix-v5", "BankHeist-v5",
        "BattleZone-v5", "Boxing-v5", "Breakout-v5", "ChopperCommand-v5", "CrazyClimber-v5",
        "DemonAttack-v5", "Freeway-v5", "Frostbite-v5", "Gopher-v5", "Hero-v5",
        "IceHockey-v5", "Jamesbond-v5", "Kangaroo-v5", "Krull-v5", "KungFuMaster-v5",
        "MsPacman-v5", "Pong-v5", "PrivateEye-v5", "Seaquest-v5", "Skiing-v5",
        "Surround-v5", "Tennis-v5", "UpNDown-v5"
    ]
    for env in env_names:
        eval_env = gym.vector.SyncVectorEnv([make_env("ALE/" + env, 1)])
        random_agent = Agent(eval_env)
        rewards = deque(maxlen=10)
        steps = deque(maxlen=10)
        for i in range(10):
            reward, step = random_agent.run()
            rewards.append(reward)
            steps.append(step)
        print(env, "\t", np.mean(rewards), np.std(rewards), "\t", np.mean(rewards), np.std(steps))
