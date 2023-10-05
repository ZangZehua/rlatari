import os
import sys
import datetime
import random
import hydra
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from omegaconf import OmegaConf

from algorithms import atari_agents
from utils.utils import Logger


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


def make_save(cfg):
    time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y.%m.%d.%H.%M.%S"))
    cfg.save_path = os.path.join(cfg.base_path, time)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    if cfg.model:
        cfg.model_path = os.path.join(cfg.save_path, "models")
        if not os.path.exists(cfg.model_path):
            os.makedirs(cfg.model_path)
    return time

def run(args, stdout):
    time = make_save(args)
    args.seed = random.randint(0, 100000)
    OmegaConf.save(args, os.path.join(args.save_path, "config.yaml"))
    sys.stdout = Logger(stdout, os.path.join(args.save_path, "logs.txt"))
    print("============================================================")
    print("saving at:", args.save_path)
    # create train env and eval env
    envs = gym.vector.SyncVectorEnv(
        [make_env("ALE/" + args.env, args.seed + i, args.resize) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    eval_env = gym.vector.SyncVectorEnv(
        [make_env("ALE/" + args.env, args.seed, args.resize)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda_deterministic

    # create agent
    agent = atari_agents[args.algo.algo_name](args, envs, eval_env, device)
    avg_reward, std_reward = agent.run()
    print("============================================================")
    print("saving at:", time, "avg reward:", avg_reward, std_reward)
    print("============================================================")
    sys.stdout.close()


@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg):
    stdout = sys.stdout
    for r in range(cfg.run):
        run(cfg, stdout)


if __name__ == "__main__":
    main()
