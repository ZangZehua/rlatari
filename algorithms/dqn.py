import os.path
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from .utils import layer_init, linear_schedule


class CNNQNetwork(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, 4, 0)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, 4, 2, 0)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, 3, 1, 0)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
        )
        self.q = nn.Sequential(
            layer_init(nn.Linear(512, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, action_shape)),
        )

    def forward(self, x):
        return self.q(self.backbone(x / 255.0))


class Policy:
    def __init__(self, args, action_space, device):
        self.args = args
        action_shape = action_space.n
        self.device = device
        self.q_network = CNNQNetwork(action_shape).to(self.device)
        self.target_network = CNNQNetwork(action_shape).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.loss_func = nn.MSELoss().to(self.device)

    def select_action(self, obs):
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def learn(self, data):
        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.args.gamma * target_max * (1 - data.dones.flatten())
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = self.loss_func(td_target, old_val)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), old_val.mean().item()

    def update_target(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
            )

    def save_model(self, save_path):
        torch.save(self.q_network.state_dict(), save_path)

    def load_model(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path))


class Agent:
    def __init__(self, args, envs, eval_env, device):
        self.paths = args
        self.args = args.algo
        self.envs = envs
        self.eval_env = eval_env
        self.device = device
        print(self.envs.single_observation_space)
        print(self.envs.single_action_space)
        print(self.device)
        print("============================================================")

        self.writer = SummaryWriter(self.paths.save_path, flush_secs=2)
        self.policy = Policy(self.args, self.envs.single_action_space, self.device)
        self.buffer = ReplayBuffer(self.args.buffer_size,
                                   self.envs.single_observation_space, self.envs.single_action_space,
                                   self.device,
                                   n_envs=args.num_envs,
                                   optimize_memory_usage=True,
                                   handle_timeout_termination=False, )

    @torch.no_grad()
    def eval(self):
        obs, _ = self.eval_env.reset()
        while True:
            action = self.policy.select_action(obs)
            obs, _, _, _, infos = self.eval_env.step(action)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if "episode" not in info:
                        continue
                    return info['episode']['r'], info['episode']['l']

    def run(self):
        start_time = time.time()
        obs, _ = self.envs.reset()
        global_step = 0
        training = True
        while training:
            epsilon = linear_schedule(self.args.start_e, self.args.end_e,
                                      self.args.exploration_fraction * self.args.exploration_exp, global_step)
            if random.random() < epsilon:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.args.num_envs)])
            else:
                actions = self.policy.select_action(obs)
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if "episode" not in info:
                        continue
                    print(f"global_step={global_step}, "
                          f"episodic_reward={info['episode']['r']}, "
                          f"episodic_length={info['episode']['l']}, "
                          f"time_used={(time.time() - start_time)}")
                    self.writer.add_scalar("charts/episodic_reward", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    self.writer.add_scalar("charts/epsilon", epsilon, global_step)
                    if global_step >= self.args.max_steps:
                        training = False
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(truncated):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.buffer.add(obs, real_next_obs, actions, rewards, terminated, infos)
            obs = next_obs
            global_step += 1

            if global_step >= self.args.learning_starts:
                if global_step % self.args.train_freq == 0:
                    data = self.buffer.sample(self.args.batch_size)
                    loss, old_val = self.policy.learn(data)
                    if global_step % 100 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar("losses/q_values", old_val, global_step)
                        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if global_step % self.args.target_network_freq == 0:
                    self.policy.update_target()
                if self.paths.model and global_step % self.paths.model_freq == 0:
                    print(f"===> saving model:{global_step}")
                    save_path = os.path.join(self.paths.model_path, "model-" + str(global_step) + ".pth")
                    self.policy.save_model(save_path)

        print("============================================================")
        print("eval")
        eval_rewards = []
        eval_path = open(os.path.join(self.paths.save_path, "eval.csv"), "a+")
        for _ in range(self.args.eval_times):
            er, es = self.eval()
            print(er, es)
            eval_rewards.append(er[0])
            eval_path.write(str(er[0]) + "," + str(es[0]) + "\n")
            eval_path.flush()
        eval_path.close()

        return np.stack(eval_rewards).mean(), np.stack(eval_rewards).std()
