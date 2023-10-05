import os.path
import math
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from labml_helpers.schedule import Piecewise

from .utils import layer_init, ReplayBuffer


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class CNNQNetwork(nn.Module):
    def __init__(self, action_shape, v_min, v_max, n_atoms, noisy_std,
                 data_efficient=False, log_softmax=False):
        super().__init__()
        self.action_shape = action_shape
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        if data_efficient:
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 32, 5, 5, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 5, 5, 0),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                layer_init(nn.Linear(576, 512)),
            )
        else:
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
        self.value = nn.Sequential(
            NoisyLinear(512, 128, noisy_std),
            nn.ReLU(inplace=True),
            NoisyLinear(128, n_atoms, noisy_std),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(512, 128, noisy_std),
            nn.ReLU(inplace=True),
            NoisyLinear(128, action_shape * n_atoms, noisy_std)
        )

        self.softmax = nn.LogSoftmax(dim=-1) if log_softmax else nn.Softmax(dim=-1)

    def forward(self, x, action=None):
        h = self.backbone(x / 255.0)
        v = self.value(h)
        a = self.advantage(h)
        v, a = v.reshape(-1, 1, self.n_atoms), a.reshape(-1, self.action_shape, self.n_atoms)
        logit = v + a - a.mean(dim=1, keepdim=True)
        logit = self.softmax(logit)
        q_values = (logit * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, logit[torch.arange(len(x)), action]


    def reset_noise(self):
        for name, module in self.named_children():
            if module is not self.softmax:
                for layer in module:
                    if type(layer) is NoisyLinear:
                        layer.reset_noise()


class Policy:
    def __init__(self, args, action_space, device):
        self.args = args
        action_shape = action_space.n
        self.device = device
        self.q_network = CNNQNetwork(action_shape, args.v_min, args.v_max, args.n_atoms, args.noisy_std,
                                          args.data_efficient, args.log_softmax).to(self.device)
        self.target_network = CNNQNetwork(action_shape, args.v_min, args.v_max, args.n_atoms, args.noisy_std,
                                          args.data_efficient, args.log_softmax).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.loss_func = nn.MSELoss().to(self.device)

    def select_action(self, obs):
        actions, _ = self.q_network(torch.Tensor(obs).to(self.device))
        actions = actions.cpu().numpy()
        return actions

    def learn(self, data):
        # print("learn")
        obs, next_obs, actions, rewards, dones = (
            data['obs'], data['next_obs'], data['action'], data['reward'], data['done'])
        with torch.no_grad():
            next_actions, _ = self.q_network(next_obs)
            # print(next_actions.shape)
            _, next_pmfs = self.target_network(next_obs, next_actions)
            next_atoms = rewards + pow(self.args.gamma, self.args.n_steps) * self.target_network.atoms * (~dones)
            # print(next_atoms.shape)
            # projection
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.args.v_min, self.args.v_max)

            b = (tz - self.args.v_min) / delta_z
            l = b.floor().clamp(0, self.args.n_atoms - 1)
            u = b.ceil().clamp(0, self.args.n_atoms - 1)
            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            # print(delta_z.shape, tz.shape, b.shape, l.shape, u.shape, d_m_l.shape, d_m_u.shape)
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
        _, old_pmfs = self.q_network(obs, actions.squeeze(-1))
        before_mean_loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1))
        loss = before_mean_loss.mean()
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        old_val = (old_pmfs * self.q_network.atoms).sum(1)
        return loss.item(), old_val.mean().item(), before_mean_loss.detach().cpu().numpy()

    def update_target(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
            )

    def reset_noise(self):
        self.q_network.reset_noise()

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
                                   self.envs.single_observation_space.shape,
                                   self.args.gamma,
                                   self.args.alpha,
                                   self.args.n_steps,
                                   self.device)

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
        prioritized_replay_beta = Piecewise(
            [
                (0, 0.4),
                (self.args.max_steps, 1)
            ], outside_value=1)
        global_step = 0
        training = True
        start_time = time.time()
        obs, _ = self.envs.reset()
        while training:
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
                    if global_step >= self.args.max_steps:
                        training = False
            self.buffer.add(obs, actions, rewards, terminated)
            obs = next_obs
            if global_step % self.args.train_freq == 0:
                self.policy.reset_noise()
            global_step += 1

            if global_step >= self.args.learning_starts:
                if global_step % self.args.train_freq == 0:
                    beta = prioritized_replay_beta(global_step)
                    data = self.buffer.sample(self.args.batch_size, beta)
                    loss, old_val, um_loss = self.policy.learn(data)
                    new_priorities = np.abs(um_loss) + 1e-6
                    # Update replay buffer priorities
                    # print(data['indexes'].shape, new_priorities.shape)
                    self.buffer.update_priorities(data['indexes'], new_priorities)
                    if global_step % 100 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar("losses/q_values", old_val, global_step)
                        self.writer.add_scalar("losses/beta", beta, global_step)
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
