import os.path

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = None
        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out


class ResPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ResPolicyNet, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = f.avg_pool1d(x, 12)  # 这里使用平均池化，你也可以根据需求使用其他池化方式
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return f.softmax(x, dim=1)


class ResValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ResValueNet, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class FnnPolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(FnnPolicyNet, self).__init__()
        # 对于连续动作，n_actions 是动作维度（这里应该是1，因为只有角速度）
        self.action_dim = 1  # 连续动作维度：角速度
        self.fc1 = nn.Linear(n_states, n_hiddens)
        # 输出均值和方差（对于对角高斯分布，只需要输出均值和log_std）
        self.fc_mean = nn.Linear(n_hiddens, self.action_dim)
        self.fc_log_std = nn.Linear(n_hiddens, self.action_dim)

    # 前向传播 - 连续动作版本
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        mean = self.fc_mean(x)  # [b,n_hiddens]-->[b,1] 动作均值
        log_std = self.fc_log_std(x)  # [b,n_hiddens]-->[b,1] log标准差
        log_std = torch.clamp(log_std, -20, 2)  # 限制log_std范围，避免方差过大或过小
        return mean, log_std


class FnnValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(FnnValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x.squeeze(1)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device, action_bound=None):
        """
        :param state_dim: 特征空间的维数
        :param hidden_dim: 隐藏层的维数
        :param action_dim: 动作空间的维数（对于连续动作，通常为1）
        :param actor_lr: actor网络的学习率
        :param critic_lr: critic网络的学习率
        :param gamma: 经验回放参数
        :param device: 用于训练的设备
        :param action_bound: 动作边界，格式为 (low, high)，例如 (-h_max, h_max)
        """
        # 策略网络 - 连续动作版本
        self.actor = FnnPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = FnnValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device
        self.action_bound = action_bound  # 动作边界，用于限制动作范围

    def take_action(self, states):
        """
        :param states: nparray, size(state_dim,) 代表智能体的状态
        :return: action (连续值), (mean, log_std) 用于记录
        """
        states_np = np.array(states)[np.newaxis, :]  # 直接使用np.array来转换
        states_tensor = torch.tensor(states_np, dtype=torch.float).to(self.device)
        mean, log_std = self.actor(states_tensor)
        std = torch.exp(log_std)
        # 使用高斯分布采样连续动作
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        
        # 如果指定了动作边界，直接使用 clip 限制动作范围（简化实现）
        if self.action_bound is not None:
            low, high = self.action_bound
            action = torch.clamp(action, low, high)
        
        return action.squeeze().cpu().item(), (mean.squeeze().cpu().item(), log_std.squeeze().cpu().item())

    def update(self, transition_dict):
        """
        :param transition_dict: dict, 包含状态,动作, 单个智能体的奖励, 下一个状态的四元组
        :return: None
        """
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], 
                              dtype=torch.float).view(-1, 1).to(self.device)  # 连续动作，使用float
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device).squeeze()
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        
        # 对于连续动作，计算 log_prob
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        action_dist = torch.distributions.Normal(mean, std)
        
        # 直接计算 log_prob（因为动作已经在边界内通过 clip 限制）
        log_probs = action_dist.log_prob(actions)
        
        log_probs = log_probs.sum(dim=1, keepdim=True)  # 如果是多维动作，需要求和

        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(f.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss, critic_loss, td_delta

    def save(self, save_dir, epoch_i):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()
        }, os.path.join(save_dir, "actor", 'actor_weights_' + str(epoch_i) + '.pth'))
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()
        }, os.path.join(save_dir, "critic", 'critic_weights_' + str(epoch_i) + '.pth'))

    def load(self, actor_path, critic_path):
        if actor_path and os.path.exists(actor_path):
            checkpoint = torch.load(actor_path)
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if critic_path and os.path.exists(critic_path):
            checkpoint = torch.load(critic_path)
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
