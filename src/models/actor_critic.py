import os

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
        # 对于离散动作，n_actions 是动作空间的大小
        self.action_dim = n_actions  # 离散动作空间大小
        self.fc1 = nn.Linear(n_states, n_hiddens)
        # 输出动作概率分布（使用softmax）
        self.fc2 = nn.Linear(n_hiddens, self.action_dim)

    # 前向传播 - 离散动作版本
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        action_probs = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        action_probs = f.softmax(action_probs, dim=1)  # 转换为概率分布
        return action_probs


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
                 gamma, device, entropy_coef=0.01):
        """
        :param state_dim: 特征空间的维数
        :param hidden_dim: 隐藏层的维数
        :param action_dim: 动作空间的维数（对于离散动作，是动作空间的大小）
        :param actor_lr: actor网络的学习率
        :param critic_lr: critic网络的学习率
        :param gamma: 经验回放参数
        :param device: 用于训练的设备
        :param entropy_coef: 熵正则化系数，用于鼓励探索
        """
        # 策略网络 - 离散动作版本
        self.actor = FnnPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = FnnValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        
        # 初始化critic的最后一层权重更小，避免初始值过大
        nn.init.orthogonal_(self.critic.fc2.weight, gain=0.01)
        nn.init.constant_(self.critic.fc2.bias, 0.0)
        
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim  # 离散动作空间大小
        self.entropy_coef = entropy_coef  # 熵正则化系数
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        self.reward_momentum = 0.995  # 使用更保守的动量（0.995），使归一化更稳定

    def take_action(self, states):
        """
        :param states: nparray, size(state_dim,) 代表智能体的状态
        :return: action_idx (离散动作索引), action_probs (动作概率分布) 用于记录
        """
        states_np = np.array(states)[np.newaxis, :]  # 直接使用np.array来转换
        states_tensor = torch.tensor(states_np, dtype=torch.float).to(self.device)
        action_probs = self.actor(states_tensor)
        # 使用分类分布采样离散动作
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        
        # 使用detach()断开梯度连接，然后转换为numpy
        return action_idx.squeeze().cpu().item(), action_probs.squeeze().detach().cpu().numpy()

    def update(self, transition_dict):
        """
        :param transition_dict: dict, 包含状态,动作, 单个智能体的奖励, 下一个状态的四元组
        :return: None
        """
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = transition_dict['actions']  # 离散动作索引列表
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device).squeeze()
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)

        # 对于离散动作，计算 log_prob
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 计算所选动作的 log_prob
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device).view(-1)
        log_probs = action_dist.log_prob(actions_tensor).unsqueeze(1)
        
        # 使用running statistics归一化奖励，避免critic loss过大
        # 更新running statistics
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item() + 1e-8
        
        if self.reward_count == 0:
            # 第一次，直接使用batch statistics
            self.reward_mean = batch_mean
            self.reward_std = max(batch_std, 0.1)  # 确保std不会太小
        else:
            # 使用指数移动平均更新running statistics
            # 使用更保守的动量（0.995），使归一化更稳定
            self.reward_momentum = 0.995
            self.reward_mean = self.reward_momentum * self.reward_mean + (1 - self.reward_momentum) * batch_mean
            # 使用更稳定的std更新方式，避免std突然变化
            new_std = max(batch_std, 0.1)  # 确保std不会太小
            self.reward_std = self.reward_momentum * self.reward_std + (1 - self.reward_momentum) * new_std
        
        self.reward_count += 1
        
        # 使用running statistics归一化奖励
        # 确保std不会太小
        effective_std = max(self.reward_std, 0.1)
        rewards_normalized = (rewards - self.reward_mean) / effective_std
        
        # 限制归一化后的奖励范围，防止极端值导致Critic Loss爆炸
        # 从[-10, 10]改为[-5, 5]，更保守的范围，进一步提高稳定性
        rewards_normalized = torch.clamp(rewards_normalized, min=-5.0, max=5.0)
        
        # 计算td_target和td_delta（使用归一化后的奖励）
        td_target_normalized = rewards_normalized + self.gamma * self.critic(next_states).detach()
        td_delta_normalized = td_target_normalized - self.critic(states)
        
        # 计算 critic loss（使用归一化后的奖励）
        critic_loss = torch.mean(f.mse_loss(self.critic(states), td_target_normalized.detach()))
        
        # 添加Critic Loss的clipping，防止极端值（设置合理的上限100）
        # 但使用更温和的clipping，只clip极端值，不影响正常训练
        if critic_loss.item() > 100.0:
            critic_loss = torch.clamp(critic_loss, min=0.0, max=100.0)
        
        # 数值稳定性保护：限制 log_probs 的范围
        # log_probs 通常应该在 [-20, 0] 范围内，如果超出则裁剪
        log_probs = torch.clamp(log_probs, min=-20.0, max=10.0)
        
        # 限制 td_delta 的范围，平衡策略梯度信号和稳定性
        # 从[-10, 10]改为[-5, 5]，更保守的范围，进一步提高稳定性
        td_delta_clipped = torch.clamp(td_delta_normalized.detach(), min=-5.0, max=5.0)
        
        # 计算熵（用于鼓励探索）
        entropy = action_dist.entropy().mean()
        
        # 检查是否有 NaN 或 Inf
        if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
            print("Warning: log_probs contains NaN or Inf, skipping actor update")
            return torch.tensor(0.0, device=self.device), critic_loss, td_delta_normalized
        
        if torch.any(torch.isnan(td_delta_clipped)) or torch.any(torch.isinf(td_delta_clipped)):
            print("Warning: td_delta contains NaN or Inf, skipping actor update")
            return torch.tensor(0.0, device=self.device), critic_loss, td_delta_normalized

        # Actor loss = policy gradient - entropy regularization
        # 熵正则化鼓励探索，防止策略过早收敛
        actor_loss = torch.mean(-log_probs * td_delta_clipped) - self.entropy_coef * entropy
        
        # 检查 actor_loss 是否有 NaN 或 Inf（虽然理论上不应该发生，但为了安全起见）
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print(f"Warning: actor_loss is NaN or Inf (value: {actor_loss.item()}), skipping this update")
            return torch.tensor(0.0, device=self.device), critic_loss, td_delta_normalized
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        # 进一步加强梯度裁剪，从1.0改为0.5，进一步提高稳定性
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss, critic_loss, td_delta_normalized

    def save(self, save_dir, epoch_i):
        # Create directories if they don't exist
        actor_dir = os.path.join(save_dir, "actor")
        critic_dir = os.path.join(save_dir, "critic")
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(critic_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()
        }, os.path.join(actor_dir, 'actor_weights_' + str(epoch_i) + '.pth'))
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()
        }, os.path.join(critic_dir, 'critic_weights_' + str(epoch_i) + '.pth'))

    def load(self, actor_path, critic_path):
        if actor_path and os.path.exists(actor_path):
            checkpoint = torch.load(actor_path)
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if critic_path and os.path.exists(critic_path):
            checkpoint = torch.load(critic_path)
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
