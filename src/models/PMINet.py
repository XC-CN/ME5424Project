import torch.optim as optim
import numpy as np
# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as f
import os


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @staticmethod
    def forward(output1, output2):
        loss = torch.mean(torch.log(1 + torch.exp(-output1)) + torch.log(1 + torch.exp(output2)))
        return loss


class PMINetwork(nn.Module):
    def __init__(self, comm_dim=5, obs_dim=4, boundary_state_dim=3, hidden_dim=64, b2_size=3000, 
                 state_dim=None):
        """
        :param comm_dim: 通信维度（UAV之间）或UAV观测维度（Protector观测UAV）
        :param obs_dim: 目标观测维度
        :param boundary_state_dim: 边界状态维度
        :param hidden_dim: 隐藏层维度
        :param b2_size: 训练批次大小
        :param state_dim: 如果提供，将使用统一的状态维度而不是分别处理各部分
        """
        super(PMINetwork, self).__init__()
        self.comm_dim = comm_dim
        self.obs_dim = obs_dim
        self.boundary_state_dim = boundary_state_dim
        self.hidden_dim = hidden_dim
        self.b2_size = b2_size
        
        # 如果提供了统一的状态维度，使用简化的架构
        if state_dim is not None:
            self.input_dim = state_dim
            self.fc_input = nn.Linear(state_dim, hidden_dim)
            self.bn_input = nn.BatchNorm1d(hidden_dim)
        else:
            # 原有的分别处理各部分的方式
            self.fc_comm = nn.Linear(comm_dim, hidden_dim)
            self.bn_comm = nn.BatchNorm1d(hidden_dim) 
            self.fc_obs = nn.Linear(obs_dim, hidden_dim)
            self.bn_obs = nn.BatchNorm1d(hidden_dim) 
            self.fc_boundary_state = nn.Linear(boundary_state_dim, hidden_dim)
            self.bn_boundary_state = nn.BatchNorm1d(hidden_dim)
            self.input_dim = None

        # 共享的特征提取层
        self.fc1 = nn.Linear(hidden_dim * (3 if state_dim is None else 1), hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        
        # 双输出头：合作PMI和对抗PMI（总是存在）
        self.fc_cooperation = nn.Linear(hidden_dim, 1)  # 友方合作PMI
        self.fc_adversarial = nn.Linear(hidden_dim, 1)   # 敌方对抗PMI
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        """
        :param x: 输入特征（两个agent状态的乘积或单个agent状态）
        :return: (cooperation_pmi, adversarial_pmi) 永远返回两个PMI值
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        
        # 如果使用统一状态维度
        if self.input_dim is not None:
            x_processed = self.fc_input(x)
            x_processed = f.relu(self.bn_input(x_processed))
            combined = x_processed
        else:
            # 原有的分别处理方式
            comm = x[:, :self.comm_dim]
            obs = x[:, self.comm_dim:self.comm_dim + self.obs_dim]
            boundary_state = x[:, self.comm_dim + self.obs_dim:self.comm_dim + self.obs_dim + self.boundary_state_dim]

            # Process each part with BatchNorm
            comm_vec = self.fc_comm(comm)
            comm_vec = f.relu(self.bn_comm(comm_vec))
            obs_vec = self.fc_obs(obs)
            obs_vec = f.relu(self.bn_obs(obs_vec))
            boundary_state_vec = self.fc_boundary_state(boundary_state)
            boundary_state_vec = f.relu(self.bn_boundary_state(boundary_state_vec))

            # Concatenate
            combined = torch.cat((comm_vec, obs_vec, boundary_state_vec), dim=1)
        
        # 共享特征提取
        x = self.fc1(combined)
        x = f.relu(self.bn1(x))
        
        # 同时输出合作和对抗PMI
        cooperation_pmi = self.fc_cooperation(x)
        adversarial_pmi = self.fc_adversarial(x)
        return cooperation_pmi, adversarial_pmi

    def inference(self, single_data, relation_type='cooperation'):
        """
        推理接口
        :param single_data: 单个样本数据
        :param relation_type: 关系类型（'cooperation'=合作, 'adversarial'=对抗）
        :return: 根据relation_type返回对应的PMI值
        """
        self.eval()
        if isinstance(single_data, np.ndarray):
            single_data = torch.tensor(single_data, dtype=torch.float32)

        if single_data.ndim == 1:
            single_data = single_data.unsqueeze(0)
        
        cooperation_pmi, adversarial_pmi = self.forward(single_data)
        
        if relation_type == 'cooperation':
            return cooperation_pmi.item()
        elif relation_type == 'adversarial':
            return adversarial_pmi.item()
        else:
            # 默认返回合作PMI
            return cooperation_pmi.item()
    
    def inference_dual(self, single_data):
        """
        双输出推理接口，返回合作和对抗两个PMI值
        :param single_data: 单个样本数据
        :return: (cooperation_pmi, adversarial_pmi)
        """
        self.eval()
        if isinstance(single_data, np.ndarray):
            single_data = torch.tensor(single_data, dtype=torch.float32)

        if single_data.ndim == 1:
            single_data = single_data.unsqueeze(0)
        
        cooperation_pmi, adversarial_pmi = self.forward(single_data)
        return cooperation_pmi.item(), adversarial_pmi.item()

    def train_pmi(self, config, train_data, n_agents, enemy_data=None, n_enemies=None,
                  mixed_training=False, protector_data=None, n_protectors=None):
        """
        训练PMI网络，同时训练合作PMI和对抗PMI
        
        :param config: 配置字典
        :param train_data: 训练数据（友方）(timesteps*n_agents, state_dim)
        :param n_agents: agent数量（友方）
        :param agent_type: 'uav' 或 'protector'，用于确定是友方还是敌方训练
        :param enemy_data: 敌方训练数据 (timesteps*n_enemies, state_dim)，可选，用于训练对抗PMI
        :param n_enemies: 敌方agent数量，如果提供enemy_data则必须提供
        :param mixed_training: 是否混合训练（同时使用UAV和Protector的友方数据训练合作PMI）
        :param protector_data: Protector的训练数据 (timesteps*n_protectors, state_dim)，仅在mixed_training=True时使用
        :param n_protectors: Protector数量，仅在mixed_training=True时使用
        """
        self.train()
        loss_function = CustomLoss()
        
        # 获取状态维度
        state_dim = train_data.size(1)
        timesteps = train_data.size(0) // n_agents
        train_data = train_data.view(timesteps, n_agents, state_dim)
        
        # 混合训练：准备UAV和Protector的数据
        protector_selected_data = None
        if mixed_training and protector_data is not None and n_protectors is not None:
            protector_state_dim = protector_data.size(1)
            protector_timesteps = protector_data.size(0) // n_protectors
            protector_data = protector_data.view(protector_timesteps, n_protectors, protector_state_dim)
            
            # 确保状态维度一致
            if protector_state_dim == state_dim:
                # 分配采样数量：一半来自UAV数据，一半来自Protector数据
                half_size = self.b2_size // 2
                protector_timestep_indices = torch.randint(low=0, high=min(protector_timesteps, timesteps), 
                                                            size=(half_size,))
                protector_agent_indices = torch.randint(low=0, high=n_protectors, size=(half_size, 2))
                protector_selected_data = torch.zeros((half_size, 2, state_dim))
                for i in range(half_size):
                    protector_selected_data[i] = protector_data[protector_timestep_indices[i], protector_agent_indices[i]]
        
        # 训练合作PMI（友方之间的依赖关系）
        # 如果使用混合训练，只采样一半来自当前agent类型的数据
        cooperation_sample_size = self.b2_size // 2 if mixed_training and protector_selected_data is not None else self.b2_size
        timestep_indices = torch.randint(low=0, high=timesteps, size=(cooperation_sample_size,))
        agent_indices = torch.randint(low=0, high=n_agents, size=(cooperation_sample_size, 2))
        selected_data = torch.zeros((cooperation_sample_size, 2, state_dim))
        for i in range(cooperation_sample_size):
            selected_data[i] = train_data[timestep_indices[i], agent_indices[i]]
        
        # 合并混合训练数据
        if mixed_training and protector_selected_data is not None:
            # 合并UAV和Protector的合作训练数据
            mixed_cooperation_data = torch.cat([selected_data, protector_selected_data], dim=0)
        else:
            mixed_cooperation_data = selected_data

        # 准备敌方数据（如果提供）
        enemy_selected_data = None
        if enemy_data is not None and n_enemies is not None:
            enemy_state_dim = enemy_data.size(1)
            enemy_timesteps = enemy_data.size(0) // n_enemies
            enemy_data = enemy_data.view(enemy_timesteps, n_enemies, enemy_state_dim)
            # 确保状态维度一致
            if enemy_state_dim == state_dim:
                # 随机选择时间步和敌方agent
                enemy_timestep_indices = torch.randint(low=0, high=min(enemy_timesteps, timesteps), 
                                                        size=(self.b2_size,))
                enemy_agent_indices = torch.randint(low=0, high=n_enemies, size=(self.b2_size,))
                # 选择对应的友方agent（与敌方在同一时间步）
                friendly_agent_indices = torch.randint(low=0, high=n_agents, size=(self.b2_size,))
                enemy_selected_data = torch.zeros((self.b2_size, 2, state_dim))
                for i in range(self.b2_size):
                    # 第一列：友方agent状态
                    enemy_selected_data[i, 0] = train_data[enemy_timestep_indices[i], friendly_agent_indices[i]]
                    # 第二列：敌方agent状态
                    enemy_selected_data[i, 1] = enemy_data[enemy_timestep_indices[i], enemy_agent_indices[i]]

        avg_loss_cooperation = 0
        avg_loss_adversarial = 0
        cooperation_data_size = mixed_cooperation_data.size(0)
        batch_count_cooperation = cooperation_data_size // config["pmi"]["batch_size"]
        batch_count_adversarial = self.b2_size // config["pmi"]["batch_size"] if enemy_selected_data is not None else 0
        
        # 1. 训练合作PMI（友方之间的依赖关系，可能包含UAV-UAV和Protector-Protector）
        for i in range(batch_count_cooperation):
            self.optimizer.zero_grad()
            batch_data = mixed_cooperation_data[i * config["pmi"]["batch_size"]:
                                                min((i + 1) * config["pmi"]["batch_size"], cooperation_data_size)]
            if batch_data.size(0) < 2:  # 确保批次至少有2个样本
                continue
            input_1_2 = batch_data[:, 0].squeeze(1)
            input_1_3 = batch_data[:, 1].squeeze(1)
            
            coop_1_2, adv_1_2 = self.forward(input_1_2)
            coop_1_3, adv_1_3 = self.forward(input_1_3)
            loss_cooperation = loss_function(coop_1_2, coop_1_3)
            
            # 反向传播和优化（仅合作PMI）
            loss_cooperation.backward()
            self.optimizer.step()
            avg_loss_cooperation += abs(loss_cooperation.item())
        
        # 2. 训练对抗PMI（友方与敌方之间的对抗关系）
        if enemy_selected_data is not None:
            for i in range(batch_count_adversarial):
                self.optimizer.zero_grad()
                enemy_batch_data = enemy_selected_data[i * config["pmi"]["batch_size"]:(i + 1) * config["pmi"]["batch_size"]]
                friendly_input = enemy_batch_data[:, 0].squeeze(1)  # 友方agent状态
                enemy_input = enemy_batch_data[:, 1].squeeze(1)    # 敌方agent状态
                
                # 计算友方和敌方状态的乘积
                adversarial_input = friendly_input * enemy_input
                
                _, adv_friendly_enemy_1 = self.forward(adversarial_input)
                # 对于对抗PMI，我们希望它能够反映对抗程度
                # 使用负值训练，因为对抗意味着负相关
                _, adv_friendly_enemy_2 = self.forward(adversarial_input)
                loss_adversarial = loss_function(-adv_friendly_enemy_1, -adv_friendly_enemy_2)
                
                # 反向传播和优化（仅对抗PMI）
                loss_adversarial.backward()
                self.optimizer.step()
                avg_loss_adversarial += abs(loss_adversarial.item()) if loss_adversarial is not None else 0
        
        avg_loss_cooperation /= max(batch_count_cooperation, 1)
        avg_loss_adversarial /= max(batch_count_adversarial, 1)
        
        return avg_loss_cooperation, avg_loss_adversarial

    def save(self, save_dir, epoch_i):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(save_dir, "pmi", 'pmi_weights_' + str(epoch_i) + '.pth'))

    def load(self, path):
        if path and os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
