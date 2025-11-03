import os.path
import csv
from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import draw_textured_animation
from torch.utils.tensorboard import SummaryWriter
import random
import collections


class ReturnValueOfTrain:
    def __init__(self):
        self.return_list = []
        self.target_tracking_return_list = []
        self.boundary_punishment_return_list = []
        self.duplicate_tracking_punishment_return_list = []
        self.average_covered_targets_list = []
        self.max_covered_targets_list = []

    def item(self):
        value_dict = {
            'return_list': self.return_list,
            'target_tracking_return_list': self.target_tracking_return_list,
            'boundary_punishment_return_list': self.boundary_punishment_return_list,
            'duplicate_tracking_punishment_return_list': self.duplicate_tracking_punishment_return_list,
            'average_covered_targets_list': self.average_covered_targets_list,
            'max_covered_targets_list': self.max_covered_targets_list
        }
        return value_dict

    def save_epoch(self, reward, tt_return, bp_return, dtp_return, average_targets, max_targets):
        self.return_list.append(reward)
        self.target_tracking_return_list.append(tt_return)
        self.boundary_punishment_return_list.append(bp_return)
        self.duplicate_tracking_punishment_return_list.append(dtp_return)
        self.average_covered_targets_list.append(average_targets)
        self.max_covered_targets_list.append(max_targets)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, transition_dict):
        # 从transition_dict中提取各个列表
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']

        # 将各个元素合并成元组，并添加到缓冲区中
        experiences = zip(states, actions, rewards, next_states)
        self.buffer.extend(experiences)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, min(batch_size, self.size()))
        states, actions, rewards, next_states = zip(*transitions)

        # 构造返回的字典
        sample_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states
        }
        return sample_dict

    def size(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']

        experiences = zip(states, actions, rewards, next_states)

        for experience in experiences:
            max_priority = self.priorities.max() if self.buffer else 1.0

            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.pos] = experience

            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return dict(states=[], actions=[], rewards=[], next_states=[]), None, None

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])

        sample_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
        }
        return sample_dict, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)


def operate_epoch(config, env, uav_agent, protector_agent, pmi, num_steps, cwriter_state=None, cwriter_prob=None):
    """
    :param config:
    :param env:
    :param uav_agent: Agent for UAVs
    :param protector_agent: Agent for Protectors
    :param pmi: 
    :param num_steps: 
    :param cwriter_state: 用于记录一个epoch内的state信息, 调试bug时使用
    :param cwriter_prob:  用于记录一个epoch内的prob信息, 调试bug时使用
    :return: 
    """
    uav_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    protector_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    episode_return = 0
    episode_target_tracking_return = 0
    episode_boundary_punishment_return = 0
    episode_duplicate_tracking_punishment_return = 0
    covered_targets_list = []

    for i in range(num_steps):
        config['step'] = i + 1
        uav_action_list = []
        protector_action_list = []

        # each uav makes choices
        if uav_agent:
            for uav in env.uav_list:
                state = uav.get_local_state()
                if cwriter_state:
                    cwriter_state.writerow(state.tolist())
                # 连续动作：直接返回连续角速度值
                action_continuous, (mean, log_std) = uav_agent.take_action(state)
                if cwriter_prob:
                    cwriter_prob.writerow([mean, log_std])  # 记录均值和log_std
                uav_transition_dict['states'].append(state)
                uav_action_list.append(action_continuous)
                # Store continuous action value for training
                uav_transition_dict['actions'].append(action_continuous)
                
                # # 离散化动作部分已注释（暂时不使用）
                # action_continuous = uav.discrete_action(action_idx.item())
                # uav_transition_dict['actions'].append(action_idx.item())
        else:
            # For C-METHOD, use heuristic actions
            for uav in env.uav_list:
                action = uav.get_action_by_direction(env.target_list, env.uav_list)
                uav_action_list.append(action)

        # each protector makes choices
        if protector_agent:
            for protector in env.protector_list:
                state = protector.get_local_state()
                # 连续动作：直接返回连续角速度值
                action_continuous, (mean, log_std) = protector_agent.take_action(state)
                protector_transition_dict['states'].append(state)
                protector_action_list.append(action_continuous)
                # Store continuous action value for training
                protector_transition_dict['actions'].append(action_continuous)
                
                # # 离散化动作部分已注释（暂时不使用）
                # action_continuous = protector.discrete_action(action_idx.item())
                # protector_transition_dict['actions'].append(action_idx.item())
        else:
            # For C-METHOD, use heuristic actions
            for protector in env.protector_list:
                action = protector.get_action_by_direction(env.target_list, env.uav_list)
                protector_action_list.append(action)

        # use action_list to update the environment
        next_state_list, next_protector_state_list, reward_list, covered_targets = env.step(config, pmi, uav_action_list, protector_action_list)
        # Note: continuous actions are already stored above for training
        uav_transition_dict['next_states'].extend(next_state_list)
        uav_transition_dict['rewards'].extend(reward_list['rewards'])
        
        if protector_agent:
            # Note: continuous actions are already stored above for training
            protector_transition_dict['next_states'].extend(next_protector_state_list)
            protector_transition_dict['rewards'].extend(reward_list.get('protector_rewards', []))

        episode_return += sum(reward_list['rewards']) + sum(reward_list.get('protector_rewards', []))
        episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
        episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
        episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])
        covered_targets_list.append(covered_targets)

    total_agents = env.n_uav + env.n_protectors
    episode_return /= num_steps * total_agents
    episode_target_tracking_return /= num_steps * env.n_uav
    episode_boundary_punishment_return /= num_steps * env.n_uav
    episode_duplicate_tracking_punishment_return /= num_steps * env.n_uav
    average_covered_targets = np.mean(covered_targets_list)
    max_covered_targets = np.max(covered_targets_list)

    return (uav_transition_dict, protector_transition_dict, episode_return, episode_target_tracking_return,
            episode_boundary_punishment_return, episode_duplicate_tracking_punishment_return,
            average_covered_targets, max_covered_targets)


def train(config, env, uav_agent, protector_agent, pmi, num_episodes, num_steps, frequency):
    """
    :param config:
    :param pmi: pmi network
    :param frequency: 打印消息的频率
    :param num_steps: 每局进行的步数
    :param env:
    :param uav_agent: Agent for UAVs (所有的无人机共享权重训练, 所以共用一个agent)
    :param protector_agent: Agent for Protectors (所有的保护者共享权重训练, 所以共用一个agent)
    :param num_episodes: 局数
    :return:
    """
    # initialize saving list
    save_dir = os.path.join(config["save_dir"], "logs")
    writer = SummaryWriter(log_dir=save_dir)  # 可以指定log存储的目录
    return_value = ReturnValueOfTrain()
    # buffer = ReplayBuffer(config["actor_critic"]["buffer_size"])
    uav_buffer = PrioritizedReplayBuffer(config["actor_critic"]["buffer_size"])
    protector_buffer = PrioritizedReplayBuffer(config["actor_critic"]["buffer_size"])
    if config["actor_critic"]["sample_size"] > 0:
        sample_size = config["actor_critic"]["sample_size"]
    else:
        sample_size = max(config["environment"]["n_uav"], config["environment"]["n_protectors"]) * num_steps

    with open(os.path.join(save_dir, 'state.csv'), mode='w', newline='') as state_file, \
            open(os.path.join(save_dir, 'prob.csv'), mode='w', newline='') as prob_file:
        cwriter_state = csv.writer(state_file)
        cwriter_prob = csv.writer(prob_file)

        cwriter_state.writerow(['state'])  # 写入state.csv的表头
        cwriter_prob.writerow(['prob'])  # 写入prob.csv的表头

        with tqdm(total=num_episodes, desc='Episodes') as pbar:
            for i in range(num_episodes):
                # reset environment from config yaml file
                env.reset(config=config)

                # episode start
                uav_transition_dict, protector_transition_dict, reward, tt_return, bp_return, \
                    dtp_return, average_targets, max_targets = operate_epoch(config, env, uav_agent, protector_agent, pmi, num_steps)
                writer.add_scalar('reward', reward, i)
                writer.add_scalar('target_tracking_return', tt_return, i)
                writer.add_scalar('boundary_punishment', bp_return, i)
                writer.add_scalar('duplicate_tracking_punishment', dtp_return, i)
                writer.add_scalar('average_covered_targets', average_targets, i)
                writer.add_scalar('max_covered_targets', max_targets, i)

                # saving return lists
                return_value.save_epoch(reward, tt_return, bp_return, dtp_return, average_targets, max_targets)

                # sample from buffer and update UAV agent
                uav_actor_loss = uav_critic_loss = None
                avg_pmi_loss = None
                if uav_agent:
                    uav_buffer.add(uav_transition_dict)
                    uav_sample_dict, uav_indices, _ = uav_buffer.sample(sample_size)
                    
                    # Check if buffer has enough samples
                    if len(uav_sample_dict.get('states', [])) > 0:
                        # update actor-critic network
                        uav_actor_loss, uav_critic_loss, uav_td_errors = uav_agent.update(uav_sample_dict)
                        writer.add_scalar('uav_actor_loss', uav_actor_loss, i)
                        writer.add_scalar('uav_critic_loss', uav_critic_loss, i)
                        
                        # update buffer
                        uav_buffer.update_priorities(uav_indices, uav_td_errors.abs().detach().cpu().numpy())
                        
                        # update pmi network (using UAV states)
                        if pmi:
                            avg_pmi_loss = pmi.train_pmi(config, torch.tensor(np.array(uav_sample_dict["states"])), env.n_uav)
                            writer.add_scalar('avg_pmi_loss', avg_pmi_loss, i)

                # sample from buffer and update Protector agent
                protector_actor_loss = protector_critic_loss = None
                if protector_agent:
                    protector_buffer.add(protector_transition_dict)
                    protector_sample_dict, protector_indices, _ = protector_buffer.sample(sample_size)
                    
                    # Check if buffer has enough samples
                    if len(protector_sample_dict.get('states', [])) > 0:
                        # update actor-critic network
                        protector_actor_loss, protector_critic_loss, protector_td_errors = protector_agent.update(protector_sample_dict)
                        writer.add_scalar('protector_actor_loss', protector_actor_loss, i)
                        writer.add_scalar('protector_critic_loss', protector_critic_loss, i)
                        
                        # update buffer
                        protector_buffer.update_priorities(protector_indices, protector_td_errors.abs().detach().cpu().numpy())

                # save & print
                if (i + 1) % frequency == 0:
                    # print some information
                    postfix_dict = {'episode': '%d' % (i + 1),
                                   'return': '%.3f' % np.mean(return_value.return_list[-frequency:])}
                    if uav_agent:
                        postfix_dict['uav_actor_loss'] = '%f' % uav_actor_loss
                        postfix_dict['uav_critic_loss'] = '%f' % uav_critic_loss
                    if protector_agent:
                        postfix_dict['prot_actor_loss'] = '%f' % protector_actor_loss
                        postfix_dict['prot_critic_loss'] = '%f' % protector_critic_loss
                    if pmi:
                        postfix_dict['avg pmi loss'] = '%f' % avg_pmi_loss
                    pbar.set_postfix(postfix_dict)

                    # save results and weights
                    draw_textured_animation(config=config, env=env, num_steps=num_steps, ep_num=i)
                    if uav_agent:
                        uav_agent.save(save_dir=config["save_dir"], epoch_i=i + 1)
                    if protector_agent:
                        protector_agent.save(save_dir=os.path.join(config["save_dir"], "protector"), epoch_i=i + 1)
                    if pmi:
                        pmi.save(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_position(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_covered_num(save_dir=config["save_dir"], epoch_i=i + 1)

                # episode end
                pbar.update(1)

    writer.close()

    return return_value.item()


def evaluate(config, env, uav_agent, protector_agent, pmi, num_steps):
    """
    :param config:
    :param pmi: pmi network
    :param num_steps: 每局进行的步数
    :param env:
    :param uav_agent: Agent for UAVs
    :param protector_agent: Agent for Protectors
    :return:
    """
    # initialize saving list
    return_value = ReturnValueOfTrain()

    # reset environment from config yaml file
    env.reset(config=config)

    # episode start
    uav_transition_dict, protector_transition_dict, reward, tt_return, bp_return, dtp_return, average_targets, max_targets = operate_epoch(config, env, uav_agent, protector_agent, pmi, num_steps)

    # saving return lists
    return_value.save_epoch(reward, tt_return, bp_return, dtp_return, average_targets, max_targets)

    # save results and weights
    draw_textured_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()

def run_epoch(config, pmi, env, num_steps):
    """
    :param config:
    :param env:
    :param num_steps:
    :return:
    """
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    episode_return = 0
    episode_target_tracking_return = 0
    episode_boundary_punishment_return = 0
    episode_duplicate_tracking_punishment_return = 0
    covered_targets_list = []

    for _ in range(num_steps):
        uav_action_list = []
        protector_action_list = []

        # each uav makes choices
        for uav in env.uav_list:
            action = uav.get_action_by_direction(env.target_list, env.uav_list)
            uav_action_list.append(action)

        # each protector makes choices
        for protector in env.protector_list:
            action = protector.get_action_by_direction(env.target_list, env.uav_list)
            protector_action_list.append(action)

        next_state_list, next_protector_state_list, reward_list, covered_targets = env.step(config, pmi, uav_action_list, protector_action_list)

        # use action_list to update the environment
        transition_dict['actions'].extend(uav_action_list)
        transition_dict['rewards'].extend(reward_list['rewards'])

        episode_return += sum(reward_list['rewards'])
        episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
        episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
        episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])
        covered_targets_list.append(covered_targets)

    average_covered_targets = np.mean(covered_targets_list)
    max_covered_targets = np.max(covered_targets_list)

    return (transition_dict, episode_return, episode_target_tracking_return,
            episode_boundary_punishment_return, episode_duplicate_tracking_punishment_return,
            average_covered_targets, max_covered_targets)

def run(config, env, pmi, num_steps):
    """
    :param config:
    :param num_steps: 每局进行的步数
    :param env:
    :return:
    """
    # initialize saving list
    return_value = ReturnValueOfTrain()

    # reset environment from config yaml file
    env.reset(config=config)

    # episode start
    transition_dict, reward, tt_return, bp_return, dtp_return, average_targets, max_targets = run_epoch(config, pmi, env, num_steps)

    # saving return lists
    return_value.save_epoch(reward, tt_return, bp_return, dtp_return, average_targets, max_targets)

    # save results and weights
    draw_textured_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()