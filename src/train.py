import os
from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import LiveRenderer
from torch.utils.tensorboard import SummaryWriter
import random
import collections


class ReturnValueOfTrain:
    def __init__(self):
        # UAV metrics
        self.return_list = []
        self.target_tracking_return_list = []
        self.boundary_punishment_return_list = []
        self.duplicate_tracking_punishment_return_list = []
        self.protector_collision_return_list = []

        # Protector metrics
        self.protector_return_list = []
        self.protector_protect_reward_list = []
        self.protector_block_reward_list = []
        self.protector_failure_penalty_list = []
        self.protector_approach_bonus_list = []
        self.protector_retreat_bonus_list = []
        self.protector_movement_penalty_list = []

        # Target metrics
        self.target_return_list = []
        self.target_safety_reward_list = []
        self.target_danger_penalty_list = []
        self.target_capture_penalty_list = []
        self.target_approach_bonus_list = []
        self.target_escape_bonus_list = []
        self.target_movement_penalty_list = []

        # Coverage stats
        self.average_covered_targets_list = []
        self.max_covered_targets_list = []

    def item(self):
        return {
            'return_list': self.return_list,
            'target_tracking_return_list': self.target_tracking_return_list,
            'boundary_punishment_return_list': self.boundary_punishment_return_list,
            'duplicate_tracking_punishment_return_list': self.duplicate_tracking_punishment_return_list,
            'protector_collision_return_list': self.protector_collision_return_list,
            'protector_return_list': self.protector_return_list,
            'protector_protect_reward_list': self.protector_protect_reward_list,
            'protector_block_reward_list': self.protector_block_reward_list,
            'protector_failure_penalty_list': self.protector_failure_penalty_list,
            'protector_approach_bonus_list': self.protector_approach_bonus_list,
            'protector_retreat_bonus_list': self.protector_retreat_bonus_list,
            'protector_movement_penalty_list': self.protector_movement_penalty_list,
            'target_return_list': self.target_return_list,
            'target_safety_reward_list': self.target_safety_reward_list,
            'target_danger_penalty_list': self.target_danger_penalty_list,
            'target_capture_penalty_list': self.target_capture_penalty_list,
            'target_approach_bonus_list': self.target_approach_bonus_list,
            'target_escape_bonus_list': self.target_escape_bonus_list,
            'target_movement_penalty_list': self.target_movement_penalty_list,
            'average_covered_targets_list': self.average_covered_targets_list,
            'max_covered_targets_list': self.max_covered_targets_list
        }

    def save_epoch(self, uav_metrics, protector_metrics, target_metrics, average_targets, max_targets):
        # UAV metrics
        self.return_list.append(uav_metrics.get('return', 0.0))
        self.target_tracking_return_list.append(uav_metrics.get('target_tracking', 0.0))
        self.boundary_punishment_return_list.append(uav_metrics.get('boundary', 0.0))
        self.duplicate_tracking_punishment_return_list.append(uav_metrics.get('duplicate', 0.0))
        self.protector_collision_return_list.append(uav_metrics.get('protector_collision', 0.0))

        # Protector metrics
        self.protector_return_list.append(protector_metrics.get('return', 0.0))
        self.protector_protect_reward_list.append(protector_metrics.get('protect', 0.0))
        self.protector_block_reward_list.append(protector_metrics.get('block', 0.0))
        self.protector_failure_penalty_list.append(protector_metrics.get('failure', 0.0))
        self.protector_approach_bonus_list.append(protector_metrics.get('approach', 0.0))
        self.protector_retreat_bonus_list.append(protector_metrics.get('retreat', 0.0))
        self.protector_movement_penalty_list.append(protector_metrics.get('movement', 0.0))

        # Target metrics
        self.target_return_list.append(target_metrics.get('return', 0.0))
        self.target_safety_reward_list.append(target_metrics.get('safety', 0.0))
        self.target_danger_penalty_list.append(target_metrics.get('danger', 0.0))
        self.target_capture_penalty_list.append(target_metrics.get('capture', 0.0))
        self.target_approach_bonus_list.append(target_metrics.get('approach', 0.0))
        self.target_escape_bonus_list.append(target_metrics.get('escape', 0.0))
        self.target_movement_penalty_list.append(target_metrics.get('movement', 0.0))

        # Coverage stats
        self.average_covered_targets_list.append(average_targets)
        self.max_covered_targets_list.append(max_targets)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, transition_dict):
        # Extract state/action/reward/next_state sequences from the transition dict
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']

        # Package them into experience tuples and append to the buffer
        experiences = zip(states, actions, rewards, next_states)
        self.buffer.extend(experiences)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, min(batch_size, self.size()))
        states, actions, rewards, next_states = zip(*transitions)

        # Build the dictionary that the learners expect
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


def operate_epoch(config, env, agents, pmi, num_steps, render_hook=None):
    roles = ['uav', 'protector', 'target']
    transition_dict = {
        role: {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
        for role in roles
    }

    uav_acc = {'return': 0.0, 'target_tracking': 0.0, 'boundary': 0.0,
               'duplicate': 0.0, 'protector_collision': 0.0}
    protector_acc = {
        'return': 0.0,
        'protect': 0.0,
        'block': 0.0,
        'failure': 0.0,
        'approach': 0.0,
        'retreat': 0.0,
        'movement': 0.0,
    }
    target_acc = {
        'return': 0.0,
        'safety': 0.0,
        'danger': 0.0,
        'capture': 0.0,
        'approach': 0.0,
        'escape': 0.0,
        'movement': 0.0,
    }
    covered_targets_list = []

    steps_run = 0

    for step in range(num_steps):
        config['step'] = step + 1
        role_states = {role: [] for role in roles}
        role_actions = {role: [] for role in roles}

        # UAV decisions
        uav_control_method = config.get('uav', {}).get('control_method', 'rl')
        for uav in env.uav_list:
            state = uav.get_local_state()
            if uav_control_method == 'rule_based':
                rule_params = config.get('uav', {}).get('rule_based_params', {})
                action = uav.get_rule_based_action(env.target_list, env.protector_list, env.uav_list, rule_params)
                role_actions['uav'].append(action)
            else:
                action, _ = agents['uav'].take_action(state)
                role_actions['uav'].append(int(action.item()))
            role_states['uav'].append(state)

        # Protector decisions
        for protector in env.protector_list:
            state = protector.get_local_state()
            action, _ = agents['protector'].take_action(state)
            role_states['protector'].append(state)
            role_actions['protector'].append(int(action.item()))

        # Target decisions
        for target in env.target_list:
            state = target.get_local_state()
            action, _ = agents['target'].take_action(state)
            role_states['target'].append(state)
            role_actions['target'].append(int(action.item()))

        next_states, reward_dict, covered_targets, done = env.step(
            config,
            pmi,
            role_actions['uav'],
            role_actions['protector'],
            role_actions['target']
        )

        if render_hook is not None:
            render_hook(step, env)
        for role in roles:
            transition_dict[role]['states'].extend(role_states[role])
            transition_dict[role]['actions'].extend(role_actions[role])
            transition_dict[role]['rewards'].extend(reward_dict[role]['rewards'])
            transition_dict[role]['next_states'].extend(next_states[role])

        uav_acc['return'] += float(np.sum(reward_dict['uav']['rewards']))
        uav_acc['target_tracking'] += float(np.sum(reward_dict['uav']['target_tracking']))
        uav_acc['boundary'] += float(np.sum(reward_dict['uav']['boundary']))
        uav_acc['duplicate'] += float(np.sum(reward_dict['uav']['duplicate']))
        uav_acc['protector_collision'] += float(np.sum(reward_dict['uav']['protector_collision']))

        protector_acc['return'] += float(np.sum(reward_dict['protector']['rewards']))
        protector_acc['protect'] += float(np.sum(reward_dict['protector']['protect_reward']))
        protector_acc['block'] += float(np.sum(reward_dict['protector']['block_reward']))
        protector_acc['failure'] += float(np.sum(reward_dict['protector']['failure_penalty']))
        protector_acc['approach'] += float(np.sum(reward_dict['protector']['approach_bonus']))
        protector_acc['retreat'] += float(np.sum(reward_dict['protector']['retreat_bonus']))
        protector_acc['movement'] += float(np.sum(reward_dict['protector']['movement_penalty']))

        target_acc['return'] += float(np.sum(reward_dict['target']['rewards']))
        target_acc['safety'] += float(np.sum(reward_dict['target']['safety_reward']))
        target_acc['danger'] += float(np.sum(reward_dict['target']['danger_penalty']))
        target_acc['capture'] += float(np.sum(reward_dict['target']['capture_penalty']))
        target_acc['approach'] += float(np.sum(reward_dict['target']['approach_bonus']))
        target_acc['escape'] += float(np.sum(reward_dict['target']['escape_bonus']))
        target_acc['movement'] += float(np.sum(reward_dict['target']['movement_penalty']))

        covered_targets_list.append(covered_targets)

        steps_run += 1
        if done:
            break

    def average(total, count):
        return total / count if count > 0 else 0.0

    effective_steps = max(steps_run, 1)

    counts = {
        'uav': max(env.n_uav, 1) * effective_steps,
        'protector': max(env.n_protectors, 1) * effective_steps,
        'target': max(env.m_targets, 1) * effective_steps
    }

    uav_metrics = {
        'return': average(uav_acc['return'], counts['uav']),
        'target_tracking': average(uav_acc['target_tracking'], counts['uav']),
        'boundary': average(uav_acc['boundary'], counts['uav']),
        'duplicate': average(uav_acc['duplicate'], counts['uav']),
        'protector_collision': average(uav_acc['protector_collision'], counts['uav'])
    }
    protector_metrics = {
        'return': average(protector_acc['return'], counts['protector']),
        'protect': average(protector_acc['protect'], counts['protector']),
        'block': average(protector_acc['block'], counts['protector']),
        'failure': average(protector_acc['failure'], counts['protector']),
        'approach': average(protector_acc['approach'], counts['protector']),
        'retreat': average(protector_acc['retreat'], counts['protector']),
        'movement': average(protector_acc['movement'], counts['protector'])
    }
    target_metrics = {
        'return': average(target_acc['return'], counts['target']),
        'safety': average(target_acc['safety'], counts['target']),
        'danger': average(target_acc['danger'], counts['target']),
        'capture': average(target_acc['capture'], counts['target']),
        'approach': average(target_acc['approach'], counts['target']),
        'escape': average(target_acc['escape'], counts['target']),
        'movement': average(target_acc['movement'], counts['target'])
    }
    average_covered_targets = float(np.mean(covered_targets_list)) if covered_targets_list else 0.0
    max_covered_targets = float(np.max(covered_targets_list)) if covered_targets_list else 0.0

    return (
        transition_dict,
        uav_metrics,
        protector_metrics,
        target_metrics,
        average_covered_targets,
        max_covered_targets,
    )



def train(config, env, agents, pmi, num_episodes, num_steps, frequency):
    roles = ['uav', 'protector', 'target']
    missing_roles = [role for role in roles if agents.get(role) is None]
    if missing_roles:
        raise ValueError(f"Missing agents for roles: {missing_roles}")

    freq_value = frequency if frequency is not None else config.get("frequency", 1)
    frequency = max(1, int(freq_value))

    log_dir = os.path.join(config["save_dir"], "logs")
    writer = SummaryWriter(log_dir=log_dir)
    return_value = ReturnValueOfTrain()

    actor_cfg_map = {
        'uav': config["actor_critic"],
        'protector': config["protector_actor_critic"],
        'target': config["target_actor_critic"]
    }
    role_counts = {
        'uav': env.n_uav,
        'protector': env.n_protectors,
        'target': env.m_targets
    }

    buffers = {}
    sample_sizes = {}
    for role in roles:
        cfg = actor_cfg_map[role]
        buffers[role] = PrioritizedReplayBuffer(cfg["buffer_size"])
        if cfg.get("sample_size", 0) > 0:
            sample_sizes[role] = cfg["sample_size"]
        else:
            sample_sizes[role] = max(role_counts[role], 1) * num_steps

    last_losses = {role: (0.0, 0.0) for role in roles}

    render_train = config.get("render_when_train", False)
    eval_cfg = config.get("evaluate", {})
    render_trail = eval_cfg.get("render_trail", 60)
    render_fps = eval_cfg.get("video_fps", 10)

    with tqdm(total=num_episodes, desc='Episodes') as pbar:
        for episode in range(num_episodes):
            env.reset(config=config)

            renderer = None
            make_recording = render_train and ((episode + 1) % frequency == 0)
            if make_recording:
                animated_dir = os.path.join(config["save_dir"], "animated")
                os.makedirs(animated_dir, exist_ok=True)
                video_path = os.path.join(animated_dir, f"train_episode_{episode + 1}.mp4")
                renderer = LiveRenderer(
                    env,
                    pause=0.0,
                    trail_steps=render_trail,
                    show=False,
                    record=True,
                    video_path=video_path,
                    video_fps=render_fps,
                )

            try:
                transitions, uav_metrics, protector_metrics, target_metrics, average_targets, max_targets = operate_epoch(
                    config, env, agents, pmi, num_steps, render_hook=renderer if renderer is not None else None)
            finally:
                if renderer is not None:
                    renderer.close()

            writer.add_scalar('uav/return', uav_metrics['return'], episode)
            writer.add_scalar('uav/target_tracking', uav_metrics['target_tracking'], episode)
            writer.add_scalar('uav/boundary', uav_metrics['boundary'], episode)
            writer.add_scalar('uav/duplicate', uav_metrics['duplicate'], episode)
            writer.add_scalar('uav/protector_collision', uav_metrics['protector_collision'], episode)

            writer.add_scalar('protector/return', protector_metrics['return'], episode)
            writer.add_scalar('protector/protect', protector_metrics['protect'], episode)
            writer.add_scalar('protector/block', protector_metrics['block'], episode)
            writer.add_scalar('protector/failure', protector_metrics['failure'], episode)
            writer.add_scalar('protector/approach_bonus', protector_metrics['approach'], episode)
            writer.add_scalar('protector/retreat_bonus', protector_metrics['retreat'], episode)
            writer.add_scalar('protector/movement_penalty', protector_metrics['movement'], episode)

            writer.add_scalar('target/return', target_metrics['return'], episode)
            writer.add_scalar('target/safety', target_metrics['safety'], episode)
            writer.add_scalar('target/danger', target_metrics['danger'], episode)
            writer.add_scalar('target/capture', target_metrics['capture'], episode)
            writer.add_scalar('target/approach_bonus', target_metrics['approach'], episode)
            writer.add_scalar('target/escape_bonus', target_metrics['escape'], episode)
            writer.add_scalar('target/movement_penalty', target_metrics['movement'], episode)

            writer.add_scalar('coverage/average', average_targets, episode)
            writer.add_scalar('coverage/max', max_targets, episode)

            if (episode + 1) % frequency == 0 or episode == num_episodes - 1:
                pbar.write(
                    f"Episode {episode + 1}/{num_episodes} - "
                    f"Protector(return={protector_metrics['return']:.3f}, approach={protector_metrics['approach']:.3f}, "
                    f"retreat={protector_metrics['retreat']:.3f}, move={protector_metrics['movement']:.3f}) | "
                    f"Target(return={target_metrics['return']:.3f}, approach={target_metrics['approach']:.3f}, "
                    f"escape={target_metrics['escape']:.3f}, move={target_metrics['movement']:.3f})"
                )

            return_value.save_epoch(uav_metrics, protector_metrics, target_metrics, average_targets, max_targets)

            uav_sample_dict = None
            uav_control_method = config.get('uav', {}).get('control_method', 'rl')
            for role in roles:
                if role == 'uav' and uav_control_method == 'rule_based':
                    continue

                buffers[role].add(transitions[role])
                sample_dict, indices, _ = buffers[role].sample(sample_sizes[role])
                if len(sample_dict['states']) == 0:
                    continue

                actor_loss, critic_loss, td_errors = agents[role].update(sample_dict)
                actor_loss_value = float(actor_loss.detach().cpu().item() if torch.is_tensor(actor_loss) else actor_loss)
                critic_loss_value = float(critic_loss.detach().cpu().item() if torch.is_tensor(critic_loss) else critic_loss)
                writer.add_scalar(f'{role}/actor_loss', actor_loss_value, episode)
                writer.add_scalar(f'{role}/critic_loss', critic_loss_value, episode)
                last_losses[role] = (actor_loss_value, critic_loss_value)

                if indices is not None and td_errors is not None:
                    buffers[role].update_priorities(indices, td_errors.abs().detach().cpu().numpy())

                if role == 'uav':
                    uav_sample_dict = sample_dict

            if pmi and uav_sample_dict and len(uav_sample_dict['states']) > 0:
                state_tensor = torch.tensor(np.array(uav_sample_dict["states"]), dtype=torch.float32)
                avg_pmi_loss = pmi.train_pmi(config, state_tensor, env.n_uav)
                writer.add_scalar('pmi/avg_loss', avg_pmi_loss, episode)
            else:
                avg_pmi_loss = 0.0

            if (episode + 1) % frequency == 0:
                postfix = {
                    'episode': f'{episode + 1}',
                    'uav_return': f'{np.mean(return_value.return_list[-frequency:]):.3f}'
                }
                for role in roles:
                    actor_loss_value, critic_loss_value = last_losses.get(role, (0.0, 0.0))
                    postfix[f'{role}_actor'] = f'{actor_loss_value:.4f}'
                    postfix[f'{role}_critic'] = f'{critic_loss_value:.4f}'
                if pmi:
                    postfix['pmi_loss'] = f'{avg_pmi_loss:.4f}'
                pbar.set_postfix(postfix)
                for role in roles:
                    agents[role].save(save_dir=config["save_dir"], epoch_i=episode + 1, tag=role)
                if pmi:
                    pmi.save(save_dir=config["save_dir"], epoch_i=episode + 1)
                env.save_position(save_dir=config["save_dir"], epoch_i=episode + 1)
                env.save_covered_num(save_dir=config["save_dir"], epoch_i=episode + 1)

            pbar.update(1)

    writer.close()
    return return_value.item()


def evaluate(config, env, agents, pmi, num_steps):
    return_value = ReturnValueOfTrain()

    env.reset(config=config)
    eval_cfg = config.get("evaluate", {})
    enable_live = eval_cfg.get("enable_live", True)
    render_pause = eval_cfg.get("render_pause", 0.05)
    trail_steps = eval_cfg.get("render_trail", 60)
    save_outputs = eval_cfg.get("save_outputs", False)

    video_path = None
    if save_outputs and eval_cfg.get("save_animation", True):
        animated_dir = os.path.join(config["save_dir"], "animated")
        os.makedirs(animated_dir, exist_ok=True)
        video_path = os.path.join(animated_dir, "evaluation_episode_0.mp4")

    renderer = None
    need_renderer = enable_live or video_path is not None
    if need_renderer:
        renderer = LiveRenderer(
            env,
            pause=render_pause if enable_live else 0.0,
            trail_steps=trail_steps,
            show=enable_live,
            record=video_path is not None,
            video_path=video_path,
            video_fps=eval_cfg.get("video_fps", 10),
        )

    try:
        _, uav_metrics, protector_metrics, target_metrics, average_targets, max_targets = operate_epoch(
            config,
            env,
            agents,
            pmi,
            num_steps,
            render_hook=renderer if renderer is not None else None
        )
    finally:
        if renderer is not None:
            renderer.close()

    return_value.save_epoch(uav_metrics, protector_metrics, target_metrics, average_targets, max_targets)

    if save_outputs:
        env.save_position(save_dir=config["save_dir"], epoch_i=0)
        env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()




def run_epoch(config, pmi, env, num_steps, render_hook=None):
    uav_acc = {'return': 0.0, 'target_tracking': 0.0, 'boundary': 0.0,
               'duplicate': 0.0, 'protector_collision': 0.0}
    protector_acc = {'return': 0.0, 'protect': 0.0, 'block': 0.0, 'failure': 0.0, 'approach': 0.0, 'retreat': 0.0, 'movement': 0.0}
    target_acc = {'return': 0.0, 'safety': 0.0, 'danger': 0.0, 'capture': 0.0, 'approach': 0.0, 'escape': 0.0, 'movement': 0.0}
    covered_targets_list = []

    steps_run = 0

    for step in range(num_steps):
        config['step'] = step + 1
        uav_actions = []
        for uav in env.uav_list:
            action = uav.get_action_by_direction(env.target_list, env.uav_list)
            uav_actions.append(int(action))

        _, reward_dict, covered_targets, done = env.step(config, pmi, uav_actions, None, None)

        if render_hook is not None:
            render_hook(step, env)

        uav_acc['return'] += float(np.sum(reward_dict['uav']['rewards']))
        uav_acc['target_tracking'] += float(np.sum(reward_dict['uav']['target_tracking']))
        uav_acc['boundary'] += float(np.sum(reward_dict['uav']['boundary']))
        uav_acc['duplicate'] += float(np.sum(reward_dict['uav']['duplicate']))
        uav_acc['protector_collision'] += float(np.sum(reward_dict['uav']['protector_collision']))

        protector_acc['return'] += float(np.sum(reward_dict['protector']['rewards']))
        protector_acc['protect'] += float(np.sum(reward_dict['protector']['protect_reward']))
        protector_acc['block'] += float(np.sum(reward_dict['protector']['block_reward']))
        protector_acc['failure'] += float(np.sum(reward_dict['protector']['failure_penalty']))
        protector_acc['approach'] += float(np.sum(reward_dict['protector']['approach_bonus']))
        protector_acc['retreat'] += float(np.sum(reward_dict['protector']['retreat_bonus']))
        protector_acc['movement'] += float(np.sum(reward_dict['protector']['movement_penalty']))

        target_acc['return'] += float(np.sum(reward_dict['target']['rewards']))
        target_acc['safety'] += float(np.sum(reward_dict['target']['safety_reward']))
        target_acc['danger'] += float(np.sum(reward_dict['target']['danger_penalty']))
        target_acc['capture'] += float(np.sum(reward_dict['target']['capture_penalty']))
        target_acc['approach'] += float(np.sum(reward_dict['target']['approach_bonus']))
        target_acc['escape'] += float(np.sum(reward_dict['target']['escape_bonus']))
        target_acc['movement'] += float(np.sum(reward_dict['target']['movement_penalty']))


        covered_targets_list.append(covered_targets)

        steps_run += 1
        if done:
            break

    def average(total, count):
        return total / count if count > 0 else 0.0

    effective_steps = max(steps_run, 1)

    counts = {
        'uav': max(env.n_uav, 1) * effective_steps,
        'protector': max(env.n_protectors, 1) * effective_steps,
        'target': max(env.m_targets, 1) * effective_steps
    }

    uav_metrics = {
        'return': average(uav_acc['return'], counts['uav']),
        'target_tracking': average(uav_acc['target_tracking'], counts['uav']),
        'boundary': average(uav_acc['boundary'], counts['uav']),
        'duplicate': average(uav_acc['duplicate'], counts['uav']),
        'protector_collision': average(uav_acc['protector_collision'], counts['uav'])
    }
    protector_metrics = {
        'return': average(protector_acc['return'], counts['protector']),
        'protect': average(protector_acc['protect'], counts['protector']),
        'block': average(protector_acc['block'], counts['protector']),
        'failure': average(protector_acc['failure'], counts['protector']),
        'approach': average(protector_acc['approach'], counts['protector']),
        'retreat': average(protector_acc['retreat'], counts['protector']),
        'movement': average(protector_acc['movement'], counts['protector'])
    }
    target_metrics = {
        'return': average(target_acc['return'], counts['target']),
        'safety': average(target_acc['safety'], counts['target']),
        'danger': average(target_acc['danger'], counts['target']),
        'capture': average(target_acc['capture'], counts['target']),
        'approach': average(target_acc['approach'], counts['target']),
        'escape': average(target_acc['escape'], counts['target']),
        'movement': average(target_acc['movement'], counts['target'])
    }

    average_covered_targets = float(np.mean(covered_targets_list)) if covered_targets_list else 0.0
    max_covered_targets = float(np.max(covered_targets_list)) if covered_targets_list else 0.0

    return uav_metrics, protector_metrics, target_metrics, average_covered_targets, max_covered_targets


def run(config, env, pmi, num_steps):
    return_value = ReturnValueOfTrain()

    env.reset(config=config)
    eval_cfg = config.get("evaluate", {})
    enable_live = eval_cfg.get("enable_live", True)
    render_pause = eval_cfg.get("render_pause", 0.05)
    trail_steps = eval_cfg.get("render_trail", 60)
    save_outputs = eval_cfg.get("save_outputs", True)

    video_path = None
    if save_outputs and eval_cfg.get("save_animation", True):
        animated_dir = os.path.join(config["save_dir"], "animated")
        os.makedirs(animated_dir, exist_ok=True)
        video_path = os.path.join(animated_dir, "run_episode_0.mp4")

    renderer = None
    need_renderer = enable_live or video_path is not None
    if need_renderer:
        renderer = LiveRenderer(
            env,
            pause=render_pause if enable_live else 0.0,
            trail_steps=trail_steps,
            show=enable_live,
            record=video_path is not None,
            video_path=video_path,
            video_fps=eval_cfg.get("video_fps", 10),
        )

    try:
        uav_metrics, protector_metrics, target_metrics, average_targets, max_targets = run_epoch(
            config, pmi, env, num_steps, render_hook=renderer if renderer is not None else None)
    finally:
        if renderer is not None:
            renderer.close()

    return_value.save_epoch(uav_metrics, protector_metrics, target_metrics, average_targets, max_targets)

    if save_outputs:
        env.save_position(save_dir=config["save_dir"], epoch_i=0)
        env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()





