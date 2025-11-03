import random
import numpy as np
from math import cos, sin, sqrt, exp, pi, e, atan2, hypot
from typing import List, Tuple
from models.PMINet import PMINetwork
from agent.target import TARGET
from agent.protector import PROTECTOR
from scipy.special import softmax
from utils.data_util import clip_and_normalize


class UAV:
    def __init__(self, x0, y0, h0, a0, v_max, h_max, na, dc, dp, dt, use_global_info=False, x_max=None, y_max=None):
        """
        :param dt: float, 采样的时间间隔
        :param x0: float, 坐标
        :param y0: float, 坐标
        :param h0: float, 朝向
        :param a0: float, 初始动作值（角度改变量，连续值）
        :param v_max: float, 最大线速度
        :param h_max: float, 最大角速度
        :param na: int, 动作空间的维度（保留用于兼容性，但不再使用）
        :param dc: float, 与无人机交流的最大距离
        :param dp: float, 捕捉目标的最大距离
        :param use_global_info: bool, 是否使用全局信息（忽略距离限制）
        :param x_max: float, 地图x轴最大尺寸（用于全局信息归一化）
        :param y_max: float, 地图y轴最大尺寸（用于全局信息归一化）
        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.Na = na  # 保留用于兼容性，但动作现在是连续的

        # action: 现在存储连续的角度改变量（角速度）
        self.a = a0

        # the maximum communication distance and maximum perception distance
        self.dc = dc  # 需要保存dc，用于observe_uav
        self.dp = dp

        # time interval
        self.dt = dt
        
        # 是否使用全局信息（忽略观测范围限制）
        self.use_global_info = use_global_info
        # 地图尺寸（用于全局信息归一化）
        self.x_max = x_max if x_max is not None else dc * 10  # 默认值
        self.y_max = y_max if y_max is not None else dp * 10  # 默认值

        # set of local information
        # self.communication = []
        self.target_observation = []
        self.uav_communication = []

        # reward
        self.raw_reward = 0
        self.reward = 0
        
        # 朝向锁定状态（用于碰撞弹开效果）
        self.lock = 0  # 剩余锁定时间（单位：步数）
        self.captured_targets_count = 0  # 本步捕获的目标数量

    def __distance(self, target) -> float:
        """
        calculate the distance from uav to target
        :param target: class UAV or class TARGET
        :return: scalar
        """
        return sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    @staticmethod
    def distance(x1, y1, x2, y2) -> float:
        """
        calculate the distance from uav to target
        :param x2:
        :param y1:
        :param x1:
        :param y2:
        :return: scalar
        """
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def discrete_action(self, a_idx: int) -> float:
        """
        from the action space index to the real difference (保留用于兼容性)
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 即角度改变量
        """
        # from action space to the real world action
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, action: 'float') -> (float, float, float):
        """
        接收连续动作值，更新当前位置
        :param action: float, 角度改变量（角速度），范围应在 [-h_max, h_max]
        :return: (x, y, h) 更新后的位置和朝向
        """
        # 将动作限制在有效范围内
        a = np.clip(action, -self.h_max, self.h_max)
        self.a = a

        dx = self.dt * self.v_max * cos(self.h)  # x 方向位移
        dy = self.dt * self.v_max * sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        
        # 如果处于朝向锁定状态，忽略动作的朝向改变
        if self.lock > 0:
            self.lock -= 1  # 减少锁定时间
            # 锁定期间不更新朝向，保持当前朝向继续运动
        else:
            # 正常更新朝向角度
            self.h += self.dt * a
        
        self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内

        return self.x, self.y, self.h  # 返回agent的位置和朝向(heading/theta)
    
    def apply_knockback(self, knockback_angle: float, lock_duration: int = 0):
        """
        应用弹开效果：改变朝向并锁定一段时间
        :param knockback_angle: float, 弹开的角度（弧度）
        :param lock_duration: int, 锁定持续时间（步数），锁定期间无法改变朝向
        """
        self.h = knockback_angle
        self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内
        self.lock = max(self.lock, lock_duration)  # 已锁定的情况下碰到新的protector

    def observe_target(self, targets_list: List['TARGET'], relative=True):
        """
        Observing targets with optional global information
        :param relative: relative to uav itself
        :param targets_list: [class TARGET]
        :return: None
        """
        self.target_observation = []  # Reset observed targets
        # 已捕获目标不再被观测
        # 根据 use_global_info 选择归一化因子
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dp
        
        for target in targets_list:
            if getattr(target, 'captured', False):
                continue
            
            # 根据 use_global_info 决定是否使用距离限制
            if not self.use_global_info:
                dist = self.__distance(target)
                if dist > self.dp:
                    continue  # 只观测范围内的目标
            
            # 使用适当的归一化因子（全局信息用地图尺寸，局部信息用dp）
            if relative:
                self.target_observation.append(((target.x - self.x) / norm_factor,
                                                (target.y - self.y) / norm_factor,
                                                cos(target.h) * target.v_max / self.v_max - cos(self.h),
                                                sin(target.h) * target.v_max / self.v_max - sin(self.h)))
            else:
                self.target_observation.append((target.x / norm_factor,
                                                target.y / norm_factor,
                                                cos(target.h) * target.v_max / self.v_max,
                                                sin(target.h) * target.v_max / self.v_max))
                    
    def observe_protector(self, protectors_list: List['PROTECTOR'], relative=True):
        """
        Observing protectors with optional global information
        :param relative: relative to uav itself
        :param protectors_list: [class PROTECTOR]
        :return: None
        """
        self.protector_observation = []  # Reset observed protectors
        # 根据 use_global_info 选择归一化因子
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dp
        
        for protector in protectors_list:
            # 根据 use_global_info 决定是否使用距离限制
            if not self.use_global_info:
                dist = self.__distance(protector)
                if dist > self.dp:
                    continue  # 只观测范围内的protector
            
            # 使用适当的归一化因子（全局信息用地图尺寸，局部信息用dp）
            if relative:
                self.protector_observation.append(((protector.x - self.x) / norm_factor,
                                                    (protector.y - self.y) / norm_factor,
                                                    cos(protector.h) * protector.v_max / self.v_max - cos(self.h),
                                                    sin(protector.h) * protector.v_max / self.v_max - sin(self.h)))
            else:
                self.protector_observation.append((protector.x / norm_factor,
                                                    protector.y / norm_factor,
                                                    cos(protector.h) * protector.v_max / self.v_max,
                                                    sin(protector.h) * protector.v_max / self.v_max))

    def observe_uav(self, uav_list: List['UAV'], relative=True):  # communication
        """
        Observe other UAVs with optional global information
        :param relative: relative to uav itself
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_communication = []  # Reset observed UAVs
        # 根据 use_global_info 选择归一化因子
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dc
        
        for uav in uav_list:
            if uav != self:
                # 根据 use_global_info 决定是否使用距离限制
                if not self.use_global_info:
                    dist = self.__distance(uav)
                    if dist > self.dc:
                        continue  # 只观测范围内的UAV
                
                # 使用适当的归一化因子（全局信息用地图尺寸，局部信息用dc）
                # add (x, y, vx, vy, a) information
                # 将动作值归一化到 [-1, 1] 范围
                uav_normalized_action = uav.a / uav.h_max if uav.h_max > 0 else 0.0
                self_normalized_action = self.a / self.h_max if self.h_max > 0 else 0.0
                if relative:
                    self.uav_communication.append(((uav.x - self.x) / norm_factor,
                                                   (uav.y - self.y) / norm_factor,
                                                   cos(uav.h) - cos(self.h),
                                                   sin(uav.h) - sin(self.h),
                                                   uav_normalized_action - self_normalized_action))
                else:
                    self.uav_communication.append((uav.x / norm_factor,
                                                   uav.y / norm_factor,
                                                   cos(uav.h),
                                                   sin(uav.h),
                                                   uav_normalized_action))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, float]],
                                        List[Tuple[float, float, float, float]], Tuple[float, float, float]):
        """
        :return: [(x, y, vx, vy, a),...] for uav, [(x, y, vx, vy)] for targets, (x, y, a) for itself
        注意：现在 a 是连续的动作值，归一化到 [-1, 1] 范围
        """
        # 将动作值归一化到 [-1, 1] 范围（基于 h_max）
        normalized_action = self.a / self.h_max if self.h_max > 0 else 0.0
        # 根据 use_global_info 选择归一化因子（自身位置归一化）
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dc
        return self.uav_communication, self.target_observation, (self.x / norm_factor, self.y / norm_factor, normalized_action)

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray: (12)
        """
        communication, observation, sb = self.__get_all_local_state()
        # 根据 use_global_info 选择归一化因子（用于距离计算）
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else max(self.dc, self.dp)

        if communication:
            d_communication = []  # store the distance from each uav to itself
            for x, y, vx, vy, na in communication:
                # x, y 已经是归一化后的相对坐标，需要乘以norm_factor还原距离
                if self.use_global_info:
                    # 在全局信息模式下，使用归一化坐标直接计算相对距离
                    d_communication.append(min(hypot(x * norm_factor, y * norm_factor) / norm_factor, 1))
                else:
                    # 在局部信息模式下，使用原始距离计算
                    d_communication.append(min(hypot(x * self.dc, y * self.dc) / max(self.dc, self.dp), 1))

            # regularization by the distance
            communication = np.array(communication)
            d_communication = np.array(d_communication)
            d_communication = np.clip(d_communication, 1e-6, 1)  # 避免除零
            communication_weighted = communication / d_communication[:, np.newaxis]
            average_communication = np.mean(communication_weighted, axis=0)
        else:
            # average_communication = np.zeros(4 + self.Na)  # empty communication
            average_communication = -np.ones(4 + 1)  # empty communication (x, y, vx, vy, a)

        if observation:
            d_observation = []  # store the distance from each target to itself
            for x, y, vx, vy in observation:
                # x, y 已经是归一化后的相对坐标
                if self.use_global_info:
                    d_observation.append(min(hypot(x * norm_factor, y * norm_factor) / norm_factor, 1))
                else:
                    d_observation.append(min(hypot(x * self.dp, y * self.dp) / max(self.dc, self.dp), 1))

            # regularization by the distance
            observation = np.array(observation)
            d_observation = np.array(d_observation)
            d_observation = np.clip(d_observation, 1e-6, 1)  # 避免除零
            observation_weighted = observation / d_observation[:, np.newaxis]
            average_observation = np.mean(observation_weighted, axis=0)
        else:
            average_observation = -np.ones(4)  # empty observation

        sb = np.array(sb)
        result = np.hstack((average_communication, average_observation, sb))
        return result

    def get_local_state(self) -> 'np.ndarray':
        """
        :return: np.ndarray
        """
        # using weighted mean method:
        return self.__get_local_state_by_weighted_mean()

    def __calculate_multi_target_tracking_reward(self, target_list) -> float:
        """
        Calculates multi-target tracking reward.
        Modified to give a very small reward for being close to targets,
        considering a larger tracking radius than `self.dp` and limiting
        the reward to only the closest few targets to prevent reward exploitation.
        :return: scalar [0, small_reward_factor * max_rewarded_targets]
        """
        track_reward = 0

        # Define parameters for tracking reward
        # The tracking radius is set to be larger than self.dp (perception/capture range).
        # This allows for a reward for being in the vicinity, even if not within capture range.
        tracking_radius = 2.5 * self.dp  # Example: 2.5 times the perception distance

        # Limit the number of closest targets for which a tracking reward is given.
        # This prevents a UAV from accumulating excessive rewards by being in the middle of many targets.
        max_rewarded_targets = 3  # Reward only the 3 closest targets within tracking_radius

        # Define a small factor for the reward magnitude.
        # This ensures the tracking reward remains small compared to capture rewards.
        small_reward_factor = 0.1

        # Calculate distances to all targets and filter those within the tracking radius
        distances_to_targets = []
        for target in target_list:
            distance = self.__distance(target)
            if distance <= tracking_radius:
                distances_to_targets.append((distance, target))

        # Sort targets by distance (closest first)
        distances_to_targets.sort(key=lambda x: x[0])

        # Calculate reward only for the closest 'max_rewarded_targets'
        for i, (distance, _) in enumerate(distances_to_targets):
            if i >= max_rewarded_targets:
                break  # Stop after processing the specified number of targets

            # Reward is scaled down and decreases linearly with distance.
            # Max reward (small_reward_factor) is given at distance 0, and 0 reward at tracking_radius.
            reward = small_reward_factor * (tracking_radius - distance) / tracking_radius
            track_reward += reward

        return track_reward

    def __calculate_target_capture_reward(self) -> float:
        """
        calculate target capture reward
        :return: scalar [0, m_targets]
        """
        capture_reward = self.captured_targets_count
        self.captured_targets_count = 0
        return capture_reward

    def __calculate_duplicate_tracking_punishment(self, uav_list: List['UAV'], radio=2) -> float:
        """
        calculate duplicate tracking punishment
        :param uav_list: [class UAV]
        :param radio: radio用来控制惩罚的范围, 超出多远才算入惩罚
        :return: scalar (-e/2, -1/2]
        """
        total_punishment = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * exp((radio * self.dp - distance) / (radio * self.dp))
                    # total_punishment += clip_and_normalize(punishment, -e/2, -1/2, -1)
                    total_punishment += punishment  # 没有clip, 在调用时外部clip
        return total_punishment

    def __calculate_boundary_punishment(self, x_max: float, y_max: float) -> float:
        """
        :param x_max: border of the map at x-axis, scalar
        :param y_max: border of the map at y-axis, scalar
        :return:
        """
        x_to_0 = self.x - 0
        x_to_max = x_max - self.x
        y_to_0 = self.y - 0
        y_to_max = y_max - self.y
        d_bdr = min(x_to_0, x_to_max, y_to_0, y_to_max)
        if 0 <= self.x <= x_max and 0 <= self.y <= y_max:
            if d_bdr < self.dp:
                boundary_punishment = -0.5 * (self.dp - d_bdr) / self.dp
            else:
                boundary_punishment = 0
        else:
            boundary_punishment = -1/2
        return boundary_punishment  # 没有clip, 在调用时外部clip
        # return clip_and_normalize(boundary_punishment, -1/2, 0, -1)

    def calculate_raw_reward(self, uav_list: List['UAV'], target__list: List['TARGET'], protector_list: List['PROTECTOR'], x_max, y_max):
        """
        calculate three parts of the reward/punishment for this uav
        :return: float, float, float
        """
        reward = self.__calculate_multi_target_tracking_reward(target__list)
        boundary_punishment = self.__calculate_boundary_punishment(x_max, y_max)
        punishment = self.__calculate_duplicate_tracking_punishment(uav_list)
        protector_punishment = self.__calculate_protector_collision_punishment(protector_list)
        return reward, boundary_punishment, punishment, protector_punishment

    def __calculate_protector_collision_punishment(self, protector_list: List['PROTECTOR']) -> float:
        """
        简单的距离阈值惩罚：
        - 若 UAV 与任意 Protector 距离 < threshold 则产生惩罚（负值）
        - 返回值取区间 [-K, 0]（未归一化），后续由 clip_and_normalize 归一化到 [-1,0]
        """
        # UAV 自身半径，可配置或默认
        uav_radius = getattr(self, "radius", 0.5)
        worst_pen = 0.0  # 惩罚为非正数（0 表示无惩罚）
        for prot in protector_list:
            # protector 里约定属性 safe_r（或使用 prot.safe_r）
            prot_safe = getattr(prot, "safe_r", None)
            if prot_safe is None:
                # 用默认或 prot.radius
                prot_safe = getattr(prot, "radius", 0.5)
            threshold = prot_safe + uav_radius  # 视为碰撞/警戒半径
            dx = prot.x - self.x
            dy = prot.y - self.y
            dist = hypot(dx, dy)
            if dist < threshold:
                # 惩罚按距离成比例：靠得越近惩罚越大（负）
                # 未归一化惩罚，例如： - (threshold - dist)  -> 介于 (-threshold, 0]
                pen = -(threshold - dist)
                if pen < worst_pen:
                    worst_pen = pen
        return worst_pen

    def __calculate_cooperative_reward_by_pmi(self, uav_list: List['UAV'], protector_list: List['PROTECTOR'],
                                               pmi_net: "PMINetwork", a_cooperation=0.5, a_adversarial=0.0) -> float:
        """
        使用PMI网络计算合作奖励和对抗奖励
        :param pmi_net: PMI网络（总是输出两个值：合作和对抗PMI）
        :param uav_list: [class UAV] 友方列表
        :param protector_list: [class PROTECTOR] 敌方列表
        :param a_cooperation: float, 友方合作的比例（自私 vs 共享）
        :param a_adversarial: float, 对抗敌方的比例（自身奖励 vs 惩罚敌方）
        :return:
        """
        from scipy.special import softmax
        
        la = self.get_local_state()
        base_reward = self.raw_reward
        
        # 1. 友方合作奖励（与其他UAV共享奖励）
        if a_cooperation > 0:
            neighbor_uav_rewards = []
            neighbor_cooperation_dependencies = []
            
            for other_uav in uav_list:
                if other_uav != self and self.__distance(other_uav) <= self.dp:
                    neighbor_uav_rewards.append(other_uav.raw_reward)
                    other_uav_la = other_uav.get_local_state()
                    _input = la * other_uav_la
                    
                    # 使用合作PMI
                    cooperation_pmi = pmi_net.inference(_input.squeeze(), relation_type='cooperation')
                    neighbor_cooperation_dependencies.append(cooperation_pmi)
            
            if len(neighbor_uav_rewards):
                neighbor_uav_rewards = np.array(neighbor_uav_rewards)
                neighbor_cooperation_dependencies = np.array(neighbor_cooperation_dependencies).astype(np.float32)
                cooperation_weights = softmax(neighbor_cooperation_dependencies)
                cooperation_reward = a_cooperation * np.sum(neighbor_uav_rewards * cooperation_weights).item()
            else:
                cooperation_reward = 0.0
        else:
            cooperation_reward = 0.0
        
        # 2. 敌方对抗奖励（基于对抗PMI，考虑protector作为敌方）
        if a_adversarial > 0:
            protector_rewards = []
            adversarial_dependencies = []
            
            for protector in protector_list:
                dist_to_protector = self.__distance(protector)
                if dist_to_protector <= self.dp:  # 在观测范围内
                    protector_la = protector.get_local_state()
                    _input = la * protector_la
                    
                    # 使用对抗PMI
                    adversarial_pmi = pmi_net.inference(_input.squeeze(), relation_type='adversarial')
                    adversarial_dependencies.append(adversarial_pmi)
                    
                    # 对抗奖励：敌方（protector）奖励越高，我们越要对抗（负相关）
                    # 对抗PMI越高，表示对抗越激烈，我们应该获得更多奖励（因为成功对抗了敌方）
                    protector_rewards.append(-protector.raw_reward)  # 敌方的负奖励作为我们的奖励
            
            if len(protector_rewards):
                protector_rewards = np.array(protector_rewards)
                adversarial_dependencies = np.array(adversarial_dependencies).astype(np.float32)
                adversarial_weights = softmax(adversarial_dependencies)
                # 对抗奖励：当对抗PMI高时，从敌方损失中获得奖励
                adversarial_reward = a_adversarial * np.sum(protector_rewards * adversarial_weights).item()
            else:
                adversarial_reward = 0.0
        else:
            adversarial_reward = 0.0
        
        # 综合奖励：自身奖励 + 友方合作奖励 + 敌方对抗奖励
        final_reward = (1 - a_cooperation - a_adversarial) * base_reward + cooperation_reward + adversarial_reward
        
        return final_reward

    def __calculate_cooperative_reward_by_mean(self, uav_list: List['UAV'], a) -> float:
        """
        calculate cooperative reward by mean
        :param uav_list: [class UAV]
        :param a: float, proportion of selfish and sharing
        :return:
        """
        if a == 0:  # 提前判断，节省计算的复杂度
            return self.raw_reward

        neighbor_rewards = []
        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
        # 没有加入PMI网络
        reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards) \
            if len(neighbor_rewards) else 0
        return reward

    def calculate_cooperative_reward(self, uav_list: List['UAV'], protector_list: List['PROTECTOR'] = None,
                                     pmi_net=None, a_cooperation=0.5, a_adversarial=0.0) -> float:
        """
        计算合作奖励（包括友方合作和敌方对抗）
        :param uav_list: UAV列表（友方）
        :param protector_list: Protector列表（敌方），可选
        :param pmi_net: PMI网络（可选）
        :param a_cooperation: 友方合作比例（0: 完全自私, 1: 完全共享）
        :param a_adversarial: 敌方对抗比例（0: 不考虑敌方, >0: 对抗敌方）
        :return:
        """
        if protector_list is None:
            protector_list = []
            
        if pmi_net:
            return self.__calculate_cooperative_reward_by_pmi(uav_list, protector_list, pmi_net, 
                                                              a_cooperation, a_adversarial)
        else:
            # 如果没有PMI网络，只使用均值方法进行友方合作
            return self.__calculate_cooperative_reward_by_mean(uav_list, a_cooperation)

    def get_action_by_direction(self, target_list, uav_list):
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # 奖励和惩罚权重
        target_reward_weight = 1.0
        repetition_penalty_weight = 0.8
        self.epsilon = 0.25
        self.continue_tracing = 0.3
        
        best_score = float('-inf')
        best_angle = 0.0

        # 随机扰动：以epsilon的概率选择随机动作
        if random.random() < self.epsilon:
            return np.random.uniform(-self.h_max, self.h_max)
        else:
            for target in target_list:
                target_x, target_y = target.x, target.y

                # 当前无人机到目标的距离
                dist_to_target = distance(self.x, self.y, target_x, target_y)

                # 重复追踪的惩罚，考虑其他无人机在重复追踪半径内是否在追踪同一目标
                repetition_penalty = 0.0
                for uav in uav_list:
                    uav_x, uav_y = uav.x, uav.y
                    if (uav_x, uav_y) != (self.x, self.y):
                        dist_to_target_from_other_uav = distance(uav_x, uav_y, target_x, target_y)
                        if dist_to_target_from_other_uav < self.dc:
                            repetition_penalty += repetition_penalty_weight

                # 计算当前目标的得分
                score = target_reward_weight / dist_to_target - repetition_penalty

                # 根据得分选择最优目标
                if score > best_score:
                    best_score = score
                    best_angle = np.arctan2(target_y - self.y, target_x - self.x) - self.h

        # 以continue_tracing的概率保持上一个动作
        if random.random() < self.continue_tracing:
            best_angle = 0
            
        # 将期望的角度差转换为连续动作值（角速度）
        # best_angle 是期望的角度差，需要转换为角速度
        a = np.clip(best_angle / self.dt, -self.h_max, self.h_max)
        return a