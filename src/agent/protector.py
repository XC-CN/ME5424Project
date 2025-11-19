import random
import numpy as np
from math import pi, sin, cos, sqrt, exp, hypot
from typing import List, Tuple


class PROTECTOR:
    def __init__(self, x0: float, y0: float, h0: float, a0: float, v_max: float, h_max: float, dt, safe_radius: float, dc: float, dp: float, use_global_info=False, x_max=None, y_max=None):
        """
        :param x0: float, 初始 x 坐标
        :param y0: float, 初始 y 坐标
        :param h0: float, 初始朝向
        :param a0: float, 初始动作值（角度改变量，连续值）
        :param v_max: float, 最大线速度
        :param h_max: float, 最大角速度
        :param dt: float, 时间间隔
        :param safe_radius: float, 安全半径（手臂长度）
        :param dc: float, 与其他保护者交流的最大距离
        :param dp: float, 观测 UAV 和目标的最大距离
        :param use_global_info: bool, 是否使用全局信息（忽略距离限制）
        :param x_max: float, 地图x轴最大尺寸（用于全局信息归一化）
        :param y_max: float, 地图y轴最大尺寸（用于全局信息归一化）
        """
        self.x, self.y = x0, y0
        self.h = h0
        self.v_max = v_max
        self.h_max = h_max
        self.dt = dt
        self.safe_r = safe_radius
        
        # 动作：现在存储连续的角度改变量（角速度）
        self.a = a0
        # Na 用于离散动作转换（从配置中获取，需要外部设置）
        self.Na = None  # 将在环境初始化时设置
        
        # 通信和观测距离
        self.dc = dc
        self.dp = dp
        
        # 是否使用全局信息（忽略观测范围限制）
        self.use_global_info = use_global_info
        # 地图尺寸（用于全局信息归一化）
        self.x_max = x_max if x_max is not None else dc * 10  # 默认值
        self.y_max = y_max if y_max is not None else dp * 10  # 默认值
        
        # 观测信息
        self.target_observation = []
        self.uav_observation = []
        self.protector_observation = []  # 使用protector_observation而不是protector_communication
        
        # 奖励
        self.raw_reward = 0
        self.reward = 0

    def discrete_action(self, a_idx: int) -> float:
        """
        from the action space index to the real difference (保留用于兼容性)
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 即角度改变量
        """
        if self.Na is None or self.Na <= 1:
            # 如果没有设置 Na 或无效值，返回随机动作
            return random.uniform(-self.h_max, self.h_max)
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
        self.h += self.dt * a
        self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内

        return self.x, self.y, self.h  # 返回protector的位置和朝向(heading/theta)

    def clamp_inside(self, x_max, y_max):
        """边界夹紧 + 调头"""
        if self.x < 0:
            self.x = 0
            self.h = pi - self.h
        if self.x > x_max:
            self.x = x_max
            self.h = pi - self.h
        if self.y < 0:
            self.y = 0
            self.h = -self.h
        if self.y > y_max:
            self.y = y_max
            self.h = -self.h

    def __distance(self, obj) -> float:
        """
        计算到目标对象的距离
        :param obj: class UAV, TARGET 或 PROTECTOR
        :return: 距离值
        """
        return sqrt((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2)

    def observe_target(self, targets_list: List['TARGET'], relative=True):
        """
        Observing targets with optional global information
        :param relative: relative to protector itself
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
        :param relative: relative to protector itself
        :param protectors_list: [class PROTECTOR]
        :return: None
        """
        self.protector_observation = []  # Reset observed protectors
        # 根据 use_global_info 选择归一化因子
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dp
        
        for protector in protectors_list:
            if protector != self:
                # 根据 use_global_info 决定是否使用距离限制
                if not self.use_global_info:
                    dist = self.__distance(protector)
                    if dist > self.dp:
                        continue  # 只观测范围内的protector
                
                # 使用适当的归一化因子（全局信息用地图尺寸，局部信息用dp）
                # add (x, y, vx, vy) information
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

    def observe_uav(self, uav_list: List['UAV'], relative=True):  # observation
        """
        Observing UAVs with optional global information
        :param relative: relative to protector itself
        :param uav_list: [class UAV]
        :return: None
        """
        self.uav_observation = []  # Reset observed UAVs
        # 根据 use_global_info 选择归一化因子
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dp
        
        for uav in uav_list:
            # 根据 use_global_info 决定是否使用距离限制
            if not self.use_global_info:
                dist = self.__distance(uav)
                if dist > self.dp:
                    continue  # 只观测范围内的UAV
            
            # 使用适当的归一化因子（全局信息用地图尺寸，局部信息用dp）
            # add (x, y, vx, vy) information
            if relative:
                self.uav_observation.append(((uav.x - self.x) / norm_factor,
                                                (uav.y - self.y) / norm_factor,
                                                cos(uav.h) * uav.v_max / self.v_max - cos(self.h),
                                                sin(uav.h) * uav.v_max / self.v_max - sin(self.h)))
            else:
                self.uav_observation.append((uav.x / norm_factor,
                                                uav.y / norm_factor,
                                                cos(uav.h) * uav.v_max / self.v_max,
                                                sin(uav.h) * uav.v_max / self.v_max))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float]],
                                        List[Tuple[float, float, float, float]], 
                                        List[Tuple[float, float, float, float]],
                                        Tuple[float, float, float]):
        """
        :return: [(x, y, vx, vy),...] for uav, [(x, y, vx, vy)] for targets, [(x, y, vx, vy)] for protectors, (x, y, a) for itself
        注意：现在 a 是连续的动作值，归一化到 [-1, 1] 范围
        """
        # 将动作值归一化到 [-1, 1] 范围（基于 h_max）
        normalized_action = self.a / self.h_max if self.h_max > 0 else 0.0
        # 根据 use_global_info 选择归一化因子（自身位置归一化）
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dc
        return self.uav_observation, self.target_observation, self.protector_observation, (self.x / norm_factor, self.y / norm_factor, normalized_action)

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray
        """
        uav_obs, target_obs, prot_obs, sb = self.__get_all_local_state()
        # 根据 use_global_info 选择归一化因子（用于距离计算）
        norm_factor = max(self.x_max, self.y_max) if self.use_global_info else self.dp

        # 处理UAV观测
        if uav_obs:
            d_uav = []
            for x, y, vx, vy in uav_obs:
                # x, y 已经是归一化后的相对坐标
                if self.use_global_info:
                    d_uav.append(min(hypot(x * norm_factor, y * norm_factor) / norm_factor, 1))
                else:
                    d_uav.append(min(hypot(x * self.dp, y * self.dp) / self.dp, 1))
            uav_obs = np.array(uav_obs)
            d_uav = np.array(d_uav)
            d_uav = np.clip(d_uav, 1e-6, 1)  # 避免除零
            uav_weighted = uav_obs / d_uav[:, np.newaxis]
            average_uav = np.mean(uav_weighted, axis=0)
        else:
            average_uav = -np.ones(4)  # empty UAV observation (x, y, vx, vy)

        # 处理目标观测
        if target_obs:
            d_target = []
            for x, y, vx, vy in target_obs:
                # x, y 已经是归一化后的相对坐标
                if self.use_global_info:
                    d_target.append(min(hypot(x * norm_factor, y * norm_factor) / norm_factor, 1))
                else:
                    d_target.append(min(hypot(x * self.dp, y * self.dp) / self.dp, 1))
            target_obs = np.array(target_obs)
            d_target = np.array(d_target)
            d_target = np.clip(d_target, 1e-6, 1)  # 避免除零
            target_weighted = target_obs / d_target[:, np.newaxis]
            average_target = np.mean(target_weighted, axis=0)
        else:
            average_target = -np.ones(4)  # empty target observation

        # 处理保护者观测
        if prot_obs:
            d_prot = []
            for x, y, vx, vy in prot_obs:
                # x, y 已经是归一化后的相对坐标
                if self.use_global_info:
                    d_prot.append(min(hypot(x * norm_factor, y * norm_factor) / norm_factor, 1))
                else:
                    d_prot.append(min(hypot(x * self.dp, y * self.dp) / self.dp, 1))
            prot_obs = np.array(prot_obs)
            d_prot = np.array(d_prot)
            d_prot = np.clip(d_prot, 1e-6, 1)  # 避免除零
            prot_weighted = prot_obs / d_prot[:, np.newaxis]
            average_prot = np.mean(prot_weighted, axis=0)
        else:
            average_prot = -np.ones(4)  # empty protector observation

        sb = np.array(sb)
        result = np.hstack((average_uav, average_target, average_prot, sb))
        return result

    def get_local_state(self) -> 'np.ndarray':
        """
        :return: np.ndarray
        """
        # using weighted mean method:
        return self.__get_local_state_by_weighted_mean()

    def __calculate_target_protection_reward(self, target_list: List['TARGET'], uav_list: List['UAV']) -> float:
        """
        计算保护目标的奖励：当protector能够阻挡UAV接近目标时给予奖励
        改进：增加接近目标的基础奖励，解决奖励稀疏问题
        :param target_list: 目标列表
        :param uav_list: UAV列表
        :return: 保护奖励 [0, reward_factor * protected_targets]
        """
        protection_reward = 0.0
        
        # 保护半径：protector的保护范围（通常大于观测距离）
        protection_radius = 2.0 * self.safe_r  # 使用安全半径的倍数作为保护范围
        # 扩展接近奖励半径：即使不在保护范围内，靠近目标也应该有小奖励
        proximity_reward_radius = 4.0 * self.safe_r  # 更大的范围给予基础奖励
        
        # 奖励因子：每保护一个目标的基础奖励
        reward_factor = 1.0
        # 接近奖励因子：即使不在保护范围内，靠近目标也有小奖励
        proximity_reward_factor = 0.3  # 较小的基础奖励，鼓励接近目标
        
        # 最大奖励目标数量：避免奖励过度集中在少数目标上
        max_rewarded_targets = 5
        
        # 对每个未被捕获的目标计算保护奖励
        protected_scores = []
        for target in target_list:
            if getattr(target, 'captured', False):
                continue
            
            # protector到目标的距离
            dist_prot_to_target = self.__distance(target)
            
            # 基础接近奖励：即使不在保护范围内，靠近目标也有奖励（解决稀疏问题）
            proximity_base_reward = 0.0
            if dist_prot_to_target <= proximity_reward_radius:
                # 距离越近，奖励越高（线性衰减）
                proximity_base_reward = proximity_reward_factor * (1.0 - dist_prot_to_target / proximity_reward_radius)
            
            # 如果protector在保护范围内，计算阻挡效果
            blocking_score = 0.0
            if dist_prot_to_target <= protection_radius:
                for uav in uav_list:
                    # UAV到目标的距离
                    dist_uav_to_target = sqrt((uav.x - target.x) ** 2 + (uav.y - target.y) ** 2)
                    
                    # protector到UAV的距离
                    dist_prot_to_uav = self.__distance(uav)
                    
                    # 如果UAV正在接近目标（距离小于某个阈值）
                    if dist_uav_to_target < 3.0 * protection_radius:
                        # 计算protector是否在UAV和目标之间（阻挡效果）
                        # 使用向量投影判断protector是否在UAV到目标的路径上
                        uav_to_target_dx = target.x - uav.x
                        uav_to_target_dy = target.y - uav.y
                        uav_to_target_dist = sqrt(uav_to_target_dx ** 2 + uav_to_target_dy ** 2)
                        
                        if uav_to_target_dist > 1e-6:
                            # 计算protector到UAV-目标连线的距离
                            prot_to_uav_dx = uav.x - self.x
                            prot_to_uav_dy = uav.y - self.y
                            
                            # 投影到UAV-目标方向
                            proj = (prot_to_uav_dx * uav_to_target_dx + prot_to_uav_dy * uav_to_target_dy) / uav_to_target_dist
                            
                            # 如果protector在UAV和目标之间（0 < proj < dist）
                            if 0 < proj < uav_to_target_dist:
                                # 计算到连线的垂直距离
                                perp_dist = abs(prot_to_uav_dx * uav_to_target_dy - prot_to_uav_dy * uav_to_target_dx) / uav_to_target_dist
                                
                                # 如果protector的safe_r能够覆盖到这条路径
                                if perp_dist < self.safe_r:
                                    # 阻挡效果：距离越近，阻挡效果越好
                                    blocking_effect = (self.safe_r - perp_dist) / self.safe_r
                                    # 根据UAV到目标的距离，越近的UAV需要越强的阻挡
                                    urgency = (3.0 * protection_radius - dist_uav_to_target) / (3.0 * protection_radius)
                                    blocking_score += blocking_effect * urgency
                
                # 综合奖励：protector靠近目标 + 阻挡UAV
                proximity_bonus = (protection_radius - dist_prot_to_target) / protection_radius
                blocking_reward = proximity_bonus * 0.3 + blocking_score * 0.7
            else:
                # 不在保护范围内，只有基础接近奖励
                blocking_reward = 0.0
            
            # 总奖励 = 基础接近奖励 + 阻挡奖励
            total_score = proximity_base_reward + blocking_reward
            protected_scores.append((total_score, dist_prot_to_target))
        
        # 按得分排序，只奖励前max_rewarded_targets个目标
        protected_scores.sort(key=lambda x: x[0], reverse=True)
        for i, (score, _) in enumerate(protected_scores):
            if i >= max_rewarded_targets:
                break
            protection_reward += reward_factor * score
        
        return protection_reward

    def __calculate_overlapping_protector_punishment(self, protector_list: List['PROTECTOR'], radio=1.5) -> float:
        """
        计算重叠保护者惩罚：如果太多保护者聚集在一起，给予轻微惩罚
        改进：弱化惩罚，因为protector需要合作，过度聚集的惩罚可能过于严格
        :param protector_list: [class PROTECTOR]
        :param radio: 控制惩罚范围的系数
        :return: 惩罚值 (-radio * punishment_factor, 0]
        """
        total_punishment = 0.0
        punishment_factor = 0.15  # 降低惩罚强度（从0.3降到0.15），鼓励合作但避免过度聚集
        
        for other_prot in protector_list:
            if other_prot != self:
                distance = self.__distance(other_prot)
                # 只在非常接近时才给予惩罚（缩小惩罚范围）
                if distance <= 0.8 * radio * self.safe_r:  # 从radio降到0.8*radio
                    # 惩罚按距离递减：靠得越近惩罚越大
                    punishment = -punishment_factor * exp((0.8 * radio * self.safe_r - distance) / (0.8 * radio * self.safe_r))
                    total_punishment += punishment
        
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

    def calculate_raw_reward(self, uav_list: List['UAV'], target_list: List['TARGET'], protector_list: List['PROTECTOR'], x_max, y_max):
        """
        计算protector的原始奖励/惩罚
        :param uav_list: UAV列表
        :param target_list: 目标列表
        :param protector_list: 保护者列表
        :param x_max: 地图x边界
        :param y_max: 地图y边界
        :return: (保护奖励, 边界惩罚, 重叠惩罚, 拦截奖励)
        """
        # 保护目标奖励：protector能够阻挡UAV接近目标
        protection_reward = self.__calculate_target_protection_reward(target_list, uav_list)
        # 边界惩罚：保持与UAV相同
        boundary_punishment = self.__calculate_boundary_punishment(x_max, y_max)
        # 重叠保护者惩罚：避免过度聚集
        overlapping_punishment = self.__calculate_overlapping_protector_punishment(protector_list)
        # 成功拦截奖励：protector成功拦截UAV并使其远离目标
        interception_reward = self.__calculate_successful_interception_reward(uav_list, target_list)
        
        return protection_reward, boundary_punishment, overlapping_punishment, interception_reward
    
    def __calculate_successful_interception_reward(self, uav_list: List['UAV'], target_list: List['TARGET']) -> float:
        """
        计算成功拦截奖励：当protector成功拦截UAV，且拦截使其远离目标时给予奖励
        改进：增加接近UAV的基础奖励，即使未成功拦截，接近威胁UAV也有奖励（解决稀疏问题）
        :param uav_list: UAV列表
        :param target_list: 目标列表
        :return: 拦截奖励 [0, reward_factor * intercepted_uavs]
        """
        interception_reward = 0.0
        
        # 拦截检测范围：protector的拦截影响范围（基于safe_r）
        interception_range = 1.5 * self.safe_r  # 考虑手臂长度和碰撞范围
        # 接近威胁奖励范围：即使未成功拦截，接近威胁UAV也有奖励
        threat_proximity_range = 3.0 * self.safe_r  # 更大的范围给予基础奖励
        
        # 奖励因子
        reward_factor = 2.0  # 拦截奖励应该比较高，因为这是直接有效的防御
        # 接近威胁奖励因子：即使未成功拦截，接近威胁UAV也有小奖励
        threat_proximity_factor = 0.4  # 较小的基础奖励，鼓励接近威胁UAV
        
        # 对每个UAV检查是否被拦截或接近威胁
        for uav in uav_list:
            dist_to_uav = self.__distance(uav)
            
            # 找到最近的未被捕获的目标
            closest_target = None
            min_dist = float('inf')
            for target in target_list:
                if getattr(target, 'captured', False):
                    continue
                dist = sqrt((uav.x - target.x) ** 2 + (uav.y - target.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_target = target
            
            if closest_target:
                # 计算UAV到目标的距离
                uav_to_target_dist = min_dist
                
                # 基础接近威胁奖励：即使未成功拦截，接近威胁UAV也有奖励（解决稀疏问题）
                if dist_to_uav <= threat_proximity_range and uav_to_target_dist < 5.0 * self.safe_r:
                    # UAV正在接近目标，protector接近UAV应该得到奖励
                    # 距离越近，奖励越高（线性衰减）
                    proximity_reward = threat_proximity_factor * (1.0 - dist_to_uav / threat_proximity_range)
                    # 根据UAV到目标的距离调整：UAV越接近目标，protector接近UAV的奖励越高
                    urgency = (5.0 * self.safe_r - uav_to_target_dist) / (5.0 * self.safe_r)
                    proximity_reward *= max(0.0, urgency)  # 确保非负
                    interception_reward += proximity_reward
                
                # 检查UAV是否处于锁定状态（被拦截）
                if getattr(uav, 'lock', 0) > 0 and dist_to_uav <= interception_range:
                    # 检查拦截后的朝向是否使UAV远离目标
                    uav_heading = uav.h  # UAV当前朝向（拦截后的朝向）
                    
                    # 计算UAV朝向目标的理想方向
                    target_dx = closest_target.x - uav.x
                    target_dy = closest_target.y - uav.y
                    target_angle = np.arctan2(target_dy, target_dx)
                    
                    # 计算UAV当前朝向与目标方向的夹角
                    angle_diff = abs(uav_heading - target_angle)
                    # 归一化到 [0, pi]
                    if angle_diff > pi:
                        angle_diff = 2 * pi - angle_diff
                    
                    # 如果夹角大于 pi/2（90度），说明UAV朝向远离目标
                    if angle_diff > pi / 2:
                        # 夹角越大，远离效果越好，奖励越高
                        # 夹角从 pi/2 到 pi，奖励从 0 到 reward_factor
                        effectiveness = (angle_diff - pi / 2) / (pi / 2)  # [0, 1]
                        
                        # 根据距离目标的远近调整奖励（越近时拦截，奖励越高）
                        current_dist = sqrt(target_dx ** 2 + target_dy ** 2)
                        # 假设危险距离为 5 * safe_r，在此范围内拦截更有效
                        danger_zone = 5.0 * self.safe_r
                        if current_dist < danger_zone:
                            urgency_bonus = (danger_zone - current_dist) / danger_zone  # [0, 1]
                            reward = reward_factor * effectiveness * (1.0 + urgency_bonus)
                        else:
                            reward = reward_factor * effectiveness
                        
                        interception_reward += reward
        
        return interception_reward


    def __calculate_cooperative_reward_by_pmi(self, protector_list: List['PROTECTOR'], uav_list: List['UAV'], 
                                               pmi_net, a_cooperation=0.5, a_adversarial=0.3) -> float:
        """
        使用PMI网络计算合作奖励和对抗奖励
        :param pmi_net: PMI网络（支持双输出模式）
        :param protector_list: [class PROTECTOR] 友方列表
        :param uav_list: [class UAV] 敌方列表
        :param a_cooperation: float, 友方合作的比例（自私 vs 共享）
        :param a_adversarial: float, 对抗敌方的比例（自身奖励 vs 惩罚敌方）
        :return:
        """
        from scipy.special import softmax
        
        la = self.get_local_state()
        base_reward = self.raw_reward
        
        # 1. 友方合作奖励（与其他protector共享奖励）
        if a_cooperation > 0:
            neighbor_protector_rewards = []
            neighbor_cooperation_dependencies = []
            
            for other_prot in protector_list:
                if other_prot != self and self.__distance(other_prot) <= self.dc:
                    neighbor_protector_rewards.append(other_prot.raw_reward)
                    other_prot_la = other_prot.get_local_state()
                    _input = la * other_prot_la
                    
                    # 使用合作PMI
                    cooperation_pmi = pmi_net.inference(_input.squeeze(), relation_type='cooperation')
                    neighbor_cooperation_dependencies.append(cooperation_pmi)
            
            if len(neighbor_protector_rewards):
                neighbor_protector_rewards = np.array(neighbor_protector_rewards)
                neighbor_cooperation_dependencies = np.array(neighbor_cooperation_dependencies).astype(np.float32)
                cooperation_weights = softmax(neighbor_cooperation_dependencies)
                cooperation_reward = a_cooperation * np.sum(neighbor_protector_rewards * cooperation_weights).item()
            else:
                cooperation_reward = 0.0
        else:
            cooperation_reward = 0.0
        
        # 2. 敌方对抗奖励（基于对抗PMI，惩罚敌方或增强自身）
        if a_adversarial > 0:
            uav_rewards = []
            adversarial_dependencies = []
            
            for uav in uav_list:
                dist_to_uav = self.__distance(uav)
                if dist_to_uav <= self.dp:  # 在观测范围内
                    uav_la = uav.get_local_state()
                    _input = la * uav_la
                    
                    # 使用对抗PMI
                    adversarial_pmi = pmi_net.inference(_input.squeeze(), relation_type='adversarial')
                    adversarial_dependencies.append(adversarial_pmi)
                    
                    # 对抗奖励：敌方奖励越高，我们越要惩罚（负相关）
                    # 或者：对抗PMI越高，表示对抗越激烈，我们应该获得更多奖励
                    # 这里采用后者：对抗PMI高时，protector应该获得额外奖励
                    uav_rewards.append(-uav.raw_reward)  # 敌方的负奖励作为我们的奖励
            
            if len(uav_rewards):
                uav_rewards = np.array(uav_rewards)
                adversarial_dependencies = np.array(adversarial_dependencies).astype(np.float32)
                adversarial_weights = softmax(adversarial_dependencies)
                # 对抗奖励：当对抗PMI高时，从敌方损失中获得奖励
                adversarial_reward = a_adversarial * np.sum(uav_rewards * adversarial_weights).item()
            else:
                adversarial_reward = 0.0
        else:
            adversarial_reward = 0.0
        
        # 综合奖励：自身奖励 + 友方合作奖励 + 敌方对抗奖励
        final_reward = (1 - a_cooperation - a_adversarial) * base_reward + cooperation_reward + adversarial_reward
        
        return final_reward

    def __calculate_cooperative_reward_by_mean(self, protector_list: List['PROTECTOR'], a) -> float:
        """
        使用平均值计算合作奖励
        :param protector_list: [class PROTECTOR]
        :param a: float, 自私与共享的比例
        :return:
        """
        if a == 0:  # 提前判断，节省计算的复杂度
            return self.raw_reward

        neighbor_rewards = []
        for other_prot in protector_list:
            if other_prot != self and self.__distance(other_prot) <= self.dc:
                neighbor_rewards.append(other_prot.raw_reward)
        # 没有加入PMI网络
        reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards) \
            if len(neighbor_rewards) else (1 - a) * self.raw_reward
        return reward

    def calculate_cooperative_reward(self, protector_list: List['PROTECTOR'], uav_list: List['UAV'] = None,
                                     pmi_net=None, a_cooperation=0.5, a_adversarial=0.0) -> float:
        """
        计算合作奖励（包括友方合作和敌方对抗）
        :param protector_list: 保护者列表（友方）
        :param uav_list: UAV列表（敌方），可选
        :param pmi_net: PMI网络（可选，支持双输出模式）
        :param a_cooperation: 友方合作比例（0: 完全自私, 1: 完全共享）
        :param a_adversarial: 敌方对抗比例（0: 不考虑敌方, >0: 对抗敌方）
        :return:
        """
        if uav_list is None:
            uav_list = []
            
        if pmi_net:
            return self.__calculate_cooperative_reward_by_pmi(protector_list, uav_list, pmi_net, 
                                                             a_cooperation, a_adversarial)
        else:
            # 如果没有PMI网络，只使用均值方法进行友方合作
            return self.__calculate_cooperative_reward_by_mean(protector_list, a_cooperation)

    def get_action_by_direction(self, target_list, uav_list):
        """
        基于目标和UAV的位置计算动作（用于启发式策略，非强化学习）
        """
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # 保护者应该移动到能够阻挡UAV接近目标的位置
        # 奖励和惩罚权重
        protection_weight = 1.0
        self.epsilon = 0.25
        self.continue_tracing = 0.3
        
        best_score = float('-inf')
        best_angle = 0.0

        # 随机扰动：以epsilon的概率选择随机动作
        if random.random() < self.epsilon:
            return np.random.uniform(-self.h_max, self.h_max)
        else:
            # 对每个未被捕获的目标，计算protector应该移动到的位置
            for target in target_list:
                if getattr(target, 'captured', False):
                    continue
                
                target_x, target_y = target.x, target.y
                
                # 找到最接近目标的UAV
                closest_uav = None
                min_uav_dist = float('inf')
                for uav in uav_list:
                    dist_uav_to_target = distance(uav.x, uav.y, target_x, target_y)
                    if dist_uav_to_target < min_uav_dist:
                        min_uav_dist = dist_uav_to_target
                        closest_uav = uav
                
                if closest_uav:
                    # protector应该移动到UAV和目标之间的位置
                    # 计算理想的阻挡位置（UAV和目标连线的中点附近）
                    ideal_x = (closest_uav.x + target_x) / 2
                    ideal_y = (closest_uav.y + target_y) / 2
                    
                    # 当前protector到理想位置的距离
                    dist_to_ideal = distance(self.x, self.y, ideal_x, ideal_y)
                    
                    # 分数：距离理想位置越近，分数越高
                    score = protection_weight / (dist_to_ideal + 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_angle = np.arctan2(ideal_y - self.y, ideal_x - self.x) - self.h

        # 以continue_tracing的概率保持上一个动作
        if random.random() < self.continue_tracing:
            best_angle = 0
        
        # 将期望的角度差转换为连续动作值（角速度）
        a = np.clip(best_angle / self.dt, -self.h_max, self.h_max)
        return a