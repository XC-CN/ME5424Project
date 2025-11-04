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
    def __init__(self, x0, y0, h0, a0, v_max, h_max, na, dc, dp, dt):
        """
        :param dt: float, 閲囨牱鐨勬椂闂撮棿闅?        :param x0: float, 鍧愭爣
        :param y0: float, 鍧愭爣
        :param h0: float, 鏈濆悜
        :param a0: float, 鍒濆鍔ㄤ綔鍊硷紙瑙掑害鏀瑰彉閲忥紝杩炵画鍊硷級
        :param v_max: float, 鏈€澶х嚎閫熷害
        :param h_max: float, 鏈€澶ц閫熷害
        :param na: int, 鍔ㄤ綔绌洪棿鐨勭淮搴︼紙淇濈暀鐢ㄤ簬鍏煎鎬э紝浣嗕笉鍐嶄娇鐢級
        :param dc: float, 涓庢棤浜烘満浜ゆ祦鐨勬渶澶ц窛绂?        :param dp: float, 鎹曟崏鐩爣鐨勬渶澶ц窛绂?        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.Na = na  # 淇濈暀鐢ㄤ簬鍏煎鎬э紝浣嗗姩浣滅幇鍦ㄦ槸杩炵画鐨?
        # action: 鐜板湪瀛樺偍杩炵画鐨勮搴︽敼鍙橀噺锛堣閫熷害锛?        self.a = a0

        # the maximum communication distance and maximum perception distance
        # self.dc = dc
        self.dp = dp

        # time interval
        self.dt = dt

        # set of local information
        # self.communication = []
        self.target_observation = []
        self.uav_communication = []

        # reward
        self.raw_reward = 0
        self.reward = 0
        
        # 鏈濆悜閿佸畾鐘舵€侊紙鐢ㄤ簬纰版挒寮瑰紑鏁堟灉锛?        self.lock = 0  # 鍓╀綑閿佸畾鏃堕棿锛堝崟浣嶏細姝ユ暟锛?        self.captured_targets_count = 0  # 鏈鎹曡幏鐨勭洰鏍囨暟閲?
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
        from the action space index to the real difference (淇濈暀鐢ㄤ簬鍏煎鎬?
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 鍗宠搴︽敼鍙橀噺
        """
        # from action space to the real world action
        na = a_idx + 1  # 浠?1 寮€濮嬬储寮?        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, action: 'float') -> (float, float, float):
        """
        鎺ユ敹杩炵画鍔ㄤ綔鍊硷紝鏇存柊褰撳墠浣嶇疆
        :param action: float, 瑙掑害鏀瑰彉閲忥紙瑙掗€熷害锛夛紝鑼冨洿搴斿湪 [-h_max, h_max]
        :return: (x, y, h) 鏇存柊鍚庣殑浣嶇疆鍜屾湞鍚?        """
        # 灏嗗姩浣滈檺鍒跺湪鏈夋晥鑼冨洿鍐?        a = np.clip(action, -self.h_max, self.h_max)
        self.a = a

        dx = self.dt * self.v_max * cos(self.h)  # x 鏂瑰悜浣嶇Щ
        dy = self.dt * self.v_max * sin(self.h)  # y 鏂瑰悜浣嶇Щ
        self.x += dx
        self.y += dy
        
        # 濡傛灉澶勪簬鏈濆悜閿佸畾鐘舵€侊紝蹇界暐鍔ㄤ綔鐨勬湞鍚戞敼鍙?        if self.lock > 0:
            self.lock -= 1  # 鍑忓皯閿佸畾鏃堕棿
            # 閿佸畾鏈熼棿涓嶆洿鏂版湞鍚戯紝淇濇寔褰撳墠鏈濆悜缁х画杩愬姩
        else:
            # 姝ｅ父鏇存柊鏈濆悜瑙掑害
            self.h += self.dt * a
        
        self.h = (self.h + pi) % (2 * pi) - pi  # 纭繚鏈濆悜瑙掑害鍦?[-pi, pi) 鑼冨洿鍐?
        return self.x, self.y, self.h  # 杩斿洖agent鐨勪綅缃拰鏈濆悜(heading/theta)
    
    def apply_knockback(self, knockback_angle: float, lock_duration: int = 0):
        """
        搴旂敤寮瑰紑鏁堟灉锛氭敼鍙樻湞鍚戝苟閿佸畾涓€娈垫椂闂?        :param knockback_angle: float, 寮瑰紑鐨勮搴︼紙寮у害锛?        :param lock_duration: int, 閿佸畾鎸佺画鏃堕棿锛堟鏁帮級锛岄攣瀹氭湡闂存棤娉曟敼鍙樻湞鍚?        """
        self.h = knockback_angle
        self.h = (self.h + pi) % (2 * pi) - pi  # 纭繚鏈濆悜瑙掑害鍦?[-pi, pi) 鑼冨洿鍐?        self.lock = max(self.lock, lock_duration)  # 宸查攣瀹氱殑鎯呭喌涓嬬鍒版柊鐨刾rotector

    def observe_target(self, targets_list: List['TARGET'], relative=True):
        """
        Observing target with a radius within dp
        :param relative: relative to uav itself
        :param targets_list: [class UAV]
        :return: None
        """
        self.target_observation = []  # Reset observed targets
        # 宸叉崟鑾风洰鏍囦笉鍐嶈瑙傛祴
        for target in targets_list:
            if getattr(target, 'captured', False):
                continue
            dist = self.__distance(target)
            if dist <= self.dp:
                if relative:
                    self.target_observation.append(((target.x - self.x) / self.dp,
                                                    (target.y - self.y) / self.dp,
                                                    cos(target.h) * target.v_max / self.v_max - cos(self.h),
                                                    sin(target.h) * target.v_max / self.v_max - sin(self.h)))
                else:
                    self.target_observation.append((target.x / self.dp,
                                                    target.y / self.dp,
                                                    cos(target.h) * target.v_max / self.v_max,
                                                    sin(target.h) * target.v_max / self.v_max))
                    
    def observe_protector(self, protectors_list: List['PROTECTOR'], relative=True):
        """
        Observing target with a radius within dp
        :param relative: relative to uav itself
        :param targets_list: [class UAV]
        :return: None
        """
        self.protector_observation = []  # Reset observed targets
        for protector in protectors_list:
            dist = self.__distance(protector)
            if dist <= self.dp:
                # add (x, y, vx, vy) information
                if relative:
                    self.target_observation.append(((protector.x - self.x) / self.dp,
                                                    (protector.y - self.y) / self.dp,
                                                    cos(protector.h) * protector.v_max / self.v_max - cos(self.h),
                                                    sin(protector.h) * protector.v_max / self.v_max - sin(self.h)))
                else:
                    self.target_observation.append((protector.x / self.dp,
                                                    protector.y / self.dp,
                                                    cos(protector.h) * protector.v_max / self.v_max,
                                                    sin(protector.h) * protector.v_max / self.v_max))

    def observe_uav(self, uav_list: List['UAV'], relative=True):  # communication
        """
        communicate with other uav_s with a radius within dp
        :param relative: relative to uav itself
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_communication = []  # Reset observed targets
        for uav in uav_list:
            dist = self.__distance(uav)
            if dist <= self.dc and uav != self:
                # add (x, y, vx, vy, a) information
                # 灏嗗姩浣滃€煎綊涓€鍖栧埌 [-1, 1] 鑼冨洿
                uav_normalized_action = uav.a / uav.h_max if uav.h_max > 0 else 0.0
                self_normalized_action = self.a / self.h_max if self.h_max > 0 else 0.0
                if relative:
                    self.uav_communication.append(((uav.x - self.x) / self.dc,
                                                   (uav.y - self.y) / self.dc,
                                                   cos(uav.h) - cos(self.h),
                                                   sin(uav.h) - sin(self.h),
                                                   uav_normalized_action - self_normalized_action))
                else:
                    self.uav_communication.append((uav.x / self.dc,
                                                   uav.y / self.dc,
                                                   cos(uav.h),
                                                   sin(uav.h),
                                                   uav_normalized_action))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, float]],
                                        List[Tuple[float, float, float, float]], Tuple[float, float, float]):
        """
        :return: [(x, y, vx, vy, a),...] for uav, [(x, y, vx, vy)] for targets, (x, y, a) for itself
        娉ㄦ剰锛氱幇鍦?a 鏄繛缁殑鍔ㄤ綔鍊硷紝褰掍竴鍖栧埌 [-1, 1] 鑼冨洿
        """
        # 灏嗗姩浣滃€煎綊涓€鍖栧埌 [-1, 1] 鑼冨洿锛堝熀浜?h_max锛?        normalized_action = self.a / self.h_max if self.h_max > 0 else 0.0
        return self.uav_communication, self.target_observation, (self.x / self.dc, self.y / self.dc, normalized_action)

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray: (12)
        """
        communication, observation, sb = self.__get_all_local_state()

        if communication:
            d_communication = []  # store the distance from each uav to itself
            for x, y, vx, vy, na in communication:
                d_communication.append(min(self.distance(x, y, self.x, self.y), 1))

            # regularization by the distance
            # communication = self.__transform_to_array2d(communication)
            communication = np.array(communication)
            communication_weighted = communication / np.array(d_communication)[:, np.newaxis]
            average_communication = np.mean(communication_weighted, axis=0)
        else:
            # average_communication = np.zeros(4 + self.Na)  # empty communication
            average_communication = -np.ones(4 + 1)  # empty communication (x, y, vx, vy, a)

        if observation:
            d_observation = []  # store the distance from each target to itself
            for x, y, vx, vy in observation:
                d_observation.append(min(self.distance(x, y, self.x, self.y), 1))

            # regularization by the distance
            observation = np.array(observation)
            observation_weighted = observation / np.array(d_observation)[:, np.newaxis]
            average_observation = np.mean(observation_weighted, axis=0)
        else:
            average_observation = -np.ones(4)  # empty observation  # TODO -1鍚堟硶鍚?
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
        :param radio: radio鐢ㄦ潵鎺у埗鎯╃綒鐨勮寖鍥? 瓒呭嚭澶氳繙鎵嶇畻鍏ユ儵缃?        :return: scalar (-e/2, -1/2]
        """
        total_punishment = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * exp((radio * self.dp - distance) / (radio * self.dp))
                    # total_punishment += clip_and_normalize(punishment, -e/2, -1/2, -1)
                    total_punishment += punishment  # 娌℃湁clip, 鍦ㄨ皟鐢ㄦ椂澶栭儴clip
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
        return boundary_punishment  # 娌℃湁clip, 鍦ㄨ皟鐢ㄦ椂澶栭儴clip
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
        绠€鍗曠殑璺濈闃堝€兼儵缃氾細
        - 鑻?UAV 涓庝换鎰?Protector 璺濈 < threshold 鍒欎骇鐢熸儵缃氾紙璐熷€硷級
        - 杩斿洖鍊煎彇鍖洪棿 [-K, 0]锛堟湭褰掍竴鍖栵級锛屽悗缁敱 clip_and_normalize 褰掍竴鍖栧埌 [-1,0]
        """
        # UAV 鑷韩鍗婂緞锛屽彲閰嶇疆鎴栭粯璁?        uav_radius = getattr(self, "radius", 0.5)
        worst_pen = 0.0  # 鎯╃綒涓洪潪姝ｆ暟锛? 琛ㄧず鏃犳儵缃氾級
        for prot in protector_list:
            # protector 閲岀害瀹氬睘鎬?safe_r锛堟垨浣跨敤 prot.safe_r锛?            prot_safe = getattr(prot, "safe_r", None)
            if prot_safe is None:
                # 鐢ㄩ粯璁ゆ垨 prot.radius
                prot_safe = getattr(prot, "radius", 0.5)
            threshold = prot_safe + uav_radius  # 瑙嗕负纰版挒/璀︽垝鍗婂緞
            dx = prot.x - self.x
            dy = prot.y - self.y
            dist = hypot(dx, dy)
            if dist < threshold:
                # 鎯╃綒鎸夎窛绂绘垚姣斾緥锛氶潬寰楄秺杩戞儵缃氳秺澶э紙璐燂級
                # 鏈綊涓€鍖栨儵缃氾紝渚嬪锛?- (threshold - dist)  -> 浠嬩簬 (-threshold, 0]
                pen = -(threshold - dist)
                if pen < worst_pen:
                    worst_pen = pen
        return worst_pen

    def __calculate_cooperative_reward_by_pmi(self, uav_list: List['UAV'], pmi_net: "PMINetwork", a) -> float:
        """
        calculate cooperative reward by pmi network
        :param pmi_net: class PMINetwork
        :param uav_list: [class UAV]
        :param a: float, proportion of selfish and sharing
        :return:
        """
        if a == 0:  # 鎻愬墠鍒ゆ柇锛岃妭鐪佽绠楃殑澶嶆潅搴?            return self.raw_reward

        neighbor_rewards = []
        neighbor_dependencies = []
        la = self.get_local_state()

        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
                other_uav_la = other_uav.get_local_state()
                _input = la * other_uav_la
                neighbor_dependencies.append(pmi_net.inference(_input.squeeze()))

        if len(neighbor_rewards):
            neighbor_rewards = np.array(neighbor_rewards)
            neighbor_dependencies = np.array(neighbor_dependencies).astype(np.float32)
            softmax_values = softmax(neighbor_dependencies)
            reward = (1 - a) * self.raw_reward + a * np.sum(neighbor_rewards * softmax_values).item()
        else:
            reward = (1 - a) * self.raw_reward
        return reward

    def __calculate_cooperative_reward_by_mean(self, uav_list: List['UAV'], a) -> float:
        """
        calculate cooperative reward by mean
        :param uav_list: [class UAV]
        :param a: float, proportion of selfish and sharing
        :return:
        """
        if a == 0:  # 鎻愬墠鍒ゆ柇锛岃妭鐪佽绠楃殑澶嶆潅搴?            return self.raw_reward

        neighbor_rewards = []
        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
        # 娌℃湁鍔犲叆PMI缃戠粶
        reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards) \
            if len(neighbor_rewards) else 0
        return reward

    def calculate_cooperative_reward(self, uav_list: List['UAV'], pmi_net=None, a=0.5) -> float:
        """
        :param uav_list:
        :param pmi_net:
        :param a: 0: selfish, 1: completely shared
        :return:
        """
        if pmi_net:
            return self.__calculate_cooperative_reward_by_pmi(uav_list, pmi_net, a)
        else:
            return self.__calculate_cooperative_reward_by_mean(uav_list, a)

    def get_action_by_direction(self, target_list, uav_list):
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # 濂栧姳鍜屾儵缃氭潈閲?        target_reward_weight = 1.0
        repetition_penalty_weight = 0.8
        self.epsilon = 0.25
        self.continue_tracing = 0.3
        
        best_score = float('-inf')
        best_angle = 0.0

        # 闅忔満鎵板姩锛氫互epsilon鐨勬鐜囬€夋嫨闅忔満鍔ㄤ綔
        if random.random() < self.epsilon:
            return np.random.uniform(-self.h_max, self.h_max)
        else:
            for target in target_list:
                target_x, target_y = target.x, target.y

                # 褰撳墠鏃犱汉鏈哄埌鐩爣鐨勮窛绂?                dist_to_target = distance(self.x, self.y, target_x, target_y)

                # 閲嶅杩借釜鐨勬儵缃氾紝鑰冭檻鍏朵粬鏃犱汉鏈哄湪閲嶅杩借釜鍗婂緞鍐呮槸鍚﹀湪杩借釜鍚屼竴鐩爣
                repetition_penalty = 0.0
                for uav in uav_list:
                    uav_x, uav_y = uav.x, uav.y
                    if (uav_x, uav_y) != (self.x, self.y):
                        dist_to_target_from_other_uav = distance(uav_x, uav_y, target_x, target_y)
                        if dist_to_target_from_other_uav < self.dc:
                            repetition_penalty += repetition_penalty_weight

                # 璁＄畻褰撳墠鐩爣鐨勫緱鍒?                score = target_reward_weight / dist_to_target - repetition_penalty

                # 鏍规嵁寰楀垎閫夋嫨鏈€浼樼洰鏍?                if score > best_score:
                    best_score = score
                    best_angle = np.arctan2(target_y - self.y, target_x - self.x) - self.h

        # 浠ontinue_tracing鐨勬鐜囦繚鎸佷笂涓€涓姩浣?        if random.random() < self.continue_tracing:
            best_angle = 0
            
        # 灏嗘湡鏈涚殑瑙掑害宸浆鎹负杩炵画鍔ㄤ綔鍊硷紙瑙掗€熷害锛?        # best_angle 鏄湡鏈涚殑瑙掑害宸紝闇€瑕佽浆鎹负瑙掗€熷害
        a = np.clip(best_angle / self.dt, -self.h_max, self.h_max)
        return a
