import random
import numpy as np
from math import pi, sin, cos, sqrt, exp, hypot
from typing import List, Tuple


class PROTECTOR:
    def __init__(self, x0: float, y0: float, h0: float, a0: float, v_max: float, h_max: float, dt, safe_radius: float, dc: float, dp: float):
        """
        :param x0: float, 鍒濆 x 鍧愭爣
        :param y0: float, 鍒濆 y 鍧愭爣
        :param h0: float, 鍒濆鏈濆悜
        :param a0: float, 鍒濆鍔ㄤ綔鍊硷紙瑙掑害鏀瑰彉閲忥紝杩炵画鍊硷級
        :param v_max: float, 鏈€澶х嚎閫熷害
        :param h_max: float, 鏈€澶ц閫熷害
        :param dt: float, 鏃堕棿闂撮殧
        :param safe_radius: float, 瀹夊叏鍗婂緞锛堟墜鑷傞暱搴︼級
        :param dc: float, 涓庡叾浠栦繚鎶よ€呬氦娴佺殑鏈€澶ц窛绂?        :param dp: float, 瑙傛祴 UAV 鍜岀洰鏍囩殑鏈€澶ц窛绂?        """
        self.x, self.y = x0, y0
        self.h = h0
        self.v_max = v_max
        self.h_max = h_max
        self.dt = dt
        self.safe_r = safe_radius
        
        # 鍔ㄤ綔锛氱幇鍦ㄥ瓨鍌ㄨ繛缁殑瑙掑害鏀瑰彉閲忥紙瑙掗€熷害锛?        self.a = a0
        
        # 閫氫俊鍜岃娴嬭窛绂?        self.dc = dc
        self.dp = dp
        
        # 瑙傛祴淇℃伅
        self.target_observation = []
        self.uav_observation = []
        self.protector_communication = []
        
        # 濂栧姳
        self.raw_reward = 0
        self.reward = 0

    def update_position(self, action_id=None):
        """闅忔満灏忚閫?+ 鍖€閫熷墠杩涳紙鍙敤浜庣畝鍗曚豢鐪燂級"""
        self.act_id = 0 if action_id is None else action_id
        a = random.uniform(-self.h_max, self.h_max)
        self.x += self.dt * self.v_max * cos(self.h)
        self.y += self.dt * self.v_max * sin(self.h)
        self.h += a * self.dt

    def clamp_inside(self, x_max, y_max):
        """杈圭晫澶圭揣 + 璋冨ご"""
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
        璁＄畻鍒扮洰鏍囧璞＄殑璺濈
        :param obj: class UAV, TARGET 鎴?PROTECTOR
        :return: 璺濈鍊?        """
        return sqrt((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2)

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
        Observing other protectors within dp
        :param relative: relative to protector itself
        :param protectors_list: [class PROTECTOR]
        :return: None
        """
        self.protector_observation = []  # Reset observed protectors
        for protector in protectors_list:
            if protector != self:
                dist = self.__distance(protector)
                if dist <= self.dp:
                    # add (x, y, vx, vy) information
                    if relative:
                        self.protector_observation.append(((protector.x - self.x) / self.dp,
                                                            (protector.y - self.y) / self.dp,
                                                            cos(protector.h) * protector.v_max / self.v_max - cos(self.h),
                                                            sin(protector.h) * protector.v_max / self.v_max - sin(self.h)))
                    else:
                        self.protector_observation.append((protector.x / self.dp,
                                                            protector.y / self.dp,
                                                            cos(protector.h) * protector.v_max / self.v_max,
                                                            sin(protector.h) * protector.v_max / self.v_max))

    def observe_uav(self, uav_list: List['UAV'], relative=True):  # observation
        """
        Observing UAVs with a radius within dp
        :param relative: relative to protector itself
        :param uav_list: [class UAV]
        :return: None
        """
        self.uav_observation = []  # Reset observed UAVs
        for uav in uav_list:
            dist = self.__distance(uav)
            if dist <= self.dp:
                # add (x, y, vx, vy) information
                if relative:
                    self.uav_observation.append(((uav.x - self.x) / self.dp,
                                                    (uav.y - self.y) / self.dp,
                                                    cos(uav.h) * uav.v_max / self.v_max - cos(self.h),
                                                    sin(uav.h) * uav.v_max / self.v_max - sin(self.h)))
                else:
                    self.uav_observation.append((uav.x / self.dp,
                                                    uav.y / self.dp,
                                                    cos(uav.h) * uav.v_max / self.v_max,
                                                    sin(uav.h) * uav.v_max / self.v_max))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float]],
                                        List[Tuple[float, float, float, float]], 
                                        List[Tuple[float, float, float, float]],
                                        Tuple[float, float, float]):
        """
        :return: [(x, y, vx, vy),...] for uav, [(x, y, vx, vy)] for targets, [(x, y, vx, vy)] for protectors, (x, y, a) for itself
        娉ㄦ剰锛氱幇鍦?a 鏄繛缁殑鍔ㄤ綔鍊硷紝褰掍竴鍖栧埌 [-1, 1] 鑼冨洿
        """
        # 灏嗗姩浣滃€煎綊涓€鍖栧埌 [-1, 1] 鑼冨洿锛堝熀浜?h_max锛?        normalized_action = self.a / self.h_max if self.h_max > 0 else 0.0
        return self.uav_observation, self.target_observation, self.protector_observation, (self.x / self.dc, self.y / self.dc, normalized_action)

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray
        """
        uav_obs, target_obs, prot_obs, sb = self.__get_all_local_state()

        # 澶勭悊UAV瑙傛祴
        if uav_obs:
            d_uav = []
            for x, y, vx, vy in uav_obs:
                d_uav.append(min(hypot(x * self.dp, y * self.dp), 1))
            uav_obs = np.array(uav_obs)
            uav_weighted = uav_obs / np.array(d_uav)[:, np.newaxis]
            average_uav = np.mean(uav_weighted, axis=0)
        else:
            average_uav = -np.ones(4)  # empty UAV observation (x, y, vx, vy)

        # 澶勭悊鐩爣瑙傛祴
        if target_obs:
            d_target = []
            for x, y, vx, vy in target_obs:
                d_target.append(min(hypot(x * self.dp, y * self.dp), 1))
            target_obs = np.array(target_obs)
            target_weighted = target_obs / np.array(d_target)[:, np.newaxis]
            average_target = np.mean(target_weighted, axis=0)
        else:
            average_target = -np.ones(4)  # empty target observation

        # 澶勭悊淇濇姢鑰呰娴?        if prot_obs:
            d_prot = []
            for x, y, vx, vy in prot_obs:
                d_prot.append(min(hypot(x * self.dp, y * self.dp), 1))
            prot_obs = np.array(prot_obs)
            prot_weighted = prot_obs / np.array(d_prot)[:, np.newaxis]
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
        璁＄畻淇濇姢鐩爣鐨勫鍔憋細褰損rotector鑳藉闃绘尅UAV鎺ヨ繎鐩爣鏃剁粰浜堝鍔?        :param target_list: 鐩爣鍒楄〃
        :param uav_list: UAV鍒楄〃
        :return: 淇濇姢濂栧姳 [0, reward_factor * protected_targets]
        """
        protection_reward = 0.0
        
        # 淇濇姢鍗婂緞锛歱rotector鐨勪繚鎶よ寖鍥达紙閫氬父澶т簬瑙傛祴璺濈锛?        protection_radius = 2.0 * self.safe_r  # 浣跨敤瀹夊叏鍗婂緞鐨勫€嶆暟浣滀负淇濇姢鑼冨洿
        
        # 濂栧姳鍥犲瓙锛氭瘡淇濇姢涓€涓洰鏍囩殑鍩虹濂栧姳
        reward_factor = 1.0
        
        # 鏈€澶у鍔辩洰鏍囨暟閲忥細閬垮厤濂栧姳杩囧害闆嗕腑鍦ㄥ皯鏁扮洰鏍囦笂
        max_rewarded_targets = 5
        
        # 瀵规瘡涓湭琚崟鑾风殑鐩爣璁＄畻淇濇姢濂栧姳
        protected_scores = []
        for target in target_list:
            if getattr(target, 'captured', False):
                continue
            
            # protector鍒扮洰鏍囩殑璺濈
            dist_prot_to_target = self.__distance(target)
            
            # 濡傛灉protector鍦ㄤ繚鎶よ寖鍥村唴
            if dist_prot_to_target <= protection_radius:
                # 璁＄畻protector闃绘尅UAV鐨勬晥鏋?                blocking_score = 0.0
                
                for uav in uav_list:
                    # UAV鍒扮洰鏍囩殑璺濈
                    dist_uav_to_target = sqrt((uav.x - target.x) ** 2 + (uav.y - target.y) ** 2)
                    
                    # protector鍒癠AV鐨勮窛绂?                    dist_prot_to_uav = self.__distance(uav)
                    
                    # 濡傛灉UAV姝ｅ湪鎺ヨ繎鐩爣锛堣窛绂诲皬浜庢煇涓槇鍊硷級
                    if dist_uav_to_target < 3.0 * protection_radius:
                        # 璁＄畻protector鏄惁鍦║AV鍜岀洰鏍囦箣闂达紙闃绘尅鏁堟灉锛?                        # 浣跨敤鍚戦噺鎶曞奖鍒ゆ柇protector鏄惁鍦║AV鍒扮洰鏍囩殑璺緞涓?                        uav_to_target_dx = target.x - uav.x
                        uav_to_target_dy = target.y - uav.y
                        uav_to_target_dist = sqrt(uav_to_target_dx ** 2 + uav_to_target_dy ** 2)
                        
                        if uav_to_target_dist > 1e-6:
                            # 璁＄畻protector鍒癠AV-鐩爣杩炵嚎鐨勮窛绂?                            prot_to_uav_dx = uav.x - self.x
                            prot_to_uav_dy = uav.y - self.y
                            
                            # 鎶曞奖鍒癠AV-鐩爣鏂瑰悜
                            proj = (prot_to_uav_dx * uav_to_target_dx + prot_to_uav_dy * uav_to_target_dy) / uav_to_target_dist
                            
                            # 濡傛灉protector鍦║AV鍜岀洰鏍囦箣闂达紙0 < proj < dist锛?                            if 0 < proj < uav_to_target_dist:
                                # 璁＄畻鍒拌繛绾跨殑鍨傜洿璺濈
                                perp_dist = abs(prot_to_uav_dx * uav_to_target_dy - prot_to_uav_dy * uav_to_target_dx) / uav_to_target_dist
                                
                                # 濡傛灉protector鐨剆afe_r鑳藉瑕嗙洊鍒拌繖鏉¤矾寰?                                if perp_dist < self.safe_r:
                                    # 闃绘尅鏁堟灉锛氳窛绂昏秺杩戯紝闃绘尅鏁堟灉瓒婂ソ
                                    blocking_effect = (self.safe_r - perp_dist) / self.safe_r
                                    # 鏍规嵁UAV鍒扮洰鏍囩殑璺濈锛岃秺杩戠殑UAV闇€瑕佽秺寮虹殑闃绘尅
                                    urgency = (3.0 * protection_radius - dist_uav_to_target) / (3.0 * protection_radius)
                                    blocking_score += blocking_effect * urgency
                
                # 缁煎悎濂栧姳锛歱rotector闈犺繎鐩爣 + 闃绘尅UAV
                proximity_bonus = (protection_radius - dist_prot_to_target) / protection_radius
                total_score = proximity_bonus * 0.3 + blocking_score * 0.7
                protected_scores.append((total_score, dist_prot_to_target))
        
        # 鎸夊緱鍒嗘帓搴忥紝鍙鍔卞墠max_rewarded_targets涓洰鏍?        protected_scores.sort(key=lambda x: x[0], reverse=True)
        for i, (score, _) in enumerate(protected_scores):
            if i >= max_rewarded_targets:
                break
            protection_reward += reward_factor * score
        
        return protection_reward

    def __calculate_overlapping_protector_punishment(self, protector_list: List['PROTECTOR'], radio=1.5) -> float:
        """
        璁＄畻閲嶅彔淇濇姢鑰呮儵缃氾細濡傛灉澶淇濇姢鑰呰仛闆嗗湪涓€璧凤紝缁欎簣杞诲井鎯╃綒
        :param protector_list: [class PROTECTOR]
        :param radio: 鎺у埗鎯╃綒鑼冨洿鐨勭郴鏁?        :return: 鎯╃綒鍊?(-radio * punishment_factor, 0]
        """
        total_punishment = 0.0
        punishment_factor = 0.3  # 鎯╃綒寮哄害锛堣緝杞伙紝榧撳姳鍚堜綔浣嗛伩鍏嶈繃搴﹁仛闆嗭級
        
        for other_prot in protector_list:
            if other_prot != self:
                distance = self.__distance(other_prot)
                if distance <= radio * self.safe_r:
                    # 鎯╃綒鎸夎窛绂婚€掑噺锛氶潬寰楄秺杩戞儵缃氳秺澶?                    punishment = -punishment_factor * exp((radio * self.safe_r - distance) / (radio * self.safe_r))
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
        return boundary_punishment  # 娌℃湁clip, 鍦ㄨ皟鐢ㄦ椂澶栭儴clip
        # return clip_and_normalize(boundary_punishment, -1/2, 0, -1)

    def calculate_raw_reward(self, uav_list: List['UAV'], target_list: List['TARGET'], protector_list: List['PROTECTOR'], x_max, y_max):
        """
        璁＄畻protector鐨勫師濮嬪鍔?鎯╃綒
        :param uav_list: UAV鍒楄〃
        :param target_list: 鐩爣鍒楄〃
        :param protector_list: 淇濇姢鑰呭垪琛?        :param x_max: 鍦板浘x杈圭晫
        :param y_max: 鍦板浘y杈圭晫
        :return: (淇濇姢濂栧姳, 杈圭晫鎯╃綒, 閲嶅彔鎯╃綒, 鎷︽埅濂栧姳)
        """
        # 淇濇姢鐩爣濂栧姳锛歱rotector鑳藉闃绘尅UAV鎺ヨ繎鐩爣
        protection_reward = self.__calculate_target_protection_reward(target_list, uav_list)
        # 杈圭晫鎯╃綒锛氫繚鎸佷笌UAV鐩稿悓
        boundary_punishment = self.__calculate_boundary_punishment(x_max, y_max)
        # 閲嶅彔淇濇姢鑰呮儵缃氾細閬垮厤杩囧害鑱氶泦
        overlapping_punishment = self.__calculate_overlapping_protector_punishment(protector_list)
        # 鎴愬姛鎷︽埅濂栧姳锛歱rotector鎴愬姛鎷︽埅UAV骞朵娇鍏惰繙绂荤洰鏍?        interception_reward = self.__calculate_successful_interception_reward(uav_list, target_list)
        
        return protection_reward, boundary_punishment, overlapping_punishment, interception_reward
    
    def __calculate_successful_interception_reward(self, uav_list: List['UAV'], target_list: List['TARGET']) -> float:
        """
        璁＄畻鎴愬姛鎷︽埅濂栧姳锛氬綋protector鎴愬姛鎷︽埅UAV锛屼笖鎷︽埅浣垮叾杩滅鐩爣鏃剁粰浜堝鍔?        :param uav_list: UAV鍒楄〃
        :param target_list: 鐩爣鍒楄〃
        :return: 鎷︽埅濂栧姳 [0, reward_factor * intercepted_uavs]
        """
        interception_reward = 0.0
        
        # 鎷︽埅妫€娴嬭寖鍥达細protector鐨勬嫤鎴奖鍝嶈寖鍥达紙鍩轰簬safe_r锛?        interception_range = 1.5 * self.safe_r  # 鑰冭檻鎵嬭噦闀垮害鍜岀鎾炶寖鍥?        
        # 濂栧姳鍥犲瓙
        reward_factor = 2.0  # 鎷︽埅濂栧姳搴旇姣旇緝楂橈紝鍥犱负杩欐槸鐩存帴鏈夋晥鐨勯槻寰?        
        # 瀵规瘡涓猆AV妫€鏌ユ槸鍚﹁鎷︽埅
        for uav in uav_list:
            # 妫€鏌AV鏄惁澶勪簬閿佸畾鐘舵€侊紙琚嫤鎴級
            if getattr(uav, 'lock', 0) > 0:
                # 妫€鏌rotector鏄惁鍦ㄦ嫤鎴寖鍥村唴锛堝彲鑳芥槸杩欎釜protector鎷︽埅鐨勶級
                dist_to_uav = self.__distance(uav)
                
                if dist_to_uav <= interception_range:
                    # 妫€鏌ユ嫤鎴悗鐨勬湞鍚戞槸鍚︿娇UAV杩滅鐩爣
                    uav_heading = uav.h  # UAV褰撳墠鏈濆悜锛堟嫤鎴悗鐨勬湞鍚戯級
                    
                    # 鎵惧埌鏈€杩戠殑鏈鎹曡幏鐨勭洰鏍?                    closest_target = None
                    min_dist = float('inf')
                    for target in target_list:
                        if getattr(target, 'captured', False):
                            continue
                        dist = sqrt((uav.x - target.x) ** 2 + (uav.y - target.y) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = target
                    
                    if closest_target:
                        # 璁＄畻UAV鏈濆悜鐩爣鐨勭悊鎯虫柟鍚?                        target_dx = closest_target.x - uav.x
                        target_dy = closest_target.y - uav.y
                        target_angle = np.arctan2(target_dy, target_dx)
                        
                        # 璁＄畻UAV褰撳墠鏈濆悜涓庣洰鏍囨柟鍚戠殑澶硅
                        angle_diff = abs(uav_heading - target_angle)
                        # 褰掍竴鍖栧埌 [0, pi]
                        if angle_diff > pi:
                            angle_diff = 2 * pi - angle_diff
                        
                        # 濡傛灉澶硅澶т簬 pi/2锛?0搴︼級锛岃鏄嶶AV鏈濆悜杩滅鐩爣
                        if angle_diff > pi / 2:
                            # 澶硅瓒婂ぇ锛岃繙绂绘晥鏋滆秺濂斤紝濂栧姳瓒婇珮
                            # 澶硅浠?pi/2 鍒?pi锛屽鍔变粠 0 鍒?reward_factor
                            effectiveness = (angle_diff - pi / 2) / (pi / 2)  # [0, 1]
                            
                            # 鏍规嵁璺濈鐩爣鐨勮繙杩戣皟鏁村鍔憋紙瓒婅繎鏃舵嫤鎴紝濂栧姳瓒婇珮锛?                            current_dist = sqrt(target_dx ** 2 + target_dy ** 2)
                            # 鍋囪鍗遍櫓璺濈涓?5 * safe_r锛屽湪姝よ寖鍥村唴鎷︽埅鏇存湁鏁?                            danger_zone = 5.0 * self.safe_r
                            if current_dist < danger_zone:
                                urgency_bonus = (danger_zone - current_dist) / danger_zone  # [0, 1]
                                reward = reward_factor * effectiveness * (1.0 + urgency_bonus)
                            else:
                                reward = reward_factor * effectiveness
                            
                            interception_reward += reward
        
        return interception_reward


    def __calculate_cooperative_reward_by_pmi(self, protector_list: List['PROTECTOR'], pmi_net, a) -> float:
        """
        浣跨敤PMI缃戠粶璁＄畻鍚堜綔濂栧姳
        :param pmi_net: PMI缃戠粶
        :param protector_list: [class PROTECTOR]
        :param a: float, 鑷涓庡叡浜殑姣斾緥
        :return:
        """
        if a == 0:  # 鎻愬墠鍒ゆ柇锛岃妭鐪佽绠楃殑澶嶆潅搴?            return self.raw_reward

        from scipy.special import softmax
        
        neighbor_rewards = []
        neighbor_dependencies = []
        la = self.get_local_state()

        for other_prot in protector_list:
            if other_prot != self and self.__distance(other_prot) <= self.dc:
                neighbor_rewards.append(other_prot.raw_reward)
                other_prot_la = other_prot.get_local_state()
                _input = la * other_prot_la
                neighbor_dependencies.append(pmi_net.inference(_input.squeeze()))

        if len(neighbor_rewards):
            neighbor_rewards = np.array(neighbor_rewards)
            neighbor_dependencies = np.array(neighbor_dependencies).astype(np.float32)
            softmax_values = softmax(neighbor_dependencies)
            reward = (1 - a) * self.raw_reward + a * np.sum(neighbor_rewards * softmax_values).item()
        else:
            reward = (1 - a) * self.raw_reward
        return reward

    def __calculate_cooperative_reward_by_mean(self, protector_list: List['PROTECTOR'], a) -> float:
        """
        浣跨敤骞冲潎鍊艰绠楀悎浣滃鍔?        :param protector_list: [class PROTECTOR]
        :param a: float, 鑷涓庡叡浜殑姣斾緥
        :return:
        """
        if a == 0:  # 鎻愬墠鍒ゆ柇锛岃妭鐪佽绠楃殑澶嶆潅搴?            return self.raw_reward

        neighbor_rewards = []
        for other_prot in protector_list:
            if other_prot != self and self.__distance(other_prot) <= self.dc:
                neighbor_rewards.append(other_prot.raw_reward)
        # 娌℃湁鍔犲叆PMI缃戠粶
        reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards) \
            if len(neighbor_rewards) else (1 - a) * self.raw_reward
        return reward

    def calculate_cooperative_reward(self, protector_list: List['PROTECTOR'], pmi_net=None, a=0.5) -> float:
        """
        璁＄畻鍚堜綔濂栧姳
        :param protector_list: 淇濇姢鑰呭垪琛?        :param pmi_net: PMI缃戠粶锛堝彲閫夛級
        :param a: 0: 瀹屽叏鑷, 1: 瀹屽叏鍏变韩
        :return:
        """
        if pmi_net:
            return self.__calculate_cooperative_reward_by_pmi(protector_list, pmi_net, a)
        else:
            return self.__calculate_cooperative_reward_by_mean(protector_list, a)

    def get_action_by_direction(self, target_list, uav_list):
        """
        鍩轰簬鐩爣鍜孶AV鐨勪綅缃绠楀姩浣滐紙鐢ㄤ簬鍚彂寮忕瓥鐣ワ紝闈炲己鍖栧涔狅級
        """
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # 淇濇姢鑰呭簲璇ョЩ鍔ㄥ埌鑳藉闃绘尅UAV鎺ヨ繎鐩爣鐨勪綅缃?        # 濂栧姳鍜屾儵缃氭潈閲?        protection_weight = 1.0
        self.epsilon = 0.25
        self.continue_tracing = 0.3
        
        best_score = float('-inf')
        best_angle = 0.0

        # 闅忔満鎵板姩锛氫互epsilon鐨勬鐜囬€夋嫨闅忔満鍔ㄤ綔
        if random.random() < self.epsilon:
            return np.random.uniform(-self.h_max, self.h_max)
        else:
            # 瀵规瘡涓湭琚崟鑾风殑鐩爣锛岃绠梡rotector搴旇绉诲姩鍒扮殑浣嶇疆
            for target in target_list:
                if getattr(target, 'captured', False):
                    continue
                
                target_x, target_y = target.x, target.y
                
                # 鎵惧埌鏈€鎺ヨ繎鐩爣鐨刄AV
                closest_uav = None
                min_uav_dist = float('inf')
                for uav in uav_list:
                    dist_uav_to_target = distance(uav.x, uav.y, target_x, target_y)
                    if dist_uav_to_target < min_uav_dist:
                        min_uav_dist = dist_uav_to_target
                        closest_uav = uav
                
                if closest_uav:
                    # protector搴旇绉诲姩鍒癠AV鍜岀洰鏍囦箣闂寸殑浣嶇疆
                    # 璁＄畻鐞嗘兂鐨勯樆鎸′綅缃紙UAV鍜岀洰鏍囪繛绾跨殑涓偣闄勮繎锛?                    ideal_x = (closest_uav.x + target_x) / 2
                    ideal_y = (closest_uav.y + target_y) / 2
                    
                    # 褰撳墠protector鍒扮悊鎯充綅缃殑璺濈
                    dist_to_ideal = distance(self.x, self.y, ideal_x, ideal_y)
                    
                    # 鍒嗘暟锛氳窛绂荤悊鎯充綅缃秺杩戯紝鍒嗘暟瓒婇珮
                    score = protection_weight / (dist_to_ideal + 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_angle = np.arctan2(ideal_y - self.y, ideal_x - self.x) - self.h

        # 浠ontinue_tracing鐨勬鐜囦繚鎸佷笂涓€涓姩浣?        if random.random() < self.continue_tracing:
            best_angle = 0
        
        # 灏嗘湡鏈涚殑瑙掑害宸浆鎹负杩炵画鍔ㄤ綔鍊硷紙瑙掗€熷害锛?        a = np.clip(best_angle / self.dt, -self.h_max, self.h_max)
        return a
