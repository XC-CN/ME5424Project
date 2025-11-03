import random
import numpy as np
from typing import List, Tuple, Optional
from models.PMINet import PMINetwork
from agent.target import Target
from agent.protector import Protector
from scipy.special import softmax
from utils.data_util import clip_and_normalize


class UAV:
    def __init__(self, x0, y0, h0, a_idx, v_max, h_max, na, dc, dp, dt):
        """
        :param dt: float, 闂備焦褰冨ú锕傛偋闁秵鍎嶉柛鏇ㄥ墯椤ρ囨⒒閸屾稒灏︽俊顐㈡健濮?
        :param x0: float, 闂佺鍕闁?
        :param y0: float, 闂佺鍕闁?
        :param h0: float, 闂佸搫鐗婄换鍌炲箖?
        :param v_max: float, 闂佸搫鐗冮崑鎾愁熆閸棗鎳庡▓鐘绘⒑椤愩倕鏋庨悗?
        :param h_max: float, 闂佸搫鐗冮崑鎾愁熆閸棗妫锟犳⒑椤愩倕鏋庨悗?
        :param na: int, 闂佸憡鏌ｉ崝瀣礊閺冨倻鐭氭繛宸簼閿涚喖鏌ｉ妸銉ヮ伀婵烇綆鍠楅幆?
        :param dc: float, 婵炴垶鎸哥€涒晛螞閵堝棛顩查柣鎴炆戠花姘瑰┃鍨偓鏍矈閿曞倹鍎嶉柛鏇ㄥ墯娴犳ê顭块崼鍡楁閻涒晝绱?
        :param dp: float, 闁荤喐鐟ラ崐鍦矈閹绢喗鍎庢い鏃傛櫕閸ㄥジ鏌ｉ妸銉ヮ仾婵炴捁鍩栧鍕樁缂佹劖绋撶划?
        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.Na = na

        # action
        self.a = a_idx

        # the maximum communication distance and maximum perception distance
        self.dc = dc
        self.dp = dp

        # time interval
        self.dt = dt

        # world bounds (闂佹眹鍨兼禍顒勭嵁閸℃ɑ娅犻柛鎰╁妽闊剟鏌涢幒鎾剁畵妞ゎ偅鍔欏畷鐘诲冀閵婏富妲柣鐘辩婢т粙鎮?
        self.world_bounds: Optional[Tuple[float, float]] = None

        # set of local information
        # self.communication = []
        self.target_observation = []
        self.uav_communication = []

        # reward
        self.raw_reward = 0
        self.reward = 0

    def __distance(self, target) -> float:
        """
        calculate the distance from uav to target
        :param target: class UAV or class Target
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

    def set_world_bounds(self, x_max: float, y_max: float):
        self.world_bounds = (max(float(x_max), 0.0), max(float(y_max), 0.0))

    def _clamp_inside(self):
        if self.world_bounds is None:
            return
        x_max, y_max = self.world_bounds
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

    def update_position(self, action: 'int') -> (float, float, float):
        """
        receive the index from action space, then update the current position
        :param action: {0,1,...,Na - 1}
        :return:
        """
        self.a = action
        a = self.discrete_action(action)
        dx = self.dt * self.v_max * cos(self.h)
        dy = self.dt * self.v_max * sin(self.h)
        self.x += dx
        self.y += dy
        self.h += self.dt * a
        self.h = (self.h + pi) % (2 * pi) - pi
        if self.world_bounds is not None:
            self._clamp_inside()
        return self.x, self.y, self.h
    def discrete_action(self, a_idx: int) -> float:
        """
        from the action space index to the real difference
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 闂佸憡顨呴悿鍥綖濡や焦鍎熼柨鏃囧Г閺嗩參鏌涘▎鎰惰€块柛?
        """
        # from action space to the real world action
        na = a_idx + 1  # 婵?1 閻庢鍠掗崑鎾斥攽椤旂⒈鍎戦柛灞诲妼椤?
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

        self.h = (self.h + pi) % (2 * pi) - pi  # 缂佺虎鍙庨崰鏇犳崲濮樿泛瀚夋繝闈涙閸婂鎮峰▎鎰瑨閻庤濞婂畷?[-pi, pi) 闂佽偐鍘ч崯顐⒚洪崸妤€绀?

        return self.x, self.y, self.h  # 闁哄鏅滈弻銊ッ洪張绯秂nt闂佹眹鍔岀€氼亞绱為崨顖滅＞妞ゆ柨鍚嬬€氭煡鏌￠崼鐔虹畵闁?heading/theta)

    def observe_target(self, targets_list: List['Target'], relative=True):
        """
        Observing target with a radius within dp
        :param relative: relative to uav itself
        :param targets_list: [class UAV]
        :return: None
        """
        self.target_observation = []  # Reset observed targets
        # 閻庡湱顭堝璺虹暦閻斿吋鍤斿瀣缁愭鏌″鍛煑缂佹顦靛畷妯侯吋閸偅钑夐柣鐔哥懃閸婂湱绮?
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
                    
    def observe_protector(self, protectors_list: List['Protector'], relative=True):
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
                if relative:
                    self.uav_communication.append(((uav.x - self.x) / self.dc,
                                                   (uav.y - self.y) / self.dc,
                                                   cos(uav.h) - cos(self.h),
                                                   sin(uav.h) - sin(self.h),
                                                   (uav.a - self.a) / self.Na))
                else:
                    self.uav_communication.append((uav.x / self.dc,
                                                   uav.y / self.dc,
                                                   cos(uav.h),
                                                   sin(uav.h),
                                                   uav.a / self.Na))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, float]],
                                        List[Tuple[float, float, float, float]], Tuple[float, float, float]):
        """
        :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
        """
        return self.uav_communication, self.target_observation, (self.x / self.dc, self.y / self.dc, self.a / self.Na)

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
            average_communication = -np.ones(4 + 1)  # empty communication  # TODO -1闂佸憡鑹鹃悧濠勬兜閸洖瑙?

        if observation:
            d_observation = []  # store the distance from each target to itself
            for x, y, vx, vy in observation:
                d_observation.append(min(self.distance(x, y, self.x, self.y), 1))

            # regularization by the distance
            observation = np.array(observation)
            observation_weighted = observation / np.array(d_observation)[:, np.newaxis]
            average_observation = np.mean(observation_weighted, axis=0)
        else:
            average_observation = -np.ones(4)  # empty observation  # TODO -1闂佸憡鑹鹃悧濠勬兜閸洖瑙?

        sb = np.array(sb)
        result = np.hstack((average_communication, average_observation, sb))
        return result

    def get_local_state(self) -> 'np.ndarray':
        """
        :return: np.ndarray
        """
        # using weighted mean method:
        return self.__get_local_state_by_weighted_mean()

    def __calculate_multi_target_tracking_reward(self, uav_list) -> float:
        """
        calculate multi target tracking reward
        :return: scalar [1, 2)
        """
        track_reward = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= self.dp:
                    reward = 1 + (self.dp - distance) / self.dp
                    # track_reward += clip_and_normalize(reward, 1, 2, 0)
                    track_reward += reward  # 濠电偛澶囬崜婵嗭耿娑旑湶ip, 闂侀潻璐熼崝蹇涙儍閻斿吋鍋ㄩ柕濠忛檮椤ρ冾熆閼哥數澧甸柛搴ｂ挅lip
        return track_reward

    def __calculate_duplicate_tracking_punishment(self, uav_list: List['UAV'], radio=2) -> float:
        """
        calculate duplicate tracking punishment
        :param uav_list: [class UAV]
        :param radio: radio闂佹椿娼块崝宥咁焽閻㈢绠崇憸宥夊春濡ゅ懎绠氶柍鍝勫暟缁嬪牓鏌ｉ妸銉ヮ伂閻庡灚鐗犲畷? 闁烩剝甯掗幊搴ㄥ吹椤撶喎绶炴慨妯虹－缁犲ジ鏌熼棃娑毿ら柣锝咁煼瀹曟濡烽敃鈧崝锔剧磽?
        :return: scalar (-e/2, -1/2]
        """
        total_punishment = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * exp((radio * self.dp - distance) / (radio * self.dp))
                    # total_punishment += clip_and_normalize(punishment, -e/2, -1/2, -1)
                    total_punishment += punishment  # 濠电偛澶囬崜婵嗭耿娑旑湶ip, 闂侀潻璐熼崝蹇涙儍閻斿吋鍋ㄩ柕濠忛檮椤ρ冾熆閼哥數澧甸柛搴ｂ挅lip
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
        return boundary_punishment  # 濠电偛澶囬崜婵嗭耿娑旑湶ip, 闂侀潻璐熼崝蹇涙儍閻斿吋鍋ㄩ柕濠忛檮椤ρ冾熆閼哥數澧甸柛搴ｂ挅lip
        # return clip_and_normalize(boundary_punishment, -1/2, 0, -1)

    def calculate_raw_reward(self, uav_list: List['UAV'], target__list: List['Target'], protector_list: List['Protector'], x_max, y_max):
        """
        calculate three parts of the reward/punishment for this uav
        :return: float, float, float
        """
        reward = self.__calculate_multi_target_tracking_reward(target__list)
        boundary_punishment = self.__calculate_boundary_punishment(x_max, y_max)
        punishment = self.__calculate_duplicate_tracking_punishment(uav_list)
        protector_punishment = self.__calculate_protector_collision_punishment(protector_list)
        return reward, boundary_punishment, punishment, protector_punishment

    def __calculate_protector_collision_punishment(self, protector_list: List['Protector']) -> float:
        """
        缂備胶濮崑鎾绘煕濡や焦绀堟繛鍫熷灩閹瑰嫬鈹戦崼顐も偓鏌ユ⒒閸愩劎澧曢柍褜鍓欓崥瀣磿閻㈢數纾炬慨妯哄⒔缁?
        - 闂?UAV 婵炴垶鎸哥花濂稿箲閵忋倕绠?Protector 闁荤姷鍎ょ换鍕€?< threshold 闂佸憡甯楅悷銈嗩殽閸ヮ剚鍋ㄩ柣鏃傤焾閸旓妇绱撻崘鈺冨暡缂佽鲸鐟ч幏褰掓偄鐏忎礁浜剧痪顓㈩棑缁€?
        - 闁哄鏅滈弻銊ッ洪弽顓炵９闁绘挸楠哥徊鍧楁煕閺嶃劍銇濇俊?[-K, 0]闂佹寧绋戦悧濠傦耿椤撀颁汗闁瑰搫绉堕閬嶆煕閺嶃劎澹勭紒杈ㄥ哺閺佸秶浠﹂懞銉モ偓鐢电磽娓氬洤骞橀柡?clip_and_normalize 閻熸粎澧楃敮濠勭博閹绢喖绀岄柡宓啰鍘?[-1,0]
        """
        # UAV 闂佺厧顨庢禍锝夋閳哄懎纭€濠电姴鍊荤粣鐐烘煥濞戞瀚扮憸鐗堢叀閺屽﹤顓奸崶鈺傜€梺鐟扮摠閻楃姷鍒掗婊勫?
        uav_radius = getattr(self, "radius", 0.5)
        worst_pen = 0.0  # 闂佽鍨弲婵堢礊閹烘挾鈻旀繛宸簼婵粍鎱ㄥ┑鎾剁ɑ闁哄棛鍠栭弫? 闁荤偞绋忛崝搴ㄥΦ濮樿泛绫嶉柣妯哄暱閸旓妇绱撻崘鈺冨暡缂?
        for prot in protector_list:
            # protector 闂備焦褰冮惉鑲┾偓鐟扮－閳ь剝顫夐懝楣冩儓濮椻偓楠炩偓?safe_r闂佹寧绋戦悧濠囧垂閵婏附濯撮悹鎭掑妽閺?prot.safe_r闂?
            prot_safe = getattr(prot, "safe_r", None)
            if prot_safe is None:
                # 闂佹椿娼块崝鎴犲垝椤栨粍濯奸柕鍫濇閻?prot.radius
                prot_safe = getattr(prot, "radius", 0.5)
            threshold = prot_safe + uav_radius  # 闁荤喐鐟ュΛ鏃傛嫻閻斿搫鏋堥柣妤€鐗婄€?闁荤姭鍋撻柨鏃囨閻忓﹪鏌涘Δ鈧敃銈囨?
            dx = prot.x - self.x
            dy = prot.y - self.y
            dist = hypot(dx, dy)
            if dist < threshold:
                # 闂佽鍨弲婵堢礊閹烘绠板璺烘捣閻涒晝绱掗崒婊呭笡闁搞劍鑹捐闁哄倸澧芥导鎰版煥濞戞瑥鍝烘繛纰卞墮椤曘儱螖閸曨厜銊╁级閳哄倸鐏﹂柛搴ｆ暩缁辨柨顫濆畷鍥嗐劌顭块崼鍡楃Ф缁€鍕偣閹邦喖娅愮紒?
                # 闂佸搫鐗滄禍婊呯礊婵犲啰鈻旈柍褜鍓熷畷鐘诲冀閵娿儱濮︾紓鍌氬暞閻旑剛妲愬┑鍥╃懝閻庯綆浜跺ú銈夋煥?- (threshold - dist)  -> 婵炲濮撮鍕姳?(-threshold, 0]
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
        if a == 0:  # 闂佸湱绮崝鏇熸櫠閻樿绀嗛柕鍫濇閻掍粙鏌ㄥ☉妯肩劮濠碘槅鍙冮幆鍥ㄦ媴缁涘鎮佺紓浣哄У椤ㄥ牆鈻撻幋鐐茬窞鐎广儱妫欑徊浠嬪箹?
            return self.raw_reward

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
        if a == 0:  # 闂佸湱绮崝鏇熸櫠閻樿绀嗛柕鍫濇閻掍粙鏌ㄥ☉妯肩劮濠碘槅鍙冮幆鍥ㄦ媴缁涘鎮佺紓浣哄У椤ㄥ牆鈻撻幋鐐茬窞鐎广儱妫欑徊浠嬪箹?
            return self.raw_reward

        neighbor_rewards = []
        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
        # 濠电偛澶囬崜婵嗭耿娓氣偓瀹曟繈鎮╅幓鎺戞辈PMI缂傚倸鍟崹鍦垝?
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

        # 婵犻潧鍊归悧鏇熸叏閹惰棄妞介悘鐐村劤閸旓妇绱撻崘鈺冾暡婵炵厧鐗撻弻?
        target_reward_weight = 1.0
        repetition_penalty_weight = 0.8
        self.epsilon = 0.25
        self.continue_tracing = 0.3
        
        best_score = float('-inf')
        best_angle = 0.0

        # 闂傚倸鎳庣换鎴濃攦閳ь剟鏌熼崹顐ｇ凡濠殿喒鏅犻弫宥咁潩椤愶紕闉峞psilon闂佹眹鍔岀€氼參銆呰閹娊宕堕钘変壕濠㈣泛顑呴銉╂⒒閸涱厾绠叉繝褉鍋撻梺鐑╂櫓閸犳鎮?
        if random.random() < self.epsilon:
            return np.random.randint(0, self.Na)
        else:
            for target in target_list:
                target_x, target_y = target.x, target.y

                # 閻熸粎澧楅幐鍛婃櫠閻樿绫嶉柣妯块哺閻粙鏌￠崼婵囨儓闁糕晛鐭傞幆鍕敊閻ｅ苯鐏遍梺姹囧妼鐎氼垳绮ｅ☉姘辩焿?
                dist_to_target = distance(self.x, self.y, target_x, target_y)

                # 闂備焦褰冪粔鎾囬崣澶嬩氦闁稿﹦鍠栭崵瀣煟閵娿儱顏柛搴ｆ暩缁辨柨顫濋崜褏顦梺鍏兼緲閸熷啿顬婃繝姘闁哄牏鏁搁柧鍌炴煛閸愵亜小婵懓顦靛鐢稿传閸曨剝鍚梻浣瑰絻缁夋挳藝閸欏浜ら柛濠勫枛閸ゅ鏌涘Δ鈧敃銈囨閻愬搫绀冮柛娑卞枟绗戦梺鍛婄啲缁犳垵锕㈤鍛氦闁稿﹦鍠栭崵瀣煕濮橆剙顨欑紒鏃€鎸抽幆鍕敊閻ｅ苯鐏?
                repetition_penalty = 0.0
                for uav in uav_list:
                    uav_x, uav_y = uav.x, uav.y
                    if (uav_x, uav_y) != (self.x, self.y):
                        dist_to_target_from_other_uav = distance(uav_x, uav_y, target_x, target_y)
                        if dist_to_target_from_other_uav < self.dc:
                            repetition_penalty += repetition_penalty_weight

                # 闁荤姳绶ょ槐鏇㈡偩閺勫繈浜归柟鎯у暱椤ゅ懘鏌ｉ埡濠傛灍闁绘牭缍侀幆鍐礋椤愩倗绠掗梺?
                score = target_reward_weight / dist_to_target - repetition_penalty

                # 闂佸搫绉烽～澶婄暤娴ｅ壊鍤楁俊銈傚亾闁搞劌閰ｉ弻鍛緞鐎ｎ亶浠撮梺鍝勭墐閸嬫挸霉閸忕⒈鐒芥繛韫嵆瀵?
                if score > best_score:
                    best_score = score
                    best_angle = np.arctan2(target_y - self.y, target_x - self.x) - self.h

        # 婵炲濯▍鎼妌tinue_tracing闂佹眹鍔岀€氼參銆呰閹娊宕堕敂鍓ь啍闂佸綊鏅插鎺旂箔閸屾稓鈻旈柍褜鍓氱粙澶愵敂閸曨倣妤€霉?
        if random.random() < self.continue_tracing:
            best_angle = 0
            
        # 闁诲繐绻愬Λ娆忥耿閿熺姴瀚夋繛鎴炵閻ｉ亶鎮峰▎鎰瑨閻庣懓鍟块蹇涱敊閹稿海銈梺纭咁嚃濠⑩偓閻犳劗鍠撶划瀣磼濡粯鐤囬梺鍛婃煟閸斿绱為弮鍌涱潟闁靛繒濮风粚?
        a = np.clip(best_angle / self.dt, -self.h_max, self.h_max)
        k = (self.Na - 1) * a / self.h_max
        a_idx = int(np.round((k + self.Na - 1) / 2))
        a_idx = max(0, min(self.Na - 1, a_idx))
        return a_idx
