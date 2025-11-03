import argparse
import os.path
import logging
from environment import Environment
from models.actor_critic import ActorCritic
from utils.args_util import get_config
from train import train, evaluate, run
from models.PMINet import PMINetwork
from utils.data_util import save_csv
from utils.draw_util import plot_reward_curve

logging.basicConfig(level=logging.WARNING)


def print_config(vdict, name="config"):
    """
    :param vdict: dict, 闂備浇顕ф鎼佹倶濮橆剦鐔嗘慨妞诲亾妤犵偛锕ㄧ粻娑㈠籍閳ь剙鈻嶉悩缁樼叆闁哄洨鍋涢埀顒€鎽滃☉鐢稿焵椤掑嫭鈷戦悷娆忓閸熷繘鏌涢悩鎰佹疁闁诡噣绠栭弻鍡楊吋閸℃せ鍋撻崹顐ょ闁瑰鍋熼幊鎰箾閸喐顥堥柡灞剧☉椤繈鏁愰崨顒€顥?
    :param name: str, 闂傚倸鍊烽懗鍫曞箠閹剧粯鍊舵慨妯挎硾缁犳壆绱掔€ｎ厽纭堕柡鍡愬€濋弻娑㈠箛闂堟稒鐏嶉梺绋块缁夊綊寮诲☉銏犲嵆闁靛鍎扮花浠嬫⒑閸︻厽娅曢柛鐘崇墵瀵鈽夊鍛澑闂佸搫娲㈤崝搴ㄥ礉閹间焦鈷戦悹鍥ｂ偓铏亶濠电偛寮剁划搴ㄥ礆?    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")


def print_args(args, name="args"):
    """
    :param args:
    :param name: str, 闂傚倸鍊烽懗鍫曞箠閹剧粯鍊舵慨妯挎硾缁犳壆绱掔€ｎ厽纭堕柡鍡愬€濋弻娑㈠箛闂堟稒鐏嶉梺绋块缁夊綊寮诲☉銏犲嵆闁靛鍎扮花浠嬫⒑閸︻厽娅曢柛鐘崇墵瀵鈽夊鍛澑闂佸搫娲㈤崝搴ㄥ礉閹间焦鈷戦悹鍥ｂ偓铏亶濠电偛寮剁划搴ㄥ礆?    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    for arg in vars(args):
        print("| {:<11} : {}".format(arg, getattr(args, arg)))
    print("-----------------------------------------")


def add_args_to_config(config, args):
    for arg in vars(args):
        # print("| {:<11} : {}".format(arg, getattr(args, arg)))
        config[str(arg)] = getattr(args, arg)


def main(args):
    # 闂傚倸鍊风粈渚€宕ョ€ｎ喖纾块柟鎯版鎼村﹪鏌ら懝鎵牚濞存粌缍婇弻娑㈠Ψ椤旂厧顫╁┑鈽嗗亝閿曘垽骞冨畡鎵虫瀻闊洦鎼╂导鈧梻渚€娼уΛ妤呮晝椤忓牆钃熼柣鏂挎惈閺嬪牊淇婇婵囥€冨瑙勬礋濮婄粯鎷呯憴鍕╀户濠电偟鍘у鈥崇暦濠靛洦鍎熼柍閿亾闁哄妫冮弻鐔告綇閸撗呮殸闂佹椿鍘介〃濠囧蓟濞戞矮娌柛鎾椻偓婵洤鈹?
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", args.method + ".yaml")
    config = get_config(config_path)
    add_args_to_config(config, args)
    print_config(config)
    print_args(args)

    # 闂傚倸鍊风粈渚€骞夐敍鍕殰婵°倕鍟畷鏌ユ煕瀹€鈧崕鎴犵礊閺嶎厽鐓欓柣妤€鐗婄欢鑼磼閳ь剙鐣濋崟顒傚幐婵犮垼娉涢敃銈夋偂閹洀ronment, agent
    env = Environment(n_uav=config["environment"]["n_uav"],
    env = Environment(n_uav=config["environment"]["n_uav"],
                      m_targets=config["environment"]["m_targets"],
                      n_protectors=config["environment"]["n_protectors"],
                      x_max=config["environment"]["x_max"],
                      y_max=config["environment"]["y_max"],
                      na=config["environment"]["na"])
    env.reset(config=config)

    state_dict = env.get_states()
    uav_state_dim = len(state_dict['uav'][0]) if state_dict['uav'] else config["actor_critic"]["hidden_dim"]
    protector_state_dim = len(state_dict['protector'][0]) if state_dict['protector'] else config["protector_actor_critic"]["hidden_dim"]
    target_state_dim = len(state_dict['target'][0]) if state_dict['target'] else config["target_actor_critic"]["hidden_dim"]

    uav_action_dim = config["environment"]["na"]
    protector_action_dim = config["protector"].get("na", uav_action_dim)
    target_action_dim = config["target"].get("na", uav_action_dim)

    if args.method == "C-METHOD":
        agents = {'uav': None, 'protector': None, 'target': None}
    else:
        device = config["devices"][0]  # 闂備礁鎲￠悷顖涚閻愬搫鏋侀柕鍫濇川缁犳梹銇勯幋锝嗙《妞ゅ繐宕埥澶愬箻鐠鸿櫣娈瞖vice
        agents = {
            'uav': ActorCritic(state_dim=uav_state_dim,
                               hidden_dim=config["actor_critic"]["hidden_dim"],
                               action_dim=uav_action_dim,
                               actor_lr=float(config["actor_critic"]["actor_lr"]),
                               critic_lr=float(config["actor_critic"]["critic_lr"]),
                               gamma=float(config["actor_critic"]["gamma"]),
                               device=device),
            'protector': ActorCritic(state_dim=protector_state_dim,
                                     hidden_dim=config["protector_actor_critic"]["hidden_dim"],
                                     action_dim=protector_action_dim,
                                     actor_lr=float(config["protector_actor_critic"]["actor_lr"]),
                                     critic_lr=float(config["protector_actor_critic"]["critic_lr"]),
                                     gamma=float(config["protector_actor_critic"]["gamma"]),
                                     device=device),
            'target': ActorCritic(state_dim=target_state_dim,
                                  hidden_dim=config["target_actor_critic"]["hidden_dim"],
                                  action_dim=target_action_dim,
                                  actor_lr=float(config["target_actor_critic"]["actor_lr"]),
                                  critic_lr=float(config["target_actor_critic"]["critic_lr"]),
                                  gamma=float(config["target_actor_critic"]["gamma"]),
                                  device=device)
        }
        agents['uav'].load(args.actor_path, args.critic_path)
        agents['protector'].load(args.protector_actor_path, args.protector_critic_path)
        agents['target'].load(args.target_actor_path, args.target_critic_path)
    if args.method == "C-METHOD" and args.phase in ("train", "evaluate"):
        raise ValueError("C-METHOD 婵☆垪鈧磭纭€闁哄棗鍊风粭澶愬绩椤栨稑鐦鑸电濞呫倝鎳楅幋鎺旂Ъ閻犱緡鍘剧划?閻犲洤瀚崣濠囨晬瀹€鍐惧殲濞达綀娉曢弫?run 闂傚啳鍩栭宀勫箣閺嶎兘鍋撻崨绮瑰亾婢跺顏ラ柛蹇旀构缁剟寮憴鍕€婇柕?)

    if args.phase == "train":
        return_list = train(config=config,
                            env=env,
                            agents=agents,
                            pmi=pmi,
                            num_episodes=args.num_episodes,
                            num_steps=args.num_steps,
                            frequency=args.frequency)
    elif args.phase == "evaluate":
        return_list = evaluate(config=config,
                               env=env,
                               agents=agents,
                               pmi=pmi,
                               num_steps=args.num_steps)
    elif args.phase == "run":
        return_list = run(config=config,
                          env=env,
                          pmi=pmi,
                          num_steps=args.num_steps)
    else:
        return
    plot_reward_curve(config, return_list["return_list"], "overall_return")
    plot_reward_curve(config, return_list["target_tracking_return_list"],
                      "target_tracking_return_list")
    plot_reward_curve(config, return_list["boundary_punishment_return_list"],
                      "boundary_punishment_return_list")
    plot_reward_curve(config, return_list["duplicate_tracking_punishment_return_list"],
                      "duplicate_tracking_punishment_return_list")
    plot_reward_curve(config, return_list["protector_collision_return_list"],
                      "protector_collision_return_list")
    plot_reward_curve(config, return_list["protector_return_list"],
                      "protector_return_list")
    plot_reward_curve(config, return_list["protector_protect_reward_list"],
                      "protector_protect_reward_list")
    plot_reward_curve(config, return_list["protector_block_reward_list"],
                      "protector_block_reward_list")
    plot_reward_curve(config, return_list["protector_failure_penalty_list"],
                      "protector_failure_penalty_list")
    plot_reward_curve(config, return_list["target_return_list"],
                      "target_return_list")
    plot_reward_curve(config, return_list["target_safety_reward_list"],
                      "target_safety_reward_list")
    plot_reward_curve(config, return_list["target_danger_penalty_list"],
                      "target_danger_penalty_list")
    plot_reward_curve(config, return_list["target_capture_penalty_list"],
                      "target_capture_penalty_list")
    plot_reward_curve(config, return_list["average_covered_targets_list"],
                      "average_covered_targets_list")
    plot_reward_curve(config, return_list["max_covered_targets_list"],
                      "max_covered_targets_list")


if __name__ == "__main__":
    # 闂傚倸鍊风粈渚€骞夐敍鍕殰婵°倕鍟伴惌娆撴煙鐎电啸缁惧彞绮欓弻鐔煎箲閹伴潧娈悗鐟版啞缁诲牓寮婚悢琛″亾濞戞顏嗙箔濮橆兘鏀芥い鏂挎惈閻忔煡鏌熼鎸庣【闁宠棄顦灒闁割煈鍣导鍐煟鎼淬埄鍟忛柛锝庡櫍瀹曟粓鎮㈡總澶婃濡炪倖鍔х粻鎴犵不椤栫偞鍊堕柣鎰嚟缁犳牠鎮介妞诲亾閹颁礁娈ㄩ梺褰掓？缁€浣虹不濞戞瑣浜滈柟鍝勭Х閸忓瞼鎲?
    parser = argparse.ArgumentParser(description="")

    # 婵犵數濮烽弫鎼佸磿閹寸姷绀婇柍褜鍓氶妵鍕即閸℃顏柛娆忕箻閺岋綁骞囬鍌欑驳闂佽鍨崕鐢稿蓟濞戞ǚ妲堥柛妤冨仜缁犵懓鈹戦埥鍡椾簼闁挎洏鍨藉?    parser.add_argument("--phase", type=str, default="train", choices=["train", "evaluate", "run"])
    parser.add_argument("-e", "--num_episodes", type=int, default=10000, help="闂傚倷娴囧畷鍨叏瀹曞洦顐介柨鐔哄Т閸屻劑鏌涢幘妤€鍟悘濠囨煟韫囨洖浠ч柛瀣尰閹便劌顭ㄩ崟鈺€绨婚梺鍝勫暙濞层倛顣挎繝?)
    parser.add_argument("-s", "--num_steps", type=int, default=200, help="婵犵數濮甸鏍闯椤栨粌绶ら柣锝呮湰瀹曞弶淇婇妶鍛櫤闁稿浜弻娑㈠焺閸忊晜鍨剁粋宥夊Χ閸℃瑧顔曢梺鐟邦嚟閸嬬喖鎯岄妶澶嬬厸闁逞屽墴閹崇偤濡烽敃鈧鍨攽閻樼粯娑ч悗姘煎墴瀹曟繂顭ㄩ崼鐔哄幐?)
    parser.add_argument("-f", "--frequency", type=int, default=100, help="闂傚倸鍊烽懗鍫曞箠閹剧粯鍊舵慨妯挎硾缁犳壆绱掔€ｎ厽纭堕柡鍡愬€濋弻娑㈠箛閵婏附鐝栭梺宕囨嚀缁夊綊寮诲澶婄厸濞达絽鎲″▓銊х磽娴ｄ粙鍝洪柣妤冨█瀵鈽夊Ο閿嬬€婚棅顐㈡储閸庢煡銆傞懖鈺冪＝濞达綀娅ｇ敮娑欐叏濡濮傞柛鈹惧亾濡炪倖甯婇懗鍫曞煀閺囩喆浜滄い鎾跺仦閹兼劙鏌嶇紒妯诲磳妤犵偛顑夐幃婊堝炊閳哄啰鍘梻鍌欒兌閹虫捇顢氶鐔稿弿闁圭虎鍠楅崵?)
    parser.add_argument("-a", "--actor_path", type=str, default=None, help="actor缂傚倸鍊搁崐鎼佸磹閹间礁鐤い鏍仜閸ㄥ倿鏌涢敂璇插箹闁搞劍绻堥弻娑㈩敃閿濆棛顦ョ紓浣哄珡閸ャ劎鍘遍梺閫涘嵆濞佳囧几濞戞瑧绠鹃柛娑卞亜閻忔煡鏌″畝瀣М濠碘剝鎮傛俊鐑藉Ψ椤旇崵纾介梻鍌欑劍濡炲潡宕㈡禒瀣？闁瑰墎鐡旈弫?)
    parser.add_argument("-c", "--critic_path", type=str, default=None, help="critic缂傚倸鍊搁崐鎼佸磹閹间礁鐤い鏍仜閸ㄥ倿鏌涢敂璇插箹闁搞劍绻堥弻娑㈩敃閿濆棛顦ョ紓浣哄珡閸ャ劎鍘遍梺閫涘嵆濞佳囧几濞戞瑧绠鹃柛娑卞亜閻忔煡鏌″畝瀣М濠碘剝鎮傛俊鐑藉Ψ椤旇崵纾介梻鍌欑劍濡炲潡宕㈡禒瀣？闁瑰墎鐡旈弫?)
    parser.add_argument("-p", "--pmi_path", type=str, default=None, help="pmi缂傚倸鍊搁崐鎼佸磹閹间礁鐤い鏍仜閸ㄥ倿鏌涢敂璇插箹闁搞劍绻堥弻娑㈩敃閿濆棛顦ョ紓浣哄珡閸ャ劎鍘遍梺閫涘嵆濞佳囧几濞戞瑧绠鹃柛娑卞亜閻忔煡鏌″畝瀣М濠碘剝鎮傛俊鐑藉Ψ椤旇崵纾介梻鍌欑劍濡炲潡宕㈡禒瀣？闁瑰墎鐡旈弫?)
    parser.add_argument("--protector_actor_path", type=str, default=None, help="婵犳鍣徊鐣岀矓閻熸噴褰掓偨椤ｅ〖or缂傚倸鍊搁崯顖炲垂閸︻厼鍨濋柛顐ｆ礀缁狙囨煕閹伴潧鏋涙繛鍛灲閹绗熼姘变哗缂?)
    parser.add_argument("--protector_critic_path", type=str, default=None, help="婵犳鍣徊鐣岀矓閻熸噴褰掓偨椤︽tic缂傚倸鍊搁崯顖炲垂閸︻厼鍨濋柛顐ｆ礀缁狙囨煕閹伴潧鏋涙繛鍛灲閹绗熼姘变哗缂?)
    parser.add_argument("--target_actor_path", type=str, default=None, help="闂佽绻愮换鎰崲濡ソ褰掓偨椤ｅ〖or缂傚倸鍊搁崯顖炲垂閸︻厼鍨濋柛顐ｆ礀缁狙囨煕閹伴潧鏋涙繛鍛灲閹绗熼姘变哗缂?)
    parser.add_argument("--target_critic_path", type=str, default=None, help="闂佽绻愮换鎰崲濡ソ褰掓偨椤︽tic缂傚倸鍊搁崯顖炲垂閸︻厼鍨濋柛顐ｆ礀缁狙囨煕閹伴潧鏋涙繛鍛灲閹绗熼姘变哗缂?)
    parser.add_argument("-m", "--method", help="", default="MAAC-R", choices=["MAAC", "MAAC-R", "MAAC-G", "C-METHOD"])
    # 闂傚倷娴囧畷鐢稿窗閹扮増鍋￠弶鍫氭櫅缁躲倕螖閿濆懎鏆為柛濠囨涧闇夐柣妯烘▕閸庡繒鈧懓鎲＄换鍫ュ蓟閻旇　鍋撳☉娅亞绮顑芥斀妞ゆ柨鎼悘鏌ユ煙椤旀寧纭鹃柍钘夘槸铻ｉ柛顭戝櫘娴煎啴鏌ｆ惔銏╁晱闁革綆鍣ｅ畷婊堟偄婵傚娈?    main_args = parser.parse_args()

    # 闂傚倷娴囧畷鍨叏閹绢噮鏁勯柛娑欐綑閻ゎ喖霉閸忓吋缍戦柡瀣╃窔閺屾洟宕煎┑鍥舵￥闂佸磭绮Λ鍐箖瑜版帒鐐婇柕濞垮劤缁佸嘲鈹戦鍡欑ɑ闁告梹鐟╁?    main(main_args)
