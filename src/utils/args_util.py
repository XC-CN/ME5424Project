import os
import random
import time
import yaml
import numpy as np
import torch


def get_config(config_file, phase="train"):
    """
    Load configuration and optionally prepare output directories.
    """
    with open(config_file, 'r', encoding="UTF-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if 'seed' in config and config['seed'] is not None:
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    run_id = str(os.getpid())
    exp_name = '_'.join([
        config['exp_name'],
        time.strftime('%Y-%b-%d-%H-%M-%S'), run_id
    ])

    save_dir = os.path.join(config['result_dir'], exp_name)
    config['save_dir'] = save_dir
    config['phase'] = phase

    mkdir(config['result_dir'])

    prepare_outputs = phase != "evaluate"
    if prepare_outputs:
        mkdir(save_dir)
        # 模型文件目录
        mkdir(os.path.join(save_dir, "actor"))
        mkdir(os.path.join(save_dir, "critic"))
        mkdir(os.path.join(save_dir, "pmi"))
        # 配置文件目录
        mkdir(os.path.join(save_dir, "config"))
        # 指标数据目录
        mkdir(os.path.join(save_dir, "metrics"))
        mkdir(os.path.join(save_dir, "metrics", "csv"))
        mkdir(os.path.join(save_dir, "metrics", "plots"))
        # 其他目录
        mkdir(os.path.join(save_dir, "animated"))
        mkdir(os.path.join(save_dir, "t_xy"))
        mkdir(os.path.join(save_dir, "u_xy"))
        mkdir(os.path.join(save_dir, "p_xy"))
        mkdir(os.path.join(save_dir, "covered_target_num"))

    set_device(config)

    if prepare_outputs:
        # 将args.yaml保存到config目录
        args_save_name = os.path.join(save_dir, "config", 'args.yaml')
        with open(args_save_name, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    return config


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def set_device(config):
    if config['gpus'] == -1 or not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print('use cpu')
        config['devices'] = [torch.device('cpu')]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(config['first_device'],
                                                                            config['first_device'] + config['gpus']))
        print('use gpus: {}'.format(config['gpus']))
        config['devices'] = [torch.device('cuda', i) for i in range(config['first_device'],
                                                                    config['first_device'] + config['gpus'])]


if __name__ == "__main__":
    example = get_config("../configs/MAAC-R.yaml")
    print(type(example))
    print(example)
