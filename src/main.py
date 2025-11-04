# -*- coding: utf-8 -*-
import argparse
import logging
import os
from pathlib import Path

from environment import Environment
from models.actor_critic import ActorCritic
from models.PMINet import PMINetwork
from train import train, evaluate, run
from utils.args_util import get_config
from utils.data_util import save_csv
from utils.draw_util import plot_reward_curve


logging.basicConfig(level=logging.WARNING)


def print_config(config, name="config"):
    """Pretty-print the top-level configuration values."""
    print("-----------------------------------------")
    print(f"|This is the summary of {name}:")
    for key, value in config.items():
        if value is None:
            continue
        print(f"|{key:11}\t: {value}")
    print("-----------------------------------------")


def print_args(args, name="args"):
    """Display command-line arguments in a formatted table."""
    print("-----------------------------------------")
    print(f"|This is the summary of {name}:")
    for arg in vars(args):
        print(f"| {arg:<11} : {getattr(args, arg)}")
    print("-----------------------------------------")


def _latest_weight_file(directory: Path, prefix: str):
    best_path = None
    best_epoch = -1
    for file in directory.glob(f"{prefix}_weights_*.pth"):
        stem_parts = file.stem.split("_")
        try:
            epoch = int(stem_parts[-1])
        except (IndexError, ValueError):
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = file
        elif epoch == best_epoch and best_path is not None:
            if file.stat().st_mtime > best_path.stat().st_mtime:
                best_path = file
    return best_path


def find_latest_checkpoints(result_dir: str, current_save_dir: str):
    """
    Locate the most recent experiment directory that contains complete actor/critic
    checkpoints for all three agent roles.
    """
    base = Path(result_dir)
    if not base.exists():
        return {}, None

    current_path = Path(current_save_dir).resolve()
    candidates = []

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.resolve() == current_path:
            continue

        actor_dir = child / "actor"
        critic_dir = child / "critic"
        if not actor_dir.exists() or not critic_dir.exists():
            continue

        role_paths = {}
        missing = False
        for role in ("uav", "protector", "target"):
            actor_file = _latest_weight_file(actor_dir, f"{role}_actor")
            critic_file = _latest_weight_file(critic_dir, f"{role}_critic")
            if actor_file is None or critic_file is None:
                missing = True
                break
            role_paths[role] = (actor_file, critic_file)

        if missing or not role_paths:
            continue

        mtimes = [path.stat().st_mtime for pair in role_paths.values() for path in pair]
        candidates.append((max(mtimes), role_paths, child))

    if not candidates:
        return {}, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_paths, source_dir = candidates[0]
    return best_paths, source_dir


def add_args_to_config(config, args):
    """Append argparse values to the configuration dictionary."""
    for arg in vars(args):
        config[str(arg)] = getattr(args, arg)


def build_environment(config):
    """Instantiate and reset the environment using configuration values."""
    env_cfg = config["environment"]
    env = Environment(
        n_uav=env_cfg["n_uav"],
        m_targets=env_cfg["m_targets"],
        n_protectors=env_cfg["n_protectors"],
        x_max=env_cfg["x_max"],
        y_max=env_cfg["y_max"],
        na=env_cfg["na"],
    )
    env.reset(config=config)
    return env


def build_pmi(config, args):
    """Create the PMI network and optionally load pretrained weights."""
    pmi_cfg = config.get("pmi", {})
    pmi = PMINetwork(
        hidden_dim=pmi_cfg.get("hidden_dim", 64),
        b2_size=pmi_cfg.get("b2_size", 3000),
    )
    if getattr(args, "pmi_path", None):
        pmi.load(args.pmi_path)
    return pmi


def build_agents(config, state_dims, action_dims, args):
    """Initialize agent policies for UAV, protector, and target roles."""
    if args.method == "C-METHOD":
        return {"uav": None, "protector": None, "target": None}

    device = config["devices"][0]
    agents = {
        "uav": ActorCritic(
            state_dim=state_dims["uav"],
            hidden_dim=config["actor_critic"]["hidden_dim"],
            action_dim=action_dims["uav"],
            actor_lr=float(config["actor_critic"]["actor_lr"]),
            critic_lr=float(config["actor_critic"]["critic_lr"]),
            gamma=float(config["actor_critic"]["gamma"]),
            device=device,
        ),
        "protector": ActorCritic(
            state_dim=state_dims["protector"],
            hidden_dim=config["protector_actor_critic"]["hidden_dim"],
            action_dim=action_dims["protector"],
            actor_lr=float(config["protector_actor_critic"]["actor_lr"]),
            critic_lr=float(config["protector_actor_critic"]["critic_lr"]),
            gamma=float(config["protector_actor_critic"]["gamma"]),
            device=device,
        ),
        "target": ActorCritic(
            state_dim=state_dims["target"],
            hidden_dim=config["target_actor_critic"]["hidden_dim"],
            action_dim=action_dims["target"],
            actor_lr=float(config["target_actor_critic"]["actor_lr"]),
            critic_lr=float(config["target_actor_critic"]["critic_lr"]),
            gamma=float(config["target_actor_critic"]["gamma"]),
            device=device,
        ),
    }

    agents["uav"].load(args.actor_path, args.critic_path)
    agents["protector"].load(args.protector_actor_path, args.protector_critic_path)
    agents["target"].load(args.target_actor_path, args.target_critic_path)

    return agents


def summarise_metrics(config, metrics):
    """Persist metric summaries to disk and draw reward curves."""
    if not metrics:
        return

    phase = config.get("phase")
    if phase == "evaluate":
        eval_cfg = config.get("evaluate", {})
        if not eval_cfg.get("save_outputs", False):
            return

    save_csv(config, metrics)

    curves = [
        ("return_list", "overall_return"),
        ("target_tracking_return_list", "target_tracking_return_list"),
        ("boundary_punishment_return_list", "boundary_punishment_return_list"),
        ("duplicate_tracking_punishment_return_list", "duplicate_tracking_punishment_return_list"),
        ("protector_collision_return_list", "protector_collision_return_list"),
        ("protector_return_list", "protector_return_list"),
        ("protector_protect_reward_list", "protector_protect_reward_list"),
        ("protector_block_reward_list", "protector_block_reward_list"),
        ("protector_failure_penalty_list", "protector_failure_penalty_list"),
        ("target_return_list", "target_return_list"),
        ("target_safety_reward_list", "target_safety_reward_list"),
        ("target_danger_penalty_list", "target_danger_penalty_list"),
        ("target_capture_penalty_list", "target_capture_penalty_list"),
        ("average_covered_targets_list", "average_covered_targets_list"),
        ("max_covered_targets_list", "max_covered_targets_list"),
    ]

    for key, filename in curves:
        values = metrics.get(key)
        if values is not None:
            plot_reward_curve(config, values, filename)


def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", f"{args.method}.yaml")
    config = get_config(config_path, phase=args.phase)
    config["phase"] = args.phase
    config["render_when_train"] = bool(args.render_when_train)

    if args.phase == "evaluate":
        defaults, source_dir = find_latest_checkpoints(config["result_dir"], config["save_dir"])
        if defaults:
            if getattr(args, "actor_path", None) is None and "uav" in defaults:
                args.actor_path = str(defaults["uav"][0])
            if getattr(args, "critic_path", None) is None and "uav" in defaults:
                args.critic_path = str(defaults["uav"][1])
            if getattr(args, "protector_actor_path", None) is None and "protector" in defaults:
                args.protector_actor_path = str(defaults["protector"][0])
            if getattr(args, "protector_critic_path", None) is None and "protector" in defaults:
                args.protector_critic_path = str(defaults["protector"][1])
            if getattr(args, "target_actor_path", None) is None and "target" in defaults:
                args.target_actor_path = str(defaults["target"][0])
            if getattr(args, "target_critic_path", None) is None and "target" in defaults:
                args.target_critic_path = str(defaults["target"][1])
            if source_dir is not None:
                config["default_checkpoint_dir"] = str(source_dir)

    add_args_to_config(config, args)
    print_config(config)
    print_args(args)

    env = build_environment(config)

    state_dict = env.get_states()
    state_dims = {
        "uav": len(state_dict["uav"][0]) if state_dict["uav"] else config["actor_critic"]["hidden_dim"],
        "protector": len(state_dict["protector"][0]) if state_dict["protector"] else config["protector_actor_critic"]["hidden_dim"],
        "target": len(state_dict["target"][0]) if state_dict["target"] else config["target_actor_critic"]["hidden_dim"],
    }

    action_dims = {
        "uav": config["environment"]["na"],
        "protector": config.get("protector", {}).get("na", config["environment"]["na"]),
        "target": config.get("target", {}).get("na", config["environment"]["na"]),
    }

    agents = build_agents(config, state_dims, action_dims, args)
    pmi = build_pmi(config, args)

    if args.method == "C-METHOD" and args.phase in {"train", "evaluate"}:
        raise ValueError("C-METHOD is not supported for train/evaluate phases in run().")

    if args.phase == "train":
        metrics = train(
            config=config,
            env=env,
            agents=agents,
            pmi=pmi,
            num_episodes=args.num_episodes,
            num_steps=args.num_steps,
            frequency=args.frequency,
        )
    elif args.phase == "evaluate":
        metrics = evaluate(
            config=config,
            env=env,
            agents=agents,
            pmi=pmi,
            num_steps=args.num_steps,
        )
    elif args.phase == "run":
        metrics = run(
            config=config,
            env=env,
            pmi=pmi,
            num_steps=args.num_steps,
        )
    else:
        raise ValueError(f"Unknown phase '{args.phase}'.")

    summarise_metrics(config, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent aerial combat training entry point.")

    parser.add_argument("--phase", type=str, default="train", choices=["train", "evaluate", "run"])
    parser.add_argument("-e", "--num_episodes", type=int, default=10000, help="Number of training episodes to run.")
    parser.add_argument("-s", "--num_steps", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("-f", "--frequency", type=int, default=100, help="Logging or evaluation frequency in episodes.")
    parser.add_argument("--render_when_train", action='store_true', default=False, help='Enable live rendering during training.')
    parser.add_argument("-a", "--actor_path", type=str, default=None, help="Path to a pretrained actor checkpoint.")
    parser.add_argument("-c", "--critic_path", type=str, default=None, help="Path to a pretrained critic checkpoint.")
    parser.add_argument("-p", "--pmi_path", type=str, default=None, help="Path to a pretrained PMI network checkpoint.")
    parser.add_argument("--protector_actor_path", type=str, default=None, help="Path to a pretrained protector actor checkpoint.")
    parser.add_argument("--protector_critic_path", type=str, default=None, help="Path to a pretrained protector critic checkpoint.")
    parser.add_argument("--target_actor_path", type=str, default=None, help="Path to a pretrained target actor checkpoint.")
    parser.add_argument("--target_critic_path", type=str, default=None, help="Path to a pretrained target critic checkpoint.")
    parser.add_argument("-m", "--method", default="MAAC-R", choices=["MAAC", "MAAC-R", "MAAC-G", "C-METHOD"])

    main(parser.parse_args())
