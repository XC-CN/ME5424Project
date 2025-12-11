# Eagle-Hen Chain Competition: Phased Training with Gymnasium + Stable-Baselines3

> The current main focus of this repository is: In a Box2D physics environment, train a hen and an eagle in two phases using **Curriculum Learning** to address the cold-start problem of "both agents being too weak to converge."

## Project Overview

- **Physics Engine**: `Box2D`, simulating the inertia and swing of the chick chain behind the hen using distance joints.
- **Reinforcement Learning Framework**: `Gymnasium` + `Stable-Baselines3 (PPO)`.
- **Core Environment Files**:
  - `src/curriculum_env.py`
    - `PhysicsConfig`: Unified physical parameter configuration (time step, world size, radii, chain length, etc.).
    - `BasePhysicsEnv`: Shared Box2D infrastructure (world, rigid bodies, distance joints, observation construction).
    - `HenTrainingEnv`: Phase 1 — Train the hen against a heuristic-controlled eagle.
    - `EagleTrainingEnv`: Phase 2 — Train the eagle against a **frozen hen policy**.
- **Training Scripts**:
  - `src/train_hen.py`: Train the hen and output `hen_stage_1.zip`.
  - `src/train_eagle.py`: Load `hen_stage_1` as a training partner, train the eagle, and output `eagle_stage_1.zip`.

The previous multi-agent MAAC / MAAC-R approach is no longer the main focus of the project. Relevant code is retained in `src/` for traceability but will not be documented in this README.

---

## Environment and Dependencies

- **Recommended Python Version**: 3.9 or higher (3.10/3.11 are preferred for better performance).
- **Key Dependencies**:
  - Numerical Computing & Visualization: `numpy`, `scipy`, `matplotlib`, `pillow`, `imageio`, `tqdm`.
  - Deep Learning & Logging: `torch`, `torchvision`, `torchaudio`, `tensorboard`.
  - Reinforcement Learning & Physics: `gymnasium`, `stable-baselines3`, `pybox2d` (install via Conda).

### Installation Steps

Execute the following commands in the project root directory (a virtual environment or Conda environment is recommended):

```bash
pip install -r requirements.txt
```

If encountering missing Box2D-related modules (e.g., `Box2D` or `pybox2d`), install via Conda:

```bash
conda install -c conda-forge pybox2d
```

For Conda users, activate the environment first (example):

```bash
conda activate ME5424Project
```

---

## Physics Environment and State Space

### Physics Configuration (`PhysicsConfig`)

`PhysicsConfig` defines critical parameters for the Box2D world. Key fields include:

- `world_size`: Half-side length of the world; the simulation space is \[-world_size, world_size\]^2.
- `dt`: Simulation time step (e.g., `1/30` seconds).
- `max_steps`: Maximum steps per episode.
- `hen_radius` / `eagle_radius` / `chick_radius`: Radii of rigid bodies for the hen, eagle, and chicks.
- `chain_links`: Number of segments in the chick chain.
- `chain_spacing`: Static distance between adjacent chicks.
- `hen_max_speed` / `eagle_max_speed`: Maximum linear speed for each agent.
- `max_force`: Maximum force applied to the center of a rigid body per step (controls acceleration).
- `catch_radius`: Distance threshold for the eagle to "catch the tail of the chick chain."

Physical properties (e.g., map size, chain length, movement speed) can be adjusted by modifying default values in `PhysicsConfig`, with all environment instances sharing this underlying logic.

### Action and Observation Space (`BasePhysicsEnv`)

- **Action Space**
  - Type: `gymnasium.spaces.Box`
  - Shape: `shape=(2,)`
  - Interpretation: 2D acceleration direction (x, y) with values in `[-1, 1]`, internally mapped to `max_force` with speed clipping.

- **Observation Space**
  - Type: `gymnasium.spaces.Box`, `shape=(12,)`.
  - Normalized vectors in the current agent's (hen/eagle) local coordinate system, including:
    - Self-position (x, y), normalized by `world_size`.
    - Self-velocity vector, normalized by the corresponding `*_max_speed`.
    - Opponent's relative position vector.
    - Relative position vector of the chain's tail.
    - Tail velocity.
    - Average chain stretch degree.
    - Ratio of current step to `max_steps`.

During training:
- **Hen training**: The environment returns observations from the **hen's perspective**.
- **Eagle training**: The environment returns observations from the **eagle's perspective**, while additionally constructing hen-perspective observations for the frozen policy when required.

---

## Phase 1: Train the Hen (`HenTrainingEnv` + `train_hen.py`)

### Training Objective

- Maintain the hen's position between the eagle and the chick chain to delay the eagle's approach to the chain's tail.
- Ensure stability while moving the physical chain, avoiding excessive stretching and violent swings.

The eagle in the environment is controlled by heuristic rules, pursuing the chain's tail with simple lateral perturbations (serving as a "rule-based training partner").

### Execution Command

Run in the project root directory:

```bash
python src/train_hen.py --total-steps 1000000
```

Or:

```bash
python src/train_hen.py --total-steps 300000 --eval-freq 10000 --save-dir results/curriculum --seed 42
```

Key Parameter Explanations:
- `--total-steps`: Total training steps for PPO (corresponding to `model.learn(total_timesteps=...)`).
- `--eval-freq`: Interval (in steps) for evaluation and best-model saving via `EvalCallback`.
- `--save-dir`: Output directory for models and logs (default: `results/curriculum`).
- `--seed`: Random seed (for environment and PPO).

### Phase 1 Behavior Visualization

To intuitively observe how the hen defends against the heuristic eagle while moving the chick chain, a visualization script is provided:

```bash
python src/visualize_hen_stage1.py --episodes 1 --fps 90
```

By default, the script loads `results/curriculum/best_model.zip` as the hen's policy (the highest-scoring model during training). Adjust the number of episodes with `--episodes` and frame rate with `--fps`; specify a custom model path with `--model`. The program exits after closing the visualization window.

- **Hen**: Orange circle.
- **Eagle**: Blue circle.
- **Chick Chain**: Green small circles (all chicks in the chain are rendered).
- Coordinate range matches the physics world (\[-world_size, world_size\]^2), with x/y axes representing position.

### Parallel Training and Hardware Acceleration

To leverage multi-core CPUs (e.g., i7-14700KF) and enhance data diversity (IID), `src/train_hen.py` defaults to **parallel training mode**:

- **Multi-environment Parallelism (`n_envs=16`)**: Uses `SubprocVecEnv` to launch 16 independent processes for simultaneous data collection, significantly improving sampling speed (FPS).
- **Decorrelating Data**: Parallel sampling effectively reduces temporal correlation in single-environment data, leading to more uniform data distribution and improved PPO stability.
- **GPU Acceleration Optimization**: For high-performance GPUs (e.g., RTX 5080), `batch_size=512` and `n_steps=256` are configured to balance memory throughput and update frequency.

> **Note**: For machines with fewer CPU cores, reduce `n_envs` in `src/train_hen.py` (e.g., to 4 or 8).

### Output Results

By default, the Phase 1 script generates:
- **Best Hen Model**:
  - Automatically saved by `EvalCallback` in `--save-dir`.
  - Explicitly saved upon script completion: `results/curriculum/hen_stage_1.zip`.
- **TensorBoard Logs**:
  - Path: `results/curriculum/tb`.
- Visualization Command:
  ```bash
  tensorboard --logdir results/curriculum
  ```

---

## Phase 2: Train the Eagle (`EagleTrainingEnv` + `train_eagle.py`)

### Training Objective

- Enable the eagle to learn effective approach and breakthrough strategies against a defensive-trained hen.
- Reliably capture the chain's tail despite the hen's interference.

In this phase, the hen's policy is **frozen (no parameter updates)** and only participates as part of the environment; the eagle's PPO is trained from scratch.

### Execution Command

Ensure Phase 1 is completed and `result/curriculum/hen_stage_1.zip` is generated, then run:

```bash
python src/train_eagle.py
```

Key Parameter Explanations:
- `--hen-model`: Path to the hen's PPO model trained in Phase 1.
- Other parameters are consistent with Phase 1: `--total-steps`, `--eval-freq`, `--save-dir`, `--seed`.

### Phase 2 Behavior Visualization

To observe the eagle's offensive behavior against the frozen defensive hen, use the Phase 2 visualization script:

```bash
python src/visualize_eagle_stage2.py --episodes 1 --fps 60
```

- **Eagle**: Blue circle (controlled by the Phase 2-trained policy).
- **Hen**: Orange circle (using the frozen Phase 1 policy).
- **Chick Chain**: Green small circles; the tail turns red when captured by the eagle.
- **Hen's Wings**: Red line segments indicating the blocking zone width (determined by `block_margin`).

Optional Parameters:
- `--model`: Path to the eagle's policy model (default: `results/curriculum/eagle_stage_1.zip`).
- `--hen-model`: Path to the hen's policy model (default: `results/curriculum/hen_stage_1.zip`).
- `--episodes`: Number of visualization episodes.
- `--fps`: Frame rate.

### Frozen Hen Policy and Perspective Switching

The internal logic of `EagleTrainingEnv.step()` is as follows:
1. Construct **hen-perspective observations** from the current physical state: `hen_obs = self._get_obs(role="hen")`.
2. Inference with the frozen hen model: `hen_action, _ = self.hen_model.predict(hen_obs, deterministic=True)`.
3. Update the hen's rigid body state using `hen_action` (no network parameter updates).
4. Update the eagle's rigid body state using the externally provided `action`.
5. Advance the Box2D world by one step, construct observations and rewards from the **eagle's perspective**, and return them.

For the training eagle, the opponent is simply "a challenging hen policy" with no need to concern itself with internal implementation details.

### Output Results

The Phase 2 script generates:
- **Best Eagle Model**: Automatically saved to `--save-dir` by `EvalCallback`.
- **Phase 2 Model Snapshot**: Saved as `results/curriculum/eagle_stage_1.zip` upon script completion for potential subsequent joint fine-tuning.
- **TensorBoard Logs**: Reuses the `results/curriculum/tb` directory.

---

## Phase 3 (Optional): Joint Fine-Tuning of Hen + Eagle

This repository **only provides training scripts for Phases 1 and 2**. Phase 3 is an optional extension:
- Load models from Phases 1 and 2: `hen_stage_1.zip` and `eagle_stage_1.zip`.
- Allow simultaneous parameter updates for both the hen and eagle in a joint environment, with moderate learning rate increases or shortened training steps for small-scale fine-tuning.
- Reward design can reuse Phase-specific objectives:
  - Hen: Protect the chain's tail and block the eagle.
  - Eagle: Capture the tail efficiently.

To implement Phase 3, create `train_stage3.py` reusing the physical environment in `curriculum_env.py`, and alternately call `predict` and `learn` for both PPO policies in a loop.

---

## Result Analysis and Visualization

### TensorBoard Curves

- Both Phases 1 and 2 generate TensorBoard logs in `--save-dir`.
- Example default path: `results/curriculum/tb`.

Launch Command Example:
```bash
tensorboard --logdir results/curriculum
```

Monitored Metrics:
- Average reward per step/episode.
- Loss function trends and other training indicators.

### Terminal Output Parameter Explanations

During training, the terminal periodically outputs log tables with the following parameter interpretations:

```text
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 287          |  <-- Average number of steps per episode
|    ep_rew_mean          | 1.68         |  <-- Average cumulative reward per episode (higher is better)
| time/                   |              |
|    fps                  | 3500         |  <-- Frames Per Second (training speed, steps sampled per second)
|    iterations           | 74           |  <-- Number of main PPO algorithm iterations
|    time_elapsed         | 86           |  <-- Total training time elapsed (seconds)
|    total_timesteps      | 303104       |  <-- Total number of environment steps sampled
| train/                  |              |
|    approx_kl            | 0.00417      |  <-- Approximate KL divergence (measures policy update magnitude; excessive values indicate overly aggressive updates)
|    clip_fraction        | 0.0267       |  <-- Fraction of samples clipped by PPO's clipping mechanism
|    clip_range           | 0.2          |  <-- PPO clipping threshold (hyperparameter)
|    entropy_loss         | -2.44        |  <-- Negative policy entropy (larger absolute values indicate more random actions/exploration)
|    explained_variance   | 0.87         |  <-- Explained variance of the value function (closer to 1 indicates better Value Net predictions)
|    learning_rate        | 0.0003       |  <-- Current learning rate
|    loss                 | 0.651        |  <-- Total loss (Policy Loss + Value Loss + Entropy Loss)
|    n_updates            | 730          |  <-- Total number of gradient updates
|    policy_gradient_loss | -0.00234     |  <-- Policy gradient loss component
|    std                  | 0.82         |  <-- Standard deviation of the action distribution (reflects exploration; larger initially, decreasing over time)
|    value_loss           | 1.54         |  <-- Value function loss component (Critic network error)
------------------------------------------
```

**Key Metrics to Monitor**:
1. **`ep_rew_mean`**: Most intuitive performance indicator; should gradually increase with training.
2. **`explained_variance`**: If persistently near 0 or negative, the Value Network fails to fit rewards—adjust network architecture or hyperparameters.
3. **`entropy_loss`**: Maintain moderate values early for exploration; decreasing absolute values indicate policy convergence. Prematurely approaching 0 suggests early convergence.

### Custom Evaluation and Playback Example

No standalone evaluation script is included, but custom evaluation/playback can be implemented as follows (example code):

```python
from stable_baselines3 import PPO
from curriculum_env import HenTrainingEnv, PhysicsConfig

cfg = PhysicsConfig()
env = HenTrainingEnv(config=cfg, seed=0)
model = PPO.load("results/curriculum/hen_stage_1.zip")

obs, _ = env.reset()
for _ in range(cfg.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # Add custom rendering or data logging logic here
    if terminated or truncated:
        break
```

Extend this code with rendering, statistics, or video recording to create an evaluation pipeline tailored to specific experimental needs.

---

## Frequently Asked Questions (FAQ)

- **Dependency Installation Failures (especially Box2D-related)**
  - Update `pip` first: `pip install --upgrade pip`.
  - Install separately: `pip install "gymnasium[box2d]" box2d-py`.
  - Avoid overly new Python versions; prioritize 3.9–3.11.

- **TensorBoard Cannot Load Logs**
  - Ensure `--save-dir` in the training command matches `tensorboard --logdir`.
  - Verify event files (`events.out.tfevents.*`) exist in `results/curriculum/tb`.

- **Training Divergence or High Volatility**
  - Increase `--total-steps` to extend training duration.
  - Adjust `PhysicsConfig` (e.g., `max_steps`, `catch_radius`) to match task difficulty.
  - Tune PPO hyperparameters (`learning_rate`, `gamma`, `batch_size`, etc.) in `train_hen.py`/`train_eagle.py` as needed.

- **OMP Library Conflict (`OMP: Error #15`)**
  - Set the environment variable in Windows PowerShell first:
    ```powershell
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python src/train_hen.py ...
    ```

---

## Directory Structure Relevant to This README

- `src/curriculum_env.py`
  - `PhysicsConfig`, `BasePhysicsEnv`
  - `HenTrainingEnv`: Hen training environment.
  - `EagleTrainingEnv`: Eagle training environment.
- `src/train_hen.py`: Phase 1 training script.
- `src/train_eagle.py`: Phase 2 training script.
- `requirements.txt`: Dependency list for the curriculum learning pipeline.

Other files (e.g., `src/environment.py`, `src/main.py`, `results/MAAC*`) are legacy implementations of the previous multi-agent approach and are not part of the current recommended workflow—retained only for reference.

## 老鹰–母鸡链条对抗：基于 Gymnasium + Stable-Baselines3 的分阶段训练

> 本仓库当前主线为：在 Box2D 物理环境中，通过 **课程学习（Curriculum Learning）** 分两阶段训练母鸡与老鹰，解决“双方都太弱、难以收敛”的冷启动问题。

### 项目概览

- **物理引擎**：`Box2D`，通过距离关节模拟母鸡身后的小鸡链条惯性与摆动。
- **强化学习框架**：`Gymnasium` + `Stable-Baselines3 (PPO)`。
- **核心环境文件**：
  - `src/curriculum_env.py`
    - `PhysicsConfig`：统一的物理参数配置（步长、世界尺寸、半径、链长等）。
    - `BasePhysicsEnv`：共享 Box2D 底层（世界、刚体、距离关节、观测构造）。
    - `HenTrainingEnv`：阶段一——训练母鸡，对手为启发式老鹰。
    - `EagleTrainingEnv`：阶段二——训练老鹰，对手为 **冻结的母鸡策略**。
- **训练脚本**：
  - `src/train_hen.py`：训练母鸡，输出 `hen_stage_1.zip`。
  - `src/train_eagle.py`：加载 `hen_stage_1` 作为陪练，训练老鹰，输出 `eagle_stage_1.zip`。

旧的多智能体 MAAC / MAAC-R 思路已不再作为项目主线，相关代码仍保留在 `src/` 中以便回溯，但 README 不再对其进行说明。

---

## 环境与依赖

- **推荐 Python 版本**：3.9 及以上（3.10/3.11 表现更好）。
- **主要依赖**：
  - 数值与可视化：`numpy`、`scipy`、`matplotlib`、`pillow`、`imageio`、`tqdm`。
  - 深度学习与日志：`torch`、`torchvision`、`torchaudio`、`tensorboard`。
  - 强化学习与物理：`gymnasium`、`stable-baselines3`、`pybox2d`（通过 Conda 安装）。

### 安装步骤

在项目根目录执行（建议使用虚拟环境或 Conda 环境）：

```bash
pip install -r requirements.txt
```

若提示缺少 Box2D 相关模块（如 `Box2D` 或 `pybox2d`），建议使用 Conda 安装：

```bash
conda install -c conda-forge pybox2d
```

如使用 Conda，可先激活环境（示例）：

```bash
conda activate ME5424Project
```

---

## 物理环境与状态空间

### 物理配置（`PhysicsConfig`）

`PhysicsConfig` 定义了 Box2D 世界的关键参数，部分字段示意：

- `world_size`：世界半边长，仿真空间为 \[-world_size, world_size\]^2。
- `dt`：仿真时间步长（例如 `1/30` 秒）。
- `max_steps`：单 episode 最大步数。
- `hen_radius` / `eagle_radius` / `chick_radius`：母鸡、老鹰、小鸡刚体半径。
- `chain_links`：小鸡链条节数。
- `chain_spacing`：相邻小鸡在静止状态下的距离。
- `hen_max_speed` / `eagle_max_speed`：各自最大线速度。
- `max_force`：每步作用在刚体中心的最大力，用于控制加速度。
- `catch_radius`：老鹰判定“抓到小鸡尾端”的距离阈值。

你可以在 `PhysicsConfig` 中通过修改默认值调整物理特性（例如地图大小、链条长度、移动速度等），所有环境实例会共享这套底层逻辑。

### 动作与观测空间（`BasePhysicsEnv`）

- **动作空间**

  - 类型：`gymnasium.spaces.Box`
  - 形状：`shape=(2,)`
  - 含义：二维平面上的加速度方向（x、y），取值范围 `[-1, 1]`，内部会映射到 `max_force` 并做速度裁剪。
- **观测空间**

  - 类型：`gymnasium.spaces.Box`，`shape=(12,)`。
  - 根据当前“视角角色”（母鸡/老鹰）不同，返回其自身坐标系下的归一化向量，主要包括：
    - 自身位置（x, y），除以 `world_size` 归一化。
    - 自身速度向量，除以对应 `*_max_speed`。
    - 对手相对位置向量。
    - 链条尾端相对位置向量。
    - 尾端速度。
    - 链条平均拉伸程度。
    - 当前时间步占 `max_steps` 的比例。

这样在训练时：

- **训练母鸡** 时，环境返回的是 **母鸡视角观测**；
- **训练老鹰** 时，环境返回的是 **老鹰视角观测**，但内部在需要时会额外构造母鸡视角观测喂给冻结模型。

---

## 阶段一：训练母鸡（`HenTrainingEnv` + `train_hen.py`）

### 训练目标

- 保持母鸡位于老鹰与小鸡链条之间，尽量延缓老鹰接近链尾。
- 在带动物理链条运动时保持相对稳定，避免过度拉伸和剧烈摆动。

环境中的老鹰由启发式规则控制，只会追击链条尾端并带有简单侧向扰动，相当于“规则陪练”。

### 运行命令

在项目根目录执行：

```
python src/train_hen.py --total-steps 1000000
```

或

```bash
python src/train_hen.py --total-steps 300000 --eval-freq 10000 --save-dir results/curriculum --seed 42
```

主要参数说明：

- `--total-steps`：PPO 总训练步数（对应 `model.learn(total_timesteps=...)`）。
- `--eval-freq`：每隔多少步在 `EvalCallback` 中评估并保存最优模型。
- `--save-dir`：模型与日志的输出目录，默认 `results/curriculum`。
- `--seed`：随机种子（用于环境和 PPO）。

### 阶段一行为可视化

为了直观观察母鸡在启发式老鹰攻击下如何带动身后的小鸡链条进行防守，本仓库提供了一个简单的可视化脚本：

```bash
python src/visualize_hen_stage1.py --episodes 1 --fps 90
```

默认情况下，可视化脚本会加载 `results/curriculum/best_model.zip` 作为母鸡策略（即训练过程中评估分数最高的模型）。你可以通过 `--episodes` 控制可视化的回合数，通过 `--fps` 控制刷新速度；如需指定其他模型，可显式传入 `--model` 参数。可视化窗口关闭后程序自动结束。

- **母鸡**：橙色圆点。
- **老鹰**：蓝色圆点。
- **小鸡链条**：绿色小圆点（完整链条上所有小鸡都会被绘制出来）。
- 坐标范围与物理世界一致（\[-world_size, world_size\]^2），横纵坐标分别表示 X/Y 位置。

### 并行训练与硬件加速

为了充分利用高性能 CPU（如 i7-14700KF）的多核优势并提升数据多样性（IID），`src/train_hen.py` 默认配置为 **并行训练模式**：

- **多环境并行 (`n_envs=16`)**：使用 `SubprocVecEnv` 开启 16 个独立进程同时收集数据，极大地提高了采样速度 (FPS)。
- **打破相关性**：并行采样能有效打破单环境中的时间相关性，使训练数据分布更均匀，提升 PPO 算法的稳定性。
- **GPU 加速优化**：针对高性能显卡（如 RTX 5080），脚本中调整了 `batch_size=512` 和 `n_steps=256`，以平衡显存吞吐量与更新频率。

> **注意**：如果你在核心数较少的机器上运行，建议在 `src/train_hen.py` 中适当减小 `n_envs`（例如改为 4 或 8）。

### 输出结果

默认情况下，阶段一脚本会产生：

- **最优母鸡模型**：

  - 由 `EvalCallback` 自动保存，位于 `--save-dir` 下。
  - 脚本结束时显式保存：`results/curriculum/hen_stage_1.zip`。
- **TensorBoard 日志**：

  - 路径：`results/curriculum/tb`。
- 可视化命令：

  ```bash
  tensorboard --logdir results/curriculum
  ```

---

## 阶段二：训练老鹰（`EagleTrainingEnv` + `train_eagle.py`）

### 训练目标

- 在面对已经学会防守的母鸡时，老鹰能够学习更合理的接近和突破策略。
- 在母鸡干扰下仍能可靠地捕获链条尾端。

此阶段母鸡策略 **冻结不更新参数**，仅作为环境的一部分参与仿真；老鹰 PPO 从零开始训练。

### 运行命令

确保已完成阶段一训练并生成 `result/curriculum/hen_stage_1.zip` 后执行：

```bash
python src/train_eagle.py
```

主要参数说明：

- `--hen-model`：阶段一训练得到的母鸡 PPO 模型路径。
- 其余参数含义与阶段一一致：`--total-steps`、`--eval-freq`、`--save-dir`、`--seed`。

### 阶段二行为可视化

为了观察老鹰在冻结母鸡防守下的进攻行为，可以使用阶段二可视化脚本：

```bash
python src/visualize_eagle_stage2.py --episodes 1 --fps 60
```

- **老鹰**：蓝色圆点（由阶段二训练得到的策略控制）。
- **母鸡**：橙色圆点（使用阶段一冻结策略）。
- **小鸡链条**：绿色小圆点，尾端被老鹰抓到时会变成红色。
- **母鸡翅膀**：红色线段，表示阻挡带宽度（由 `block_margin` 决定）。

可选参数：

- `--model`：老鹰策略模型路径，默认 `results/curriculum/eagle_stage_1.zip`。
- `--hen-model`：母鸡策略模型路径，默认 `results/curriculum/hen_stage_1.zip`。
- `--episodes`：可视化回合数。
- `--fps`：刷新帧率。

### 冻结母鸡策略与视角转换

在 `EagleTrainingEnv.step()` 内部，逻辑大致为：

1. 根据当前物理状态构造 **母鸡视角观测**：`hen_obs = self._get_obs(role="hen")`。
2. 使用冻结的母鸡模型进行推理：`hen_action, _ = self.hen_model.predict(hen_obs, deterministic=True)`。
3. 使用 `hen_action` 更新母鸡刚体状态（不更新网络参数）。
4. 使用从外部传入的 `action` 更新老鹰刚体状态。
5. 推进 Box2D 世界一步，根据 **老鹰视角** 构造观测与奖励并返回。

这样，对正在训练的老鹰来说，对手只是“一个难缠的母鸡策略”，无需关心其内部实现细节。

### 输出结果

阶段二脚本会产生：

- **最优老鹰模型**：由 `EvalCallback` 自动保存到 `--save-dir`。
- **阶段二模型快照**：脚本结束时保存为 `results/curriculum/eagle_stage_1.zip`，用于后续可能的联合微调。
- **TensorBoard 日志**：复用 `results/curriculum/tb` 目录。

---

## 阶段三（可选）：母鸡 + 老鹰联合微调

当前仓库 **只提供阶段一与阶段二的训练脚本**，第三阶段为可选扩展思路：

- 同时加载阶段一和阶段二产出的模型：`hen_stage_1.zip` 与 `eagle_stage_1.zip`。
- 在联合环境中允许母鸡与老鹰同时更新参数，适度提高学习率或缩短训练步数进行小规模微调。
- 奖励设计可沿用当前阶段的目标：
  - 母鸡：保护链条尾端、阻挡老鹰。
  - 老鹰：尽快有效地抓到尾端。

如需要，可以在此基础上实现一个 `train_stage3.py`，复用 `curriculum_env.py` 中的物理环境，并在循环中交替调用两个 PPO 策略的 `predict` 与 `learn`。

---

## 结果分析与可视化

### TensorBoard 曲线

- 阶段一与阶段二均会在 `--save-dir` 下创建 TensorBoard 日志目录。
- 默认示例路径：`results/curriculum/tb`。

启动命令示例：

```bash
tensorboard --logdir results/curriculum
```

可以查看：

- 每步/每回合的平均回报。
- 损失函数变化趋势等训练指标。

### 终端输出参数说明

在训练过程中，终端会定期输出如下格式的日志表格，各参数含义解释如下：

```text
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 287          |  <-- 平均每个回合（Episode）的步数长度
|    ep_rew_mean          | 1.68         |  <-- 平均每个回合的累积奖励（越高越好）
| time/                   |              |
|    fps                  | 3500         |  <-- Frames Per Second，每秒采样的环境步数（训练速度）
|    iterations           | 74           |  <-- PPO 算法的主循环迭代次数
|    time_elapsed         | 86           |  <-- 训练已花费的总时间（秒）
|    total_timesteps      | 303104       |  <-- 累计采样的环境总步数
| train/                  |              |
|    approx_kl            | 0.00417      |  <-- 近似 KL 散度，衡量新旧策略的差异（过大说明更新太激进）
|    clip_fraction        | 0.0267       |  <-- 被 PPO 截断（clip）机制处理的样本比例
|    clip_range           | 0.2          |  <-- PPO 的截断阈值（超参数）
|    entropy_loss         | -2.44        |  <-- 策略熵的负值，绝对值越大表示动作越随机（探索性越强）
|    explained_variance   | 0.87         |  <-- 价值函数的解释方差，越接近 1 表示 Value Net 预测越准
|    learning_rate        | 0.0003       |  <-- 当前学习率
|    loss                 | 0.651        |  <-- 总损失值（Policy Loss + Value Loss + Entropy Loss）
|    n_updates            | 730          |  <-- 梯度更新的总次数
|    policy_gradient_loss | -0.00234     |  <-- 策略梯度部分的损失
|    std                  | 0.82         |  <-- 动作分布的标准差（反映探索程度，训练初期较大，后期应减小）
|    value_loss           | 1.54         |  <-- 价值函数部分的损失（Critic 网络的误差）
------------------------------------------
```

  **重点关注指标：**

1. **`ep_rew_mean`**：最直观的性能指标，应随训练逐渐上升。
2. **`explained_variance`**：如果长期接近 0 或为负，说明 Value Network 未能有效拟合回报，可能需要调整网络结构或超参数。
3. **`entropy_loss`**：训练初期应保持一定大小以确保探索，后期随策略确定性增加而（绝对值）减小。如果过早接近 0，可能发生了过早收敛（Early Convergence）。

### 自定义评估与回放示例

当前仓库未内置独立评估脚本，如需自定义评估/回放，可以参考以下基本流程（示意代码）：

```python
from stable_baselines3 import PPO
from curriculum_env import HenTrainingEnv, PhysicsConfig

cfg = PhysicsConfig()
env = HenTrainingEnv(config=cfg, seed=0)
model = PPO.load("results/curriculum/hen_stage_1.zip")

obs, _ = env.reset()
for _ in range(cfg.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # 这里可以添加自定义渲染或数据记录逻辑
    if terminated or truncated:
        break
```

你可以在此基础上加入渲染、统计或录像等逻辑，形成适配自己实验需求的评估工具链。

---

---

## 常见问题（FAQ）

- **依赖安装失败（特别是 Box2D 相关）**

  - 建议先升级 `pip`：`pip install --upgrade pip`。
  - 再单独安装：`pip install "gymnasium[box2d]" box2d-py`。
  - 避免使用过新的 Python 版本，可以优先尝试 3.9–3.11。
- **TensorBoard 无法看到日志**

  - 确认训练命令中的 `--save-dir` 与 `tensorboard --logdir` 一致。
  - 检查 `results/curriculum/tb` 下是否生成了事件文件（`events.out.tfevents.*`）。
- **训练不收敛或波动很大**

  - 增大 `--total-steps` 以延长训练时间。
  - 调整 `PhysicsConfig`（如 `max_steps`、`catch_radius` 等）匹配任务难度。
  - 根据需要在 `train_hen.py`、`train_eagle.py` 中调整 PPO 超参数（`learning_rate`、`gamma`、`batch_size` 等）。
- **OMP 库冲突（`OMP: Error #15`）**

  - 在 Windows PowerShell 下可以先设置环境变量：

    ```powershell
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python src/train_hen.py ...
    ```

---

## 与本 README 相关的目录结构

- `src/curriculum_env.py`
  - `PhysicsConfig`、`BasePhysicsEnv`
  - `HenTrainingEnv`：母鸡训练环境。
  - `EagleTrainingEnv`：老鹰训练环境。
- `src/train_hen.py`：阶段一训练脚本。
- `src/train_eagle.py`：阶段二训练脚本。
- `requirements.txt`：依赖列表，与课程学习主线训练脚本对应。

其他文件（如 `src/environment.py`、`src/main.py`、`results/MAAC*` 等）为旧版多智能体训练方案的遗留实现，不再作为当前推荐路径的一部分，仅供回溯参考使用。
