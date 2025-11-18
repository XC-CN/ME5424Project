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

```bash
python src/train_hen.py --total-steps 300000 --eval-freq 10000 --save-dir results/curriculum --seed 42
```

主要参数说明：

- `--total-steps`：PPO 总训练步数（对应 `model.learn(total_timesteps=...)`）。
- `--eval-freq`：每隔多少步在 `EvalCallback` 中评估并保存最优模型。
- `--save-dir`：模型与日志的输出目录，默认 `results/curriculum`。
- `--seed`：随机种子（用于环境和 PPO）。

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

确保已完成阶段一训练并生成 `results/curriculum/hen_stage_1.zip` 后执行：

```bash
python src/train_eagle.py \
  --hen-model results/curriculum/hen_stage_1.zip \
  --total-steps 300000 \
  --eval-freq 10000 \
  --save-dir results/curriculum \
  --seed 123
```

主要参数说明：

- `--hen-model`：阶段一训练得到的母鸡 PPO 模型路径。
- 其余参数含义与阶段一一致：`--total-steps`、`--eval-freq`、`--save-dir`、`--seed`。

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

### 阶段一行为可视化（母鸡 + 老鹰 + 小鸡链条）

为了直观观察母鸡在启发式老鹰攻击下如何带动身后的小鸡链条进行防守，本仓库提供了一个简单的可视化脚本：

```bash
python src/visualize_hen_stage1.py --episodes 5 --fps 60
```

- **母鸡**：橙色圆点。
- **老鹰**：蓝色圆点。
- **小鸡链条**：绿色小圆点（完整链条上所有小鸡都会被绘制出来）。
- 坐标范围与物理世界一致（\[-world_size, world_size\]^2），横纵坐标分别表示 X/Y 位置。

默认情况下，可视化脚本会加载 `results/curriculum/best_model.zip` 作为母鸡策略（即训练过程中评估分数最高的模型）。你可以通过 `--episodes` 控制可视化的回合数，通过 `--fps` 控制刷新速度；如需指定其他模型，可显式传入 `--model` 参数。可视化窗口关闭后程序自动结束。

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
