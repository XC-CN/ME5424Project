# 🦅 MARL Eagles-Chicks-Tracking

> 本项目基于多智能体强化学习（Multi-Agent Actor-Critic, MAAC），构建“老鹰-小鸡-母鸡”对抗场景。老鹰负责协同追捕，小鸡学习逃逸，母鸡负责保护，实现攻防博弈与策略协作。

---

## 📚 文档导航

- [功能概览](#功能概览)
- [环境依赖](#环境依赖)
- [运行指南](#运行指南)
  - [命令参数详解](#命令参数详解)
  - [常用命令示例与说明](#常用命令示例与说明)
  - [训练输出与评估指引](#训练输出与评估指引)
  - [多智能体训练说明](#多智能体训练说明)
- [碰撞弹开效果](#碰撞弹开效果)
- [智能体介绍](#智能体介绍)
- [算法框架对比](#算法框架对比)
- [使用技巧](#使用技巧)
- [致谢与许可](#致谢与许可)
- [联系方式](#联系方式)

---

## 功能概览

- 🤝 **多老鹰协同**：共享观测，离散动作控制抓捕。
- 🐤 **小鸡捕获判定**：`capture_radius` 判断与覆盖率统计。
- 🐔 **母鸡保护机制**：安全半径惩罚、物理弹开、阻挡奖励。
- 🧠 **奖励体系**：支持 PMI、自利/全局奖励等多种协作方式。
- 🎥 **可视化工具链**：动画渲染、轨迹导出、TensorBoard 日志。

---

## 环境依赖

| 组件                        | 说明                         |
| --------------------------- | ---------------------------- |
| Python                      | 3.9 及以上（推荐 3.10/3.11） |
| PyTorch                     | 深度学习框架                 |
| NumPy / SciPy               | 数值计算                     |
| Matplotlib                  | 可视化绘图                   |
| Pillow / imageio            | 图片、动画处理               |
| tqdm / PyYAML / TensorBoard | 进度条、配置解析、训练日志   |

安装方式：

```bash
pip install -r requirements.txt
# pip install: Python 包管理工具安装命令
# -r requirements.txt: 从 requirements.txt 文件中读取依赖包列表并批量安装

# 或者逐项安装
pip install numpy scipy matplotlib pillow imageio torch torchvision torchaudio tqdm pyyaml tensorboard
# pip install: 安装指定的 Python 包
# numpy scipy matplotlib pillow imageio: 数值计算、科学计算、绘图、图像处理等库
# torch torchvision torchaudio: PyTorch 深度学习框架及相关组件
# tqdm pyyaml tensorboard: 进度条、YAML 配置解析、训练可视化工具
```

---

## 运行指南

所有命令默认在仓库根目录 `ME5424/` 下执行：

```bash
python src/main.py --phase <模式> --method <算法> [其它参数]
# python: Python 解释器
# src/main.py: 主程序入口脚本路径
# --phase <模式>: 必填参数，指定运行模式（train/evaluate/run）
# --method <算法>: 必填参数，指定算法配置（MAAC-R/MAAC/MAAC-G/C-METHOD）
# [其它参数]: 可选参数，如 -e（轮数）、-s（步数）、-f（保存频率）等
```

若出现 `OMP: Error #15` 等提示，可先设置 `KMP_DUPLICATE_LIB_OK=TRUE` 再运行。

### 命令参数详解

| 参数                                                     | 说明                                                        | 是否必填 | 默认值     | 示例                                   |
| -------------------------------------------------------- | ----------------------------------------------------------- | -------- | ---------- | -------------------------------------- |
| `--phase`                                              | 运行模式：`train` / `evaluate` / `run`                | 必填     | `train`  | `--phase run`                        |
| `-m`                                                   | 算法配置：`MAAC-R` / `MAAC` / `MAAC-G` / `C-METHOD` | 必填     | `MAAC-R` | `-m MAAC`                            |
| `-e`                                                   | 训练轮数（仅训练）                                          | 可选     | `10000`  | `-e 50`                              |
| `-s`                                                   | 每轮步数                                                    | 可选     | `200`    | `-s 300`                             |
| `-f`                                                   | 保存频率（训练）                                            | 可选     | `100`    | `-f 20`                              |
| `-a` / `-c`                                          | 老鹰 Actor / Critic 权重路径                                | 可选     | 最近权重   | `-a results/.../actor_100.pth`       |
| `--protector_actor_path` / `--protector_critic_path` | 母鸡网络加载/保存路径                                       | 可选     | 最近权重   | `--protector_actor_path results/...` |
| `--target_actor_path` / `--target_critic_path`       | 小鸡网络加载/保存路径                                       | 可选     | 最近权重   | `--target_actor_path results/...`    |
| `-p`                                                   | PMI 网络权重路径                                            | 可选     | `None`   | `-p results/.../pmi_100.pth`         |

### 常用命令示例与说明

激活conda环境：

```bash
conda activate ME5424Project  # 激活名为 ME5424Project 的 conda 虚拟环境
```

#### 1️⃣ 演示模式（无需训练、权重）

```bash
python src/main.py --phase run --method MAAC-R -s 300
# --phase run: 指定运行模式为演示模式（使用启发式策略，无需训练）
# --method MAAC-R: 选择 MAAC-R 算法配置文件
# -s 300: 设置每轮仿真步数为 300 步
```

- 使用内置启发式策略快速播放场景，适用于初次体验或演示。
- 输出动画保存在 `results/MAAC-R/{experiment}/animated/`，可直接查看。

#### 2️⃣ 训练模式（自动保存最新模型）

```bash
python src/main.py --phase train --method MAAC-R -e 50 -s 300 -f 20
# --phase train: 指定运行模式为训练模式（使用神经网络进行学习）
# --method MAAC-R: 选择 MAAC-R 算法配置文件
# -e 50: 设置训练轮数（episodes）为 50 轮
# -s 300: 设置每轮仿真步数为 300 步
# -f 20: 设置保存频率为每 20 轮保存一次模型和日志
```

- 训练 50 局、每局 300 步，每 20 局保存一次模型与日志。
- 训练结束后，`results/MAAC-R/{experiment}/` 将包含：
  - `actor/`、`critic/`、`pmi/`：按保存频率生成的权重快照；
  - `logs/`：TensorBoard 日志，可实时观察奖励与损失；
  - `frames/`、`animated/`：关键帧与合成视频；
  - 各类 CSV 指标（例如 `return_list.csv`、`protector_return_list.csv`、`target_return_list.csv`）。

#### 3️⃣ 评估模式（默认加载最近训练结果）

```bash
python src/main.py --phase evaluate --method MAAC-R -s 5000
# --phase evaluate: 指定运行模式为评估模式（加载已训练权重进行测试）
# --method MAAC-R: 选择 MAAC-R 算法配置文件
# -s 500: 设置每轮仿真步数为 500 步
```

- 不指定权重路径时，会自动寻找 `results/MAAC-R/` 下最近一次训练输出的最新权重进行评估。
- 如需复现特定检查点，可显式指定：
  ```bash
  python src/main.py --phase evaluate --method MAAC-R -s 500 \
         --actor_path results/MAAC-R/.../actor_100.pth \
         --critic_path results/MAAC-R/.../critic_100.pth \
         --protector_actor_path results/MAAC-R/.../protector_actor_weights_100.pth \
         --protector_critic_path results/MAAC-R/.../protector_critic_weights_100.pth \
         --target_actor_path results/MAAC-R/.../target_actor_weights_100.pth \
         --target_critic_path results/MAAC-R/.../target_critic_weights_100.pth
  # --phase evaluate: 评估模式
  # --method MAAC-R: 使用 MAAC-R 配置
  # -s 500: 仿真步数 500 步
  # --actor_path: 指定老鹰 Actor 网络权重文件路径
  # --critic_path: 指定老鹰 Critic 网络权重文件路径
  # --protector_actor_path: 指定母鸡 Actor 网络权重文件路径
  # --protector_critic_path: 指定母鸡 Critic 网络权重文件路径
  # --target_actor_path: 指定小鸡 Actor 网络权重文件路径
  # --target_critic_path: 指定小鸡 Critic 网络权重文件路径
  ```
- 评估会再次输出动画与指标，可与训练阶段对比。

### 训练输出与评估指引

- **结果目录**：`results/MAAC-R/{experiment}/` 包含权重、日志、动画、轨迹与各类 CSV 指标。
- **日志可视化**：

  ```bash
  tensorboard --logdir results/MAAC-R/{experiment}/logs
  # tensorboard: 启动 TensorBoard 可视化工具
  # --logdir: 指定日志文件所在目录路径
  ```

  查看老鹰/母鸡/小鸡三类智能体的奖励、损失、覆盖率等曲线。
- **指标文件**：`*_return_list.csv`、`protector_block_reward_list.csv`、`target_capture_penalty_list.csv` 等，可用于绘制图表或对比实验。

### 多智能体训练说明

- `train` 阶段默认同时训练老鹰、母鸡、小鸡三套 Actor-Critic 网络。
- CLI 新增参数可分别加载/保存母鸡、小鸡的 Actor、Critic。
- `configs/MAAC-R.yaml` 增加了 `protector_actor_critic`、`target_actor_critic` 超参区块（包含动作空间、观测半径、奖励权重等），便于独立调参。
- TensorBoard 日志与 CSV 指标会分别记录三类智能体的奖励、损失、覆盖率，便于分析协作与对抗效果。

---

## 碰撞弹开效果

当老鹰侵入母鸡防护臂时：

1. 计算老鹰到手臂线段的最短距离；
2. 距离小于 `arm_thickness` 即视为碰撞；
3. 按 `knockback` 沿法线方向推离并限制在地图范围内；
4. 刷新所有智能体的观测与奖励。

配置示例：

```yaml
protector:
  safe_radius: 200.0      # 防护臂长度
  knockback: 200.0        # 弹开强度
  arm_thickness: 100.0    # 碰撞判定厚度
```

---

## 智能体介绍

### 🦅 老鹰（UAV - 追击者）

- 速度：20 单位/步
- 观测半径：200，通信半径：500
- 动作空间：12 个离散转向动作
- 任务：协同追踪小鸡，避免母鸡惩罚

### 🐤 小鸡（Chick - 被追击者）

- 速度：5
- 捕获半径：120
- 行为：可训练逃逸策略（默认随机）
- 状态：被老鹰进入捕获半径即判定被捕

### 🐔 母鸡（Protector - 防御者）

- 速度：5
- 安全半径：200
- 弹开机制：老鹰入侵时触发惩罚与物理推离
- 任务：保护小鸡、阻挡老鹰、降低被捕率

---

## 算法框架对比

| 方法             | 描述             | 协作机制    | 适用场景     | 推荐度     |
| ---------------- | ---------------- | ----------- | ------------ | ---------- |
| MAAC             | 老鹰自利策略     | ❌ 无协作   | 简单对抗     | ⭐⭐⭐     |
| MAAC-G           | 全局奖励共享     | 🤝 平均分配 | 完全协作     | ⭐⭐⭐⭐   |
| **MAAC-R** | PMI + 自适应奖励 | 🧠 智能分配 | 复杂协作对抗 | ⭐⭐⭐⭐⭐ |
| C-METHOD         | 对比基线         | -           | 性能对照     | ⭐⭐       |

---

## 使用技巧

1. **先演示后训练**：建议先运行演示模式，再按短程训练观察曲线。
2. **调参与扩展**：可调整智能体数量、奖励权重或替换自定义策略。
3. **性能优化**：启用 GPU、增大 batch size、扩充经验池或多 GPU 并行以缩短训练时间。

---

## 致谢与许可

项目用于 **ME5424 Swarm Robotics and Aerial Robotics** 课程实验，欢迎在此基础上扩展更复杂的协作策略与物理交互。项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

---

## 联系方式

- 📧 Email: [your-email@example.com]
- 🐛 Issue: [GitHub Issues](https://github.com/XC-CN/5424Project/issues)
- 💬 Discussion: [GitHub Discussions](https://github.com/XC-CN/5424Project/discussions)

---

<div align="center">

**如果这个项目对您有帮助，请给我们一颗 Star！⭐**

Made with ❤️ by ME5424 Team

</div>
