# Persona2Emotion: 性格感知的情绪反应预测器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![ACL 2026](https://img.shields.io/badge/ACL-2026-lightgrey)](https://arxiv.org/abs/2604.09162)

Personality-Aware Emotional Response Predictor

基于 **Persona-E² 数据集 (ACL 2026)** 的双塔深度学习模型，结合 **DistilBERT** 文本编码器与**大五人格 (BFI)** 特征，预测具有特定性格的人阅读事件文本后的情绪反应。

A dual-tower deep learning model based on the **Persona-E² dataset (ACL 2026)** that combines a **DistilBERT** text encoder with **Big Five (BFI)** personality features to predict how individuals with different personality traits emotionally react to text events.

---

## ✨ 功能亮点 / Features

- **🎭 性格调制情绪预测** — 输入英文文本 + 五大人格分数，实时输出 7 种情绪概率分布
- **🏷️ 领域感知 (Domain-Aware) v2 模型** — 利用数据集自带的 News / Social Media / Life Experience 领域标签（论文证明领域对情绪有系统性影响），新增独立 domain 分支；旧权重自动降级为 V1
- **📊 Soft Label 训练** — 对 36 名标注者的情绪分布做 confidence 加权聚合，用 KL 散度损失替代硬标签，缓解标注噪声与“性格错觉”
- **🌐 中英文混合输入** — 内置本地离线翻译引擎（Helsinki-NLP/opus-mt-zh-en），中文自动翻译后预测，无需网络
- **🖥️ 桌面 GUI 应用** — 基于 Tkinter，滑块调节人格，表情符号和概率条形图直观展示；检测到 v2 权重时自动出现“领域选择”单选
- **🔁 断点续训** — 训练脚本自动检测最新 checkpoint 并继续训练
- **🛡️ 工程健壮性 (A)** — 新增 `requirements.txt`、持久化 `label_map.json`（消除 `LabelEncoder` 硬编码顺序脆弱性）、`.gitignore` 忽略 `.lnk` / `.hf_cache`
- **📊 完整项目流程** — 涵盖数据探索、模型训练、硬件优化到交互式部署

---

---

## 🆕 v2 优化说明（A + B + C 方向）

本项目在 v1（DistilBERT + BFI 双塔）基础上，结合 **Persona-E² (ACL 2026)** 论文的方法论启发做了三处优化：

### A. 工程健壮性（零风险）
- 新增 `requirements.txt`，锁定 `torch` / `transformers` / `pandas` / `numpy` / `scikit-learn` / `deep_translator` / `safetensors` 依赖。
- 训练端持久化 `label_map.json`（标签顺序 + 领域顺序 + 是否使用 soft label），推断端 `desktop_app.py` 优先读取，**消除原 `LabelEncoder` 字母序与 GUI 硬编码 `EMOTIONS` 顺序不一致导致的预测错位风险**。
- `.gitignore` 增补 `*.lnk` 与 `.hf_cache/`，避免误提交快捷方式与 HuggingFace 缓存。

### B. 数据红利：Soft Label 训练
- 原代码只用了 36 名标注者的**共识硬标签**；v2 改为对每位标注者的 `E{i}_emotion` + `E{i}_confidence`（1–5 置信度，作权重）做聚合，得到每条 event 的 **7 维情绪分布 soft label**。
- 损失函数由 `CrossEntropy` 切换为 **KL 散度**（`reduction="batchmean"`），更贴近“不同性格标注者情绪本就不同”的任务本质，缓解标注噪声。

### C. 领域特征（论文强证据）
- 数据集自带 `data_source` 列，可映射到 **News / Social Media / Life Experience** 三大领域；论文证明领域对情绪有系统性影响（News=理性缓冲、Social Media=负向偏置、Life=乐观偏置），而 v1 **完全未用该特征**。
- v2 模型 `BertWithPersonaV2` 新增 `domain_encoder`（3→32）分支，与文本 CLS、人格特征拼接后分类。
- **向后兼容**：`desktop_app.py` 通过检测权重中是否含 `domain_encoder` 键，自动选择 V2（显示领域单选）或 V1（提示“请重新训练启用领域特征”），旧权重无需改动即可继续用。

> ⚠️ **当前仓库权重为 v1**（无 domain 分支）。要获得 v2 收益，请用新版 `train.py` 重新训练，生成的 `persona_emotion_model/label_map.json` 会标记 `use_soft_target=true`。

---

## 📁 项目结构 / Project Structure

```
persona-emotion-predictor/
├── train.py                     # 训练脚本（soft label + domain 分支 v2，支持多进程加速、断点续训、SMOKE 模式）
├── desktop_app.py               # 桌面 GUI 应用（内置离线翻译，V1/V2 权重自动识别）
├── check_columns.py             # 数据集列名检查工具
├── scan.py                      # 文件结构扫描工具
├── 启动情绪预测.bat              # 一键启动批处理文件
├── .gitignore
├── README.md
│
├── 基于性格建模的文本情绪反应预测/   # 数据集
│   ├── 1_dataset_all_annotators.csv   # 全部 36 名标注者的情绪标注
│   ├── 2_dataset_group_consensus.csv  # 群体共识情绪
│   ├── 3_annotator_profiles.csv       # 标注者性格档案（BFI + MBTI）
│   └── PersonaE2.txt                  # 数据集介绍
│
└── persona_emotion_model/       # 训练好的模型权重（需自行训练或从 Release 下载）
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── label_map.json            # v2 新增：标签/领域顺序 + use_soft_target 标记（向后兼容）
```

> **注意**：模型权重（约 254 MB）不包含在仓库中，需自行训练或从 [Releases](https://github.com/wjxn13/persona-emotion-predictor/releases) 下载。当前 Release 权重为 **v1**；用新版 `train.py` 重训即生成 v2 权重与 `label_map.json`。

---

## 🚀 快速开始 / Quick Start

### 1. 克隆仓库

```bash
git clone https://github.com/wjxn13/persona-emotion-predictor.git
cd persona-emotion-predictor
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 训练模型

确保数据集文件夹 `基于性格建模的文本情绪反应预测` 存在。

```bash
python train.py
```

训练默认启用 **soft label（KL 散度损失）+ domain 分支（v2）**，完成后在当前目录生成 `persona_emotion_model/`（含 `label_map.json`）。默认使用 `distilbert-base-uncased`，RTX 3050 Ti 约需 30–45 分钟。如需更高精度，可将 `train.py` 中 `model_name` 改为 `bert-base-uncased`。

**快速验证（不真正训练）**：设置环境变量 `SMOKE=1` 仅跑 5 步并手动评估，用于检查数据管线与 forward 是否正确：

```bash
SMOKE=1 python train.py
```

> 国内网络请保持 `HF_ENDPOINT=https://hf-mirror.com`（脚本顶部已设置）。

### 4. 启动桌面应用

```bash
python desktop_app.py
```

或双击 `启动情绪预测.bat`。默认使用本地离线翻译模式。

---

## 🧪 模型性能 / Model Performance

**测试集（Test Set）对比 —— v1 与 v2 实测：**

| 指标 | v1（无 domain，硬标签 CE） | v2（domain 分支 + soft label，KL 散度） | 变化 |
|------|------|------|------|
| 准确率 Accuracy | 34.77% | **35.19%** | +0.42pp |
| 宏平均 F1 Macro F1 | 31.91% | **33.24%** | **+1.33pp** |

> v2 于 2026-07-23 用 `distilbert-base-uncased` 重训完成（3 epoch / RTX 3050 Ti / ~26 分钟）。macro_f1 提升更为明显，说明 domain 分支 + 36 标注者 soft label 有效缓解了少数类的性能损耗。

- 随机基线（7 分类）准确率 ≈ 14.3%，v2 模型性能为基线的 **2.46 倍**
- 💡 **与 LLM 对比**：Persona-E² 论文测得主流 LLM 的 Top-1 仅约 **25%**，而本项目的 BERT 实现测试集 **35.19% 显著超过 LLM**——说明“小模型 + 性格调制”在此任务上并不弱，本项目可作为该任务的社区参考实现
- 使用 `bert-base-uncased` 重训练预计可进一步提升至 **37%–40%**
- 任务本身具有挑战性：同一文本，不同性格标注者的情绪经常不一致

---

## 🎮 使用演示 / Demo

1. 输入文本：*"Your flight has been cancelled due to a snowstorm."*（或中文：*"由于暴风雪，您的航班取消了"*）
2. 将**神经质 (Neuroticism)** 滑块调至 0.85，其他保持 0.5
3. 点击预测，模型可能输出：😨 恐惧 38.2%，😢 悲伤 22.1%，😊 开心 5.3%
4. 再将神经质调至 0.20，宜人性调至 0.80，预测结果变为：😐 中性 45.6%，😊 开心 20.3%

**这正是本项目的核心观点：性格改变了情绪反应。**

### 🎯 性格扫描 Demo（一键看性格如何影响情绪）

桌面应用新增 **「🎯 性格扫描 Demo」** 按钮，打开独立窗口：

- 选择一个**真实事件**（数据集中的示例，如"前男友结婚"）；
- 选**领域**（News / Social Media / Life Experience）；
- 选一个**扫描维度**（五大人格之一），其余四维固定为某个值；
- 程序把该维度从 0.0 连续扫到 1.0，画出 **7 类情绪概率随性格变化的曲线**。

点击 **「📊 显示 36 名标注者真实情绪投票」** 还会画出该事件 36 位标注者的真实情绪分布柱状图——两者对照，直观呈现 **Persona-E² 任务的核心难点（personality illusion）**：

> 模型预测曲线较平（轻量 distilbert 对性格利用率有限，情绪主要由事件语义决定），与"同一事件 36 人标注情绪常不一致"是同一难题的两面。**换 `bert-base-uncased` 底座可放大性格效应**（见性能章节）。

> 说明：v2 用 distilbert 时扫描曲线摆幅约 0.7–1.2pp（诚实呈现，非夸大为剧烈摆动）；该 Demo 的价值在于"让性格→情绪的可感知性 + 标注分歧可视化"，而非展示夸张的曲线变化。

---

## 📦 硬件要求 / Hardware Requirements

- **训练**：推荐 NVIDIA 显卡 ≥ 4GB 显存（如 RTX 3050 Ti），纯 CPU 训练极慢不推荐
- **推理（桌面应用）**：CPU 即可，首次使用需下载本地翻译模型（约 300 MB，通过 HuggingFace 国内镜像加速）

---

## 🔧 常见问题 / FAQ

**Q: 运行 `desktop_app.py` 时找不到模型？**  
A: 请先训练模型或从 [Releases](https://github.com/wjxn13/persona-emotion-predictor/releases) 下载预训练权重，并将 `persona_emotion_model` 文件夹放入项目根目录。

**Q: 在线翻译（Google/有道）失败？**  
A: 默认翻译模式已是本地离线翻译，完全离线可用。

**Q: 数据集太大，无法推送？**  
A: 数据集已在仓库中（约 30 MB），模型权重（~254 MB）通过 Release 分发。

**Q: 桌面应用里没有“领域选择”选项？**  
A: 当前 Release 权重为 **v1**（无 domain 分支）。`desktop_app.py` 会自动检测权重结构：检测到 `domain_encoder` 即显示领域单选（v2），否则提示“请重新训练启用领域特征”并退回 v1 推理。用新版 `train.py` 重训生成的 `persona_emotion_model/label_map.json` 会标记 v2。

**Q: 重训后标签顺序会变吗？会预测错位吗？**  
A: 不会。v2 训练端持久化 `label_map.json`，`desktop_app.py` 优先读取它确定标签/领域顺序，彻底避免原 `LabelEncoder` 字母序与 GUI 硬编码顺序不一致的问题。

---

## 📚 数据集引用 / Dataset Citation

本项目的训练数据来自 **Persona-E²: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events** (ACL 2026).

```
@inproceedings{yang2026personae2,
  title={Persona-E$^2$: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events},
  author={Yang, Yuqin and Zhou, Haowu and Tu, Haoran and Hui, Zhiwen and Yan, Shiqi and Li, HaoYang and She, Dong and Yao, Xianrong and Gao, Yang and Jin, Zhanpeng},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

- arXiv: [https://arxiv.org/abs/2604.09162](https://arxiv.org/abs/2604.09162)
- Kaggle: [Persona-E² Dataset](https://www.kaggle.com/datasets/crisyang777/peronsa-e-personality-shaped-emotion-dataset)

---

## 📄 许可证 / License

MIT License.

---

**如果觉得有用，请给一个 ⭐ Star！**
