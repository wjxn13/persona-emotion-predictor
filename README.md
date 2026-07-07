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
- **🌐 中英文混合输入** — 内置本地离线翻译引擎（Helsinki-NLP/opus-mt-zh-en），中文自动翻译后预测，无需网络
- **🖥️ 桌面 GUI 应用** — 基于 Tkinter，滑块调节人格，表情符号和概率条形图直观展示
- **🔁 断点续训** — 训练脚本自动检测最新 checkpoint 并继续训练
- **📊 完整项目流程** — 涵盖数据探索、模型训练、硬件优化到交互式部署

---

## 📁 项目结构 / Project Structure

```
persona-emotion-predictor/
├── train.py                     # 训练脚本（支持多进程加速、断点续训）
├── desktop_app.py               # 桌面 GUI 应用（内置离线翻译）
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
    └── tokenizer_config.json
```

> **注意**：模型权重（约 254 MB）不包含在仓库中，需自行训练或从 [Releases](https://github.com/wjxn13/persona-emotion-predictor/releases) 下载。

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

训练完成后会在当前目录生成 `persona_emotion_model/`。默认使用 `distilbert-base-uncased`，RTX 3050 Ti 约需 30–45 分钟。如需更高精度，可将 `train.py` 中 `model_name` 改为 `bert-base-uncased`。

### 4. 启动桌面应用

```bash
python desktop_app.py
```

或双击 `启动情绪预测.bat`。默认使用本地离线翻译模式。

---

## 🧪 模型性能 / Model Performance

| 指标 | 验证集 | 测试集 |
|------|--------|--------|
| 准确率 Accuracy | 36.15% | **34.77%** |
| 宏平均 F1 Macro F1 | 34.55% | **31.91%** |

- 随机基线（7 分类）准确率 ≈ 14.3%，模型性能为基线的 **2.4 倍**
- 使用 `bert-base-uncased` 重训练预计可提升至 **37%–40%**
- 任务本身具有挑战性：同一文本，不同性格标注者的情绪经常不一致

---

## 🎮 使用演示 / Demo

1. 输入文本：*"Your flight has been cancelled due to a snowstorm."*（或中文：*"由于暴风雪，您的航班取消了"*）
2. 将**神经质 (Neuroticism)** 滑块调至 0.85，其他保持 0.5
3. 点击预测，模型可能输出：😨 恐惧 38.2%，😢 悲伤 22.1%，😊 开心 5.3%
4. 再将神经质调至 0.20，宜人性调至 0.80，预测结果变为：😐 中性 45.6%，😊 开心 20.3%

**这正是本项目的核心观点：性格改变了情绪反应。**

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
