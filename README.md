我明白了，之前给你的 README 内容被包裹在 ` ```markdown ` 代码块中，导致你复制时把 ` ``` ` 也一起复制了，所以 GitHub 把代码块的标记当成普通文字显示，才出现了渲染错误。


---



# Persona2Emotion: 性格感知的情绪反应预测器 / Personality-Aware Emotional Response Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![条款:](https://img.shields.io/badge/License-MIT-yellow.svg)] (https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)[！[Python 3.11](https://img.shields.io/badge/python-3.11 -blue.svg)]（https://www.python.org/）[！[Python 3.11](https://img.shields.io/badge/python-3.11 -blue.svg)](https://www.python.org/)[！[Python 3.11](https://img.shields.io/badge/python-3.11 -blue.svg)]（https://www.python.org/）

基于 **Persona-E² 数据集 (ACL 2026)** 的双塔深度学习模型，结合 **DistilBERT** 文本编码器与 **大五人格 (BFI)** 特征，预测具有特定性格的人阅读事件文本后的情绪反应。

A dual-tower deep learning model based on the **Persona-E² dataset (ACL 2026)** that combines a **DistilBERT** text encoder with a **Big Five (BFI)** personality encoder to predict how individuals with different personality traits emotionally react to text events.基于**Persona-E²；数据集(ACL 2026)**的双塔深度学习模型，将**DistilBERT**文本编码器与**Big Five (BFI)**人格编码器相结合，预测具有不同人格特质的个体对文本事件的情感反应。

---

## ✨ 功能亮点 / Features

- 🎭 **性格调制情绪预测** / **Personality-modulated prediction**：输入英文文本 + 五大人格分数，实时输出 7 种情绪概率分布。
- 🌐 **中英文混合输入** / **Bilingual input**：内置本地离线翻译引擎 (Helsinki-NLP/opus-mt-zh-en)，中文自动翻译后预测，无需网络。
- 🖥️ **桌面 GUI 应用** / **Desktop GUI**：基于 Tkinter，滑块调节人格，表情符号和概率条形图直观展示。
- 🔄 **断点续训** / **Resumable training**：训练脚本自动检测最新 checkpoint 并继续训练。
- 📊 **完整项目流程** / **End-to-end project**：涵盖数据探索、模型训练、硬件优化到交互式部署。

---

## 📁 项目结构 / Project Structure

```
persona-emotion-predictor/
├── train.py                          # 训练脚本 (支持多进程加速、断点续训)
├── desktop_app.py                    # 桌面 GUI 应用 (内置离线翻译)
├── check_columns.py                  # 数据集列名检查工具
├── scan.py                           # 文件结构扫描工具
├── 启动情绪预测.bat                   # 一键启动批处理文件
├── requirements.txt                  # Python 依赖列表
├── .gitignore                        # Git 忽略规则
├── README.md                         # 本文件
└── 基于性格建模的文本情绪反应预测/      # 数据集文件夹 / Dataset folder
    ├── 1_dataset_all_annotators.csv  # 全部标注者情绪标注
    ├── 2_dataset_group_consensus.csv # 群体共识情绪
    ├── 3_annotator_profiles.csv      # 标注者性格档案 (BFI + MBTI)├── 3_annotator_profiles.csv      # 标注者性格档案 (BFI   MBTI)
    └── PersonaE2.txt                 # 数据集介绍
```

> **注意 / Note**：训练好的模型权重 (约 254 MB) **不会** 包含在仓库中，请从 [Releases](https://github.com/wjxn13/persona-emotion-predictor/releases) 下载或自行训练。> **注意 / Note**：训练好的模型权重 (约 254 MB) **不会** 包含在仓库中，请从 [Releases   释放](https://github.com/wjxn13/persona-emotion-predictor/releases) 下载或自行训练。

---

## 🚀 快速开始 / Quick Start

### 1. 克隆仓库 / Clone
```bash   ”“bash
git clone https://github.com/wjxn13/persona-emotion-predictor.gitGit克隆https://github.com/wjxn13/persona-emotion-predictor.git
cd persona-emotion-predictor
```

### 2. 安装依赖 / Install Dependencies
```bash   ”“bash
pip install -r requirements.txtPIP install -r requirements.txt
```

### 3. 训练模型 / Train
确保数据集文件夹 `基于性格建模的文本情绪反应预测` 存在。
```bash   ”“bash
python train.py
```
训练完成后会在当前目录生成 `persona_emotion_model/`。默认使用 `distilbert-base-uncased`，RTX 3050 Ti 约需 30–45 分钟。如需更高精度，可将 `train.py` 中 `model_name` 改为 `bert-base-uncased`。

### 4. 启动桌面应用 / Launch Desktop App
```bash   ”“bash   “bash”;“bash
python desktop_app.py
```
或双击 `启动情绪预测.bat`。默认使用本地离线翻译模式。

---

## 🧪 模型性能 / Model Performance

| 指标 / Metric | 验证集 / Val (Epoch 3) | 测试集 / Test |
|---------------|-------------------------|----------------|
| 准确率 Accuracy | 36.15% | **34.77%** |
| 宏平均 F1 Macro F1 | 34.55% | **31.91%** |

- 随机基线 (7 分类) 准确率 ≈ 14.3%，模型性能为基线的 **2.4 倍**。
- 使用 `bert-base-uncased` 重训练预计可提升至 **37%–40%**。- 使用 `bert-base-uncased` 重训练预计可提升至 **37%–40%   37%, 40%**。- 使用 `bert-base-uncased` 重训练预计可提升至 **37%–40%**。- 使用 `bert-base-uncased` 重训练预计可提升至 **37%–40%   37%, 40%**。
- 任务本身具有挑战性：同一文本，不同性格标注者的情绪经常不一致。

---

## 🎮 使用演示 / Demo

1. 输入文本：*"Your flight has been cancelled due to a snowstorm."*1. 中文：由于暴风雪，您的航班取消了1. 输入文本：*"Your flight has been cancelled due to a snowstorm."*1. 中文：由于暴风雪，您的航班取消了
2. 将 **神经质 (Neuroticism)** 滑块调至 0.85，其他保持 0.5。
3. 点击预测，模型可能输出：😨 恐惧 (fear) 38.2%, 😢 悲伤 (sadness) 22.1%, 😊 开心 (joy) 5.3% ...
4. 再将神经质调至 0.20，宜人性调至 0.80，预测结果变为：😐 中性 (neutral) 45.6%, 😊 开心 (joy) 20.3% ...

**这正是本项目的核心观点：性格改变了情绪反应。**
**This is the core idea: personality modulates emotional reactions.**这是核心思想：个性调节情绪反应

---

## 📦 硬件要求 / Hardware Requirements

- **训练**：推荐 NVIDIA 显卡 ≥ 4GB 显存 (如 RTX 3050 Ti)，纯 CPU 训练极慢不推荐。
- **推理 (桌面应用)**：CPU 即可，首次使用需下载本地翻译模型 (约 300 MB，通过 HF 镜像加速)。

---

## 🔧 常见问题 / FAQ

**Q: 运行 `desktop_app.py` 时找不到模型？**  
A: 请先训练模型或从 [Releases](https://github.com/wjxn13/persona-emotion-predictor/releases) 下载预训练权重，并将 `persona_emotion_model` 文件夹放入项目根目录。

**Q: 在线翻译 (Google/有道) 失败？**  
A: 默认翻译模式已是本地离线翻译，完全离线可用。若需在线翻译，请确保网络可访问相应服务。

**Q: 如何上传自己的模型？**  
A: 模型文件已在 `.gitignore` 中忽略。可通过 [Releases](https://github.com/wjxn13/persona-emotion-predictor/releases) 发布或使用 Git LFS。

---

## 📚 数据集引用 / Dataset Citation

本项目的训练数据来自 **Persona-E²: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events** (ACL 2026).  本项目的训练数据来自 **Persona-E²: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Eventspersona - e：一个以人为基础的数据集，用于研究文本事件中性格塑造的情绪反应** (ACL 2026).
- arXiv: [https://arxiv.org/abs/2604.09162](https://arxiv.org/abs/2604.09162)
- 数据集包含 3,111 个文本事件和 111,996 条由 36 位具备 BFI/MBTI 性格测量的标注者产生的情绪标注。

---

## 📄 许可证 / License

MIT License. 详见 [LICENSE](LICENSE) 文件。

---

**如果觉得有用，请给一个 ⭐ Star！ / If you find this project useful, please consider giving it a star!**** ** ** ** ** ** ** */如果你觉得这个项目很有用，请考虑给它打个星！**

---

