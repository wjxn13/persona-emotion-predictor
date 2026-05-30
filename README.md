---
---

# 🧠 性格感知的情绪反应预测器 (Persona2Emotion)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)(!(许可证)(https://img.shields.io/badge/license-MIT-blue.svg))(许可证)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)[！[Python 3.11](https://img.shields.io/badge/python-3.11 -green.svg)]（https://www.python.org/）[！[Python 3.11](https://img.shields.io/badge/python-3.11 -green.svg)](https://www.python.org/)[！[Python 3.11](https://img.shields.io/badge/python-3.11 -green.svg)]（https://www.python.org/）

基于 **Persona-E² 数据集 (ACL 2026)** 训练的双塔深度学习模型，结合 **DistilBERT** 文本编码器与 **大五人格 (BFI)** 特征，预测具有特定性格的人在阅读事件文本后最可能产生的情绪反应。

> **核心理念**：同一段文本，不同性格的人会产生不同的情绪。本项目从数据到模型完整实现这一心理学现象的计算建模。

## ✨ 功能亮点

- **🎭 性格调制情绪预测**：输入任意英文文本 + 五大人格维度（开放性、尽责性、外向性、宜人性、神经质），实时输出 7 种情绪的概率分布。
- **🌐 中英文混合输入**：内置**本地离线翻译**引擎 (Helsinki-NLP/opus-mt-zh-en)，中文输入会自动翻译为英文再预测，无需网络。
- **🖥️ 友好的桌面 GUI**：基于 Tkinter 的图形界面，滑块调节人格，表情符号和概率条形图直观展示结果。
- **🔄 断点续训**：训练脚本支持自动检测最新 checkpoint 继续训练，不怕意外中断。
- **📊 完整项目流程**：从数据探索、模型训练、硬件优化到交互式应用部署，适合深度学习学习者和研究者。

## 📁 项目结构

```
persona-emotion-predictor/
├── train.py                          # 模型训练脚本（支持多进程加速、断点续训）
├── desktop_app.py                    # 桌面 GUI 应用（内置本地翻译）
├── check_columns.py                  # 数据集列名检查工具
├── scan.py                           # 文件结构扫描工具
├── 启动情绪预测.bat                   # 一键启动批处理文件
├── requirements.txt                  # Python 依赖列表
├── .gitignore                        # Git 忽略规则
├── README.md                         # 本文件
└── 基于性格建模的文本情绪反应预测/      # 数据集文件夹
    ├── 1_dataset_all_annotators.csv  # 全部标注者情绪标注
    ├── 2_dataset_group_consensus.csv # 群体共识情绪
    ├── 3_annotator_profiles.csv      # 标注者性格档案（BFI + MBTI）├── 3_annotator_profiles.csv      # 标注者性格档案（BFI   MBTI）├── 3_annotator_profiles.csv      # 标注者性格档案（BFI   MBTI）├── 3_annotator_profiles.csv      # 标注者性格档案（BFI   MBTI）
    └── PersonaE2.txt                 # 数据集介绍
```

> **注意**：训练好的模型权重文件（约 254 MB）**不会**包含在仓库中，请按照下方说明自行训练。

## 🚀 快速开始

### 1. 克隆仓库
```bash   ”“bash   “bash”;“bash
git clone https://github.com/wjxn13/persona-emotion-predictor.gitGit克隆https://github.com/wjxn13/persona-emotion-predictor.git
cd persona-emotion-predictor
```

### 2. 安装依赖
```bash   ”“bash
pip install -r requirements.txtPIP install -r requirements.txt
```

### 3. 训练模型
确保 `基于性格建模的文本情绪反应预测/` 文件夹中的数据集完整，然后运行：
```bash   ”“bash
python train.py
```
训练会在当前目录生成 `persona_emotion_model/` 文件夹（包含模型权重和配置文件）。  
- 使用 `distilbert-base-uncased` 轻量模型，RTX 3050 Ti 约需 30–45 分钟。  - 使用 `distilbert-base-uncased` 轻量模型，RTX 3050 Ti 约需 30–45 分钟。
- 如需更高精度，可修改 `train.py` 中的 `model_name` 为 `bert-base-uncased`（训练时间相应增加）。

### 4. 启动桌面应用
```bash   ”“bash
python desktop_app.py
```
或直接双击 `启动情绪预测.bat`。  
界面启动后，选择“本地翻译 (离线模型)”模式，即可输入中文或英文文本开始预测。

## 🧪 模型性能

| 指标 | 验证集 (Epoch 3) | 测试集 |
|------|------------------|--------|
| 准确率 | 36.15% | **34.77%** |
| 宏平均 F1 | 34.55% | **31.91%** |

- 基线 (随机猜测 7 分类) 准确率 ≈ 14.3%，模型性能为基线的 **2.4 倍**。
- 使用 `bert-base-uncased` 重新训练预计可提升至 **37%–40%**。- 使用 `bert-base-uncased` 重新训练预计可提升至 **37%–40%   37%, 40%**。
- 任务本身极具挑战性：同一文本，36 位不同性格标注者的情绪往往不一致，模型需要学习“性格→情绪”的映射，远非传统情感分类可比。

## 🎮 使用演示

1. 输入一段文本，例如：  
   *“Your flight has been cancelled due to a snowstorm.”*由于暴风雪，您的航班取消了
2. 拖动 **神经质 (Neuroticism)** 滑块至 0.85，其他保持 0.5。
3. 点击“预测情绪”，模型输出：  
   - 😨 恐惧 (fear) 38.2%  
   - 😢 悲伤 (sadness) 22.1%  
   - 😊 开心 (joy) 5.3% …-😊（joy） 5.3% & help；
4. 再将神经质调至 0.20，宜人性调至 0.80，再次预测：  
   - 😐 中性 (neutral) 45.6%  
   - 😊 开心 (joy) 20.3% …-😊（joy） 20.3% & help；

**这正是本项目想表达的：性格改变了情绪反应。**

## 📦 硬件要求

- **训练**：推荐 NVIDIA 显卡 ≥4GB 显存（如 RTX 3050 Ti），纯 CPU 训练极慢不推荐。
- **推理（桌面应用）**：CPU 即可，但首次使用需下载本地翻译模型（约 300 MB，通过镜像自动加速）。

## 🔧 常见问题

**Q: 运行 `desktop_app.py` 时找不到模型？**  
A: 请先运行 `train.py` 完成训练，或在 `desktop_app.py` 中修改 `MODEL_PATH` 指向已有模型文件夹的绝对路径。

**Q: 在线翻译（Google/有道）失败？**  
A: 默认翻译模式已改为“本地翻译 (离线模型)”，完全离线可用。若需在线翻译，请确保网络可访问相应服务。

**Q: 如何上传自己的模型？**  
A: 模型权重文件较大，已在 `.gitignore` 中忽略。如需分享，可以上传至 Releases 页面或使用 Git LFS。

## 📚 数据集引用

本项目的训练数据来自 **Persona-E²: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events** (ACL 2026)。  本项目的训练数据来自 **Persona-E²: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Eventspersona - e：一个以人为基础的数据集，用于研究文本事件中性格塑造的情绪反应** (ACL 2026)。
- arXiv: https://arxiv.org/abs/2604.09162  
- 数据集由真实性格测量的标注者产生，包含 3111 个文本事件和 111,996 条情绪标注。

## 🤝 贡献与许可

本项目仅供学习和研究使用。欢迎提出 Issue 或 Pull Request。  
MIT License © 2026 wjxn13MIT许可证和副本；2026 wjxn13

---

**如果觉得有用，请给一个 ⭐ Star！**
```

---

