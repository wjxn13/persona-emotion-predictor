import os
import time

total_start = time.time()

# ===== 环境变量（必须在导入 torch 前设置） =====
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 尊重启动脚本（如 .bat）的离线设置：仅当未显式设置时才默认允许联网下载缓存。
# 已缓存翻译模型后，把 .bat 里的 HF_HUB_OFFLINE=1 打开即可彻底离线运行。
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

print(f"[计时] 环境变量设置完成，耗时 {time.time() - total_start:.3f} 秒")

# ===== 最小化导入：避开 auto_factory，直接加载所需具体类 =====
import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
import re
import logging
import json
from datetime import datetime
import csv
import tkinter.ttk as ttk
from collections import Counter

# 直接导入 distilbert 具体类，不触发 sklearn/accelerate 依赖
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator

print(f"[计时] 所有库导入完成，耗时 {time.time() - total_start:.3f} 秒")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===== 默认标签/领域（兼容旧权重无 label_map 的情况） =====
DEFAULT_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
DEFAULT_DOMAINS = ['News', 'Social Media', 'Life Experience']
EMOJI_MAP = {'anger': '😠', 'disgust': '🤢', 'fear': '😨', 'joy': '😊',
             'neutral': '😐', 'sadness': '😢', 'surprise': '😲'}
CN_MAP = {'anger': '愤怒', 'disgust': '厌恶', 'fear': '恐惧', 'joy': '开心',
          'neutral': '中性', 'sadness': '悲伤', 'surprise': '惊讶'}


# ================= 自定义模型 V1（旧结构，无 domain） =================
class BertWithPersona(PreTrainedModel):
    def __init__(self, config, num_labels=7, bfi_dim=5):
        super().__init__(config)
        # 直接使用 DistilBertModel，不再经过 AutoModel
        self.bert = DistilBertModel(config)
        self.persona_encoder = torch.nn.Sequential(
            torch.nn.Linear(bfi_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.classifier = torch.nn.Linear(config.hidden_size + 64, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, bfi, domain=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]  # CLS token
        persona_feat = self.persona_encoder(bfi)
        combined = torch.cat([text_feat, persona_feat], dim=1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ================= 自定义模型 V2（新结构，含 domain 分支） =================
class BertWithPersonaV2(PreTrainedModel):
    def __init__(self, config, num_labels=7, bfi_dim=5, domain_dim=3):
        super().__init__(config)
        self.bert = DistilBertModel(config)
        self.persona_encoder = torch.nn.Sequential(
            torch.nn.Linear(bfi_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.domain_encoder = torch.nn.Sequential(
            torch.nn.Linear(domain_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.classifier = torch.nn.Linear(config.hidden_size + 64 + 32, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, bfi, domain=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]
        persona_feat = self.persona_encoder(bfi)
        combined = torch.cat([text_feat, persona_feat], dim=1)
        if domain is not None:
            domain_feat = self.domain_encoder(domain)
            combined = torch.cat([combined, domain_feat], dim=1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ================= 加载情绪预测模型 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "persona_emotion_model")

logger.info("开始加载情绪预测模型...")
t0 = time.time()

# 从 label_map.json 读取标签/领域顺序（消除硬编码脆弱性；旧权重无此文件则用默认）
LABELS = DEFAULT_LABELS
DOMAIN_LIST = DEFAULT_DOMAINS
label_map_path = os.path.join(MODEL_PATH, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    LABELS = lm.get("labels", DEFAULT_LABELS)
    DOMAIN_LIST = lm.get("domain_list", DEFAULT_DOMAINS)
    logger.info("已加载 label_map.json，标签顺序与训练端一致")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
config = DistilBertConfig.from_pretrained(MODEL_PATH)

if os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
else:
    state_dict = torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location="cpu")

# 检测权重是否含 domain 分支 → 选择 V1 / V2
has_domain = any(k.startswith("domain_encoder") for k in state_dict.keys())
if has_domain:
    model = BertWithPersonaV2(config, num_labels=len(LABELS))
    domain_enabled = True
    logger.info("检测到 domain 分支，使用 V2 模型（领域特征已启用）")
else:
    model = BertWithPersona(config, num_labels=len(LABELS))
    domain_enabled = False
    logger.info("未检测到 domain 分支，使用 V1 模型（领域特征不可用，请重新训练启用）")

model.load_state_dict(state_dict, strict=False)
model.eval()
logger.info(f"情绪预测模型加载完成，耗时 {time.time() - t0:.3f} 秒。domain_enabled={domain_enabled}")

EMOTIONS = LABELS  # 顺序与模型输出严格一致

# 本地翻译模型名称
LOCAL_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
local_translator = None


# ================= GUI 应用 =================
class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("性格情绪预测器")
        root.geometry("650x850")
        root.resizable(False, False)

        # 翻译模式选择
        mode_frame = tk.LabelFrame(root, text="翻译模式", font=("微软雅黑", 9))
        mode_frame.pack(pady=(10, 0), fill="x", padx=10)

        self.trans_mode = tk.StringVar(value="local")
        tk.Radiobutton(mode_frame, text="本地翻译 (离线模型)", variable=self.trans_mode,
                       value="local", font=("微软雅黑", 9)).pack(side="left", padx=10)
        tk.Radiobutton(mode_frame, text="在线翻译 (Google)", variable=self.trans_mode,
                       value="google", font=("微软雅黑", 9)).pack(side="left", padx=10)

        # 事件领域选择（论文证明领域对情绪有系统性影响）
        if domain_enabled:
            domain_frame = tk.LabelFrame(root, text="事件领域 (Domain)", font=("微软雅黑", 9))
            domain_frame.pack(pady=(8, 0), fill="x", padx=10)
            self.domain_var = tk.StringVar(value="Social Media")
            for d in DOMAIN_LIST:
                tk.Radiobutton(domain_frame, text=d, variable=self.domain_var,
                               value=d, font=("微软雅黑", 9)).pack(side="left", padx=10)
        else:
            tk.Label(root, text="⚠ 当前模型未含领域特征；用新 train.py 重新训练后，"
                                 "即可在此选择 News / Social Media / Life Experience",
                     font=("微软雅黑", 8), fg="gray").pack(pady=(4, 0))

        # 文本输入
        tk.Label(root, text="请输入英文或中文文本事件:", font=("微软雅黑", 10)).pack(pady=(10, 0))
        self.text_box = tk.Text(root, height=5, width=60, font=("Consolas", 10))
        self.text_box.pack(pady=5)

        # 性格滑块
        slider_frame = tk.Frame(root)
        slider_frame.pack(pady=10)
        self.sliders = {}
        traits = ["开放性 (Openness)", "尽责性 (Conscientiousness)", "外向性 (Extraversion)",
                  "宜人性 (Agreeableness)", "神经质 (Neuroticism)"]
        for i, trait in enumerate(traits):
            tk.Label(slider_frame, text=trait, font=("微软雅黑", 9)).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(slider_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=300)
            scale.set(0.5)
            scale.grid(row=i, column=1, padx=5)
            self.sliders[trait] = scale

        # 预测按钮
        tk.Button(root, text="预测情绪", command=self.predict, font=("微软雅黑", 10),
                  bg="#4CAF50", fg="white").pack(pady=10)

        # 性格扫描 Demo 入口
        tk.Button(root, text="🎯 性格扫描 Demo（看性格如何改变情绪）",
                  command=self.open_scan_demo, font=("微软雅黑", 10),
                  bg="#534AB7", fg="white").pack(pady=(0, 6))

        # 翻译结果显示
        tk.Label(root, text="翻译结果 (英文):", font=("微软雅黑", 10)).pack(pady=(5, 0))
        self.translated_text = tk.Text(root, height=2, width=55, font=("Consolas", 10), state="disabled", bg="#f0f0f0")
        self.translated_text.pack(pady=5)

        # 预测结果显示
        self.result_label = tk.Label(root, text="", font=("微软雅黑", 14, "bold"))
        self.result_label.pack(pady=5)

        self.prob_text = tk.Text(root, height=10, width=55, font=("Consolas", 10), state="disabled")
        self.prob_text.pack(pady=5)

        # 启动后预热本地翻译模型（不阻塞界面）
        self.root.after(100, self.warmup_translator)

    def warmup_translator(self):
        logger.info("开始预热本地翻译模型...")
        t0 = time.time()
        self.load_local_translator()
        logger.info(f"本地翻译模型预热完成，耗时 {time.time() - t0:.3f} 秒")

    def is_chinese(self, text):
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def load_local_translator(self):
        global local_translator
        if local_translator is not None:
            return local_translator
        try:
            tok = MarianTokenizer.from_pretrained(LOCAL_MODEL_NAME)
            mod = MarianMTModel.from_pretrained(LOCAL_MODEL_NAME)
            local_translator = (tok, mod)
            logger.info("本地翻译模型加载成功。")
            return local_translator
        except Exception as e:
            logger.error(f"本地翻译模型加载失败：{e}")
            return None

    def translate_google(self, text):
        logger.info(f"[Google翻译] 开始翻译：{text[:30]}...")
        try:
            result = GoogleTranslator(source='zh-CN', target='en').translate(text)
            logger.info(f"[Google翻译] 成功：{result[:30]}...")
            return result
        except Exception as e:
            logger.error(f"[Google翻译] 失败：{e}")
            return None

    def translate_local(self, text):
        translator = self.load_local_translator()
        if translator is None:
            return None
        tok, mod = translator
        logger.info(f"[本地翻译] 开始翻译：{text[:30]}...")
        inputs = tok(text, return_tensors="pt", padding=True, truncation=True)
        outputs = mod.generate(**inputs)
        result = tok.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"[本地翻译] 成功：{result[:30]}...")
        return result

    def translate_to_english(self, text):
        mode = self.trans_mode.get()
        logger.info(f"当前翻译模式：{mode}")
        if mode == "google":
            result = self.translate_google(text)
            if result is None:
                messagebox.showinfo("翻译切换", "Google 翻译失败，尝试本地翻译...")
                self.trans_mode.set("local")
                return self.translate_local(text)
            return result
        else:  # local
            result = self.translate_local(text)
            if result is None:
                messagebox.showinfo("翻译切换", "本地翻译不可用，尝试 Google 翻译...")
                self.trans_mode.set("google")
                return self.translate_google(text)
            return result

    def predict(self):
        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入文本")
            return

        logger.info(f"收到输入文本：{text[:50]}...")
        self.translated_text.config(state="normal")
        self.translated_text.delete("1.0", tk.END)

        if self.is_chinese(text):
            self.result_label.config(text="翻译中，请稍候...", fg="gray")
            self.root.update()
            translated = self.translate_to_english(text)
            if translated is None:
                self.translated_text.insert("1.0", "翻译失败，请检查网络或切换为本地翻译")
                self.translated_text.config(state="disabled")
                logger.error("翻译失败，预测终止。")
                self.result_label.config(text="预测失败", fg="red")
                return
            text_for_model = translated
            self.translated_text.insert("1.0", translated)
        else:
            text_for_model = text
            self.translated_text.insert("1.0", "（无需翻译）")
        self.translated_text.config(state="disabled")

        bfi_vals = [self.sliders[trait].get() for trait in self.sliders]
        bfi = torch.tensor([bfi_vals], dtype=torch.float)

        # 领域特征（仅 V2 模型生效）
        domain_vec = None
        if domain_enabled:
            dname = self.domain_var.get()
            didx = DOMAIN_LIST.index(dname)
            domain_vec = torch.zeros(1, len(DOMAIN_LIST))
            domain_vec[0, didx] = 1.0

        encoding = tokenizer(text_for_model, truncation=True, padding="max_length",
                             max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"],
                            bfi=bfi, domain=domain_vec)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

        pred_idx = np.argmax(probs)
        pred_emotion = EMOTIONS[pred_idx]
        logger.info(f"预测结果：{CN_MAP.get(pred_emotion, pred_emotion)} ({pred_emotion})，概率 {probs[pred_idx]*100:.1f}%")

        self.result_label.config(
            text=f"预测情绪：{EMOJI_MAP.get(pred_emotion, '')} {CN_MAP.get(pred_emotion, pred_emotion)} ({pred_emotion})  {probs[pred_idx]*100:.1f}%",
            fg="blue"
        )

        lines = []
        for emo, prob in zip(EMOTIONS, probs):
            bar_len = int(prob * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            percent = prob * 100
            prefix = "★ " if emo == pred_emotion else "  "
            line = f"{prefix}{EMOJI_MAP.get(emo, '')} {CN_MAP.get(emo, emo):<6} |{bar}| {percent:5.1f}%"
            lines.append(line)

        self.prob_text.config(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert("1.0", "\n".join(lines))
        self.prob_text.config(state="disabled")

    # ============ 性格扫描 Demo：固定事件+领域，扫 BFI 一维看 7 类情绪曲线 ============
    def open_scan_demo(self):
        ScanDemoWindow(self.root, self)


# ================= 性格扫描 / 标注者分布 Demo 窗口 =================
class ScanDemoWindow:
    TRAITS = ["开放性 Openness", "尽责性 Conscientiousness", "外向性 Extraversion",
              "宜人性 Agreeableness", "神经质 Neuroticism"]
    TRAIT_KEYS = ["open", "cons", "extr", "agre", "neur"]
    EVENTS = [
        ("被 Wheelchair 碾过（自嘲没存在感）", "Today, I got run over. Not by a car, but by a lady in a wheelchair..."),
        ("前男友结婚，新娘是当年吵架那个", "Today, my ex-boyfriend got married. Normally this wouldn't bother me... had his bride not been the girl we constantly argued about..."),
        ("球衣被印上 NO NAME 还多收费", "Today, I received my soccer team jacket... My jacket now has 'NO NAME' spelled out on it..."),
        ("股市暴跌，投资全亏", "The stock market crashed today and I lost all my investments."),
        ("中彩票大奖", "I just won the lottery and became a millionaire!"),
    ]

    def __init__(self, parent, app):
        self.app = app
        self.scan_cache = {}
        win = tk.Toplevel(parent)
        win.title("🎯 性格扫描 Demo")
        win.geometry("900x720")
        win.resizable(False, False)
        self.win = win

        # 顶部：事件选择
        top = tk.Frame(win)
        top.pack(fill="x", padx=10, pady=8)
        tk.Label(top, text="事件：", font=("微软雅黑", 10)).pack(side="left")
        self.event_var = tk.StringVar(value=self.EVENTS[0][0])
        cb = ttk.Combobox(top, textvariable=self.event_var, width=58, font=("微软雅黑", 9), state="readonly")
        cb["values"] = [e[0] for e in self.EVENTS]
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        # 领域选择（仅 V2）
        dom_frame = tk.Frame(top)
        dom_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(dom_frame, text="领域：", font=("微软雅黑", 10)).pack(side="left")
        self.scan_domain_var = tk.StringVar(value="Social Media")
        for d in DOMAIN_LIST:
            tk.Radiobutton(dom_frame, text=d, variable=self.scan_domain_var, value=d,
                           font=("微软雅黑", 9), command=self.refresh).pack(side="left", padx=8)

        # 扫描维度选择
        dim_frame = tk.Frame(win)
        dim_frame.pack(fill="x", padx=10, pady=2)
        tk.Label(dim_frame, text="扫描维度：", font=("微软雅黑", 10)).pack(side="left")
        self.scan_dim = tk.StringVar(value=self.TRAITS[4])
        for t in self.TRAITS:
            tk.Radiobutton(dim_frame, text=t, variable=self.scan_dim, value=t,
                           font=("微软雅黑", 9), command=self.refresh).pack(side="left", padx=6)

        # 控制：固定其余四维
        ctrl = tk.Frame(win)
        ctrl.pack(fill="x", padx=10, pady=2)
        tk.Label(ctrl, text="其余四维固定值：", font=("微软雅黑", 9)).pack(side="left")
        self.base_var = tk.DoubleVar(value=0.5)
        tk.Scale(ctrl, variable=self.base_var, from_=0.0, to=1.0, resolution=0.05,
                 orient=tk.HORIZONTAL, length=160, font=("微软雅黑", 9),
                 command=lambda v: self.refresh()).pack(side="left", padx=6)

        # 画布
        self.canvas = tk.Canvas(win, bg="white", height=360, width=860)
        self.canvas.pack(fill="x", padx=10, pady=6)

        # 解释
        self.note = tk.Text(win, height=4, width=110, font=("微软雅黑", 9), state="disabled", bg="#f6f6f6")
        self.note.pack(padx=10, pady=4)

        # 36 标注者分布按钮
        tk.Button(win, text="📊 显示该事件 36 名标注者真实情绪投票", command=self.show_annotators,
                  font=("微软雅黑", 9), bg="#1D9E75", fg="white").pack(pady=4)

        self.refresh()

    def _scan(self, event_text, domain_name, dim_idx, base):
        key = (event_text, domain_name, dim_idx, round(base, 2))
        if key in self.scan_cache:
            return self.scan_cache[key]
        bfi = [base] * 5
        feats = ScanDemoWindow.TRAIT_KEYS
        enc = tokenizer(event_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        domain_vec = None
        if domain_enabled:
            didx = DOMAIN_LIST.index(domain_name)
            domain_vec = torch.zeros(1, len(DOMAIN_LIST))
            domain_vec[0, didx] = 1.0
        curves = {emo: [] for emo in EMOTIONS}
        with torch.no_grad():
            for s in np.linspace(0.0, 1.0, 21):
                bfi[dim_idx] = float(s)
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            bfi=torch.tensor([bfi], dtype=torch.float), domain=domain_vec)
                probs = torch.softmax(out.logits, dim=-1).squeeze().tolist()
                for emo, p in zip(EMOTIONS, probs):
                    curves[emo].append(p)
        self.scan_cache[key] = curves
        return curves

    def refresh(self):
        event_text = dict(self.EVENTS)[self.event_var.get()]
        domain_name = self.scan_domain_var.get()
        dim_idx = self.TRAITS.index(self.scan_dim.get())
        base = self.base_var.get()
        curves = self._scan(event_text, domain_name, dim_idx, base)

        W, H = 860, 360
        pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 45
        cw, ch = W - pad_l - pad_r, H - pad_t - pad_b
        cv = self.canvas
        cv.delete("all")
        # 背景网格 + Y 轴刻度
        for i in range(6):
            y = pad_t + ch * i / 5
            cv.create_line(pad_l, y, W - pad_r, y, fill="#e0e0e0")
            cv.create_text(pad_l - 8, y, text=f"{100 - i * 20}%", anchor="e", font=("微软雅黑", 8), fill="#888")
        # 曲线
        n = 21
        colors = ["#E24B4A", "#BA7517", "#854F0B", "#639922", "#5F5E5A", "#378ADD", "#534AB7"]
        for idx, emo in enumerate(EMOTIONS):
            pts = []
            for j in range(n):
                x = pad_l + cw * j / (n - 1)
                y = pad_t + ch * (1 - curves[emo][j])
                pts.append((x, y))
            cv.create_line(*[c for p in pts for c in p], fill=colors[idx % len(colors)], width=2, smooth=True)
        # 图例
        lx = pad_l + 5
        for idx, emo in enumerate(EMOTIONS):
            y = pad_t + 6 + idx * 16
            cv.create_rectangle(lx, y, lx + 12, y + 10, fill=colors[idx % len(colors)], outline="")
            cv.create_text(lx + 16, y + 5, text=f"{emo}", anchor="w", font=("微软雅黑", 9), fill="#333")
        # X 轴标签
        cv.create_text((pad_l + W - pad_r) / 2, H - 12, text=f"{self.scan_dim.get()} 从 0.0 → 1.0",
                       font=("微软雅黑", 9), fill="#333")

        # 解释文字（诚实版：distilbert 轻量底座下性格利用率有限，曲线摆幅小是真实局限）
        argmax_0 = max(EMOTIONS, key=lambda e: curves[e][0])
        argmax_1 = max(EMOTIONS, key=lambda e: curves[e][-1])
        max_swing = max((max(curves[e]) - min(curves[e])) * 100 for e in EMOTIONS)
        self.note.config(state="normal")
        self.note.delete("1.0", tk.END)
        self.note.insert("1.0",
            f"横轴为「{self.scan_dim.get()}」从 0 到 1 连续扫描，其余四维固定 {base:.2f}；领域={domain_name}。\n"
            f"主导情绪在 {self.scan_dim.get()}=0.0 时为 {argmax_0}（{curves[argmax_0][0]*100:.1f}%），"
            f"=1.0 时为 {argmax_1}（{curves[argmax_1][-1]*100:.1f}%）；整条曲线最大摆幅仅 {max_swing:.1f}pp。\n"
            f"曲线较平说明：当前轻量模型对性格的利用率有限，情绪主要由事件语义决定——这与数据集里"
            f"「同一事件 36 人标注情绪常不一致」（personality illusion）是同一难题的两面。换 bert-base 底座可放大性格效应。")
        self.note.config(state="disabled")

    def show_annotators(self):
        # 在数据集里按 english_item 模糊匹配，聚合 36 标注者真实情绪投票
        event_text = dict(self.EVENTS)[self.event_var.get()]
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, "基于性格建模的文本情绪反应预测", "1_dataset_all_annotators.csv")
        votes = Counter()
        if not os.path.exists(csv_path):
            messagebox.showinfo("提示", "未找到数据集 1_dataset_all_annotators.csv，无法展示标注者分布。")
            return
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["english_item"].strip().startswith(event_text.split("（")[0][:20]):
                        for i in range(1, 37):
                            e = row.get(f"E{i}_emotion")
                            if e:
                                votes[e.strip().lower()] += 1
                        break
        except Exception as ex:
            messagebox.showerror("错误", f"读取数据集失败：{ex}")
            return

        top = tk.Toplevel(self.win)
        top.title("36 名标注者真实情绪投票")
        top.geometry("560x420")
        top.resizable(False, False)
        tk.Label(top, text=f"事件：{event_text}", font=("微软雅黑", 9), wraplength=520, justify="left").pack(padx=10, pady=6)
        tk.Label(top, text="同一事件，36 名标注者给出的情绪并不一致 —— 这正是预测准确率上限只有约 35% 的原因（personality illusion）。",
                 font=("微软雅黑", 9), fg="#888", wraplength=520, justify="left").pack(padx=10, pady=4)
        cv2 = tk.Canvas(top, bg="white", height=320, width=520)
        cv2.pack(padx=10, pady=6)
        if not votes:
            cv2.create_text(260, 160, text="（该事件未在数据集中精确匹配到标注记录）", font=("微软雅黑", 10), fill="#999")
            return
        maxv = max(votes.values())
        palette = ["#E24B4A", "#BA7517", "#854F0B", "#639922", "#5F5E5A", "#378ADD", "#534AB7"]
        order = EMOTIONS
        bh = 320 / max(len(order), 1)
        for i, emo in enumerate(order):
            v = votes.get(emo, 0)
            w = (v / maxv) * 420 if maxv else 0
            y = i * bh + 10
            cv2.create_rectangle(120, y, 120 + w, y + bh - 8, fill=palette[i % len(palette)], outline="")
            cv2.create_text(112, y + bh / 2 - 4, text=EMO, anchor="e", font=("微软雅黑", 9), fill="#333")
            cv2.create_text(126 + w, y + bh / 2 - 4, text=str(v), anchor="w", font=("微软雅黑", 9), fill="#333")


if __name__ == "__main__":
    logger.info(f"总启动耗时 {time.time() - total_start:.3f} 秒，准备启动 GUI...")
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
