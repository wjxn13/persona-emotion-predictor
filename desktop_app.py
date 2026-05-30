import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator
import re
import logging
from datetime import datetime

try:
    import translators as ts
    HAS_TRANSLATORS = True
except ImportError:
    HAS_TRANSLATORS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BertWithPersona(PreTrainedModel):
    def __init__(self, config, num_labels=7, bfi_dim=5):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.persona_encoder = torch.nn.Sequential(
            torch.nn.Linear(bfi_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.classifier = torch.nn.Linear(config.hidden_size + 64, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, bfi, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]
        persona_feat = self.persona_encoder(bfi)
        combined = torch.cat([text_feat, persona_feat], dim=1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "persona_emotion_model")
logger.info("正在加载情绪预测模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
model = BertWithPersona(config, num_labels=7, bfi_dim=5)

if os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
else:
    state_dict = torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location="cpu")

model.load_state_dict(state_dict, strict=False)
model.eval()
logger.info("情绪预测模型加载完成。")
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

EMOJI = {
    'anger': '😠', 'disgust': '🤢', 'fear': '😨', 'joy': '😊',
    'neutral': '😐', 'sadness': '😢', 'surprise': '😲'
}
CN_LABEL = {
    'anger': '愤怒', 'disgust': '厌恶', 'fear': '恐惧', 'joy': '开心',
    'neutral': '中性', 'sadness': '悲伤', 'surprise': '惊讶'
}

LOCAL_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
local_translator = None

class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("性格情绪预测器")
        root.geometry("650x800")
        root.resizable(False, False)

        mode_frame = tk.LabelFrame(root, text="翻译模式", font=("微软雅黑", 9))
        mode_frame.pack(pady=(10, 0), fill="x", padx=10)

        self.trans_mode = tk.StringVar(value="local")
        tk.Radiobutton(mode_frame, text="本地翻译 (离线模型)", variable=self.trans_mode,
                       value="local", font=("微软雅黑", 9)).pack(side="left", padx=10)
        tk.Radiobutton(mode_frame, text="在线翻译 (Google)", variable=self.trans_mode,
                       value="google", font=("微软雅黑", 9)).pack(side="left", padx=10)
        tk.Radiobutton(mode_frame, text="国内翻译 (有道)", variable=self.trans_mode,
                       value="youdao", font=("微软雅黑", 9)).pack(side="left", padx=10)

        tk.Label(root, text="请输入英文或中文文本事件:", font=("微软雅黑", 10)).pack(pady=(10, 0))
        self.text_box = tk.Text(root, height=5, width=60, font=("Consolas", 10))
        self.text_box.pack(pady=5)

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

        tk.Button(root, text="预测情绪", command=self.predict, font=("微软雅黑", 10),
                  bg="#4CAF50", fg="white").pack(pady=10)

        tk.Label(root, text="翻译结果 (英文):", font=("微软雅黑", 10)).pack(pady=(5, 0))
        self.translated_text = tk.Text(root, height=2, width=55, font=("Consolas", 10), state="disabled", bg="#f0f0f0")
        self.translated_text.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("微软雅黑", 14, "bold"))
        self.result_label.pack(pady=5)

        self.prob_text = tk.Text(root, height=10, width=55, font=("Consolas", 10), state="disabled")
        self.prob_text.pack(pady=5)

    def is_chinese(self, text):
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def load_local_translator(self):
        global local_translator
        if local_translator is not None:
            return local_translator
        try:
            logger.info("正在加载本地翻译模型（镜像加速）...")
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

    def translate_youdao(self, text):
        if not HAS_TRANSLATORS:
            raise RuntimeError("translators 库未安装")
        logger.info(f"[有道翻译] 开始翻译：{text[:30]}...")
        result = ts.youdao(text, from_language='zh', to_language='en')
        logger.info(f"[有道翻译] 成功：{result[:30]}...")
        return result

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

        # 智能回退链：在线模式失败 -> 尝试本地 -> 都失败才报错
        if mode in ("youdao", "google"):
            # 先尝试当前选择的在线翻译
            if mode == "youdao":
                try:
                    return self.translate_youdao(text)
                except Exception as e:
                    logger.warning(f"有道翻译失败：{e}，尝试回退到本地翻译...")
                    local_result = self.translate_local(text)
                    if local_result is not None:
                        self.trans_mode.set("local")
                        messagebox.showinfo("翻译切换", "有道翻译失败，已自动切换为本地翻译。")
                        return local_result
                    # 本地也失败，再试 Google
                    logger.warning("本地翻译不可用，尝试 Google...")
                    google_result = self.translate_google(text)
                    if google_result is not None:
                        self.trans_mode.set("google")
                        messagebox.showinfo("翻译切换", "已自动切换为 Google 翻译。")
                        return google_result
                    return None
            else:  # google
                result = self.translate_google(text)
                if result is not None:
                    return result
                logger.warning("Google 翻译失败，尝试回退到本地翻译...")
                local_result = self.translate_local(text)
                if local_result is not None:
                    self.trans_mode.set("local")
                    messagebox.showinfo("翻译切换", "Google 翻译失败，已自动切换为本地翻译。")
                    return local_result
                return None
        else:  # local
            return self.translate_local(text)

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

        encoding = tokenizer(text_for_model, truncation=True, padding="max_length",
                             max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"], bfi=bfi)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

        pred_idx = np.argmax(probs)
        pred_emotion = EMOTIONS[pred_idx]
        logger.info(f"预测结果：{CN_LABEL[pred_emotion]} ({pred_emotion})，概率 {probs[pred_idx]*100:.1f}%")

        self.result_label.config(
            text=f"预测情绪：{EMOJI[pred_emotion]} {CN_LABEL[pred_emotion]} ({pred_emotion})  {probs[pred_idx]*100:.1f}%",
            fg="blue"
        )

        lines = []
        for emo, prob in zip(EMOTIONS, probs):
            bar_len = int(prob * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            percent = prob * 100
            prefix = "★ " if emo == pred_emotion else "  "
            line = f"{prefix}{EMOJI[emo]} {CN_LABEL[emo]:<6} |{bar}| {percent:5.1f}%"
            lines.append(line)

        self.prob_text.config(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert("1.0", "\n".join(lines))
        self.prob_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    logger.info("GUI 启动完成，默认翻译模式：本地离线，等待用户操作...")
    root.mainloop()