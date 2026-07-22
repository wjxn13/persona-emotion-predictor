import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    PreTrainedModel, TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
import os
import glob
import re
import json
warnings.filterwarnings("ignore")

# ============================================================
# 固定情绪类别顺序（不再依赖 LabelEncoder 字母序，消除推断端硬编码脆弱性）
# ============================================================
LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
NUM_LABELS = len(LABELS)
LABEL2IDX = {lab: i for i, lab in enumerate(LABELS)}

# ============================================================
# 领域(domain)映射：data_source -> News / Social Media / Life Experience
# 论文表明领域对情绪有系统性影响，这是此前完全未使用的强特征。
# ============================================================
NEWS_SOURCES = {'今日头条', '澎湃新闻', 'bbc_news', 'abc_news', 'independent_news'}
SOCIAL_SOURCES = {'微博', 'reddit', 'SocialChemistry', '微信推文'}
LIFE_SOURCES = {'FMylife', 'benignexistence', 'kindlife', 'IUTB'}

def map_domain(src):
    if src in NEWS_SOURCES:
        return 'News'
    if src in SOCIAL_SOURCES:
        return 'Social Media'
    if src in LIFE_SOURCES:
        return 'Life Experience'
    return 'News'  # 未知来源兜底

DOMAIN_LIST = ['News', 'Social Media', 'Life Experience']
DOMAIN2IDX = {d: i for i, d in enumerate(DOMAIN_LIST)}

# 是否使用 soft label（基于 36 标注者分布 + confidence 加权）。关掉则退回 hard label。
USE_SOFT_LABELS = True

# ============================================================
# 数据集
# ============================================================
class PersonaEmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, bfi_cols, use_soft=True):
        self.texts = df["english_item"].tolist()
        self.bfi = df[bfi_cols].values.astype(np.float32)
        # domain 预计算为 one-hot(3)
        self.domain = np.eye(len(DOMAIN_LIST), dtype=np.float32)[df["domain_idx"].values]
        # soft_label: (N, 7) 概率分布；hard_label: (N,) 整数（仅用于评估指标）
        self.soft = np.vstack(df["soft_label"].values).astype(np.float32)
        self.hard = df["hard_label"].values.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_soft = use_soft

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["bfi"] = torch.tensor(self.bfi[idx], dtype=torch.float)
        item["domain"] = torch.tensor(self.domain[idx], dtype=torch.float)
        # 始终携带 hard label 供评估指标使用；soft label 优先用于训练损失
        item["labels"] = torch.tensor(self.hard[idx], dtype=torch.long)
        if self.use_soft:
            item["soft_target"] = torch.tensor(self.soft[idx], dtype=torch.float)
        return item


# ============================================================
# 模型 V2：在 V1(文本+性格) 基础上加入 domain 分支
# ============================================================
class BertWithPersona(PreTrainedModel):
    def __init__(self, model_name, num_labels=NUM_LABELS, bfi_dim=5, domain_dim=3):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(model_name)
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

    def forward(self, input_ids, attention_mask, bfi, domain, labels=None, soft_target=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]  # [CLS]
        persona_feat = self.persona_encoder(bfi)
        domain_feat = self.domain_encoder(domain)
        combined = torch.cat([text_feat, persona_feat, domain_feat], dim=1)
        logits = self.classifier(combined)

        loss = None
        if soft_target is not None:
            # soft label：KL 散度（模型输出经 log_softmax）
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, soft_target, reduction="batchmean")
        elif labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ============================================================
# 主程序
# ============================================================
def build_soft_label(row):
    """用 36 标注者的情绪 + confidence 加权，构建该 event 的 7 类分布。"""
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    total = 0.0
    for i in range(1, 37):
        e = row.get(f"E{i}_emotion")
        c = row.get(f"E{i}_confidence")
        if pd.isna(e):
            continue
        e = str(e).strip()
        if e not in LABEL2IDX:
            continue
        w = 0.0 if pd.isna(c) else float(c)
        vec[LABEL2IDX[e]] += w
        total += w
    if total > 0:
        vec = vec / total
    else:
        vec[:] = 1.0 / NUM_LABELS
    return vec


def main():
    SMOKE = os.environ.get("SMOKE") == "1"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "基于性格建模的文本情绪反应预测")

    # ================= 加速参数 =================
    model_name = "distilbert-base-uncased"
    max_len = 64
    batch_size = 16
    gradient_accumulation_steps = 1
    epochs = 3
    learning_rate = 2e-5
    # Windows 下多进程 DataLoader 易卡死，自动降级为 0；Linux 保持 4 加速
    dataloader_num_workers = 0 if os.name == "nt" else 4
    # ===========================================

    # 读取数据
    df_ann = pd.read_csv(os.path.join(data_dir, "1_dataset_all_annotators.csv"))
    df_prof = pd.read_csv(os.path.join(data_dir, "3_annotator_profiles.csv"))

    df_prof = df_prof.rename(columns={
        "ID": "annotator_id",
        "Open.": "Openness",
        "Cons.": "Conscientiousness",
        "Extra.": "Extraversion",
        "Agree.": "Agreeableness",
        "Neuro.": "Neuroticism"
    })
    bfi_cols = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    # ---- 预计算 event 级 soft label 与 domain ----
    print("构建 event 级 soft label 与 domain ...")
    soft_map = {}
    for _, row in df_ann.iterrows():
        soft_map[row["id"]] = build_soft_label(row)
    df_ann["domain_name"] = df_ann["data_source"].apply(map_domain)
    df_ann["domain_idx"] = df_ann["domain_name"].map(DOMAIN2IDX)

    # 宽表转长表（每个 annotator×event 一行）
    annotator_ids = [f"E{i}" for i in range(1, 37)]
    rows = []
    for eid in annotator_ids:
        emotion_col = f"{eid}_emotion"
        confidence_col = f"{eid}_confidence"
        if emotion_col not in df_ann.columns:
            continue
        tmp = df_ann[["id", "english_item", emotion_col, confidence_col]].copy()
        tmp = tmp.rename(columns={emotion_col: "emotion", confidence_col: "confidence"})
        tmp["annotator_id"] = eid
        rows.append(tmp)

    df_long = pd.concat(rows, ignore_index=True)
    df_long = df_long.dropna(subset=["emotion"])
    df = df_long.merge(df_prof, on="annotator_id", how="left")

    # 合并 domain_idx 与 soft_label
    df["domain_idx"] = df["id"].map(df_ann.set_index("id")["domain_idx"])
    df["soft_label"] = df["id"].map(soft_map)
    df["hard_label"] = df["emotion"].map(LABEL2IDX)

    print(f"总样本: {len(df)}，情绪类别数: {NUM_LABELS}，领域: {DOMAIN_LIST}")

    # 按事件划分（防止同一事件泄漏到训练/测试）
    event_ids = df["id"].unique()
    train_ids, temp_ids = train_test_split(event_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    df_train = df[df["id"].isin(train_ids)]
    df_val = df[df["id"].isin(val_ids)]
    df_test = df[df["id"].isin(test_ids)]

    print(f"训练样本: {len(df_train)}  验证样本: {len(df_val)}  测试样本: {len(df_test)}")

    # 构建 Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PersonaEmotionDataset(df_train, tokenizer, max_len, bfi_cols, use_soft=USE_SOFT_LABELS)
    val_dataset = PersonaEmotionDataset(df_val, tokenizer, max_len, bfi_cols, use_soft=USE_SOFT_LABELS)
    test_dataset = PersonaEmotionDataset(df_test, tokenizer, max_len, bfi_cols, use_soft=USE_SOFT_LABELS)

    # smoke 测试：极小样本 + 极少步数，仅验证逻辑
    if SMOKE:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(min(64, len(train_dataset)))))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(32, len(val_dataset)))))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(min(32, len(test_dataset)))))
        epochs = 1

    model = BertWithPersona(model_name, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir="./persona_results",
        eval_strategy="epoch" if not SMOKE else "steps",
        eval_steps=5 if SMOKE else None,
        save_strategy="epoch" if not SMOKE else "no",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=5 if SMOKE else -1,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=(not SMOKE),
        load_best_model_at_end=(not SMOKE),
        metric_for_best_model="macro_f1" if not SMOKE else "loss",
        greater_is_better=(not SMOKE),
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0 if SMOKE else dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[] if SMOKE else [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---------- 断点续训：自动查找最新 checkpoint ----------
    def find_latest_checkpoint(output_dir):
        if not os.path.isdir(output_dir):
            return None
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        max_num = -1
        best_ckpt = None
        for ckpt in checkpoints:
            m = re.match(r"checkpoint-(\d+)", os.path.basename(ckpt))
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num
                    best_ckpt = ckpt
        return best_ckpt

    latest_checkpoint = find_latest_checkpoint("./persona_results")
    if latest_checkpoint:
        print(f"找到 checkpoint: {latest_checkpoint}，从断点继续训练")
    else:
        print("未找到 checkpoint，从头开始训练")

    if SMOKE:
        print("[SMOKE] 仅验证数据管线与 forward，不真正训练")
        trainer.train()
        # 手动评估，验证指标逻辑（绕过 Trainer.evaluate 内部细节）
        from torch.utils.data import DataLoader
        model.eval()
        dl = DataLoader(test_dataset, batch_size=16)
        all_preds, all_labels = [], []
        for batch in dl:
            with torch.no_grad():
                out = model(input_ids=batch["input_ids"].to(model.device),
                            attention_mask=batch["attention_mask"].to(model.device),
                            bfi=batch["bfi"].to(model.device),
                            domain=batch["domain"].to(model.device))
            all_preds.append(out.logits.argmax(-1).cpu().numpy())
            all_labels.append(batch["labels"].numpy())
        preds = np.concatenate(all_preds); labels = np.concatenate(all_labels)
        print(f"[SMOKE] 手动评估 acc={accuracy_score(labels, preds):.4f} "
              f"macro_f1={f1_score(labels, preds, average='macro'):.4f}")
    else:
        print("开始训练...")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
        print("\n测试集结果:")
        print(trainer.evaluate(test_dataset))

    # 保存最终模型 + label_map（供 desktop_app 推断端使用，消除硬编码顺序）
    if SMOKE:
        print("[SMOKE] 跳过模型保存，避免覆盖真实权重 persona_emotion_model/")
    else:
        os.makedirs("./persona_emotion_model", exist_ok=True)
        model.save_pretrained("./persona_emotion_model")
        tokenizer.save_pretrained("./persona_emotion_model")
        label_map = {
            "labels": LABELS,
            "label2idx": LABEL2IDX,
            "domain_list": DOMAIN_LIST,
            "domain2idx": DOMAIN2IDX,
            "use_soft_target": USE_SOFT_LABELS,
        }
        with open(os.path.join("./persona_emotion_model", "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print("已保存模型与 label_map.json")


if __name__ == '__main__':
    main()
