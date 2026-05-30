import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    PreTrainedModel, TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
import os
import glob
import re
warnings.filterwarnings("ignore")

# ==============================================
# 全局定义：Dataset 类、模型类、评估函数
# ==============================================

class PersonaEmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, bfi_cols):
        self.texts = df["english_item"].tolist()
        self.bfi = df[bfi_cols].values.astype(np.float32)
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["bfi"] = torch.tensor(self.bfi[idx], dtype=torch.float)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BertWithPersona(PreTrainedModel):
    def __init__(self, model_name, num_labels, bfi_dim=5):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(model_name)
        self.persona_encoder = torch.nn.Sequential(
            torch.nn.Linear(bfi_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.classifier = torch.nn.Linear(config.hidden_size + 64, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, bfi, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]  # [CLS]
        persona_feat = self.persona_encoder(bfi)
        combined = torch.cat([text_feat, persona_feat], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ==============================================
# 主程序
# ==============================================

def main():
    # 路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "基于性格建模的文本情绪反应预测")

    # ================= 加速参数 =================
    model_name = "distilbert-base-uncased"   # ★ 轻量模型，快2倍
    max_len = 64                              # ★ 缩短文本，计算量减半
    batch_size = 16                           # ★ 加大batch，让GPU吃饱
    gradient_accumulation_steps = 1           # 等效 batch=16，无需累积
    epochs = 3
    learning_rate = 2e-5
    dataloader_num_workers = 4               # ★ 更多进程并行加载
    # ===========================================

    # 读取数据
    df_ann = pd.read_csv(os.path.join(data_dir, "1_dataset_all_annotators.csv"))
    df_prof = pd.read_csv(os.path.join(data_dir, "3_annotator_profiles.csv"))

    # 重命名 BFI 列
    df_prof = df_prof.rename(columns={
        "ID": "annotator_id",
        "Open.": "Openness",
        "Cons.": "Conscientiousness",
        "Extra.": "Extraversion",
        "Agree.": "Agreeableness",
        "Neuro.": "Neuroticism"
    })
    bfi_cols = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    # 宽表转长表
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

    # 标签编码
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    num_labels = len(le.classes_)
    print(f"情绪类别数: {num_labels}")
    print(f"类别: {le.classes_}")

    # 按事件划分
    event_ids = df["id"].unique()
    train_ids, temp_ids = train_test_split(event_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    df_train = df[df["id"].isin(train_ids)]
    df_val = df[df["id"].isin(val_ids)]
    df_test = df[df["id"].isin(test_ids)]

    print(f"训练样本: {len(df_train)}")
    print(f"验证样本: {len(df_val)}")
    print(f"测试样本: {len(df_test)}")

    # 构建 Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PersonaEmotionDataset(df_train, tokenizer, max_len, bfi_cols)
    val_dataset = PersonaEmotionDataset(df_val, tokenizer, max_len, bfi_cols)
    test_dataset = PersonaEmotionDataset(df_test, tokenizer, max_len, bfi_cols)

    # 模型
    model = BertWithPersona(model_name, num_labels)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./persona_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---------- 断点续训：自动查找最新 checkpoint ----------
    def find_latest_checkpoint(output_dir):
        """返回 output_dir 下最新的 checkpoint 路径，若没有则返回 None"""
        if not os.path.isdir(output_dir):
            return None
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        max_num = -1
        best_ckpt = None
        for ckpt in checkpoints:
            basename = os.path.basename(ckpt)
            match = re.match(r"checkpoint-(\d+)", basename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
                    best_ckpt = ckpt
        return best_ckpt

    latest_checkpoint = find_latest_checkpoint("./persona_results")
    if latest_checkpoint:
        print(f"找到 checkpoint: {latest_checkpoint}，从断点继续训练")
    else:
        print("未找到 checkpoint，从头开始训练")

    print("开始训练...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    print("\n测试集结果:")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    # 保存最终模型
    model.save_pretrained("./persona_emotion_model")
    tokenizer.save_pretrained("./persona_emotion_model")


if __name__ == '__main__':
    main()