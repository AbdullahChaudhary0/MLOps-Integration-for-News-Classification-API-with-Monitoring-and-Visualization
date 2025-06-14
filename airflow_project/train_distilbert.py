import os
import time
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ──────────── 1. Paths & Load ────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ──────────── 2. Prepare text & labels ────────────
X_train = pd.concat([train_df["processed_headline"], val_df["processed_headline"]],
                     ignore_index=True).fillna("")
y_train = pd.concat([train_df["category"], val_df["category"]], ignore_index=True)

X_test  = test_df["processed_headline"].fillna("")
y_test  = test_df["category"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
num_labels = len(le.classes_)

# ──────────── 3. Tokenizer & Dataset ────────────
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
MAX_LEN = 32
MAX_SAMPLES = 10000

class HeadlineDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(),
                                   truncation=True,
                                   padding="max_length",
                                   max_length=MAX_LEN)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

X_train = X_train[:MAX_SAMPLES]
y_train_enc = y_train_enc[:MAX_SAMPLES]

train_ds = HeadlineDataset(X_train, y_train_enc)
test_ds  = HeadlineDataset(X_test,  y_test_enc)
print(f"Training on {len(train_ds)} examples, evaluation on {len(test_ds)} examples")

# ──────────── 4. Model & Freezing ────────────
device = torch.device("cpu")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
for param in model.distilbert.parameters():
    param.requires_grad = False
model.to(device)

# ──────────── 5. Training Utilities ────────────
EPOCHS       = 1
BATCH_SIZE   = 32
LEARNING_RATE = 5e-5

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ──────────── 6. MLflow Setup ────────────
mlflow.set_experiment("Text_Classification_Transformer")
best_f1 = 0.0
best_run_id = None

for run_name in [f"DistilBERT_finetune_head_{EPOCHS}epochs"]:
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("freeze_backbone", True)
        mlflow.log_param("train_size", len(train_ds))
        mlflow.log_param("test_size",  len(test_ds))

        # TRAINING LOOP WITH PROGRESS BAR
        t0 = time.time()
        model.train()
        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
            for batch in tqdm(train_loader, desc="Training batches"):
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
            print(f"Finished epoch {epoch+1}/{EPOCHS}")
        train_time = time.time() - t0
        mlflow.log_metric("training_time", train_time)

        # INFERENCE
        t1 = time.time()
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation batches"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].cpu().numpy()
                logits = model(**inputs).logits.cpu().numpy()
                preds = np.argmax(logits, axis=1)
                all_preds.extend(preds)
                all_labels.extend(labels)
        infer_time = time.time() - t1
        mlflow.log_metric("inference_time", infer_time)

        # METRICS & ARTIFACTS
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="weighted")
        cm  = confusion_matrix(all_labels, all_preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.title("Confusion Matrix – Transformer")
        cm_path = "cm_transformer.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # MODEL LOGGING & REGISTRATION
        mlflow.pytorch.log_model(model, "model")
        if f1 > best_f1:
            best_f1 = f1
            best_run_id = mlflow.active_run().info.run_id

# ──────────── 7. Register Best Model ────────────
if best_run_id:
    uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(uri, "Transformer_Head_Finetuned")

print("\n✅ Transformer-based model training complete.")
