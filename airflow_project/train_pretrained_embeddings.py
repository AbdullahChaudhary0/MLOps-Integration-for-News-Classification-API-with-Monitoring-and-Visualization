import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ────────── 1. Paths & Load ──────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# CSVs from your Airflow output
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# GloVe file (download e.g. from https://nlp.stanford.edu/data/glove.6B.zip and place in data/)
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.100d.txt")
EMBEDDING_DIM = 100

# ────────── 2. Prepare text & labels ──────────
X_train = pd.concat([
    train_df["processed_headline"],
    val_df["processed_headline"]
], ignore_index=True).fillna("")
y_train = pd.concat([
    train_df["category"],
    val_df["category"]
], ignore_index=True)

X_test  = test_df["processed_headline"].fillna("")
y_test  = test_df["category"]

# ────────── 3. Load GloVe Embeddings ──────────
print("Loading GloVe embeddings…")
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vec = np.asarray(parts[1:], dtype="float32")
        embeddings_index[word] = vec
print(f"Loaded {len(embeddings_index)} word vectors.")

# ────────── 4. Text → Average Embedding ──────────
def avg_embedding(text):
    tokens = text.split()
    vectors = [embeddings_index[w] for w in tokens if w in embeddings_index]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(EMBEDDING_DIM, dtype="float32")

# Transform all texts
X_train_emb = np.vstack([avg_embedding(t) for t in X_train])
X_test_emb  = np.vstack([avg_embedding(t) for t in X_test])

# ────────── 5. MLflow + Model Training ──────────
mlflow.set_experiment("Text_Classification_Pretrained_Embeddings")

with mlflow.start_run(run_name="LogisticRegression_GloVe100"):
    # Log data stats
    mlflow.log_param("train_rows", X_train_emb.shape[0])
    mlflow.log_param("test_rows",  X_test_emb.shape[0])
    mlflow.log_param("embedding_dim", EMBEDDING_DIM)

    # Build & train classifier
    clf = LogisticRegression(max_iter=200, random_state=42, n_jobs=-1)
    t0 = time.time()
    clf.fit(X_train_emb, y_train)
    train_time = time.time() - t0
    mlflow.log_metric("training_time", train_time)

    # Inference
    t1 = time.time()
    y_pred = clf.predict(X_test_emb)
    infer_time = time.time() - t1
    mlflow.log_metric("inference_time", infer_time)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title("Confusion Matrix - GloVe + LR")
    cm_path = "cm_glove_lr.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Log & register model
    mlflow.sklearn.log_model(clf, "model")
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model",
                           "Pretrained_Embeddings_Model")

print("✅ Pre-trained embeddings + LR model trained and registered.")
