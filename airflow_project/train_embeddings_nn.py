import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

# ────────── 1. Paths & Load ──────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
train_df  = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df    = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ────────── 2. Prepare text & labels ──────────
X_train = pd.concat([train_df["processed_headline"], val_df["processed_headline"]], ignore_index=True).fillna("")
y_train = pd.concat([train_df["category"],        val_df["category"]],        ignore_index=True)

X_test  = test_df["processed_headline"].fillna("")
y_test  = test_df["category"]

# Encode labels to integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
num_classes = len(le.classes_)

# ────────── 3. Tokenize & Pad ──────────
MAX_VOCAB_SIZE   = 10_000
MAX_SEQUENCE_LEN = 50

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
train_pad = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

test_seq  = tokenizer.texts_to_sequences(X_test)
test_pad  = pad_sequences(test_seq,  maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

# ────────── 4. Build Model Factory ──────────
def build_model(embedding_dim=64, dropout_rate=0.5):
    model = Sequential([
        Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_SEQUENCE_LEN),
        GlobalAveragePooling1D(),
        Dropout(dropout_rate),
        Dense(64, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model

# ────────── 5. MLflow Setup ──────────
mlflow.set_experiment("Text_Classification_Embeddings_NN")

# We'll run two variants and register the best by F1
best_f1 = 0.0
best_run_id = None

for emb_dim, dr in [(32, 0.3), (64, 0.5)]:
    run_name = f"Embedding{emb_dim}_Drop{dr}"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparams
        mlflow.log_param("max_vocab_size", MAX_VOCAB_SIZE)
        mlflow.log_param("max_seq_len",    MAX_SEQUENCE_LEN)
        mlflow.log_param("embedding_dim",  emb_dim)
        mlflow.log_param("dropout_rate",   dr)
        mlflow.log_param("batch_size",     64)
        mlflow.log_param("epochs",         10)

        model = build_model(embedding_dim=emb_dim, dropout_rate=dr)

        # Log TensorFlow model graph & signature
        mlflow.tensorflow.autolog(log_models=False)

        # Train
        t0 = time.time()
        history = model.fit(
            train_pad, y_train_enc,
            validation_split=0.1,
            batch_size=64,
            epochs=10,
            verbose=2
        )
        train_time = time.time() - t0
        mlflow.log_metric("training_time", train_time)

        # Predict & evaluate
        t1 = time.time()
        y_pred_probs = model.predict(test_pad, batch_size=64, verbose=0)
        infer_time = time.time() - t1
        mlflow.log_metric("inference_time", infer_time)

        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_test_enc, y_pred)
        f1  = f1_score(y_test_enc, y_pred, average="weighted")
        cm  = confusion_matrix(y_test_enc, y_pred)

        mlflow.log_metric("accuracy",   acc)
        mlflow.log_metric("f1_score",   f1)

        # Confusion matrix artifact
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.title(f"CM {run_name}")
        cm_path = f"cm_{run_name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Log the trained model
        mlflow.tensorflow.log_model(model, "model")

        # Track dataset size/version
        mlflow.log_param("train_rows", len(train_pad))
        mlflow.log_param("test_rows",  len(test_pad))

        # Update “best” and register if top
        if f1 > best_f1:
            best_f1    = f1
            best_run_id = mlflow.active_run().info.run_id

# ────────── 6. Model Registry ──────────
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "Embeddings_NN_Model")

print("✅ Embeddings NN training complete. ✅")
