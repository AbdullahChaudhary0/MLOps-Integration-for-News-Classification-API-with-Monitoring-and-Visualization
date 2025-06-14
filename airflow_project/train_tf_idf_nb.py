import pandas as pd
import time
import os
import json
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

train_path = os.path.join(DATA_DIR, "train.csv")
val_path   = os.path.join(DATA_DIR, "val.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")

# Load data
train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

X_train = pd.concat([
    train_df["processed_headline"],
    val_df["processed_headline"]
], ignore_index=True)
y_train = pd.concat([
    train_df["category"],
    val_df["category"]
], ignore_index=True)

X_test = test_df["processed_headline"]
y_test = test_df["category"]

# DROP ROWS WITH MISSING TEXT (in sync with labels)
train_all = pd.DataFrame({"text": X_train, "label": y_train}) \
              .dropna(subset=["text"])
X_train, y_train = train_all["text"], train_all["label"]

test_all = pd.DataFrame({"text": X_test, "label": y_test}) \
              .dropna(subset=["text"])
X_test, y_test = test_all["text"], test_all["label"]

# Define models
models = {
    "NaiveBayes": MultinomialNB(),
    "LinearSVM": LinearSVC()
}

# Start MLflow experiment
mlflow.set_experiment("Text_Classification_TFIDF_Traditional")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", model)
        ])

        start_train = time.time()
        pipeline.fit(X_train, y_train)
        end_train = time.time()

        start_pred = time.time()
        y_pred = pipeline.predict(X_test)
        end_pred = time.time()

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tfidf_max_features", 5000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", end_train - start_train)
        mlflow.log_metric("inference_time", end_pred - start_pred)

        # Save confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        cm_path = f"conf_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # Save and log labels.json and register model only for LinearSVM
        if model_name == "LinearSVM":
            labels = list(y_train.unique())
            with open("labels.json", "w") as f:
                json.dump(labels, f)
            mlflow.log_artifact("labels.json")

            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", f"{model_name}_Model")

print("✅ Training complete. Check MLflow UI for results.")
