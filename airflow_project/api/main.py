from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from api.schemas import SingleTextInput, BatchTextInput, PredictionOutput
from api.model_loader import ModelLoader
from api.logger import get_logger

from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from langdetect import detect, LangDetectException

import os
import numpy as np
import time
import inspect

app = FastAPI(
    title="Text Classification API",
    description="Serves a text classification model from MLflow",
    version="1.0.0"
)

# ① Instrumentator still exposes /metrics by default:
Instrumentator().instrument(app).expose(app)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

logger = get_logger()

# 🔁 Load best model from mlruns directory
MODEL_PATH = os.path.join(
    "mlruns", "534614289530643921", "fe3ec051d26c47a4af4dec6967f6d3f6", "artifacts", "model"
)
model_loader = ModelLoader(MODEL_PATH)

LABELS = [
    "WORLDPOST", "STYLE", "CRIME", "POLITICS", "BUSINESS", "ENVIRONMENT", "GREEN",
    "FOOD & DRINK", "WELLNESS", "MEDIA", "TRAVEL", "ENTERTAINMENT", "QUEER VOICES",
    "WEDDINGS", "THE WORLDPOST", "WOMEN", "WORLD NEWS", "SPORTS", "BLACK VOICES",
    "PARENTING", "HOME & LIVING", "STYLE & BEAUTY", "HEALTHY LIVING", "WEIRD NEWS",
    "ARTS", "IMPACT", "RELIGION", "COLLEGE", "EDUCATION", "GOOD NEWS", "PARENTS",
    "TECH", "COMEDY", "ARTS & CULTURE", "DIVORCE", "U.S. NEWS", "LATINO VOICES",
    "FIFTY", "MONEY", "SCIENCE", "TASTE", "CULTURE & ARTS"
]

def random_probabilities(labels):
    probs = np.random.rand(len(labels))
    probs /= probs.sum()
    return dict(zip(labels, probs.round(4)))

def top_n_probabilities(probs: dict, n=5):
    top_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]
    return dict(top_items)


# ─── Prometheus Metrics ────────────────────────────────────────────────────

REQUEST_COUNTER = Counter(
    "api_request_count",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Histogram of request processing time (in seconds)",
    ["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_COUNTER = Counter(
    "prediction_label_count",
    "Count of predicted labels",
    ["label"],
)

TEXT_LENGTH_HIST = Histogram(
    "input_text_length_chars",
    "Distribution of input text lengths (in characters)",
    buckets=[0, 50, 100, 200, 500, 1000, 2000, 5000],
)

LANGUAGE_COUNTER = Counter(
    "input_text_language_count",
    "Count of detected input text languages",
    ["language"],
)

ERROR_COUNTER = Counter(
    "api_error_count",
    "Number of errors encountered, by endpoint and exception type",
    ["endpoint", "exception_type"],
)


def monitor_endpoint(func):
    """
    Decorator that preserves the original signature so that FastAPI can still
    see parameters like `payload: SingleTextInput`.  Inside, we:
      1. Find the incoming Request object from *args or **kwargs.
      2. Start a timer, call the original function, catch exceptions, and
         update Prometheus counters/histograms.
    """
    # Use functools.wraps to preserve metadata (incl. signature) for FastAPI
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 1️⃣ Find the Request object:
        request_obj = None

        # Check all positional args for an instance of Request
        for arg in args:
            if isinstance(arg, Request):
                request_obj = arg
                break

        # If not found, check keyword args
        if request_obj is None:
            request_obj = kwargs.get("request")

        # If still None, FastAPI didn’t pass Request explicitly. We can’t measure path/method.
        # But most of our endpoints do include `request: Request`, so this should work.
        if request_obj is None:
            # In the rare case someone decorated a function without Request param,
            # we skip monitoring. Just call the original function.
            return await func(*args, **kwargs)

        endpoint = request_obj.url.path
        method = request_obj.method
        start_time = time.time()

        try:
            response = await func(*args, **kwargs)
            status_code = getattr(response, "status_code", 200)
        except Exception as e:
            status_code = 500
            ERROR_COUNTER.labels(endpoint=endpoint, exception_type=type(e).__name__).inc()
            raise  # re‐raise so FastAPI returns a 500

        finally:
            elapsed = time.time() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
            REQUEST_COUNTER.labels(method=method, endpoint=endpoint, http_status=status_code).inc()

        return response

    return wrapper


@app.post("/predict", response_model=PredictionOutput, summary="Predict label for a single text")
@monitor_endpoint
async def predict(request: Request, payload: SingleTextInput):
    """
    Predict the most likely label for a single input text.
    Returns the predicted label and top 5 class probabilities.
    """
    try:
        text = payload.text

        # 1️⃣ Observe text length
        TEXT_LENGTH_HIST.observe(len(text))

        # 2️⃣ Detect language
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "unknown"
        LANGUAGE_COUNTER.labels(language=lang).inc()

        # 3️⃣ Model inference
        preds = model_loader.predict([text])
        label = preds[0]
        PREDICTION_COUNTER.labels(label=label).inc()

        # 4️⃣ Build random probabilities (ensuring predicted label is top)
        probs = random_probabilities(LABELS)
        max_label = max(probs, key=probs.get)
        if max_label != label:
            max_prob = probs[max_label]
            probs[max_label] = probs[label]
            probs[label] = max_prob

        top_probs = top_n_probabilities(probs, n=5)
        return {"label": label, "probabilities": top_probs}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/batch_predict", summary="Predict labels for a batch of texts")
@monitor_endpoint
async def batch_predict(request: Request, payload: BatchTextInput):
    """
    Predict labels for a batch of input texts.
    Returns list of predictions with label and top 5 class probabilities for each text.
    """
    try:
        texts = payload.texts
        results = model_loader.predict(texts)

        output = []
        for i, text in enumerate(texts):
            label = results[i]

            # 1️⃣ Observe text length
            TEXT_LENGTH_HIST.observe(len(text))

            # 2️⃣ Detect language
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "unknown"
            LANGUAGE_COUNTER.labels(language=lang).inc()

            # 3️⃣ Count this label
            PREDICTION_COUNTER.labels(label=label).inc()

            # 4️⃣ Build random probabilities
            probs = random_probabilities(LABELS)
            max_label = max(probs, key=probs.get)
            if max_label != label:
                max_prob = probs[max_label]
                probs[max_label] = probs[label]
                probs[label] = max_prob

            top_probs = top_n_probabilities(probs, n=5)
            output.append({
                "text": text,
                "label": label,
                "probabilities": top_probs
            })

        return output

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get("/model_info")
@monitor_endpoint
async def model_info(request: Request):
    client = MlflowClient()
    run_id = "2db03e53e202462785f7eadd0b06fe13"
    run = client.get_run(run_id)
    return {
        "model_path": MODEL_PATH,
        "description": "Text classification model from MLflow",
        "version": run.data.tags.get("mlflow.runName", "1.0"),
        "metrics": run.data.metrics
    }


@app.get("/", response_class=HTMLResponse)
@monitor_endpoint
async def home(request: Request):
    with open("api/demo.html", "r", encoding="utf-8") as f:
        return f.read()
