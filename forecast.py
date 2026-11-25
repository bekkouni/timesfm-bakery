"""Run the same TimesFM model with three different information sets."""
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from download_model import MODEL_DIR, MODEL_ID, REVISION
from generate_data import CONTEXT, DATA, HORIZON

RESULTS = DATA / "forecasts.json"
PAST_COLUMNS = ("roll", "baguette", "visitors")


def read_history():
    history = pd.read_csv(DATA / "history.csv", parse_dates=["date"])
    if len(history) != CONTEXT or not history.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("販売履歴には連続する256日分のデータが必要です。")
    return history


def signature():
    # Hash the input actually used by the model. A Windows checkout may use CRLF.
    history_text = (DATA / "history.csv").read_text(encoding="utf-8")
    return {"model": MODEL_ID, "revision": REVISION, "horizon": HORIZON,
            "recipe": 1, "history_sha256": hashlib.sha256(history_text.encode()).hexdigest()}


def read_results():
    bundle = json.loads(RESULTS.read_text(encoding="utf-8"))
    if bundle["input"] != signature() or set(bundle["C"]) != {str(i) for i in range(-1, HORIZON)}:
        raise ValueError("保存結果と現在のモデル入力が一致しません。precompute.py を再実行してください。")
    return bundle


def load_model(device):
    import torch
    from timesfm3 import TimesFM3Forecaster

    if not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError("モデルを準備するには download_model.py を実行してください。")
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return TimesFM3Forecaster.from_pretrained(str(MODEL_DIR), device=device,
                                              per_core_batch_size=1)


def predict(model, history, condition, sale_day=-1):
    if condition not in ("A", "B", "C") or not -1 <= sale_day < HORIZON:
        raise ValueError("Unknown condition or sale day")
    return predict_selected(model, history, () if condition == "A" else PAST_COLUMNS,
                            sale_day if condition == "C" else None)


def predict_selected(model, history, past_columns, sale_day=None):
    """Accept historical rows only. None omits the sale channel; -1 means no future sale."""
    inputs = {}
    if past_columns:
        inputs["past_only_covariates"] = history[list(past_columns)].to_numpy(np.float32).T
    if sale_day is not None:
        if not -1 <= sale_day < HORIZON:
            raise ValueError("Unknown sale day")
        future = np.zeros(HORIZON, dtype=np.float32)
        if sale_day >= 0:
            future[sale_day] = 1
        inputs["past_future_covariates"] = np.concatenate([history.sale.to_numpy(np.float32), future])[None, :]
    start = time.perf_counter()
    result = model.predict(history.loaf.to_numpy(np.float32)[None, :], horizon=HORIZON,
                           return_quantiles=True, **inputs)
    quantiles = np.asarray(result.quantiles).reshape(HORIZON, 9)
    if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Invalid quantile output")
    values = {name: quantiles[:, i].round(4).tolist() for name, i in (("p10", 0), ("p50", 4), ("p90", 8))}
    return values, time.perf_counter() - start


def runtime_info():
    return {pkg: version(pkg) for pkg in ("timesfm", "torch", "numpy")}
