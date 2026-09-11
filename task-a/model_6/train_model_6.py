from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

TASK_A_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_A_ROOT.parent
SRC = TASK_A_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task_a.dataio import load_series_csv, split_temporal
from task_a.evaluation import (
    compose_task_a_score,
    load_detections,
    load_forecasts,
    load_peaks,
    score_detection,
    score_forecasts,
    score_peaks,
)
from task_a.labels import peak_label, peak_threshold, valid_signal_label
from task_a.submission import validate_detection_csv, validate_forecast_csv, validate_peak_csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgbm_core import generate_forecasts as generate_lgbm_core_forecasts

CUTOFF = "2026-02-18T14:00:00+00:00"
PRED_COLUMNS = ["energy_wh_pred", "cfp_g_pred"]

def format_utc(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(series, utc=True)
        .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        .str.replace(r"(\+0000)$", "+00:00", regex=True)
    )

def load_site_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "series_id" not in df.columns:
        df = df.rename(columns={"site_id": "series_id"})
    df["series_id"] = df["series_id"].astype(str)
    df["bucket_15m"] = pd.to_datetime(df["bucket_15m"], utc=True)
    for col in ["records", "energy_wh", "cfp_g", "work", "ncores"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return (
        df.groupby(["series_id", "bucket_15m"], as_index=False)[
            ["records", "energy_wh", "cfp_g", "work", "ncores"]
        ]
        .sum()
        .sort_values(["series_id", "bucket_15m"])
        .reset_index(drop=True)
    )

def build_detection(site_frame: pd.DataFrame) -> pd.DataFrame:
    train = site_frame[site_frame["bucket_15m"] <= pd.Timestamp(CUTOFF)].copy()
    test = site_frame.loc[site_frame["bucket_15m"] > pd.Timestamp(CUTOFF), ["series_id", "bucket_15m"]].copy()
    add_detection_time_features(train)
    add_detection_time_features(test)

    train["_valid"] = train.apply(valid_signal_label, axis=1).astype(int)
    global_rate = float(train["_valid"].mean()) if len(train) else 1.0
    site_rate = train.groupby("series_id")["_valid"].mean()
    qod_rate = train.groupby("qod")["_valid"].mean()

    scores = []
    for row in test[["series_id", "qod"]].itertuples(index=False):
        site = str(row.series_id)
        qod = int(row.qod)
        site_component = site_rate.get(site, global_rate)
        qod_component = qod_rate.get(qod, global_rate)
        scores.append(float((site_component + (8.0 * qod_component) + global_rate) / 10.0))
    score = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

    out = test[["series_id", "bucket_15m"]].copy()
    out["valid_signal_score"] = score.astype(float)
    out["valid_signal_pred"] = (score >= 0.5).astype(int)
    out["bucket_15m"] = format_utc(out["bucket_15m"])
    return out.sort_values(["bucket_15m", "series_id"])

def add_detection_time_features(frame: pd.DataFrame) -> None:
    ts = pd.to_datetime(frame["bucket_15m"], utc=True)
    frame["qod"] = ((ts.dt.hour * 4) + (ts.dt.minute // 15)).astype(np.int16)

def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (scores >= threshold).astype(int)
        tp = int(((labels == 1) & (pred == 1)).sum())
        fp = int(((labels == 0) & (pred == 1)).sum())
        fn = int(((labels == 1) & (pred == 0)).sum())
        denom = (2 * tp) + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold

def calibrate_peak_threshold(train_rows: list, thresholds: dict[str, float]) -> float:
    frame = pd.DataFrame(
        {
            "series_id": [row.series_id for row in train_rows],
            "bucket_15m": [row.bucket_15m for row in train_rows],
            "energy_wh": [row.energy_wh for row in train_rows],
            "cfp_g": [row.cfp_g for row in train_rows],
        }
    ).sort_values(["series_id", "bucket_15m"])
    frame["_actual_signal"] = frame[["energy_wh", "cfp_g"]].max(axis=1)
    grouped = frame.groupby("series_id", sort=False)
    lag4 = grouped["_actual_signal"].shift(4)
    lag96 = grouped["_actual_signal"].shift(96)
    pred_signal = pd.concat([lag4, lag96], axis=1).mean(axis=1).fillna(lag4).fillna(0.0)
    denom = frame["series_id"].map(thresholds).fillna(0.0).clip(lower=1e-9)
    ratio = pred_signal.to_numpy(dtype=float) / denom.to_numpy(dtype=float)
    scores = np.clip(ratio / (1.0 + ratio), 0.0, 1.0)
    labels = np.array([peak_label(row, thresholds) for row in train_rows], dtype=int)
    return best_f1_threshold(scores, labels)

def build_peaks(train_rows: list, test_rows: list, forecast: pd.DataFrame) -> pd.DataFrame:
    thresholds = peak_threshold(train_rows, 0.95)
    
    raw_cutoff = calibrate_peak_threshold(train_rows, thresholds)
    cutoff = max(0.01, min(raw_cutoff - 0.05, 0.20))
    
    forecast_frame = forecast.copy()
    forecast_frame["forecast_timestamp_utc"] = pd.to_datetime(
        forecast_frame["forecast_timestamp_utc"], utc=True
    )
    forecast_frame["_signal"] = forecast_frame[PRED_COLUMNS].max(axis=1)
    pred_signal = forecast_frame.groupby(["series_id", "forecast_timestamp_utc"])["_signal"].mean().to_dict()

    rows = []
    for row in test_rows:
        threshold = max(1e-9, float(thresholds.get(row.series_id, 0.0)))
        signal = max(0.0, float(pred_signal.get((row.series_id, pd.Timestamp(row.bucket_15m)), 0.0)))
        ratio = signal / threshold
        score = float(np.clip(ratio / (1.0 + ratio), 0.0, 1.0))
        rows.append(
            {
                "series_id": row.series_id,
                "bucket_15m": row.bucket_15m,
                "peak_score": score,
                "peak_pred": int(score >= cutoff),
            }
        )
    out = pd.DataFrame(rows)
    out["bucket_15m"] = format_utc(out["bucket_15m"])
    return out.sort_values(["bucket_15m", "series_id"])

def evaluate_outputs(input_path: Path, output_dir: Path) -> dict[str, float]:
    rows = load_series_csv(input_path)
    train, test = split_temporal(rows, CUTOFF)
    forecast_path = output_dir / "forecast_submission.csv"
    detection_path = output_dir / "detection_submission.csv"
    peak_path = output_dir / "peak_submission.csv"
    validate_forecast_csv(forecast_path)
    validate_detection_csv(detection_path)
    validate_peak_csv(peak_path)
    parts = {}
    parts.update(score_forecasts(test, load_forecasts(forecast_path), require_complete=True))
    parts.update(score_detection(test, load_detections(detection_path)))
    parts.update(score_peaks(train, test, predictions=load_peaks(peak_path)))
    parts["ScoreTA"] = compose_task_a_score(parts)
    return parts

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-controlled Task A model_6 outputs.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/raw_metrics/summary_sites_15m.csv")
    parser.add_argument("--output-dir", type=Path, default=TASK_A_ROOT / "model_6/outputs")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_series_csv(args.input)
    train_rows, test_rows = split_temporal(rows, CUTOFF)
    site_frame = load_site_frame(args.input)

    forecast = generate_lgbm_core_forecasts(site_frame, args.seed)
    detection = build_detection(site_frame)
    peaks = build_peaks(train_rows, test_rows, forecast)
    
    forecast = forecast.sort_values(["series_id", "forecast_timestamp_utc"])
    detection = detection.sort_values(["series_id", "bucket_15m"])
    peaks = peaks.sort_values(["series_id", "bucket_15m"])

    forecast["forecast_timestamp_utc"] = pd.to_datetime(forecast["forecast_timestamp_utc"]).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    detection["bucket_15m"] = pd.to_datetime(detection["bucket_15m"]).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    peaks["bucket_15m"] = pd.to_datetime(peaks["bucket_15m"]).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')

    forecast.to_csv(args.output_dir / "forecast_submission.csv", index=False)
    detection.to_csv(args.output_dir / "detection_submission.csv", index=False)
    peaks.to_csv(args.output_dir / "peak_submission.csv", index=False)

    metrics = evaluate_outputs(args.input, args.output_dir)
    with (args.output_dir / "metrics_model_6.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote model_6 outputs to {args.output_dir}")

if __name__ == "__main__":
    main()