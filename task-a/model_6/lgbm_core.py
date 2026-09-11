from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "LightGBM is required for this script. Install it with `python -m pip install lightgbm`."
    ) from exc

CUTOFF = pd.Timestamp("2026-02-18T14:00:00+00:00")
STEP = pd.Timedelta(minutes=15)
HORIZONS = (4, 96)
TARGETS = ("energy_wh", "cfp_g")
LAGS = (1, 4, 96)
EXTRA_LAGS = (2, 8, 16, 192, 672)
ROLL_3H = 12
EPS = 1e-6


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    out = numer / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["series_id", "bucket_15m"]).copy()
    ts = out["bucket_15m"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["tod_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * ts.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * ts.dt.dayofweek / 7.0)
    out["hour"] = ts.dt.hour.astype(np.int16)
    out["dayofweek"] = ts.dt.dayofweek.astype(np.int16)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)

    out["energy_per_core"] = _safe_div(out["energy_wh"], out["ncores"])
    out["cfp_per_core"] = _safe_div(out["cfp_g"], out["ncores"])
    out["work_per_core"] = _safe_div(out["work"], out["ncores"])

    grouped = out.groupby("series_id", sort=False, group_keys=False)
    for col in ["energy_wh", "cfp_g", "ncores"]:
        for lag in (*LAGS, *EXTRA_LAGS):
            out[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
            
    out["energy_roll_std_3h"] = grouped["energy_wh"].transform(
        lambda s: s.shift(1).rolling(ROLL_3H, min_periods=2).std()
    )
    out["energy_roll_mean_3h"] = grouped["energy_wh"].transform(
        lambda s: s.shift(1).rolling(ROLL_3H, min_periods=1).mean()
    )
    out["energy_roll_q95_24h"] = grouped["energy_wh"].transform(
        lambda s: s.shift(1).rolling(96, min_periods=8).quantile(0.95)
    )
    out["ncores_roll_mean_3h"] = grouped["ncores"].transform(
        lambda s: s.shift(1).rolling(ROLL_3H, min_periods=1).mean()
    )
    out["cfp_roll_mean_3h"] = grouped["cfp_g"].transform(
        lambda s: s.shift(1).rolling(ROLL_3H, min_periods=1).mean()
    )
    
    for col in ["energy_wh", "cfp_g", "ncores"]:
        shifted = grouped[col].shift(1)
        out[f"{col}_roll_mean_24h"] = shifted.groupby(out["series_id"]).transform(
            lambda s: s.rolling(96, min_periods=8).mean()
        )
        out[f"{col}_roll_std_24h"] = shifted.groupby(out["series_id"]).transform(
            lambda s: s.rolling(96, min_periods=8).std()
        )
        
    out["energy_diff_1_4"] = out["energy_wh_lag_1"] - out["energy_wh_lag_4"]
    out["energy_diff_4_96"] = out["energy_wh_lag_4"] - out["energy_wh_lag_96"]
    out["cfp_diff_1_4"] = out["cfp_g_lag_1"] - out["cfp_g_lag_4"]
    out["cfp_diff_4_96"] = out["cfp_g_lag_4"] - out["cfp_g_lag_96"]

    series_codes = {sid: idx for idx, sid in enumerate(sorted(out["series_id"].unique()))}
    out["series_code"] = out["series_id"].map(series_codes).astype("int32")
    return out


def add_direct_targets(features: pd.DataFrame, horizon: int, target: str) -> pd.DataFrame:
    frame = features.copy()
    future = frame.groupby("series_id", sort=False)[target].shift(-horizon)
    frame["target"] = future
    frame["forecast_timestamp_utc"] = frame["bucket_15m"] + horizon * STEP
    frame["horizon_steps_15m"] = horizon
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "series_id",
        "bucket_15m",
        "forecast_timestamp_utc",
        "horizon_steps_15m",
        "target",
    }
    return [col for col in frame.columns if col not in blocked]


def smooth_smape_objective(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_log = dataset.get_label()
    err = preds - y_log
    delta = 0.18
    scaled = err / delta
    asym = np.where(err < 0.0, 1.06, 0.94)
    grad = asym * err / np.sqrt(1.0 + scaled * scaled)
    hess = asym * np.power(1.0 + scaled * scaled, -1.5)
    hess = np.maximum(hess, 0.08)
    return grad, hess


def smape_eval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    y = np.expm1(dataset.get_label())
    pred = np.expm1(np.clip(preds, 0.0, None))
    value = np.mean(np.abs(pred - y) / np.maximum(EPS, (np.abs(y) + np.abs(pred)) * 0.5))
    return "smooth_smape", float(value), False


def train_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    cols: list[str],
    categorical: Iterable[str],
    seed: int,
) -> lgb.Booster:
    params = {
        "boosting_type": "gbdt",
        "metric": "None",
        "learning_rate": 0.025,
        "num_leaves": 47,
        "min_data_in_leaf": 24,
        "feature_fraction": 0.86,
        "bagging_fraction": 0.82,
        "bagging_freq": 3,
        "max_cat_threshold": 64,
        "lambda_l1": 0.05,
        "lambda_l2": 0.35,
        "verbosity": -1,
        "seed": seed,
        "feature_pre_filter": False,
    }
    dtrain = lgb.Dataset(
        train[cols],
        label=np.log1p(np.clip(train["target"].to_numpy(dtype=float), 0.0, None)),
        categorical_feature=list(categorical),
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        valid[cols],
        label=np.log1p(np.clip(valid["target"].to_numpy(dtype=float), 0.0, None)),
        categorical_feature=list(categorical),
        reference=dtrain,
        free_raw_data=False,
    )
    callbacks = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=0)]
    try:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2500,
            valid_sets=[dvalid],
            feval=smape_eval,
            fobj=smooth_smape_objective,
            callbacks=callbacks,
        )
    except TypeError:
        params = {**params, "objective": smooth_smape_objective}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2500,
            valid_sets=[dvalid],
            feval=smape_eval,
            callbacks=callbacks,
        )
    best_iter = max(50, int(booster.best_iteration or booster.current_iteration()))
    full = pd.concat([train, valid], ignore_index=True)
    dfull = lgb.Dataset(
        full[cols],
        label=np.log1p(np.clip(full["target"].to_numpy(dtype=float), 0.0, None)),
        categorical_feature=list(categorical),
        free_raw_data=False,
    )
    try:
        return lgb.train(
            {k: v for k, v in params.items() if k != "objective"},
            dfull,
            num_boost_round=best_iter,
            fobj=smooth_smape_objective,
            callbacks=[lgb.log_evaluation(period=0)],
        )
    except TypeError:
        return lgb.train(
            params,
            dfull,
            num_boost_round=best_iter,
            callbacks=[lgb.log_evaluation(period=0)],
        )


def seasonal_fallback(train_df: pd.DataFrame, target: str) -> tuple[dict, dict, float]:
    means = train_df.groupby("series_id")[target].mean().to_dict()
    global_mean = float(train_df[target].mean()) if len(train_df) else 0.0
    keyed: dict[tuple[str, int, int, int], float] = {}
    tmp = train_df.copy()
    tmp["dow"] = tmp["bucket_15m"].dt.dayofweek
    tmp["hour"] = tmp["bucket_15m"].dt.hour
    tmp["minute"] = tmp["bucket_15m"].dt.minute
    grouped = tmp.groupby(["series_id", "dow", "hour", "minute"])[target].median()
    for key, value in grouped.items():
        keyed[key] = float(value)
    return keyed, means, global_mean


def fallback_predict(row: pd.Series, seasonal: tuple[dict, dict, float]) -> float:
    keyed, means, global_mean = seasonal
    ts = row["forecast_timestamp_utc"]
    key = (row["series_id"], int(ts.dayofweek), int(ts.hour), int(ts.minute))
    return max(0.0, float(keyed.get(key, means.get(row["series_id"], global_mean))))


def add_causal_postprocess_columns(pred_shell: pd.DataFrame, train_base: pd.DataFrame) -> pd.DataFrame:
    ratio = train_base["cfp_g"] / train_base["energy_wh"].replace(0.0, np.nan)
    global_ratio = float(ratio.replace([np.inf, -np.inf], np.nan).median())
    if not np.isfinite(global_ratio):
        global_ratio = 0.0
    site_ratio = (
        train_base.assign(_ratio=ratio)
        .groupby("series_id")["_ratio"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_ratio)
        .to_dict()
    )
    out = pred_shell.copy()
    observed_ratio = out["cfp_g"] / out["energy_wh"].replace(0.0, np.nan)
    fallback = out["series_id"].map(site_ratio).fillna(global_ratio)
    out["_origin_cfp_energy_ratio"] = observed_ratio.replace([np.inf, -np.inf], np.nan).fillna(fallback)
    out["_origin_cfp_energy_ratio"] = out["_origin_cfp_energy_ratio"].clip(lower=0.0, upper=10.0)
    return out


def generate_forecasts(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    features = engineer_features(df)
    train_base = df[df["bucket_15m"] <= CUTOFF].copy()
    test_df = df[df["bucket_15m"] > CUTOFF].copy()
    test_truth = set(zip(test_df["series_id"], test_df["bucket_15m"]))
    test_origins = sorted(test_df["bucket_15m"].unique())
    series_ids = sorted(df["series_id"].unique())

    all_parts = []
    for horizon in HORIZONS:
        required = []
        for origin in test_origins:
            target_ts = pd.Timestamp(origin) + horizon * STEP
            for sid in series_ids:
                if (sid, target_ts) in test_truth:
                    required.append(
                        {
                            "series_id": sid,
                            "bucket_15m": pd.Timestamp(origin),
                            "forecast_timestamp_utc": target_ts,
                            "horizon_steps_15m": horizon,
                        }
                    )
        if not required:
            continue

        required_df = pd.DataFrame(required).sort_values(["series_id", "bucket_15m"])
        shells = []
        for sid, req_g in required_df.groupby("series_id", sort=False):
            hist = (
                features[features["series_id"] == sid]
                .drop(columns=["series_id"])
                .sort_values("bucket_15m")
            )
            merged = pd.merge_asof(
                req_g.sort_values("bucket_15m"),
                hist,
                on="bucket_15m",
                direction="backward",
            )
            shells.append(merged)
        pred_shell = pd.concat(shells, ignore_index=True)
        if pred_shell.empty:
            continue
        pred_shell = add_causal_postprocess_columns(pred_shell, train_base)

        part = pred_shell[["series_id", "forecast_timestamp_utc", "horizon_steps_15m"]].copy()
        for target in TARGETS:
            horizon_frame = add_direct_targets(features, horizon, target)
            train_rows = horizon_frame[
                (horizon_frame["bucket_15m"] <= CUTOFF)
                & (horizon_frame["forecast_timestamp_utc"] <= CUTOFF)
            ].copy()
            valid_start = CUTOFF - pd.Timedelta(days=10)
            valid_rows = train_rows[train_rows["bucket_15m"] > valid_start].copy()
            fit_rows = train_rows[train_rows["bucket_15m"] <= valid_start].copy()
            if len(fit_rows) < 200:
                fit_rows = train_rows.sample(frac=0.8, random_state=seed)
                valid_rows = train_rows.drop(fit_rows.index)

            cols = feature_columns(horizon_frame)
            fill_values = fit_rows[cols].median(numeric_only=True).to_dict()
            fit_rows[cols] = fit_rows[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_values).fillna(0.0)
            valid_rows[cols] = valid_rows[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_values).fillna(0.0)
            pred_x = pred_shell[cols].replace([np.inf, -np.inf], np.nan).fillna(fill_values).fillna(0.0)

            seasonal = seasonal_fallback(train_base, target)
            if len(fit_rows) >= 50 and len(valid_rows) >= 10:
                ensemble_preds = []
                for s in [seed, seed + 42, seed + 999]:
                    model = train_lgbm(fit_rows, valid_rows, cols, ["series_code"], s + horizon)
                    p = np.expm1(np.clip(model.predict(pred_x, num_iteration=model.best_iteration), 0.0, None))
                    ensemble_preds.append(p)
                
                pred = np.mean(ensemble_preds, axis=0)
                pred = np.clip(pred, 0.0, None)
                
                fallback = pred_shell.assign(forecast_timestamp_utc=part["forecast_timestamp_utc"]).apply(
                    fallback_predict, axis=1, seasonal=seasonal
                )
                pred = np.where(np.isfinite(pred), pred, fallback.to_numpy(dtype=float))
            else:
                pred = pred_shell.assign(forecast_timestamp_utc=part["forecast_timestamp_utc"]).apply(
                    fallback_predict, axis=1, seasonal=seasonal
                ).to_numpy(dtype=float)

            if target == "energy_wh" and horizon == 96:
                yesterday_target_energy = pred_shell["energy_wh"].to_numpy(dtype=float)
                pred = 0.65 * pred + 0.35 * yesterday_target_energy
            if target == "energy_wh" and horizon == 4:
                pred = 0.95 * pred + 0.05 * pred_shell["energy_wh"].to_numpy(dtype=float)
            if target == "cfp_g":
                ratio_pred = np.asarray(part.get("energy_wh_pred", pred), dtype=float) * pred_shell[
                    "_origin_cfp_energy_ratio"
                ].to_numpy(dtype=float)
                weight = 0.18 if horizon == 4 else 0.12
                pred = weight * pred + (1.0 - weight) * ratio_pred
                
            part[f"{target}_pred"] = np.clip(pred, 0.0, None)
        all_parts.append(part)

    forecast = pd.concat(all_parts, ignore_index=True)
    forecast = forecast.rename(columns={"energy_wh_pred": "energy_wh_pred", "cfp_g_pred": "cfp_g_pred"})
    forecast = forecast[
        ["series_id", "forecast_timestamp_utc", "horizon_steps_15m", "energy_wh_pred", "cfp_g_pred"]
    ].drop_duplicates(["series_id", "forecast_timestamp_utc", "horizon_steps_15m"])
    forecast["forecast_timestamp_utc"] = pd.to_datetime(forecast["forecast_timestamp_utc"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    forecast["forecast_timestamp_utc"] = forecast["forecast_timestamp_utc"].str.replace(
        r"(\+0000)$", "+00:00", regex=True
    )
    return forecast.sort_values(["forecast_timestamp_utc", "series_id", "horizon_steps_15m"])