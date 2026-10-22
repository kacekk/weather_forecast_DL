"""Data preparation, model building and training for the next-day temperature project.

Used by both `experiments.py` (systematic search) and `weather_forecast.ipynb` (the report),
so the exact same pipeline runs in both places.
"""
import hashlib
import os
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    keras.utils.set_random_seed(seed)               # Python, NumPy, TensorFlow
    tf.config.experimental.enable_op_determinism()  # deterministic ops


# ---------------------------------------------------------------- cleaning

RENAME = {
    "Temperature (C)": "temp", "Humidity": "humidity", "Wind Speed (km/h)": "wind_speed",
    "Wind Bearing (degrees)": "wind_bearing", "Visibility (km)": "visibility",
    "Pressure (millibars)": "pressure", "Precip Type": "precip_type",
    "Summary": "summary", "Daily Summary": "daily_summary",
}


def load_clean(cfg, verbose=False):
    """Read the CSV, check its hash and fix the data issues found during exploration."""
    path = Path(cfg["data"]["path"])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == cfg["data"]["sha256"], f"Dataset changed! hash = {sha}"
    df = pd.read_csv(path)
    log = {}
    # Mixed +01:00/+02:00 offsets (daylight saving time): parse as UTC, then convert to local time.
    df["time"] = pd.to_datetime(df["Formatted Date"], utc=True).dt.tz_convert(cfg["data"]["timezone"])
    log["sorted_in_file"] = bool(df["time"].is_monotonic_increasing)
    df = df.sort_values("time")
    log["duplicate_timestamps"] = int(df["time"].duplicated().sum())
    df = df.drop_duplicates(subset="time").set_index("time")
    log["loud_cover_unique"] = df["Loud Cover"].unique().tolist()
    log["pressure_zero_rows"] = int((df["Pressure (millibars)"] == 0).sum())
    log["humidity_zero_rows"] = int((df["Humidity"] == 0).sum())
    for col in ["Pressure (millibars)", "Humidity"]:   # 0 mbar / 0 % humidity = sensor fault
        df[col] = df[col].replace(0, np.nan).interpolate(method="time")
    # The forbidden column is dropped immediately so it can't leak anywhere.
    df = df.drop(columns=["Apparent Temperature (C)", "Loud Cover", "Formatted Date"]).rename(columns=RENAME)
    if verbose:
        for k, v in log.items():
            print(f"{k}: {v}")
    return df


# ---------------------------------------------------------------- features

CLOUD = {"Partly Cloudy": 1, "Mostly Cloudy": 2, "Overcast": 3, "Clear": 0}


def _cloudiness(s):
    for name, v in CLOUD.items():   # "Partly/Mostly Cloudy" checked before plain words
        if name in s:
            return v
    return np.nan


def hourly_features(df):
    """Numeric hourly features (used both for daily aggregates and the hourly model input)."""
    h = pd.DataFrame(index=df.index)
    h["temp"] = df["temp"]
    h["humidity"] = df["humidity"]
    # Dew point (Magnus formula): the temperature at which air becomes saturated.
    # Combines temperature and humidity into the amount of moisture in the air.
    a, b = 17.62, 243.12
    gamma = np.log(df["humidity"].clip(lower=0.01)) + a * df["temp"] / (b + df["temp"])
    h["dew_point"] = b * gamma / (a - gamma)
    h["visibility"] = df["visibility"]
    h["pressure"] = df["pressure"]
    rad = np.deg2rad(df["wind_bearing"])        # angle -> vector, so 359° and 1° are close
    h["wind_speed"] = df["wind_speed"]
    h["wind_u"] = df["wind_speed"] * np.sin(rad)
    h["wind_v"] = df["wind_speed"] * np.cos(rad)
    h["cloudiness"] = df["summary"].map(_cloudiness)
    h["cloudiness"] = h["cloudiness"].fillna(h["cloudiness"].rolling(6, min_periods=1).mean()).fillna(1.5)
    h["fog"] = df["summary"].str.contains("Fog").astype(float)
    h["windy"] = df["summary"].str.contains("Breezy|Windy").astype(float)
    h["rain"] = df["summary"].str.contains("Rain|Drizzle").astype(float)
    return h


def daily_table(df, h):
    """One row per local calendar day."""
    g = h.resample("D")
    d = pd.DataFrame({
        "temp_mean": g["temp"].mean(), "temp_min": g["temp"].min(), "temp_max": g["temp"].max(),
        "temp_last": g["temp"].last(),
        # temperature change over the last 6 hours of the day (evening trend)
        "temp_trend_6h": g["temp"].last() - h["temp"].resample("D").apply(lambda s: s.iloc[-7] if len(s) >= 7 else s.iloc[0]),
        "humidity": g["humidity"].mean(), "dew_point": g["dew_point"].mean(),
        "visibility": g["visibility"].mean(),
        "pressure": g["pressure"].mean(), "pressure_change": g["pressure"].last() - g["pressure"].first(),
        "wind_speed": g["wind_speed"].mean(), "wind_u": g["wind_u"].mean(), "wind_v": g["wind_v"].mean(),
        "cloudiness": g["cloudiness"].mean(), "fog": g["fog"].mean(), "windy": g["windy"].mean(),
        "rain": g["rain"].mean(),
    })
    d["daily_summary"] = df["daily_summary"].resample("D").first()
    d.index = d.index.tz_localize(None).normalize()
    num = d.columns.drop("daily_summary")
    d[num] = d[num].interpolate()
    d["daily_summary"] = d["daily_summary"].ffill()
    return d


FEATURE_GROUPS = {
    "temperature": ["temp_mean", "temp_min", "temp_max", "temp_last", "temp_trend_6h"],
    "moisture": ["humidity", "dew_point"],
    "visibility": ["visibility"],
    "pressure": ["pressure", "pressure_change"],
    "wind": ["wind_speed", "wind_u", "wind_v"],
    "sky": ["cloudiness", "fog", "windy", "rain"],
}
HOURLY_COLS = ["temp", "humidity", "dew_point", "visibility", "pressure",
               "wind_speed", "wind_u", "wind_v", "cloudiness", "fog", "windy", "rain"]


# ---------------------------------------------------------------- dataset

class Dataset:
    """All model inputs for every target day, plus the time-based split.

    Sample for target day t: days t-3..t-1 (daily rows, 72 hourly rows, 3 daily summaries)
    + day-of-year of t  ->  mean temperature of day t.
    """

    def __init__(self, cfg, verbose=False):
        self.cfg = cfg
        df = load_clean(cfg, verbose=verbose)
        self.hourly_raw = df
        h = hourly_features(df)
        self.daily = daily_table(df, h)
        W = cfg["data"]["window_days"]

        # Regular UTC hourly grid (no DST jumps); fill the few missing hours by interpolation.
        hu = h.tz_convert("UTC")
        hu = hu.reindex(pd.date_range(hu.index.min(), hu.index.max(), freq="h")).interpolate()
        hour_local = hu.index.tz_convert(cfg["data"]["timezone"]).hour
        hu["hour_sin"] = np.sin(2 * np.pi * hour_local / 24)
        hu["hour_cos"] = np.cos(2 * np.pi * hour_local / 24)
        self.hourly_cols = HOURLY_COLS + ["hour_sin", "hour_cos"]
        hv = hu[self.hourly_cols].to_numpy("float32")
        h_index = hu.index

        days = self.daily.index
        self.dates = days[W:]
        all_feats = [c for g in FEATURE_GROUPS.values() for c in g]
        dv = self.daily[all_feats].to_numpy("float32")
        self.daily_cols = all_feats
        self.X_daily = np.stack([dv[t - W:t] for t in range(W, len(days))])

        # 72 hours before local midnight of the target day
        midnights = (self.dates.tz_localize(cfg["data"]["timezone"]).tz_convert("UTC"))
        pos = h_index.get_indexer(midnights)
        assert (pos >= W * 24).all() and (pos >= 0).all()
        self.X_hourly = np.stack([hv[p - W * 24:p] for p in pos])

        doy = self.dates.dayofyear.to_numpy()
        self.X_cal = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1).astype("float32")
        texts = self.daily["daily_summary"].to_numpy()
        self.texts = np.array([" | ".join(texts[t - W:t]) for t in range(W, len(days))])
        self.y = self.daily["temp_mean"].to_numpy("float32")[W:]
        self.last_mean = self.X_daily[:, -1, all_feats.index("temp_mean")]

    def mask(self, part, years=None):
        d, c = self.dates, self.cfg["data"]
        if part == "train":
            return d <= c["train_end"]
        if part == "val":
            return (d > c["train_end"]) & (d <= c["val_end"])
        if part == "test":
            return d > c["val_end"]
        if part == "trainval":
            return d <= c["val_end"]
        if part == "years":
            return np.isin(d.year, years)
        raise ValueError(part)


# ---------------------------------------------------------------- inputs per experiment

class Prepared:
    """Scales inputs/target with statistics from the training rows only."""

    def __init__(self, ds, train_mask, p):
        self.ds, self.p = ds, p
        cols = [c for g, cs in FEATURE_GROUPS.items() if g not in p.get("drop_groups", []) for c in cs]
        self.col_idx = [ds.daily_cols.index(c) for c in cols]
        hcols = [c for c in ds.hourly_cols if not _hourly_in_dropped(c, p.get("drop_groups", []))]
        self.hcol_idx = [ds.hourly_cols.index(c) for c in hcols]
        Xd = ds.X_daily[train_mask][:, :, self.col_idx]
        Xh = ds.X_hourly[train_mask][:, :, self.hcol_idx]
        self.mu_d, self.sd_d = Xd.reshape(-1, Xd.shape[-1]).mean(0), Xd.reshape(-1, Xd.shape[-1]).std(0) + 1e-6
        self.mu_h, self.sd_h = Xh.reshape(-1, Xh.shape[-1]).mean(0), Xh.reshape(-1, Xh.shape[-1]).std(0) + 1e-6
        t = self._raw_target(train_mask)
        self.mu_y, self.sd_y = float(t.mean()), float(t.std())   # target scaling (Class 2, regression)
        if p.get("text"):
            self.vec = keras.layers.TextVectorization(max_tokens=300, output_mode="multi_hot")
            self.vec.adapt(ds.texts[train_mask])

    def _raw_target(self, m):
        y = self.ds.y[m]
        return y - self.ds.last_mean[m] if self.p["target"] == "delta" else y

    def inputs(self, m):
        ds, p, x = self.ds, self.p, {}
        if p["input"] in ("daily", "both"):
            x["daily"] = (ds.X_daily[m][:, :, self.col_idx] - self.mu_d) / self.sd_d
        if p["input"] in ("hourly", "both"):
            x["hourly"] = (ds.X_hourly[m][:, :, self.hcol_idx] - self.mu_h) / self.sd_h
        if p.get("calendar", True):
            x["cal"] = ds.X_cal[m]
        if p.get("text"):
            x["text"] = self.vec(ds.texts[m]).numpy().astype("float32")
        return x

    def target(self, m):
        return ((self._raw_target(m) - self.mu_y) / self.sd_y).astype("float32")

    def to_celsius(self, pred_scaled, m):
        pred = pred_scaled.ravel() * self.sd_y + self.mu_y
        return pred + self.ds.last_mean[m] if self.p["target"] == "delta" else pred


def _hourly_in_dropped(col, groups):
    owner = {"temp": "temperature", "humidity": "moisture", "dew_point": "moisture", "visibility": "visibility",
             "pressure": "pressure", "wind_speed": "wind", "wind_u": "wind", "wind_v": "wind",
             "cloudiness": "sky", "fog": "sky", "windy": "sky", "rain": "sky"}
    return owner.get(col) in groups


# ---------------------------------------------------------------- model

def build_model(p, shapes, seed):
    """Functional-API model (Class 3) with up to four inputs."""
    L = keras.layers
    init = lambda: keras.initializers.GlorotUniform(seed=seed)
    reg = keras.regularizers.l2(p["l2"]) if p.get("l2") else None
    ins, parts = [], []

    def seq_branch(name, arch, units):
        i = keras.Input(shape=shapes[name], name=name); ins.append(i)
        if arch == "flatten":
            return L.Flatten()(i)
        if arch == "conv1d":
            x = L.Conv1D(units, 5, activation="relu", padding="same", kernel_initializer=init())(i)
            x = L.MaxPooling1D(3)(x)
            x = L.Conv1D(units, 3, activation="relu", padding="same", kernel_initializer=init())(x)
            return L.GlobalAveragePooling1D()(x)
        rnn = {"lstm": L.LSTM, "gru": L.GRU}[arch]
        layer = rnn(units, kernel_initializer=init(), seed=seed, dropout=p.get("rnn_dropout", 0.0))
        return (L.Bidirectional(layer) if p.get("bidirectional") else layer)(i)

    if p["input"] in ("daily", "both"):
        parts.append(seq_branch("daily", p["daily_arch"], p["units"]))
    if p["input"] in ("hourly", "both"):
        parts.append(seq_branch("hourly", p["hourly_arch"], p["units"]))
    for name in ("cal", "text"):
        if name in shapes:
            i = keras.Input(shape=shapes[name], name=name); ins.append(i)
            parts.append(L.Dense(16, activation="relu", kernel_initializer=init())(i) if name == "text" else i)
    x = parts[0] if len(parts) == 1 else L.Concatenate()(parts)
    for u in p["dense"]:
        x = L.Dense(u, activation="relu", kernel_initializer=init(), kernel_regularizer=reg)(x)
        if p.get("dropout"):
            x = L.Dropout(p["dropout"], seed=seed)(x)
    out = L.Dense(1, kernel_initializer=init())(x)   # regression: no activation
    return keras.Model(ins, out)


def train_eval(ds, p, seed, train_m, val_m=None, epochs=None, log_csv=None, return_model=False):
    """Train one model. With val_m: early stopping on it and MAE (°C) returned.
    Without val_m: train for a fixed number of epochs (final retraining on train+val)."""
    set_seed(seed)
    prep = Prepared(ds, train_m, p)
    xtr, ytr = prep.inputs(train_m), prep.target(train_m)
    shapes = {k: v.shape[1:] for k, v in xtr.items()}
    model = build_model(p, shapes, seed)
    loss = keras.losses.Huber(delta=1.0) if p.get("loss") == "huber" else "mse"
    model.compile(optimizer=keras.optimizers.Adam(p["lr"]), loss=loss, metrics=["mae"])
    cbs = []
    if log_csv:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        cbs.append(keras.callbacks.CSVLogger(str(log_csv)))
    fit_kw = dict(epochs=epochs or p["epochs"], batch_size=p["batch_size"], verbose=0, shuffle=True)
    if val_m is not None:
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=p["patience"], restore_best_weights=True))
        if p.get("reduce_lr"):
            cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, p["patience"] // 3)))
        fit_kw["validation_data"] = (prep.inputs(val_m), prep.target(val_m))
    elif p.get("reduce_lr"):   # final retraining has no validation set: follow training loss instead
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=max(2, p["patience"] // 3)))
    hist = model.fit(xtr, ytr, callbacks=cbs, **fit_kw)
    res = {"epochs_run": len(hist.history["loss"])}
    if val_m is not None:
        res["best_epoch"] = int(np.argmin(hist.history["val_loss"]) + 1)
        pred = prep.to_celsius(model.predict(prep.inputs(val_m), verbose=0, batch_size=1024), val_m)
        res["val_mae"] = float(np.mean(np.abs(pred - ds.y[val_m])))
    if return_model:
        return res, model, prep, hist
    return res


def predict_celsius(model, prep, m):
    return prep.to_celsius(model.predict(prep.inputs(m), verbose=0, batch_size=1024), m)
