from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datos" / "datos"
OUTPUT_DIR = ROOT / "experiment_outputs"
DISTRICTS: Dict[int, tuple[str, int]] = {
    4: ("Salamanca", 48),
    6: ("Tetuan", 38),
    18: ("Villa de Vallecas", 40),
}
POLLUTANTS = ["NO", "NO2", "NOx", "PM10", "PM2_5"]
WEATHER_COLS = [
    "TEMP",
    "HUM_REL",
    "PRECIPITACION",
    "PRES_BARIOMETRICA",
    "RAD_SOL",
    "DIR_VIENT",
    "VEL_VIENT",
]
TIME_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]
FULL_START = pd.Timestamp("2021-01-01 00:00:00")
POST_START = pd.Timestamp("2022-01-01 00:00:00")
FULL_END = pd.Timestamp("2023-12-31 23:00:00")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


@dataclass
class RunConfig:
    models: list[str]
    windows: list[int]
    lstm_epochs: int
    lstm_hidden_dim: int
    batch_size: int
    learning_rate: float
    seed: int
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run predictive counterfactual experiments for the Madrid LEZ paper. "
            "The script rebuilds a clean modeling panel for Salamanca, Tetuan, "
            "and Villa de Vallecas, compares Ridge, LightGBM, and LSTM over "
            "windows 6 and 24, and runs lightweight ablations for the best setup."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ridge", "lightgbm", "lstm"],
        default=["ridge", "lightgbm", "lstm"],
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[6, 24],
    )
    parser.add_argument("--lstm-epochs", type=int, default=8)
    parser.add_argument("--lstm-hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


def add_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    doy = index.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
            "dow_sin": np.sin(2 * np.pi * weekday / 7),
            "dow_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "doy_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
            "doy_cos": np.cos(2 * np.pi * (doy - 1) / 365.25),
        },
        index=index,
    )


def load_city_weather() -> pd.DataFrame:
    weather = pd.read_csv(
        DATA_DIR / "meteorologia2021_2023.csv",
        parse_dates=["FECHA_HORA"],
        usecols=["FECHA_HORA", *WEATHER_COLS],
    )
    city_weather = (
        weather.groupby("FECHA_HORA", as_index=True)[WEATHER_COLS]
        .mean()
        .sort_index()
    )
    return city_weather


def load_traffic() -> pd.DataFrame:
    traffic = pd.read_csv(
        DATA_DIR / "trafico_calculado_por_distrito.csv",
        parse_dates=["fecha"],
        usecols=["distrito", "fecha", "intensidad"],
    )
    traffic["distrito"] = traffic["distrito"].astype(int)
    traffic = traffic.rename(columns={"distrito": "DISTRITO", "fecha": "FECHA_HORA"})
    return traffic


def load_pollution() -> pd.DataFrame:
    stations = [station for _, station in DISTRICTS.values()]
    pollution = pd.read_csv(
        DATA_DIR / "contaminacion2021_2023.csv",
        parse_dates=["FECHA_HORA"],
        usecols=["ESTACION", "FECHA_HORA", *POLLUTANTS],
    )
    pollution = pollution[pollution["ESTACION"].isin(stations)].copy()
    return pollution


def build_district_panels() -> dict[int, pd.DataFrame]:
    full_index = pd.date_range(FULL_START, FULL_END, freq="h")
    weather = load_city_weather()
    traffic = load_traffic()
    pollution = load_pollution()
    panels: dict[int, pd.DataFrame] = {}

    for district_id, (district_name, station_id) in DISTRICTS.items():
        panel = pd.DataFrame(index=full_index)

        district_pollution = (
            pollution.loc[pollution["ESTACION"] == station_id, ["FECHA_HORA", *POLLUTANTS]]
            .drop_duplicates(subset=["FECHA_HORA"])
            .set_index("FECHA_HORA")
            .sort_index()
        )
        district_traffic = (
            traffic.loc[traffic["DISTRITO"] == district_id, ["FECHA_HORA", "intensidad"]]
            .drop_duplicates(subset=["FECHA_HORA"])
            .set_index("FECHA_HORA")
            .sort_index()
        )

        panel = panel.join(district_pollution, how="left")
        panel = panel.join(district_traffic, how="left")
        panel = panel.join(weather, how="left")
        panel = panel.join(add_time_features(panel.index), how="left")
        panel["district_id"] = district_id
        panel["district_name"] = district_name

        for col in ["intensidad", *WEATHER_COLS]:
            panel[col] = panel[col].interpolate(method="time", limit_direction="both")
            panel[col] = panel[col].ffill().bfill()

        for pollutant in POLLUTANTS:
            hist_col = f"{pollutant}_hist"
            panel[hist_col] = panel[pollutant].interpolate(method="time", limit_direction="both")
            panel[hist_col] = panel[hist_col].ffill().bfill()

        panels[district_id] = panel

    return panels


def get_variant_columns(variant: str) -> list[str]:
    cols = ["intensidad", *WEATHER_COLS, *TIME_COLS]
    if variant == "no_traffic":
        cols = [col for col in cols if col != "intensidad"]
    elif variant == "no_weather":
        cols = [col for col in cols if col not in WEATHER_COLS]
    return cols


def clip_with_iqr(
    frame: pd.DataFrame,
    cols: Sequence[str],
    mask: pd.Series,
) -> pd.DataFrame:
    clipped = frame.copy()
    pre_stats = clipped.loc[mask, list(cols)]
    q1 = pre_stats.quantile(0.25)
    q3 = pre_stats.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clipped.loc[:, list(cols)] = clipped.loc[:, list(cols)].clip(lower=lower, upper=upper, axis=1)
    return clipped


def make_training_sequences(
    frame: pd.DataFrame,
    pollutant: str,
    exog_cols: Sequence[str],
    window: int,
    train_end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    hist_col = f"{pollutant}_hist"
    feature_cols = [hist_col, *exog_cols]
    features = frame.loc[:, feature_cols].to_numpy(dtype=np.float32)
    target = frame.loc[:, pollutant].to_numpy(dtype=np.float32)
    xs: list[np.ndarray] = []
    ys: list[float] = []

    for idx in range(window, train_end_idx):
        if np.isnan(target[idx]):
            continue
        window_values = features[idx - window : idx]
        if np.isnan(window_values).any():
            continue
        xs.append(window_values)
        ys.append(float(target[idx]))

    if not xs:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xs), np.asarray(ys, dtype=np.float32)


class BaseForecaster:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, x_seq: np.ndarray) -> float:
        raise NotImplementedError


class RidgeForecaster(BaseForecaster):
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=10.0)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_flat = x_train.reshape(len(x_train), -1)
        self.scaler.fit(x_flat)
        self.model.fit(self.scaler.transform(x_flat), y_train)

    def predict(self, x_seq: np.ndarray) -> float:
        x_flat = x_seq.reshape(1, -1)
        pred = self.model.predict(self.scaler.transform(x_flat))
        return float(pred[0])


class LightGBMForecaster(BaseForecaster):
    def __init__(self, seed: int) -> None:
        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_flat = x_train.reshape(len(x_train), -1)
        self.model.fit(x_flat, y_train)

    def predict(self, x_seq: np.ndarray) -> float:
        x_flat = x_seq.reshape(1, -1)
        pred = self.model.predict(x_flat)
        return float(pred[0])


class OriginalStyleSequenceRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=previous_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )
            )
            self.dropouts.append(nn.Dropout(p=0.2))
            previous_dim = hidden_dim
        self.head = nn.Linear(previous_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for lstm_layer, dropout in zip(self.lstm_layers, self.dropouts):
            outputs, _ = lstm_layer(outputs)
            outputs = dropout(outputs)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden)


class MultiOutputLSTMForecaster:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
    ) -> None:
        self.device = torch.device("cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.model = OriginalStyleSequenceRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        ).to(self.device)
        self.x_scaler = StandardScaler()

    def _transform_x(self, array: np.ndarray, fit: bool = False) -> np.ndarray:
        n_samples, window, n_features = array.shape
        flattened = array.reshape(-1, n_features)
        if fit:
            self.x_scaler.fit(flattened)
        transformed = self.x_scaler.transform(flattened)
        return transformed.reshape(n_samples, window, n_features)

    @staticmethod
    def _masked_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squared_error = ((preds - targets) ** 2) * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        return squared_error.sum() / denom

    @staticmethod
    def _masked_mae(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
        valid = mask.astype(bool)
        if not valid.any():
            return math.inf
        return float(np.mean(np.abs(preds[valid] - targets[valid])))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, y_mask: np.ndarray) -> None:
        if len(x_train) < 32:
            raise ValueError("Not enough samples to train the LSTM.")

        torch.manual_seed(self.seed)
        x_train_scaled = self._transform_x(x_train, fit=True).astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_mask = y_mask.astype(np.float32)

        split_idx = max(int(len(x_train_scaled) * 0.9), 1)
        if split_idx >= len(x_train_scaled):
            split_idx = len(x_train_scaled) - 1

        x_fit = x_train_scaled[:split_idx]
        y_fit = y_train[:split_idx]
        mask_fit = y_mask[:split_idx]
        x_val = x_train_scaled[split_idx:]
        y_val = y_train[split_idx:]
        mask_val = y_mask[split_idx:]

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_fit),
                torch.from_numpy(y_fit),
                torch.from_numpy(mask_fit),
            ),
            batch_size=min(self.batch_size, len(x_fit)),
            shuffle=True,
        )

        optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        best_state = None
        best_val = math.inf
        patience = 0

        for _ in range(self.epochs):
            self.model.train()
            for x_batch, y_batch, mask_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self._masked_mse(preds, y_batch, mask_batch)
                loss.backward()
                optimizer.step()

            if len(x_val) == 0:
                continue

            self.model.eval()
            with torch.no_grad():
                val_inputs = torch.from_numpy(x_val).to(self.device)
                val_preds = self.model(val_inputs).cpu().numpy()
            val_mae = self._masked_mae(val_preds, y_val, mask_val)

            if val_mae < best_val:
                best_val = val_mae
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        x_batch = self._transform_x(x_seq[np.newaxis, :, :], fit=False).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x_batch).to(self.device)
            preds = self.model(tensor).cpu().numpy()
        return preds.ravel()


def build_model(model_name: str, input_dim: int, config: RunConfig) -> BaseForecaster:
    if model_name == "ridge":
        return RidgeForecaster()
    if model_name == "lightgbm":
        return LightGBMForecaster(seed=config.seed)
    raise ValueError(f"Unsupported model: {model_name}")


def get_lstm_hidden_dims(config: RunConfig) -> list[int]:
    return [512, 256, 128, 64, config.lstm_hidden_dim]


def make_multitarget_training_sequences(
    frame: pd.DataFrame,
    exog_cols: Sequence[str],
    window: int,
    train_end_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame.loc[:, list(exog_cols)].to_numpy(dtype=np.float32)
    targets = frame.loc[:, POLLUTANTS].to_numpy(dtype=np.float32)
    target_mask = (~np.isnan(targets)).astype(np.float32)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for idx in range(window, train_end_idx):
        if target_mask[idx].sum() == 0:
            continue
        window_values = features[idx - window : idx]
        if np.isnan(window_values).any():
            continue
        xs.append(window_values)
        ys.append(np.nan_to_num(targets[idx], nan=0.0))
        masks.append(target_mask[idx])

    if not xs:
        empty_x = np.empty((0, window, len(exog_cols)), dtype=np.float32)
        empty_y = np.empty((0, len(POLLUTANTS)), dtype=np.float32)
        return empty_x, empty_y, empty_y.copy()

    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.stack(masks).astype(np.float32),
    )


def direct_multitarget_forecast(
    frame: pd.DataFrame,
    exog_cols: Sequence[str],
    window: int,
    start_idx: int,
    end_idx: int,
    model: MultiOutputLSTMForecaster,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    features = frame.loc[:, list(exog_cols)].to_numpy(dtype=np.float32)
    targets = frame.loc[:, POLLUTANTS].to_numpy(dtype=np.float32)
    preds_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []

    for idx in range(start_idx, end_idx):
        mask = ~np.isnan(targets[idx])
        if not mask.any():
            continue
        window_values = features[idx - window : idx]
        if np.isnan(window_values).any():
            continue
        preds_rows.append(model.predict(window_values).astype(np.float32))
        target_rows.append(np.nan_to_num(targets[idx], nan=0.0).astype(np.float32))
        mask_rows.append(mask.astype(np.float32))
        timestamps.append(frame.index[idx])

    if not preds_rows:
        empty = np.empty((0, len(POLLUTANTS)), dtype=np.float32)
        return empty, empty.copy(), empty.copy(), []

    return (
        np.stack(preds_rows),
        np.stack(target_rows),
        np.stack(mask_rows),
        timestamps,
    )


def run_multitarget_lstm_configuration(
    panel: pd.DataFrame,
    district_id: int,
    window: int,
    variant: str,
    config: RunConfig,
) -> list[dict[str, object]]:
    pre_mask = panel.index < POST_START
    exog_cols = get_variant_columns(variant)
    work_panel = panel.copy()
    if variant != "no_iqr":
        work_panel = clip_with_iqr(work_panel, exog_cols, pre_mask)

    pre_len = int(pre_mask.sum())
    train_end_idx = max(int(pre_len * 0.7), window + 50)
    if train_end_idx >= pre_len:
        train_end_idx = pre_len - 1

    x_train, y_train, y_mask = make_multitarget_training_sequences(
        frame=work_panel,
        exog_cols=exog_cols,
        window=window,
        train_end_idx=train_end_idx,
    )
    if len(x_train) < 100:
        return []

    model = MultiOutputLSTMForecaster(
        input_dim=len(exog_cols),
        output_dim=len(POLLUTANTS),
        hidden_dims=get_lstm_hidden_dims(config),
        epochs=config.lstm_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        seed=config.seed,
    )
    model.fit(x_train=x_train, y_train=y_train, y_mask=y_mask)

    val_preds, val_actuals, val_mask, _ = direct_multitarget_forecast(
        frame=work_panel,
        exog_cols=exog_cols,
        window=window,
        start_idx=train_end_idx,
        end_idx=pre_len,
        model=model,
    )
    post_preds, post_actuals, post_mask, _ = direct_multitarget_forecast(
        frame=work_panel,
        exog_cols=exog_cols,
        window=window,
        start_idx=pre_len,
        end_idx=len(work_panel),
        model=model,
    )
    if len(val_preds) == 0 or len(post_preds) == 0:
        return []

    district_name = DISTRICTS[district_id][0]
    rows: list[dict[str, object]] = []
    for pollutant_idx, pollutant in enumerate(POLLUTANTS):
        val_valid = val_mask[:, pollutant_idx].astype(bool)
        post_valid = post_mask[:, pollutant_idx].astype(bool)
        if val_valid.sum() < 100 or post_valid.sum() < 100:
            continue
        validation_metrics = evaluate_predictions(
            val_preds[val_valid, pollutant_idx],
            val_actuals[val_valid, pollutant_idx],
        )
        post_metrics = evaluate_predictions(
            post_preds[post_valid, pollutant_idx],
            post_actuals[post_valid, pollutant_idx],
        )
        rows.append(
            {
                "district_id": district_id,
                "district_name": district_name,
                "pollutant": pollutant,
                "model": "lstm",
                "window": window,
                "variant": variant,
                "pre_mae": validation_metrics["mae"],
                "post_pip": post_metrics["pip"],
                "post_mean_residual": post_metrics["mean_residual"],
                "post_median_residual": post_metrics["median_residual"],
                "n_train": int(len(x_train)),
                "n_validation": int(val_valid.sum()),
                "n_post": int(post_valid.sum()),
            }
        )

    return rows


def recursive_forecast(
    frame: pd.DataFrame,
    pollutant: str,
    exog_cols: Sequence[str],
    window: int,
    start_idx: int,
    end_idx: int,
    model: BaseForecaster,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    hist_col = f"{pollutant}_hist"
    hist_values = frame.loc[:, hist_col].to_numpy(dtype=np.float64).copy()
    exog_values = frame.loc[:, list(exog_cols)].to_numpy(dtype=np.float64)
    actual_values = frame.loc[:, pollutant].to_numpy(dtype=np.float64)
    preds: list[float] = []
    actuals: list[float] = []
    timestamps: list[pd.Timestamp] = []
    lower_bound, upper_bound = clip_bounds

    for idx in range(start_idx, end_idx):
        window_target = hist_values[idx - window : idx].reshape(-1, 1)
        window_exog = exog_values[idx - window : idx]
        window_values = np.concatenate([window_target, window_exog], axis=1)
        if np.isnan(window_values).any():
            continue

        pred = model.predict(window_values)
        pred = float(np.clip(pred, lower_bound, upper_bound))
        hist_values[idx] = pred

        if not np.isnan(actual_values[idx]):
            preds.append(pred)
            actuals.append(float(actual_values[idx]))
            timestamps.append(frame.index[idx])

    return np.asarray(preds, dtype=np.float32), np.asarray(actuals, dtype=np.float32), timestamps


def evaluate_predictions(preds: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
    residuals = preds - actuals
    return {
        "mae": float(mean_absolute_error(actuals, preds)),
        "pip": float(np.mean(preds > actuals) * 100.0),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
    }


def run_single_configuration(
    panel: pd.DataFrame,
    district_id: int,
    pollutant: str,
    model_name: str,
    window: int,
    variant: str,
    config: RunConfig,
) -> dict[str, object] | None:
    pre_mask = panel.index < POST_START
    hist_col = f"{pollutant}_hist"
    exog_cols = get_variant_columns(variant)
    used_cols = [hist_col, *exog_cols]

    if panel.loc[:, pollutant].notna().sum() < 500:
        return None
    if panel.loc[pre_mask, pollutant].notna().sum() < 250 or panel.loc[~pre_mask, pollutant].notna().sum() < 250:
        return None
    if panel.loc[:, hist_col].isna().all():
        return None

    work_panel = panel.copy()
    if variant != "no_iqr":
        work_panel = clip_with_iqr(work_panel, used_cols, pre_mask)

    target_pre = work_panel.loc[pre_mask, hist_col].dropna()
    if target_pre.empty:
        return None
    clip_bounds = (max(0.0, float(target_pre.min())), float(target_pre.max()))

    pre_len = int(pre_mask.sum())
    train_end_idx = max(int(pre_len * 0.7), window + 50)
    if train_end_idx >= pre_len:
        train_end_idx = pre_len - 1

    x_train, y_train = make_training_sequences(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        train_end_idx=train_end_idx,
    )
    if len(x_train) < 100:
        return None

    model = build_model(model_name=model_name, input_dim=len(used_cols), config=config)
    model.fit(x_train=x_train, y_train=y_train)

    val_preds, val_actuals, _ = recursive_forecast(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        start_idx=train_end_idx,
        end_idx=pre_len,
        model=model,
        clip_bounds=clip_bounds,
    )
    if len(val_preds) < 100:
        return None

    post_preds, post_actuals, _ = recursive_forecast(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        start_idx=pre_len,
        end_idx=len(work_panel),
        model=model,
        clip_bounds=clip_bounds,
    )
    if len(post_preds) < 100:
        return None

    validation_metrics = evaluate_predictions(val_preds, val_actuals)
    post_metrics = evaluate_predictions(post_preds, post_actuals)
    district_name = DISTRICTS[district_id][0]

    return {
        "district_id": district_id,
        "district_name": district_name,
        "pollutant": pollutant,
        "model": model_name,
        "window": window,
        "variant": variant,
        "pre_mae": validation_metrics["mae"],
        "post_pip": post_metrics["pip"],
        "post_mean_residual": post_metrics["mean_residual"],
        "post_median_residual": post_metrics["median_residual"],
        "n_train": int(len(x_train)),
        "n_validation": int(len(val_preds)),
        "n_post": int(len(post_preds)),
    }


def make_heatmap(best_results: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = best_results.copy()
    plot_frame["district_pollutant"] = plot_frame["district_name"] + " | " + plot_frame["pollutant"]
    pivot = plot_frame.pivot_table(
        index="district_name",
        columns="pollutant",
        values="post_mean_residual",
        aggfunc="mean",
    ).reindex(index=[name for name, _ in DISTRICTS.values()])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    image = ax.imshow(pivot.values, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Post-intervention mean residuals by district and pollutant")

    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iloc[row_idx, col_idx]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Prediction - observation")
    fig.tight_layout()
    fig.savefig(output_dir / "best_full_variant_heatmap.png", dpi=200)
    plt.close(fig)


def run_experiments(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    panels = build_district_panels()

    metadata = {
        "post_start": str(POST_START),
        "districts": {district_id: name for district_id, (name, _) in DISTRICTS.items()},
        "models": config.models,
        "windows": config.windows,
        "pollutants": POLLUTANTS,
        "variants": ["full", "no_traffic", "no_weather", "no_iqr"],
    }
    (config.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    selection_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for district_id, panel in panels.items():
        for model_name in config.models:
            for window in config.windows:
                if model_name == "lstm":
                    rows = run_multitarget_lstm_configuration(
                        panel=panel,
                        district_id=district_id,
                        window=window,
                        variant="full",
                        config=config,
                    )
                    if rows:
                        selection_rows.extend(rows)
                        for row in rows:
                            print(
                                f"[selection] district={row['district_name']:<19} "
                                f"pollutant={row['pollutant']:<5} model={model_name:<8} window={window:<2} "
                                f"pre_mae={row['pre_mae']:.3f} post_pip={row['post_pip']:.1f}"
                            )
                    else:
                        print(
                            f"[selection] district={DISTRICTS[district_id][0]:<19} "
                            f"model={model_name:<8} window={window:<2} skipped"
                        )
                    continue

                for pollutant in POLLUTANTS:
                    row = run_single_configuration(
                        panel=panel,
                        district_id=district_id,
                        pollutant=pollutant,
                        model_name=model_name,
                        window=window,
                        variant="full",
                        config=config,
                    )
                    if row is not None:
                        selection_rows.append(row)
                        print(
                            f"[selection] district={row['district_name']:<19} "
                            f"pollutant={pollutant:<5} model={model_name:<8} window={window:<2} "
                            f"pre_mae={row['pre_mae']:.3f} post_pip={row['post_pip']:.1f}"
                        )
                    else:
                        print(
                            f"[selection] district={DISTRICTS[district_id][0]:<19} "
                            f"pollutant={pollutant:<5} model={model_name:<8} window={window:<2} skipped"
                        )

    if not selection_rows:
        raise RuntimeError("No valid experiment completed during the model-selection stage.")

    selection_df = pd.DataFrame(selection_rows).sort_values(
        by=["pre_mae", "district_name", "pollutant", "model", "window"]
    )
    selection_df.to_csv(config.output_dir / "model_selection_results.csv", index=False)

    summary_df = (
        selection_df.groupby(["model", "window"], as_index=False)
        .agg(
            mean_pre_mae=("pre_mae", "mean"),
            median_pre_mae=("pre_mae", "median"),
            mean_post_pip=("post_pip", "mean"),
            mean_post_residual=("post_mean_residual", "mean"),
            n_runs=("pre_mae", "size"),
        )
        .sort_values(by=["mean_pre_mae", "model", "window"])
    )
    summary_df.to_csv(config.output_dir / "model_selection_summary.csv", index=False)

    best_model = str(summary_df.iloc[0]["model"])
    best_window = int(summary_df.iloc[0]["window"])
    print(f"\nBest setup based on mean pre-intervention MAE: model={best_model}, window={best_window}")

    ablation_rows: list[dict[str, object]] = []
    for variant in ["full", "no_traffic", "no_weather", "no_iqr"]:
        for district_id, panel in panels.items():
            if best_model == "lstm":
                rows = run_multitarget_lstm_configuration(
                    panel=panel,
                    district_id=district_id,
                    window=best_window,
                    variant=variant,
                    config=config,
                )
                if rows:
                    ablation_rows.extend(rows)
                    for row in rows:
                        print(
                            f"[ablation] variant={variant:<11} district={row['district_name']:<19} "
                            f"pollutant={row['pollutant']:<5} pre_mae={row['pre_mae']:.3f} post_pip={row['post_pip']:.1f}"
                        )
                else:
                    print(
                        f"[ablation] variant={variant:<11} district={DISTRICTS[district_id][0]:<19} skipped"
                    )
                continue

            for pollutant in POLLUTANTS:
                row = run_single_configuration(
                    panel=panel,
                    district_id=district_id,
                    pollutant=pollutant,
                    model_name=best_model,
                    window=best_window,
                    variant=variant,
                    config=config,
                )
                if row is not None:
                    ablation_rows.append(row)
                    print(
                        f"[ablation] variant={variant:<11} district={row['district_name']:<19} "
                        f"pollutant={pollutant:<5} pre_mae={row['pre_mae']:.3f} post_pip={row['post_pip']:.1f}"
                    )
                else:
                    print(
                        f"[ablation] variant={variant:<11} district={DISTRICTS[district_id][0]:<19} "
                        f"pollutant={pollutant:<5} skipped"
                    )

    ablation_df = pd.DataFrame(ablation_rows).sort_values(
        by=["variant", "district_name", "pollutant"]
    )
    ablation_df.to_csv(config.output_dir / "ablation_results.csv", index=False)

    ablation_summary = (
        ablation_df.groupby("variant", as_index=False)
        .agg(
            mean_pre_mae=("pre_mae", "mean"),
            mean_post_pip=("post_pip", "mean"),
            mean_post_residual=("post_mean_residual", "mean"),
            median_post_residual=("post_median_residual", "median"),
            n_runs=("pre_mae", "size"),
        )
        .sort_values(by=["variant"])
    )
    ablation_summary.to_csv(config.output_dir / "ablation_summary.csv", index=False)

    best_full_variant = ablation_df.loc[ablation_df["variant"] == "full"].copy()
    best_full_variant.to_csv(config.output_dir / "best_full_variant_results.csv", index=False)
    make_heatmap(best_full_variant, config.output_dir)

    elapsed = time.perf_counter() - started
    print(f"\nCompleted in {elapsed / 60:.2f} minutes. Outputs saved to {config.output_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = RunConfig(
        models=args.models,
        windows=args.windows,
        lstm_epochs=args.lstm_epochs,
        lstm_hidden_dim=args.lstm_hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    run_experiments(config)


if __name__ == "__main__":
    main()
