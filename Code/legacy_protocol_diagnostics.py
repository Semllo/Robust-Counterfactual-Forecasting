from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_counterfactual_experiments import (
    DATA_DIR,
    DISTRICTS,
    FULL_END,
    FULL_START,
    POLLUTANTS,
    POST_START,
    WEATHER_COLS,
    load_city_weather,
    load_traffic,
    set_seed,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "legacy_protocol_diagnostics"
POINT_TRAFFIC_COL = "traffic_point"
DISTRICT_TRAFFIC_COL = "traffic_district"
TIMESTAMP_COL = "timestamp_hours"
FEATURE_COLS = [
    TIMESTAMP_COL,
    "VEL_VIENT",
    "DIR_VIENT",
    "TEMP",
    "HUM_REL",
    "PRES_BARIOMETRICA",
    "RAD_SOL",
    "PRECIPITACION",
    POINT_TRAFFIC_COL,
    DISTRICT_TRAFFIC_COL,
]
CHAPTER_DISTRICTS = [4, 6, 18]
CHAPTER_STATIONS = [DISTRICTS[d][1] for d in CHAPTER_DISTRICTS]


@dataclass
class Config:
    windows: list[int]
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class Variant:
    name: str
    split_kind: str
    complete_case: bool


VARIANTS = [
    Variant(name="chapterish_pre70_masked", split_kind="pre70", complete_case=False),
    Variant(name="chapterish_pre70_complete", split_kind="pre70", complete_case=True),
    Variant(name="chapterish_pre70_random_complete", split_kind="pre70_random", complete_case=True),
    Variant(name="chapterish_full70_complete", split_kind="full70", complete_case=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replicate more literal variants of the original chapter protocol to diagnose "
            "why the reported MAE was substantially lower than the extended-paper reruns."
        )
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 6, 24])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_legacy_final_rows() -> pd.DataFrame:
    legacy = pd.read_csv(
        DATA_DIR / "DATOS_FINALES.csv",
        parse_dates=["FECHA_HORA"],
        usecols=["ESTACION", "DISTRITO", "FECHA_HORA", *POLLUTANTS, "intensidad"],
    )
    legacy = legacy.rename(columns={"intensidad": POINT_TRAFFIC_COL})
    legacy = legacy[
        legacy["DISTRITO"].isin(CHAPTER_DISTRICTS) & legacy["ESTACION"].isin(CHAPTER_STATIONS)
    ].copy()
    legacy["DISTRITO"] = legacy["DISTRITO"].astype(int)
    legacy["ESTACION"] = legacy["ESTACION"].astype(int)
    return legacy


def build_chapterish_frame() -> pd.DataFrame:
    legacy = load_legacy_final_rows()
    weather = load_city_weather().reset_index()
    traffic = load_traffic().rename(columns={"intensidad": DISTRICT_TRAFFIC_COL})
    traffic = traffic[traffic["DISTRITO"].isin(CHAPTER_DISTRICTS)].copy()

    frame = legacy.merge(weather, on="FECHA_HORA", how="left")
    frame = frame.merge(traffic, on=["DISTRITO", "FECHA_HORA"], how="left")
    frame[TIMESTAMP_COL] = (
        (frame["FECHA_HORA"] - FULL_START).dt.total_seconds() / 3600.0
    ).astype(np.float32)
    frame = frame.sort_values(["DISTRITO", "FECHA_HORA"]).reset_index(drop=True)
    return frame


class ChapterStyleSequenceRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
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
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(p=0.2))
            previous_dim = hidden_dim
        self.head = nn.Linear(previous_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for lstm, activation, dropout in zip(self.lstm_layers, self.activations, self.dropouts):
            outputs, _ = lstm(outputs)
            outputs = activation(outputs)
            outputs = dropout(outputs)
        return self.head(outputs[:, -1, :])


class LegacyLSTM:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.device = torch.device("cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = ChapterStyleSequenceRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[512, 256, 128, 64, 32],
        ).to(self.device)
        self.x_scaler = MinMaxScaler()

    def _transform_x(self, x_array: np.ndarray, fit: bool = False) -> np.ndarray:
        n_samples, window, n_features = x_array.shape
        flattened = x_array.reshape(-1, n_features)
        if fit:
            self.x_scaler.fit(flattened)
        transformed = self.x_scaler.transform(flattened)
        return transformed.reshape(n_samples, window, n_features).astype(np.float32)

    @staticmethod
    def _masked_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squared_error = ((preds - targets) ** 2) * mask
        return squared_error.sum() / torch.clamp(mask.sum(), min=1.0)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, y_mask: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        x_train = self._transform_x(x_train, fit=True)
        y_train = y_train.astype(np.float32)
        y_mask = y_mask.astype(np.float32)

        split_idx = max(int(len(x_train) * 0.9), 1)
        if split_idx >= len(x_train):
            split_idx = len(x_train) - 1

        x_fit = x_train[:split_idx]
        y_fit = y_train[:split_idx]
        m_fit = y_mask[:split_idx]
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        m_val = y_mask[split_idx:]

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_fit),
                torch.from_numpy(y_fit),
                torch.from_numpy(m_fit),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.learning_rate)
        best_state = None
        best_loss = math.inf

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb, mb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                preds = self.model(xb)
                loss = self._masked_mse(preds, yb, mb)
                loss.backward()
                optimizer.step()

            if len(x_val) == 0:
                continue

            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.from_numpy(x_val).to(self.device))
                val_loss = float(
                    self._masked_mse(
                        preds,
                        torch.from_numpy(y_val).to(self.device),
                        torch.from_numpy(m_val).to(self.device),
                    ).cpu()
                )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_eval = self._transform_x(x_eval, fit=False)
        with torch.no_grad():
            preds = self.model(torch.from_numpy(x_eval).to(self.device)).cpu().numpy()
        return preds.astype(np.float32)


def make_group_sequences(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
    split_kind: str,
    complete_case: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_train_rows: list[np.ndarray] = []
    y_train_rows: list[np.ndarray] = []
    m_train_rows: list[np.ndarray] = []
    x_eval_rows: list[np.ndarray] = []
    y_eval_rows: list[np.ndarray] = []
    m_eval_rows: list[np.ndarray] = []
    coverage: list[dict[str, object]] = []

    for district_id, group in frame.groupby("DISTRITO", sort=True):
        group = group.sort_values("FECHA_HORA").reset_index(drop=True)
        if split_kind in {"pre70", "pre70_random"}:
            group = group[group["FECHA_HORA"] < POST_START].reset_index(drop=True)
        elif split_kind != "full70":
            raise ValueError(f"Unsupported split kind: {split_kind}")

        if len(group) <= window + 20:
            continue

        features = group.loc[:, list(feature_cols)].to_numpy(dtype=np.float32)
        targets = group.loc[:, POLLUTANTS].to_numpy(dtype=np.float32)
        mask = (~np.isnan(targets)).astype(np.float32)

        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        eval_kept = 0
        train_kept = 0

        for idx in range(window, len(group)):
            window_values = features[idx - window : idx]
            if np.isnan(window_values).any():
                continue
            target_row = targets[idx]
            target_mask = mask[idx]
            if complete_case:
                if target_mask.min() < 1.0:
                    continue
                target_row = np.nan_to_num(target_row, nan=0.0)
                target_mask = np.ones_like(target_mask, dtype=np.float32)
            else:
                if target_mask.sum() == 0:
                    continue
                target_row = np.nan_to_num(target_row, nan=0.0)

            samples.append((window_values, target_row, target_mask))

        if not samples:
            continue

        split_idx = max(int(len(samples) * 0.7), 1)
        if split_idx >= len(samples):
            split_idx = len(samples) - 1

        sample_order = np.arange(len(samples))
        if split_kind == "pre70_random":
            rng = np.random.default_rng(seed + int(district_id))
            sample_order = rng.permutation(sample_order)

        train_ids = set(sample_order[:split_idx].tolist())
        for sample_idx, (window_values, target_row, target_mask) in enumerate(samples):
            if sample_idx in train_ids:
                x_train_rows.append(window_values)
                y_train_rows.append(target_row)
                m_train_rows.append(target_mask)
                train_kept += 1
            else:
                x_eval_rows.append(window_values)
                y_eval_rows.append(target_row)
                m_eval_rows.append(target_mask)
                eval_kept += 1

        coverage.append(
            {
                "district_id": int(district_id),
                "district_name": DISTRICTS[int(district_id)][0],
                "rows_after_split_filter": int(len(group)),
                "train_sequences": int(train_kept),
                "eval_sequences": int(eval_kept),
                "split_kind": split_kind,
                "complete_case": complete_case,
                "start": str(group["FECHA_HORA"].min()),
                "end": str(group["FECHA_HORA"].max()),
            }
        )

    if not x_train_rows or not x_eval_rows:
        empty_x = np.empty((0, window, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0, len(POLLUTANTS)), dtype=np.float32)
        return empty_x, empty_y, empty_y.copy(), empty_x.copy(), empty_y.copy(), coverage

    return (
        np.stack(x_train_rows).astype(np.float32),
        np.stack(y_train_rows).astype(np.float32),
        np.stack(m_train_rows).astype(np.float32),
        np.stack(x_eval_rows).astype(np.float32),
        np.stack(y_eval_rows).astype(np.float32),
        np.stack(m_eval_rows).astype(np.float32),
        coverage,
    )


def evaluate_multitarget(
    preds: np.ndarray,
    actuals: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    mae_values: list[float] = []
    for idx, pollutant in enumerate(POLLUTANTS):
        valid = mask[:, idx].astype(bool)
        if not valid.any():
            continue
        mae = float(mean_absolute_error(actuals[valid, idx], preds[valid, idx]))
        mae_values.append(mae)
        rows.append(
            {
                "pollutant": pollutant,
                "mae": mae,
                "n_eval": int(valid.sum()),
            }
        )
    mean_mae = float(np.mean(mae_values)) if mae_values else math.inf
    return rows, mean_mae


def run_variant(
    frame: pd.DataFrame,
    variant: Variant,
    window: int,
    config: Config,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    sequence_data = make_group_sequences(
        frame=frame,
        feature_cols=FEATURE_COLS,
        window=window,
        split_kind=variant.split_kind,
        complete_case=variant.complete_case,
        seed=config.seed,
    )
    x_train, y_train, m_train, x_eval, y_eval, m_eval, coverage = sequence_data
    if len(x_train) == 0 or len(x_eval) == 0:
        raise RuntimeError(f"No valid sequences for variant={variant.name} window={window}")

    model = LegacyLSTM(
        input_dim=len(FEATURE_COLS),
        output_dim=len(POLLUTANTS),
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    model.fit(x_train=x_train, y_train=y_train, y_mask=m_train)
    preds = model.predict(x_eval)
    metric_rows, mean_mae = evaluate_multitarget(preds=preds, actuals=y_eval, mask=m_eval)
    available_districts = [row["district_name"] for row in coverage if row["eval_sequences"] > 0]
    summary = {
        "variant": variant.name,
        "split_kind": variant.split_kind,
        "complete_case": variant.complete_case,
        "window": int(window),
        "mean_mae": mean_mae,
        "n_train_sequences": int(len(x_train)),
        "n_eval_sequences": int(len(x_eval)),
        "districts_with_eval_data": ", ".join(available_districts),
    }
    for row in metric_rows:
        row.update(
            {
                "variant": variant.name,
                "split_kind": variant.split_kind,
                "complete_case": variant.complete_case,
                "window": int(window),
            }
        )
    return metric_rows, coverage, summary


def build_data_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    pre_rows = int((frame["FECHA_HORA"] < POST_START).sum())
    total_rows = int(len(frame))
    by_district = {}
    for district_id, group in frame.groupby("DISTRITO", sort=True):
        district_name = DISTRICTS[int(district_id)][0]
        by_district[str(district_id)] = {
            "district_name": district_name,
            "rows": int(len(group)),
            "pre_rows": int((group["FECHA_HORA"] < POST_START).sum()),
            "full_complete_case_rows": int(group[FEATURE_COLS + POLLUTANTS].dropna().shape[0]),
        }
    return {
        "chapter_districts": CHAPTER_DISTRICTS,
        "chapter_stations": CHAPTER_STATIONS,
        "date_start": str(frame["FECHA_HORA"].min()),
        "date_end": str(frame["FECHA_HORA"].max()),
        "total_rows": total_rows,
        "pre_rows": pre_rows,
        "pre_share_of_total": pre_rows / total_rows if total_rows else None,
        "note": (
            "If the effective dataset is 2021-2023 with POST_START at 2022-01-01, "
            "then pre-intervention rows are roughly one third of the total horizon. "
            "A claim that training used only pre-policy rows while also representing 70% "
            "of the full dataset is internally inconsistent."
        ),
        "by_district": by_district,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = Config(
        windows=args.windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_chapterish_frame()
    data_diagnostics = build_data_diagnostics(frame)
    (config.output_dir / "data_diagnostics.json").write_text(
        json.dumps(data_diagnostics, indent=2),
        encoding="utf-8",
    )

    metric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        for window in config.windows:
            rows, coverage, summary = run_variant(
                frame=frame,
                variant=variant,
                window=window,
                config=config,
            )
            metric_rows.extend(rows)
            coverage_rows.extend(coverage)
            summary_rows.append(summary)
            print(
                f"[legacy] variant={variant.name:<24} window={window:<2} "
                f"mean_mae={summary['mean_mae']:.3f} eval={summary['n_eval_sequences']:<5} "
                f"districts={summary['districts_with_eval_data']}"
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(by=["variant", "window", "pollutant"])
    coverage_df = pd.DataFrame(coverage_rows).sort_values(
        by=["split_kind", "complete_case", "district_id"]
    )
    summary_df = pd.DataFrame(summary_rows).sort_values(by=["mean_mae", "variant", "window"])

    metrics_df.to_csv(config.output_dir / "legacy_protocol_metrics.csv", index=False)
    coverage_df.to_csv(config.output_dir / "legacy_protocol_coverage.csv", index=False)
    summary_df.to_csv(config.output_dir / "legacy_protocol_summary.csv", index=False)


if __name__ == "__main__":
    main()
