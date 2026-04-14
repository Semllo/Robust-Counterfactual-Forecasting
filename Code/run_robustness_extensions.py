from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from run_counterfactual_experiments import (
    DISTRICTS,
    POLLUTANTS,
    POST_START,
    RunConfig,
    build_district_panels,
    build_model,
    clip_with_iqr,
    evaluate_predictions,
    get_variant_columns,
    make_training_sequences,
    recursive_forecast,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "robustness_extensions"
PLACEBO_START = pd.Timestamp("2021-10-01 00:00:00")
BOOTSTRAP_REPS = 500


@dataclass(frozen=True)
class NaiveSpec:
    name: str
    lag: int
    window: int


class SeasonalNaiveForecaster:
    def __init__(self, lag_steps: int) -> None:
        self.lag_steps = lag_steps

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        return None

    def predict(self, x_seq: np.ndarray) -> float:
        hist_series = x_seq[:, 0]
        if self.lag_steps == 1:
            return float(hist_series[-1])
        return float(hist_series[0])


def make_config(output_dir: Path, seed: int = 42) -> RunConfig:
    return RunConfig(
        models=["lightgbm"],
        windows=[6, 24],
        lstm_epochs=4,
        lstm_hidden_dim=32,
        batch_size=256,
        learning_rate=1e-3,
        seed=seed,
        output_dir=output_dir,
    )


def get_best_setup() -> tuple[str, int]:
    summary = pd.read_csv(ROOT / "experiment_outputs_full_origlike_lstm" / "model_selection_summary.csv")
    best_row = summary.sort_values(by=["mean_pre_mae", "model", "window"]).iloc[0]
    return str(best_row["model"]), int(best_row["window"])


def run_univariate_with_details(
    panel: pd.DataFrame,
    district_id: int,
    pollutant: str,
    window: int,
    variant: str,
    split_point: pd.Timestamp,
    config: RunConfig,
    model_factory: Callable[[], object],
    model_label: str,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    pre_mask = panel.index < split_point
    hist_col = f"{pollutant}_hist"
    exog_cols = get_variant_columns(variant)
    used_cols = [hist_col, *exog_cols]

    if panel.loc[:, pollutant].notna().sum() < 500:
        return None, pd.DataFrame()
    if panel.loc[pre_mask, pollutant].notna().sum() < 250 or panel.loc[~pre_mask, pollutant].notna().sum() < 250:
        return None, pd.DataFrame()
    if panel.loc[:, hist_col].isna().all():
        return None, pd.DataFrame()

    work_panel = panel.copy()
    if variant != "no_iqr":
        work_panel = clip_with_iqr(work_panel, used_cols, pre_mask)

    target_pre = work_panel.loc[pre_mask, hist_col].dropna()
    if target_pre.empty:
        return None, pd.DataFrame()
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
        return None, pd.DataFrame()

    model = model_factory()
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
        return None, pd.DataFrame()

    post_preds, post_actuals, post_timestamps = recursive_forecast(
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
        return None, pd.DataFrame()

    validation_metrics = evaluate_predictions(val_preds, val_actuals)
    post_metrics = evaluate_predictions(post_preds, post_actuals)
    district_name = DISTRICTS[district_id][0]

    metrics_row = {
        "district_id": district_id,
        "district_name": district_name,
        "pollutant": pollutant,
        "model": model_label,
        "window": window,
        "variant": variant,
        "split_point": str(split_point),
        "pre_mae": validation_metrics["mae"],
        "post_pip": post_metrics["pip"],
        "post_mean_residual": post_metrics["mean_residual"],
        "post_median_residual": post_metrics["median_residual"],
        "n_train": int(len(x_train)),
        "n_validation": int(len(val_preds)),
        "n_post": int(len(post_preds)),
    }

    detail_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(post_timestamps),
            "district_id": district_id,
            "district_name": district_name,
            "pollutant": pollutant,
            "model": model_label,
            "variant": variant,
            "window": window,
            "prediction": post_preds,
            "actual": post_actuals,
        }
    )
    detail_df["residual"] = detail_df["prediction"] - detail_df["actual"]
    detail_df["is_positive"] = (detail_df["prediction"] > detail_df["actual"]).astype(int)
    detail_df["day"] = detail_df["timestamp"].dt.floor("D")
    return metrics_row, detail_df


def summarize_rows(rows: list[dict[str, object]], group_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(group_cols, as_index=False)
        .agg(
            mean_pre_mae=("pre_mae", "mean"),
            median_pre_mae=("pre_mae", "median"),
            mean_post_pip=("post_pip", "mean"),
            mean_post_residual=("post_mean_residual", "mean"),
            median_post_residual=("post_median_residual", "median"),
            n_runs=("pre_mae", "size"),
        )
        .sort_values(by=group_cols)
    )


def bootstrap_intervals(detail_df: pd.DataFrame, seed: int, reps: int = BOOTSTRAP_REPS) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique_days = np.asarray(sorted(detail_df["day"].unique()))
    if len(unique_days) == 0:
        raise ValueError("No post-intervention days available for bootstrap.")

    by_day = {day: detail_df.loc[detail_df["day"] == day] for day in unique_days}
    pip_values: list[float] = []
    mean_residual_values: list[float] = []
    median_residual_values: list[float] = []

    for _ in range(reps):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        sampled = pd.concat([by_day[day] for day in sampled_days], ignore_index=True)
        grouped = (
            sampled.groupby(["district_name", "pollutant"], as_index=False)
            .agg(
                post_pip=("is_positive", lambda values: float(np.mean(values) * 100.0)),
                post_mean_residual=("residual", "mean"),
                post_median_residual=("residual", "median"),
            )
        )
        pip_values.append(float(grouped["post_pip"].mean()))
        mean_residual_values.append(float(grouped["post_mean_residual"].mean()))
        median_residual_values.append(float(grouped["post_median_residual"].median()))

    return {
        "pip_low": float(np.quantile(pip_values, 0.025)),
        "pip_high": float(np.quantile(pip_values, 0.975)),
        "mean_residual_low": float(np.quantile(mean_residual_values, 0.025)),
        "mean_residual_high": float(np.quantile(mean_residual_values, 0.975)),
        "median_residual_low": float(np.quantile(median_residual_values, 0.025)),
        "median_residual_high": float(np.quantile(median_residual_values, 0.975)),
        "bootstrap_reps": reps,
    }


def main() -> None:
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(output_dir=output_dir, seed=42)
    set_seed(config.seed)
    best_model, best_window = get_best_setup()
    if best_model != "lightgbm":
        raise RuntimeError(f"Expected LightGBM as current best setup, got {best_model!r}.")

    panels = build_district_panels()

    naive_specs = [
        NaiveSpec(name="persistence_1h", lag=1, window=1),
        NaiveSpec(name="seasonal_naive_24h", lag=24, window=24),
    ]
    naive_rows: list[dict[str, object]] = []
    for spec in naive_specs:
        for district_id, panel in panels.items():
            for pollutant in POLLUTANTS:
                row, _ = run_univariate_with_details(
                    panel=panel,
                    district_id=district_id,
                    pollutant=pollutant,
                    window=spec.window,
                    variant="full",
                    split_point=POST_START,
                    config=config,
                    model_factory=lambda lag=spec.lag: SeasonalNaiveForecaster(lag_steps=lag),
                    model_label=spec.name,
                )
                if row is not None:
                    naive_rows.append(row)

    naive_summary = summarize_rows(naive_rows, ["model", "window"])
    naive_summary.to_csv(output_dir / "naive_baseline_summary.csv", index=False)
    pd.DataFrame(naive_rows).to_csv(output_dir / "naive_baseline_results.csv", index=False)

    placebo_rows: list[dict[str, object]] = []
    placebo_details: list[pd.DataFrame] = []
    for district_id, panel in panels.items():
        for pollutant in POLLUTANTS:
            row, detail_df = run_univariate_with_details(
                panel=panel,
                district_id=district_id,
                pollutant=pollutant,
                window=best_window,
                variant="full",
                split_point=PLACEBO_START,
                config=config,
                model_factory=lambda: build_model("lightgbm", input_dim=1, config=config),
                model_label="lightgbm_placebo",
            )
            if row is not None:
                placebo_rows.append(row)
                placebo_details.append(detail_df)

    placebo_summary = summarize_rows(placebo_rows, ["model", "window"])
    placebo_summary.to_csv(output_dir / "placebo_summary.csv", index=False)
    pd.DataFrame(placebo_rows).to_csv(output_dir / "placebo_results.csv", index=False)
    pd.concat(placebo_details, ignore_index=True).to_csv(output_dir / "placebo_post_details.csv", index=False)

    interval_rows: list[dict[str, object]] = []
    best_variant_rows: list[dict[str, object]] = []
    best_variant_details: list[pd.DataFrame] = []
    for variant in ["full", "no_traffic", "no_weather", "no_iqr"]:
        variant_rows: list[dict[str, object]] = []
        variant_details: list[pd.DataFrame] = []
        for district_id, panel in panels.items():
            for pollutant in POLLUTANTS:
                row, detail_df = run_univariate_with_details(
                    panel=panel,
                    district_id=district_id,
                    pollutant=pollutant,
                    window=best_window,
                    variant=variant,
                    split_point=POST_START,
                    config=config,
                    model_factory=lambda: build_model("lightgbm", input_dim=1, config=config),
                    model_label="lightgbm",
                )
                if row is not None:
                    variant_rows.append(row)
                    variant_details.append(detail_df)
                    best_variant_rows.append(row)
                    best_variant_details.append(detail_df.assign(variant=variant))

        variant_detail_df = pd.concat(variant_details, ignore_index=True)
        summary = summarize_rows(variant_rows, ["variant"]).iloc[0].to_dict()
        summary.update(bootstrap_intervals(variant_detail_df, seed=config.seed))
        interval_rows.append(summary)

    interval_summary = pd.DataFrame(interval_rows).sort_values(by="variant")
    interval_summary.to_csv(output_dir / "post_interval_summary.csv", index=False)
    pd.DataFrame(best_variant_rows).to_csv(output_dir / "best_model_variant_results.csv", index=False)
    pd.concat(best_variant_details, ignore_index=True).to_csv(output_dir / "best_model_post_details.csv", index=False)

    metadata = {
        "best_model": best_model,
        "best_window": best_window,
        "placebo_start": str(PLACEBO_START),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "naive_baselines": [spec.name for spec in naive_specs],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
