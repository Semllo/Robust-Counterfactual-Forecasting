from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
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
    run_multitarget_lstm_configuration,
    set_seed,
)
from run_robustness_extensions import SeasonalNaiveForecaster


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "manuscript_diagnostics"
PLACEBO_BOUNDARIES = [
    pd.Timestamp("2021-10-01 00:00:00"),
    pd.Timestamp("2021-11-01 00:00:00"),
    pd.Timestamp("2022-01-01 00:00:00"),
]
PLACEBO_HORIZON_DAYS = 60
ROLLING_CUTOFFS = [
    pd.Timestamp("2021-07-01 00:00:00"),
    pd.Timestamp("2021-09-01 00:00:00"),
    pd.Timestamp("2021-11-01 00:00:00"),
]
ROLLING_HORIZON_DAYS = 30


def make_config(seed: int = 42) -> RunConfig:
    return RunConfig(
        models=["ridge", "lightgbm", "lstm"],
        windows=[6, 24],
        lstm_epochs=4,
        lstm_hidden_dim=32,
        batch_size=256,
        learning_rate=1e-3,
        seed=seed,
        output_dir=OUTPUT_DIR,
    )


def summarize_metric_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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


def summarize_details(detail_df: pd.DataFrame) -> dict[str, float]:
    grouped = (
        detail_df.groupby(["district_name", "pollutant"], as_index=False)
        .agg(
            post_pip=("is_positive", lambda values: float(np.mean(values) * 100.0)),
            post_mean_residual=("residual", "mean"),
            post_median_residual=("residual", "median"),
        )
    )
    return {
        "mean_post_pip": float(grouped["post_pip"].mean()),
        "mean_post_residual": float(grouped["post_mean_residual"].mean()),
        "median_post_residual": float(grouped["post_median_residual"].median()),
        "n_runs": int(len(grouped)),
    }


def run_univariate_with_boundary(
    panel: pd.DataFrame,
    district_id: int,
    pollutant: str,
    model_factory: Callable[[], object],
    model_label: str,
    window: int,
    boundary: pd.Timestamp,
    variant: str,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    pre_mask = panel.index < boundary
    hist_col = f"{pollutant}_hist"
    exog_cols = get_variant_columns(variant)
    used_cols = [hist_col, *exog_cols]

    if panel.loc[:, pollutant].notna().sum() < 500:
        return None, pd.DataFrame()
    if panel.loc[pre_mask, pollutant].notna().sum() < 250:
        return None, pd.DataFrame()
    if panel.loc[~pre_mask, pollutant].notna().sum() < 250:
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

    cutoff_idx = int(pre_mask.sum())
    train_end_idx = max(int(cutoff_idx * 0.7), window + 50)
    if train_end_idx >= cutoff_idx:
        train_end_idx = cutoff_idx - 1

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
        end_idx=cutoff_idx,
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
        start_idx=cutoff_idx,
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
        "boundary": str(boundary),
        "pre_mae": validation_metrics["mae"],
        "post_pip": post_metrics["pip"],
        "post_mean_residual": post_metrics["mean_residual"],
        "post_median_residual": post_metrics["median_residual"],
        "n_validation": int(len(val_preds)),
        "n_post": int(len(post_preds)),
    }

    detail_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(post_timestamps),
            "district_id": district_id,
            "district_name": district_name,
            "pollutant": pollutant,
            "boundary": pd.Timestamp(boundary),
            "model": model_label,
            "window": window,
            "prediction": post_preds,
            "actual": post_actuals,
        }
    )
    detail_df["residual"] = detail_df["prediction"] - detail_df["actual"]
    detail_df["is_positive"] = (detail_df["prediction"] > detail_df["actual"]).astype(int)
    return metrics_row, detail_df


def direct_univariate_forecast(
    frame: pd.DataFrame,
    pollutant: str,
    exog_cols: list[str],
    window: int,
    start_idx: int,
    end_idx: int,
    model: object,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    hist_col = f"{pollutant}_hist"
    hist_values = frame.loc[:, hist_col].to_numpy(dtype=np.float64)
    exog_values = frame.loc[:, exog_cols].to_numpy(dtype=np.float64)
    actual_values = frame.loc[:, pollutant].to_numpy(dtype=np.float64)
    lower_bound, upper_bound = clip_bounds
    preds: list[float] = []
    actuals: list[float] = []

    for idx in range(start_idx, end_idx):
        if np.isnan(actual_values[idx]):
            continue
        window_target = hist_values[idx - window : idx].reshape(-1, 1)
        window_exog = exog_values[idx - window : idx]
        window_values = np.concatenate([window_target, window_exog], axis=1)
        if np.isnan(window_values).any():
            continue
        pred = float(model.predict(window_values))
        pred = float(np.clip(pred, lower_bound, upper_bound))
        preds.append(pred)
        actuals.append(float(actual_values[idx]))

    return np.asarray(preds, dtype=np.float32), np.asarray(actuals, dtype=np.float32)


def run_univariate_direct_configuration(
    panel: pd.DataFrame,
    district_id: int,
    pollutant: str,
    model_name: str,
    window: int,
    config: RunConfig,
) -> dict[str, object] | None:
    pre_mask = panel.index < POST_START
    hist_col = f"{pollutant}_hist"
    exog_cols = get_variant_columns("full")
    used_cols = [hist_col, *exog_cols]

    if panel.loc[:, pollutant].notna().sum() < 500:
        return None
    if panel.loc[pre_mask, pollutant].notna().sum() < 250 or panel.loc[~pre_mask, pollutant].notna().sum() < 250:
        return None
    if panel.loc[:, hist_col].isna().all():
        return None

    work_panel = clip_with_iqr(panel.copy(), used_cols, pre_mask)
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

    val_preds, val_actuals = direct_univariate_forecast(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        start_idx=train_end_idx,
        end_idx=pre_len,
        model=model,
        clip_bounds=clip_bounds,
    )
    post_preds, post_actuals = direct_univariate_forecast(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        start_idx=pre_len,
        end_idx=len(work_panel),
        model=model,
        clip_bounds=clip_bounds,
    )
    if len(val_preds) < 100 or len(post_preds) < 100:
        return None

    validation_metrics = evaluate_predictions(val_preds, val_actuals)
    post_metrics = evaluate_predictions(post_preds, post_actuals)
    return {
        "district_id": district_id,
        "district_name": DISTRICTS[district_id][0],
        "pollutant": pollutant,
        "model": model_name,
        "window": window,
        "evaluation": "direct",
        "pre_mae": validation_metrics["mae"],
        "post_pip": post_metrics["pip"],
        "post_mean_residual": post_metrics["mean_residual"],
        "post_median_residual": post_metrics["median_residual"],
    }


def run_rolling_origin_slice(
    panel: pd.DataFrame,
    district_id: int,
    pollutant: str,
    model_factory: Callable[[], object],
    model_label: str,
    window: int,
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> dict[str, object] | None:
    train_mask = panel.index < cutoff
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)
    hist_col = f"{pollutant}_hist"
    exog_cols = get_variant_columns("full")
    used_cols = [hist_col, *exog_cols]

    if panel.loc[train_mask, pollutant].notna().sum() < 250:
        return None
    if panel.loc[(panel.index >= cutoff) & (panel.index < horizon_end), pollutant].notna().sum() < 100:
        return None
    if panel.loc[:, hist_col].isna().all():
        return None

    work_panel = clip_with_iqr(panel.copy(), used_cols, train_mask)
    target_train = work_panel.loc[train_mask, hist_col].dropna()
    if target_train.empty:
        return None
    clip_bounds = (max(0.0, float(target_train.min())), float(target_train.max()))

    cutoff_idx = int(train_mask.sum())
    end_idx = int(panel.index.searchsorted(horizon_end))
    x_train, y_train = make_training_sequences(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        train_end_idx=cutoff_idx,
    )
    if len(x_train) < 100:
        return None

    model = model_factory()
    model.fit(x_train=x_train, y_train=y_train)
    preds, actuals, _ = recursive_forecast(
        frame=work_panel,
        pollutant=pollutant,
        exog_cols=exog_cols,
        window=window,
        start_idx=cutoff_idx,
        end_idx=end_idx,
        model=model,
        clip_bounds=clip_bounds,
    )
    if len(preds) < 100:
        return None

    metrics = evaluate_predictions(preds, actuals)
    return {
        "district_id": district_id,
        "district_name": DISTRICTS[district_id][0],
        "pollutant": pollutant,
        "model": model_label,
        "window": window,
        "cutoff": str(cutoff),
        "horizon_days": horizon_days,
        "pre_mae": metrics["mae"],
        "post_pip": metrics["pip"],
        "post_mean_residual": metrics["mean_residual"],
        "post_median_residual": metrics["median_residual"],
    }


def run_fixed_horizon_placebos(panels: dict[int, pd.DataFrame], config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for boundary in PLACEBO_BOUNDARIES:
        boundary_label = "policy_2022_01" if boundary == POST_START else f"placebo_{boundary:%Y_%m}"
        for district_id, panel in panels.items():
            for pollutant in POLLUTANTS:
                row, detail_df = run_univariate_with_boundary(
                    panel=panel,
                    district_id=district_id,
                    pollutant=pollutant,
                    model_factory=lambda: build_model("lightgbm", input_dim=1, config=config),
                    model_label="lightgbm",
                    window=6,
                    boundary=boundary,
                    variant="full",
                )
                if row is None or detail_df.empty:
                    continue
                horizon_end = boundary + pd.Timedelta(days=PLACEBO_HORIZON_DAYS)
                detail_df = detail_df[detail_df["timestamp"] < horizon_end].copy()
                if len(detail_df) < 24:
                    continue
                grouped_metrics = summarize_details(detail_df)
                row["post_pip"] = grouped_metrics["mean_post_pip"]
                row["post_mean_residual"] = grouped_metrics["mean_post_residual"]
                row["post_median_residual"] = grouped_metrics["median_post_residual"]
                row["n_post_groups"] = grouped_metrics["n_runs"]
                row["boundary_label"] = boundary_label
                rows.append(row)
                detail_df["boundary_label"] = boundary_label
                details.append(detail_df)

    rows_df = pd.DataFrame(rows)
    details_df = pd.concat(details, ignore_index=True)
    summary_df = summarize_metric_rows(rows_df, ["boundary_label"])
    label_to_boundary = rows_df.groupby("boundary_label", as_index=False)["boundary"].first()
    summary_df = summary_df.merge(label_to_boundary, on="boundary_label", how="left")
    summary_df["boundary"] = pd.to_datetime(summary_df["boundary"])
    summary_df = summary_df.sort_values(by="boundary")
    summary_df.to_csv(OUTPUT_DIR / "placebo_fixed_horizon_summary.csv", index=False)
    rows_df.to_csv(OUTPUT_DIR / "placebo_fixed_horizon_results.csv", index=False)
    details_df.to_csv(OUTPUT_DIR / "placebo_fixed_horizon_details.csv", index=False)
    return summary_df, details_df


def run_rolling_origin_sensitivity(panels: dict[int, pd.DataFrame], config: RunConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("lightgbm", 6, lambda: build_model("lightgbm", input_dim=1, config=config)),
        ("lightgbm", 12, lambda: build_model("lightgbm", input_dim=1, config=config)),
        ("lightgbm", 24, lambda: build_model("lightgbm", input_dim=1, config=config)),
        ("seasonal_naive", 24, lambda: SeasonalNaiveForecaster(lag_steps=24)),
    ]
    for cutoff in ROLLING_CUTOFFS:
        for model_label, window, model_factory in specs:
            for district_id, panel in panels.items():
                for pollutant in POLLUTANTS:
                    row = run_rolling_origin_slice(
                        panel=panel,
                        district_id=district_id,
                        pollutant=pollutant,
                        model_factory=model_factory,
                        model_label=model_label,
                        window=window,
                        cutoff=cutoff,
                        horizon_days=ROLLING_HORIZON_DAYS,
                    )
                    if row is not None:
                        rows.append(row)

    rows_df = pd.DataFrame(rows)
    summary_df = summarize_metric_rows(rows_df, ["model", "window"])
    summary_df.to_csv(OUTPUT_DIR / "rolling_origin_summary.csv", index=False)
    rows_df.to_csv(OUTPUT_DIR / "rolling_origin_results.csv", index=False)
    return summary_df


def run_direct_evaluation_sensitivity(panels: dict[int, pd.DataFrame], config: RunConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for district_id, panel in panels.items():
        for model_name in ["ridge", "lightgbm"]:
            for window in [6, 24]:
                for pollutant in POLLUTANTS:
                    row = run_univariate_direct_configuration(
                        panel=panel,
                        district_id=district_id,
                        pollutant=pollutant,
                        model_name=model_name,
                        window=window,
                        config=config,
                    )
                    if row is not None:
                        rows.append(row)

        for window in [6, 24]:
            rows.extend(
                run_multitarget_lstm_configuration(
                    panel=panel,
                    district_id=district_id,
                    window=window,
                    variant="full",
                    config=config,
                )
            )

    rows_df = pd.DataFrame(rows)
    summary_df = summarize_metric_rows(rows_df, ["model", "window"])
    summary_df.to_csv(OUTPUT_DIR / "direct_evaluation_summary.csv", index=False)
    rows_df.to_csv(OUTPUT_DIR / "direct_evaluation_results.csv", index=False)
    return summary_df


def make_diagnostic_figures(placebo_summary: pd.DataFrame) -> None:
    main_summary = pd.read_csv(ROOT / "experiment_outputs_full_origlike_lstm" / "model_selection_summary.csv")
    naive_summary = pd.read_csv(ROOT / "robustness_extensions" / "naive_baseline_summary.csv")
    naive_summary = naive_summary.rename(columns={"model": "model_name"})
    main_summary = main_summary.rename(columns={"model": "model_name"})
    credibility = pd.concat(
        [
            main_summary[["model_name", "window", "mean_pre_mae", "mean_post_residual"]],
            naive_summary[["model_name", "window", "mean_pre_mae", "mean_post_residual"]],
        ],
        ignore_index=True,
    )
    credibility["label"] = credibility.apply(
        lambda row: (
            "Seasonal naive"
            if row["model_name"] == "seasonal_naive_24h"
            else "Persistence"
            if row["model_name"] == "persistence_1h"
            else f"{str(row['model_name']).capitalize()} w={int(row['window'])}"
        ),
        axis=1,
    )

    placebo_plot = placebo_summary.copy()
    placebo_plot["short_label"] = placebo_plot["boundary"].dt.strftime("%Y-%m")
    colors = ["#b0b0b0", "#7a7a7a", "#1f77b4"]
    x = np.arange(len(placebo_plot))
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(x, placebo_plot["mean_post_residual"], color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(placebo_plot["short_label"])
    ax.set_ylabel("Mean residual")
    ax.set_title("60-day policy vs. placebo boundaries")
    ax2 = ax.twinx()
    ax2.plot(x, placebo_plot["mean_post_pip"], color="#d62728", marker="o", linewidth=2)
    ax2.set_ylabel("Mean PIP (%)")
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=min(placebo_plot["mean_post_pip"]) - 2.5, top=max(placebo_plot["mean_post_pip"]) + 3.0)
    for idx, row in placebo_plot.iterrows():
        ax.text(idx, row["mean_post_residual"] + 0.22, f"{row['mean_post_residual']:.2f}", ha="center", va="bottom", fontsize=9)
        ax2.text(idx, row["mean_post_pip"] + 0.45, f"{row['mean_post_pip']:.1f}", ha="center", va="bottom", fontsize=9, color="#d62728")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#7a7a7a", linewidth=8, alpha=0.85, label="Mean residual"),
            plt.Line2D([0], [0], color="#d62728", marker="o", linewidth=2, label="Mean PIP"),
        ],
        loc="upper left",
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "placebo_boundary_comparison.png", dpi=200)
    plt.close(fig)

    family_colors = {
        "lightgbm": "#1f77b4",
        "ridge": "#ff7f0e",
        "lstm": "#2ca02c",
        "seasonal_naive_24h": "#9467bd",
        "persistence_1h": "#8c564b",
    }
    window_markers = {1: "D", 6: "o", 24: "s"}
    display_names = {
        "lightgbm": "LightGBM",
        "ridge": "Ridge",
        "lstm": "LSTM",
        "seasonal_naive_24h": "Seasonal naive",
        "persistence_1h": "Persistence",
    }
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for _, row in credibility.iterrows():
        ax.scatter(
            row["mean_pre_mae"],
            row["mean_post_residual"],
            s=80,
            color=family_colors[str(row["model_name"])],
            marker=window_markers.get(int(row["window"]), "o"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xlabel("Mean pre-intervention MAE")
    ax.set_ylabel("Mean post residual")
    ax.set_title("Predictive credibility vs. post gap")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.2)

    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=display_names[key],
            markerfacecolor=value,
            markeredgecolor="black",
            markersize=8,
        )
        for key, value in family_colors.items()
    ]
    window_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            label=f"w={window}",
            linestyle="None",
            markersize=7,
        )
        for window, marker in window_markers.items()
    ]
    family_legend = ax.legend(handles=family_handles, loc="upper left", frameon=False, title="Model")
    ax.add_artist(family_legend)
    ax.legend(handles=window_handles, loc="lower right", frameon=False, title="Window")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "credibility_tradeoff.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = make_config(seed=42)
    set_seed(config.seed)
    panels = build_district_panels()

    placebo_summary, _ = run_fixed_horizon_placebos(panels, config)
    rolling_summary = run_rolling_origin_sensitivity(panels, config)
    direct_summary = run_direct_evaluation_sensitivity(panels, config)
    make_diagnostic_figures(placebo_summary)

    metadata = {
        "placebo_boundaries": [str(boundary) for boundary in PLACEBO_BOUNDARIES],
        "placebo_horizon_days": PLACEBO_HORIZON_DAYS,
        "rolling_cutoffs": [str(cutoff) for cutoff in ROLLING_CUTOFFS],
        "rolling_horizon_days": ROLLING_HORIZON_DAYS,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Placebo summary")
    print(placebo_summary.to_string(index=False))
    print("\nRolling-origin summary")
    print(rolling_summary.to_string(index=False))
    print("\nDirect evaluation summary")
    print(direct_summary.to_string(index=False))


if __name__ == "__main__":
    main()
