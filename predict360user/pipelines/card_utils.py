from typing import Any, Dict, List

from metaflow import current
from metaflow.cards import Image, Markdown, Table


def render_experiment_card(
    model_name: str,
    dataset: str,
    epochs: int,
    h_window: int,
    run_records: List[Dict[str, Any]],
    all_loss_curves: List[List[float]],
    all_val_loss_curves: List[List[float]],
    all_eval_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_ep = max(len(c) for c in all_loss_curves) if all_loss_curves else 0
    loss_matrix = np.full((len(all_loss_curves), max_ep), np.nan)
    val_loss_matrix = np.full((len(all_val_loss_curves), max_ep), np.nan)

    for i, c in enumerate(all_loss_curves):
        loss_matrix[i, :len(c)] = c
    for i, c in enumerate(all_val_loss_curves):
        val_loss_matrix[i, :len(c)] = c

    mean_loss = np.nanmean(loss_matrix, axis=0) if max_ep > 0 else np.array([])
    std_loss = np.nanstd(loss_matrix, axis=0) if max_ep > 0 else np.array([])
    mean_val_loss = np.nanmean(val_loss_matrix, axis=0) if max_ep > 0 else np.array([])
    std_val_loss = np.nanstd(val_loss_matrix, axis=0) if max_ep > 0 else np.array([])

    aggregate_metrics = {
        "mean_test_err_all": float(np.mean([r["test_err_all"] for r in run_records])),
        "std_test_err_all": float(np.std([r["test_err_all"] for r in run_records])),
        "mean_test_err_low": float(np.mean([r["test_err_low"] for r in run_records])),
        "std_test_err_low": float(np.std([r["test_err_low"] for r in run_records])),
        "mean_test_err_medium": float(np.mean([r["test_err_medium"] for r in run_records])),
        "std_test_err_medium": float(np.std([r["test_err_medium"] for r in run_records])),
        "mean_test_err_high": float(np.mean([r["test_err_high"] for r in run_records])),
        "std_test_err_high": float(np.std([r["test_err_high"] for r in run_records])),
    }

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    epochs_arr = np.arange(1, max_ep + 1)

    for idx, curve in enumerate(all_loss_curves):
        ax[0].plot(
            np.arange(1, len(curve) + 1),
            curve,
            linestyle="--",
            alpha=0.5,
            label=f"Run {idx} (seed {run_records[idx]['seed']})",
        )
    if len(mean_loss) > 0:
        ax[0].plot(epochs_arr, mean_loss, color="navy", linewidth=2.5, label="Mean Loss")
        ax[0].fill_between(epochs_arr, mean_loss - std_loss, mean_loss + std_loss, color="blue", alpha=0.15)
    ax[0].set_title(f"Training Learning Curves ({model_name})")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, linestyle=":", alpha=0.6)
    ax[0].legend(loc="best")

    for idx, curve in enumerate(all_val_loss_curves):
        ax[1].plot(
            np.arange(1, len(curve) + 1),
            curve,
            linestyle="--",
            alpha=0.5,
            label=f"Run {idx} (seed {run_records[idx]['seed']})",
        )
    if len(mean_val_loss) > 0:
        ax[1].plot(epochs_arr, mean_val_loss, color="darkgreen", linewidth=2.5, label="Mean Val Loss")
        ax[1].fill_between(epochs_arr, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color="green", alpha=0.15)
    ax[1].set_title(f"Validation Learning Curves ({model_name})")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Validation Loss")
    ax[1].grid(True, linestyle=":", alpha=0.6)
    ax[1].legend(loc="best")
    plt.tight_layout()

    fig_hor, ax_hor = plt.subplots(figsize=(8, 5))
    classes = ["all", "low", "medium", "high"]
    colors = {"all": "black", "low": "blue", "medium": "orange", "high": "red"}

    for cls in classes:
        per_t_matrix = []
        for ev in all_eval_results:
            if cls in ev and "per_t" in ev[cls]:
                per_t_matrix.append(ev[cls]["per_t"])
        if per_t_matrix:
            arr = np.array(per_t_matrix)
            m = np.mean(arr, axis=0)
            s = np.std(arr, axis=0)
            t_steps = np.arange(len(m))
            ax_hor.plot(t_steps, m, label=f"Class: {cls}", color=colors.get(cls, "gray"), linewidth=2.0)
            ax_hor.fill_between(t_steps, m - s, m + s, color=colors.get(cls, "gray"), alpha=0.12)

    ax_hor.set_title(f"Prediction Error per Horizon Step Across {len(run_records)} Runs")
    ax_hor.set_xlabel("Prediction Horizon Step (t)")
    ax_hor.set_ylabel("Mean Orthodromic Error")
    ax_hor.grid(True, linestyle=":", alpha=0.6)
    ax_hor.legend(loc="best")
    plt.tight_layout()

    if current.card:
        current.card.append(Markdown(f"# Metaflow Experiment Card: {model_name.upper()}"))
        current.card.append(
            Markdown(
                f"**Dataset:** `{dataset}` | **Runs Configured:** `{len(run_records)}` | **Epochs:** `{epochs}` | **Horizon:** `{h_window}`"
            )
        )
        current.card.append(Markdown("## Learning Curves Across Runs"))
        current.card.append(Image.from_matplotlib(fig))
        current.card.append(Markdown("## Test Error Across Prediction Horizon"))
        current.card.append(Image.from_matplotlib(fig_hor))

        table_rows = []
        for r in run_records:
            table_rows.append(
                [
                    str(r["seed"]),
                    f"{r['final_loss']:.4f}",
                    f"{r['final_val_loss']:.4f}",
                    f"{r['test_err_all']:.4f}",
                    f"{r['test_err_low']:.4f}",
                    f"{r['test_err_medium']:.4f}",
                    f"{r['test_err_high']:.4f}",
                ]
            )
        table_rows.append(
            [
                "MEAN +/- STD",
                f"{float(np.mean([r['final_loss'] for r in run_records])):.4f} +/- {float(np.std([r['final_loss'] for r in run_records])):.4f}",
                f"{float(np.mean([r['final_val_loss'] for r in run_records])):.4f} +/- {float(np.std([r['final_val_loss'] for r in run_records])):.4f}",
                f"{aggregate_metrics['mean_test_err_all']:.4f} +/- {aggregate_metrics['std_test_err_all']:.4f}",
                f"{aggregate_metrics['mean_test_err_low']:.4f} +/- {aggregate_metrics['std_test_err_low']:.4f}",
                f"{aggregate_metrics['mean_test_err_medium']:.4f} +/- {aggregate_metrics['std_test_err_medium']:.4f}",
                f"{aggregate_metrics['mean_test_err_high']:.4f} +/- {aggregate_metrics['std_test_err_high']:.4f}",
            ]
        )

        current.card.append(Markdown("## Runs Summary"))
        current.card.append(
            Table(
                table_rows,
                headers=["Seed", "Loss", "Val Loss", "Test Err All", "Err Low", "Err Med", "Err High"],
            )
        )

    plt.close(fig)
    plt.close(fig_hor)
    return aggregate_metrics


def render_benchmark_card(
    dataset: str,
    benchmark_records: List[Dict[str, Any]],
    output_dir: str = "saved/figures",
) -> Dict[str, Any]:
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models = [rec["model"] for rec in benchmark_records]
    clusters = ["all", "low", "medium", "high"]

    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))
    x = np.arange(len(clusters))
    n_models = max(1, len(models))
    total_width = 0.8
    bar_width = total_width / n_models

    color_palette = plt.cm.tab10(np.linspace(0, 1, max(10, n_models)))

    for i, rec in enumerate(benchmark_records):
        model_name = rec["model"]
        metrics = rec.get("aggregate_metrics", {})
        means = [metrics.get(f"mean_test_err_{c}", 0.0) for c in clusters]
        stds = [metrics.get(f"std_test_err_{c}", 0.0) for c in clusters]
        offsets = x - (total_width / 2) + (i + 0.5) * bar_width
        bars = ax_cluster.bar(
            offsets,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=model_name,
            color=color_palette[i % len(color_palette)],
            alpha=0.85,
        )

    ax_cluster.set_xticks(x)
    ax_cluster.set_xticklabels([c.capitalize() for c in clusters], fontsize=11, fontweight="bold")
    ax_cluster.set_ylabel("Mean Orthodromic Error (rad)", fontsize=11)
    ax_cluster.set_title(f"Model Comparison Across Entropy Clusters ({dataset.capitalize()} Dataset)", fontsize=13, fontweight="bold")
    ax_cluster.legend(loc="upper right", frameon=True)
    ax_cluster.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    cluster_fig_path = os.path.join(output_dir, f"benchmark_{dataset}_error_per_cluster.png")
    fig_cluster.savefig(cluster_fig_path, dpi=200)

    fig_curves, ax_curves = plt.subplots(1, 2, figsize=(14, 5))
    for i, rec in enumerate(benchmark_records):
        model_name = rec["model"]
        color = color_palette[i % len(color_palette)]

        loss_curves = rec.get("all_loss_curves", [])
        if loss_curves:
            max_ep = max(len(c) for c in loss_curves)
            m = np.full((len(loss_curves), max_ep), np.nan)
            for r_idx, c in enumerate(loss_curves):
                m[r_idx, :len(c)] = c
            mean_loss = np.nanmean(m, axis=0)
            ax_curves[0].plot(np.arange(1, len(mean_loss) + 1), mean_loss, label=model_name, color=color, linewidth=2.0)

        val_curves = rec.get("all_val_loss_curves", [])
        if val_curves:
            max_ep = max(len(c) for c in val_curves)
            m_val = np.full((len(val_curves), max_ep), np.nan)
            for r_idx, c in enumerate(val_curves):
                m_val[r_idx, :len(c)] = c
            mean_val = np.nanmean(m_val, axis=0)
            ax_curves[1].plot(np.arange(1, len(mean_val) + 1), mean_val, label=model_name, color=color, linewidth=2.0)

    ax_curves[0].set_title("Training Loss Comparison")
    ax_curves[0].set_xlabel("Epoch")
    ax_curves[0].set_ylabel("Loss")
    ax_curves[0].grid(True, linestyle=":", alpha=0.6)
    ax_curves[0].legend(loc="best")

    ax_curves[1].set_title("Validation Loss Comparison")
    ax_curves[1].set_xlabel("Epoch")
    ax_curves[1].set_ylabel("Validation Loss")
    ax_curves[1].grid(True, linestyle=":", alpha=0.6)
    ax_curves[1].legend(loc="best")
    plt.tight_layout()

    curves_fig_path = os.path.join(output_dir, f"benchmark_{dataset}_learning_curves.png")
    fig_curves.savefig(curves_fig_path, dpi=200)

    fig_horizon, ax_horizon = plt.subplots(figsize=(10, 5))
    for i, rec in enumerate(benchmark_records):
        model_name = rec["model"]
        eval_results = rec.get("all_eval_results", [])
        per_t_matrix = []
        for ev in eval_results:
            if "all" in ev and "per_t" in ev["all"]:
                per_t_matrix.append(ev["all"]["per_t"])
        if per_t_matrix:
            arr = np.array(per_t_matrix)
            m = np.mean(arr, axis=0)
            s = np.std(arr, axis=0)
            t_steps = np.arange(len(m))
            color = color_palette[i % len(color_palette)]
            ax_horizon.plot(t_steps, m, label=model_name, color=color, linewidth=2.2)
            ax_horizon.fill_between(t_steps, m - s, m + s, color=color, alpha=0.1)

    ax_horizon.set_title(f"Prediction Error per Horizon Step Across Models ({dataset.capitalize()})")
    ax_horizon.set_xlabel("Prediction Horizon Step (t)")
    ax_horizon.set_ylabel("Mean Orthodromic Error (rad)")
    ax_horizon.grid(True, linestyle=":", alpha=0.6)
    ax_horizon.legend(loc="best")
    plt.tight_layout()

    horizon_fig_path = os.path.join(output_dir, f"benchmark_{dataset}_horizon_error.png")
    fig_horizon.savefig(horizon_fig_path, dpi=200)

    if current.card:
        current.card.append(Markdown(f"# Metaflow Benchmark Report: {dataset.upper()} Dataset"))
        current.card.append(Markdown(f"**Models Evaluated:** {', '.join(f'`{m}`' for m in models)}"))
        current.card.append(Markdown("## Error per Entropy Cluster Comparison"))
        current.card.append(Image.from_matplotlib(fig_cluster))
        current.card.append(Markdown("## Learning Curves Comparison"))
        current.card.append(Image.from_matplotlib(fig_curves))
        current.card.append(Markdown("## Prediction Horizon Error Comparison"))
        current.card.append(Image.from_matplotlib(fig_horizon))

        table_rows = []
        for rec in benchmark_records:
            m = rec["model"]
            met = rec.get("aggregate_metrics", {})
            table_rows.append(
                [
                    m,
                    f"{met.get('mean_test_err_all', 0.0):.4f} +/- {met.get('std_test_err_all', 0.0):.4f}",
                    f"{met.get('mean_test_err_low', 0.0):.4f} +/- {met.get('std_test_err_low', 0.0):.4f}",
                    f"{met.get('mean_test_err_medium', 0.0):.4f} +/- {met.get('std_test_err_medium', 0.0):.4f}",
                    f"{met.get('mean_test_err_high', 0.0):.4f} +/- {met.get('std_test_err_high', 0.0):.4f}",
                ]
            )

        current.card.append(Markdown("## Benchmark Summary Table"))
        current.card.append(
            Table(
                table_rows,
                headers=["Model", "Test Err (All)", "Test Err (Low)", "Test Err (Medium)", "Test Err (High)"],
            )
        )

    plt.close(fig_cluster)
    plt.close(fig_curves)
    plt.close(fig_horizon)

    return {
        "dataset": dataset,
        "models": models,
        "cluster_fig_path": cluster_fig_path,
        "curves_fig_path": curves_fig_path,
        "horizon_fig_path": horizon_fig_path,
    }

