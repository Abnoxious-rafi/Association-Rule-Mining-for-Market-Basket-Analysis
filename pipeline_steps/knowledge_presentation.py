"""Stage 8: save result tables and visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def present_knowledge(
    output_dir: Path,
    performance: pd.DataFrame,
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
    top_products: pd.DataFrame,
) -> dict[str, str]:
    """Save analysis tables and comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(output_dir / "performance_comparison.csv", index=False)
    frequent_itemsets.to_csv(output_dir / "frequent_itemsets.csv", index=False)
    rules.to_csv(output_dir / "association_rules.csv", index=False)
    top_products.to_csv(output_dir / "top_products.csv", index=False)

    support_results = performance.loc[
        performance["setting"].eq("support_sensitivity")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for algorithm, group in support_results.groupby("algorithm"):
        group = group.sort_values("support", ascending=False)
        axes[0].plot(
            group["support"],
            group["mean_run_seconds"],
            marker="o",
            label=algorithm,
        )
        axes[1].plot(
            group["support"],
            group["mean_peak_memory_mb"],
            marker="o",
            label=algorithm,
        )
    axes[0].set(title="Runtime vs. minimum support", xlabel="Minimum support")
    axes[0].set_ylabel("Mean runtime (seconds)")
    axes[1].set(title="Peak memory vs. minimum support", xlabel="Minimum support")
    axes[1].set_ylabel("Mean Python peak memory (MB)")
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    support_chart = output_dir / "support_sensitivity.png"
    figure.savefig(support_chart, dpi=160)
    plt.close(figure)

    size_results = performance.loc[performance["setting"].eq("dataset_size")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for algorithm, group in size_results.groupby("algorithm"):
        group = group.sort_values("dataset_pct")
        axes[0].plot(
            group["dataset_pct"],
            group["mean_run_seconds"],
            marker="o",
            label=algorithm,
        )
        axes[1].plot(
            group["dataset_pct"],
            group["mean_peak_memory_mb"],
            marker="o",
            label=algorithm,
        )
    axes[0].set(title="Runtime vs. dataset size", xlabel="Baskets used (%)")
    axes[0].set_ylabel("Mean runtime (seconds)")
    axes[1].set(title="Peak memory vs. dataset size", xlabel="Baskets used (%)")
    axes[1].set_ylabel("Mean Python peak memory (MB)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    size_chart = output_dir / "dataset_scalability.png"
    figure.savefig(size_chart, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    top = top_products.head(10).sort_values("basket_count")
    labels = top["stock_code"].astype(str) + " | " + top["description"].astype(str)
    axis.barh(labels, top["basket_count"], color="#4472C4")
    axis.set(title="Top products by number of baskets", xlabel="Baskets")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    products_chart = output_dir / "top_products.png"
    figure.savefig(products_chart, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    if rules.empty:
        axis.text(0.5, 0.5, "No rules met the configured thresholds", ha="center")
    else:
        points = axis.scatter(
            rules["support"],
            rules["confidence"],
            c=rules["lift"],
            cmap="viridis",
            alpha=0.75,
        )
        figure.colorbar(points, ax=axis, label="Lift")
    axis.set(
        title="Association-rule quality",
        xlabel="Support",
        ylabel="Confidence",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    rules_chart = output_dir / "rule_quality.png"
    figure.savefig(rules_chart, dpi=160)
    plt.close(figure)

    files = {
        "performance": str(output_dir / "performance_comparison.csv"),
        "frequent_itemsets": str(output_dir / "frequent_itemsets.csv"),
        "association_rules": str(output_dir / "association_rules.csv"),
        "top_products": str(output_dir / "top_products.csv"),
        "support_sensitivity_chart": str(support_chart),
        "dataset_scalability_chart": str(size_chart),
        "top_products_chart": str(products_chart),
        "rule_quality_chart": str(rules_chart),
    }
    print(f"[8/9] Knowledge Presentation: saved tables and charts to {output_dir}")
    return files
