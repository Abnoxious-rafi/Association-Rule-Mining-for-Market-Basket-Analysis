"""Stage 9: write findings and reproducibility details."""

import json
from pathlib import Path

import pandas as pd


def create_useful_knowledge(
    output_dir: Path,
    rules: pd.DataFrame,
    top_products: pd.DataFrame,
    performance: pd.DataFrame,
    evaluation: dict[str, object],
    mlxtend_check: dict[str, int],
    summary: dict[str, object],
    presentation_files: dict[str, str],
) -> Path:
    """Write concise, evidence-based findings and practical interpretation."""
    lines = [
        "ONLINE RETAIL MARKET BASKET ANALYSIS",
        "",
        "Dataset and preparation",
        f"- Raw rows: {summary['raw_rows']:,}",
        f"- Cleaned product lines: {summary['cleaned_rows']:,}",
        f"- Baskets analyzed: {summary['basket_count']:,}",
        f"- Distinct products: {summary['product_count']:,}",
        f"- Missing customer IDs retained: "
        f"{summary['missing_customer_ids_preserved']:,}",
        "",
        "Method and verification",
        "- Apriori and FP-Growth were run on the same baskets and thresholds.",
        "- Both algorithms' frequent itemsets were required to match exactly.",
        "- MLxtend Apriori and FP-Growth independently matched the full-data baseline.",
        f"- MLxtend cross-check itemsets: {mlxtend_check}",
        f"- Rule evaluation: confidence >= {summary['min_confidence']:.3f}, lift > 1.",
        "- Timing is miner-only; memory is Python peak allocation measured by tracemalloc.",
        "",
        "Pattern evaluation",
        f"- Frequent itemsets: {evaluation['frequent_itemset_count']:,}",
        f"- Association rules passing the filters: "
        f"{evaluation['association_rule_count']:,}",
        f"- Highest rule lift: {evaluation['maximum_rule_lift']}",
        "",
        "Performance comparison (mean measurements)",
    ]
    for setting, group in performance.groupby("setting", sort=True):
        lines.append(f"- {setting.replace('_', ' ').title()}:")
        for _, row in group.iterrows():
            lines.append(
                "  "
                f"{row['algorithm']}, support={row['support']:.3f}, "
                f"dataset={row['dataset_pct']:.0f}%: "
                f"{row['mean_run_seconds']:.3f}s, "
                f"{row['mean_peak_memory_mb']:.2f}MB, "
                f"{int(row['frequent_itemsets']):,} itemsets, "
                f"{int(row['association_rules']):,} rules"
            )

    lines.extend(["", "Most frequent products by basket"])
    for _, row in top_products.head(5).iterrows():
        lines.append(
            f"- {row['stock_code']} | {row['description']}: "
            f"{int(row['basket_count']):,} baskets "
            f"({row['transaction_support']:.1%} of baskets)"
        )

    lines.extend(["", "Strongest positive associations by lift"])
    if rules.empty:
        lines.append("- No rules passed the configured confidence and lift filters.")
    else:
        for _, rule in rules.head(10).iterrows():
            lines.append(
                f"- If a basket contains [{rule['antecedents']}], it also contains "
                f"[{rule['consequents']}] in {rule['confidence']:.1%} of cases "
                f"(support={rule['support']:.3f}, lift={rule['lift']:.3f})."
            )
        lines.extend(
            [
                "",
                "Potential use",
                "- Consider testing the strongest co-purchases for product placement, "
                "bundles, or cross-promotions.",
                "- These rules show co-occurrence, not causation; validate any retail "
                "action with a controlled experiment.",
            ]
        )

    report_path = output_dir / "useful_knowledge.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = output_dir / "pipeline_summary.json"
    summary["evaluation"] = evaluation
    summary["mlxtend_crosscheck_itemsets"] = mlxtend_check
    summary["output_files"] = {
        **presentation_files,
        "useful_knowledge": str(report_path),
        "pipeline_summary": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"[9/9] Useful Knowledge: wrote findings to {report_path}")
    return report_path
