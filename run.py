"""Run all nine data-mining stages in order."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from pipeline_steps.data_cleaning import clean_data
from pipeline_steps.data_integration import integrate_data
from pipeline_steps.data_mining import crosscheck_with_mlxtend, mine_data
from pipeline_steps.data_selection import select_data
from pipeline_steps.data_sources import load_data_sources
from pipeline_steps.data_transformation import transform_data
from pipeline_steps.knowledge_presentation import present_knowledge
from pipeline_steps.pattern_evaluation import evaluate_patterns
from pipeline_steps.useful_knowledge import create_useful_knowledge


def _parse_float_list(value: str, option_name: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be comma-separated numbers."
        ) from error
    if not values or any(not 0 < number <= 1 for number in values):
        raise argparse.ArgumentTypeError(
            f"Every {option_name} value must be greater than 0 and at most 1."
        )
    return values


def parse_arguments() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the full Online Retail Apriori vs. FP-Growth project."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "Online Retail.xlsx",
        help="Path to Online Retail.xlsx (default: next to run.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "results",
        help="Directory for CSV tables, charts, and the summary report.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter; by default all countries are analyzed.",
    )
    parser.add_argument(
        "--supports",
        default="0.05,0.03,0.02",
        help="Support thresholds for the full-dataset sensitivity experiment.",
    )
    parser.add_argument(
        "--fixed-support",
        type=float,
        default=0.03,
        help="Support threshold for the dataset-size experiment and final rules.",
    )
    parser.add_argument(
        "--sizes",
        default="0.25,0.50,0.75,1.00",
        help="Fractions of baskets for the dataset-size experiment.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum confidence for association rules.",
    )
    parser.add_argument(
        "--max-itemset-length",
        type=int,
        default=3,
        help="Maximum pattern length for runtime control; use 0 for no limit.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs per algorithm and experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to select reproducible dataset-size subsets.",
    )
    args = parser.parse_args()
    args.supports = _parse_float_list(args.supports, "--supports")
    args.sizes = _parse_float_list(args.sizes, "--sizes")

    if not 0 < args.fixed_support <= 1:
        parser.error("--fixed-support must be greater than 0 and at most 1.")
    if not 0 < args.min_confidence <= 1:
        parser.error("--min-confidence must be greater than 0 and at most 1.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if args.max_itemset_length < 0:
        parser.error("--max-itemset-length cannot be negative.")
    if 1.0 not in args.sizes:
        parser.error("--sizes must include 1.0 to produce full-dataset final rules.")
    args.max_itemset_length = args.max_itemset_length or None
    return args


def main() -> None:
    args = parse_arguments()

    data = load_data_sources(args.input)
    cleaned, cleaning_stats = clean_data(data)
    integrated = integrate_data(cleaned)
    selected = select_data(integrated, args.country)
    baskets, descriptions, top_products = transform_data(selected)

    performance, frequent_itemsets, mining_details = mine_data(
        baskets=baskets,
        supports=args.supports,
        fixed_support=args.fixed_support,
        dataset_sizes=args.sizes,
        repeats=args.repeats,
        seed=args.seed,
        max_length=args.max_itemset_length,
        min_confidence=args.min_confidence,
    )
    raw_rules = mining_details["baseline_rules"]
    if not isinstance(raw_rules, list):
        raise RuntimeError("Mining did not produce the baseline association rules.")

    mlxtend_check = crosscheck_with_mlxtend(
        baskets,
        args.fixed_support,
        args.max_itemset_length,
        frequent_itemsets,
    )
    frequent_frame, rules_frame, evaluation = evaluate_patterns(
        frequent_itemsets, raw_rules, descriptions
    )

    summary: dict[str, object] = {
        **cleaning_stats,
        "selected_rows": len(selected),
        "basket_count": len(baskets),
        "product_count": len(descriptions),
        "country_filter": args.country,
        "support_thresholds": args.supports,
        "fixed_support": args.fixed_support,
        "dataset_size_fractions": args.sizes,
        "min_confidence": args.min_confidence,
        "max_itemset_length": args.max_itemset_length,
        "repeats": args.repeats,
        "random_seed": args.seed,
        "performance_measurement_rows": len(performance),
        "package_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "pandas": version("pandas"),
            "openpyxl": version("openpyxl"),
            "matplotlib": version("matplotlib"),
            "mlxtend": version("mlxtend"),
        },
    }
    presentation_files = present_knowledge(
        args.output,
        performance,
        frequent_frame,
        rules_frame,
        top_products,
    )
    report_path = create_useful_knowledge(
        args.output,
        rules_frame,
        top_products,
        performance,
        evaluation,
        mlxtend_check,
        summary,
        presentation_files,
    )
    print(f"\nProject complete. Open {report_path} for the findings.")


if __name__ == "__main__":
    main()
