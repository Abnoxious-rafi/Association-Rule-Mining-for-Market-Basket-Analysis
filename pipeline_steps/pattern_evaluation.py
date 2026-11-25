"""Stage 7: evaluate frequent patterns and association rules."""

import pandas as pd

from pipeline_steps.data_mining import Itemsets


def evaluate_patterns(
    itemsets: Itemsets,
    raw_rules: list[dict[str, object]],
    descriptions: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Format patterns and summarize support, confidence, and lift."""

    def display_item(item: str) -> str:
        return f"{item} | {descriptions.get(item, 'Unknown product')}"

    frequent_rows = [
        {
            "itemset_size": len(itemset),
            "items": " ; ".join(display_item(item) for item in sorted(itemset)),
            "support": support,
        }
        for itemset, support in itemsets.items()
    ]
    frequent_frame = pd.DataFrame(frequent_rows)
    if frequent_frame.empty:
        frequent_frame = pd.DataFrame(
            columns=["itemset_size", "items", "support"]
        )
    if not frequent_frame.empty:
        frequent_frame = frequent_frame.sort_values(
            ["itemset_size", "support"], ascending=[True, False]
        )

    rule_rows = [
        {
            "antecedents": " ; ".join(
                display_item(item) for item in sorted(rule["antecedents"])
            ),
            "consequents": " ; ".join(
                display_item(item) for item in sorted(rule["consequents"])
            ),
            "support": rule["support"],
            "confidence": rule["confidence"],
            "lift": rule["lift"],
        }
        for rule in raw_rules
    ]
    rules_frame = pd.DataFrame(
        rule_rows,
        columns=["antecedents", "consequents", "support", "confidence", "lift"],
    )
    if not rules_frame.empty:
        rules_frame = rules_frame.sort_values(
            ["lift", "confidence", "support"], ascending=False
        )

    evaluation = {
        "frequent_itemset_count": len(frequent_frame),
        "association_rule_count": len(rules_frame),
        "positive_lift_rule_count": int((rules_frame["lift"] > 1).sum())
        if not rules_frame.empty
        else 0,
        "maximum_rule_lift": float(rules_frame["lift"].max())
        if not rules_frame.empty
        else None,
    }
    print(
        "[7/9] Pattern Evaluation: "
        f"{len(frequent_frame):,} frequent itemsets and "
        f"{len(rules_frame):,} rules (confidence threshold and lift > 1)"
    )
    return frequent_frame, rules_frame, evaluation
