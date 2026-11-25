"""Stage 5: transform product lines into invoice baskets."""

from collections import Counter

import pandas as pd


Transaction = frozenset[str]


def transform_data(
    selected: pd.DataFrame,
) -> tuple[list[Transaction], dict[str, str], pd.DataFrame]:
    """Convert invoice lines into unique-item baskets and product summaries."""
    grouped = selected.groupby("InvoiceNo", sort=False)
    baskets = [
        frozenset(group["StockCode"].astype(str))
        for _, group in grouped
    ]
    if not baskets:
        raise ValueError("No invoice baskets remain after data selection.")

    descriptions = (
        selected.dropna(subset=["Description"])
        .drop_duplicates(subset=["StockCode"])
        .set_index("StockCode")["Description"]
        .astype(str)
        .to_dict()
    )
    basket_counts = Counter(item for basket in baskets for item in basket)
    quantities = selected.groupby("StockCode")["Quantity"].sum()
    product_rows = [
        {
            "stock_code": code,
            "description": descriptions.get(code, "Unknown product"),
            "basket_count": count,
            "transaction_support": count / len(baskets),
            "quantity_sold": float(quantities.get(code, 0)),
        }
        for code, count in basket_counts.items()
    ]
    top_products = pd.DataFrame(product_rows).sort_values(
        ["basket_count", "stock_code"], ascending=[False, True]
    )

    print(
        "[5/9] Data Transformation: "
        f"{len(baskets):,} baskets and "
        f"{len(basket_counts):,} distinct products (presence/absence)"
    )
    return baskets, descriptions, top_products
