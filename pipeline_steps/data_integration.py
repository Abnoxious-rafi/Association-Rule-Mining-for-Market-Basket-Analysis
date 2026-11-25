"""Stage 3: join invoice-level details back to retail line items."""

import pandas as pd


def integrate_data(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Join invoice-level attributes to the product-line fact table."""

    def first_present(values: pd.Series) -> object:
        present = values.dropna()
        return present.iloc[0] if not present.empty else pd.NA

    invoice_dimension = (
        cleaned.groupby("InvoiceNo", sort=False, as_index=False)
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            Country=("Country", "first"),
            CustomerID=("CustomerID", first_present),
        )
    )
    line_items = cleaned[
        ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"]
    ].copy()
    line_items["LineRevenue"] = line_items["Quantity"] * line_items["UnitPrice"]

    integrated = line_items.merge(
        invoice_dimension,
        on="InvoiceNo",
        how="inner",
        validate="many_to_one",
    )
    if len(integrated) != len(cleaned):
        raise RuntimeError("Invoice integration changed the product-line row count.")

    print(
        "[3/9] Data Integration: joined product lines to "
        f"{len(invoice_dimension):,} invoice records"
    )
    return integrated
