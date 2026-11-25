"""Stage 2: clean invalid, cancelled, and duplicate retail records."""

import pandas as pd


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove invalid retail records while retaining valid baskets."""
    cleaned = data.copy()
    original_rows = len(cleaned)
    missing_customer_ids = int(cleaned["CustomerID"].isna().sum())

    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype("string").str.strip()
    cleaned["StockCode"] = cleaned["StockCode"].astype("string").str.strip()
    cleaned["Description"] = cleaned["Description"].astype("string").str.strip()
    cleaned["Country"] = cleaned["Country"].astype("string").str.strip()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    required = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "Country",
    ]
    missing_required = int(cleaned[required].isna().any(axis=1).sum())
    cleaned = cleaned.dropna(subset=required)
    nonblank = (
        cleaned["InvoiceNo"].ne("")
        & cleaned["StockCode"].ne("")
        & cleaned["Description"].ne("")
        & cleaned["Country"].ne("")
    )
    missing_required += int((~nonblank).sum())
    cleaned = cleaned.loc[nonblank].copy()

    cancellation_mask = cleaned["InvoiceNo"].str.upper().str.startswith("C")
    cancellation_rows = int(cancellation_mask.sum())
    cleaned = cleaned.loc[~cancellation_mask].copy()

    invalid_value_mask = (cleaned["Quantity"] <= 0) | (cleaned["UnitPrice"] <= 0)
    nonpositive_rows = int(invalid_value_mask.sum())
    cleaned = cleaned.loc[~invalid_value_mask].copy()

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    stats = {
        "raw_rows": original_rows,
        "missing_customer_ids_preserved": missing_customer_ids,
        "missing_or_blank_required_rows_removed": missing_required,
        "cancelled_invoice_rows_removed": cancellation_rows,
        "nonpositive_quantity_or_price_rows_removed": nonpositive_rows,
        "duplicate_rows_removed": duplicate_rows,
        "cleaned_rows": len(cleaned),
    }
    print(
        "[2/9] Data Cleaning: "
        f"{len(cleaned):,} valid rows; "
        f"{cancellation_rows:,} cancellations and "
        f"{nonpositive_rows:,} nonpositive rows removed"
    )
    return cleaned, stats
