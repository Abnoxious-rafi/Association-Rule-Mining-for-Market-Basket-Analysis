"""Stage 1: load and validate the source dataset."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_data_sources(input_path: Path) -> pd.DataFrame:
    """Load and validate the Excel source."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    data = pd.read_excel(input_path, engine="openpyxl")
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"[1/9] Data Sources: loaded {len(data):,} rows from {input_path.name}")
    return data
