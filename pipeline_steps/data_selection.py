"""Stage 4: select the population used for the experiments."""

import pandas as pd


def select_data(data: pd.DataFrame, country: str | None) -> pd.DataFrame:
    """Select all countries or one requested country."""
    if country:
        country_mask = data["Country"].str.casefold().eq(country.strip().casefold())
        selected = data.loc[country_mask].copy()
        if selected.empty:
            available = ", ".join(sorted(data["Country"].dropna().unique()))
            raise ValueError(
                f"No rows found for country {country!r}. Available countries: {available}"
            )
    else:
        selected = data.copy()

    print(
        "[4/9] Data Selection: "
        f"{len(selected):,} product lines from "
        f"{selected['InvoiceNo'].nunique():,} invoices"
        + (f" in {country}" if country else " across all countries")
    )
    return selected
