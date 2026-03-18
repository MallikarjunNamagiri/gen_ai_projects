# ============================================================
# backend/pipeline/loader.py
# ============================================================

import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Loads the CLT CSV with light normalisation:
    - Strips whitespace from all string columns
    - Sorts by tenure ascending so band slicing is correct
    """
    df = pd.read_csv(filepath)
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    df = df.sort_values("tenure").reset_index(drop=True)
    return df
