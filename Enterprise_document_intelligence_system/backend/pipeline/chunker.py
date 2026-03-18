# ============================================================
# backend/pipeline/chunker.py
# Converts CLT CSV rows → natural-language chunks for embedding.
# ============================================================

import pandas as pd
from backend.pipeline import Chunk
from backend.config import TENURE_BAND_SIZE

DIMENSION_COLS = [
    "project_type", "project_sub_type", "report_type",
    "report_sub_type", "customer_type", "product",
    "channel", "history_pulled", "reporting_month",
    "date_inserted", "max_tenure_adj_prediction"
]


def chunk_csv_by_tenure(
    df: pd.DataFrame,
    filepath: str,
    tenure_band_size: int = TENURE_BAND_SIZE
) -> list[Chunk]:
    """
    Groups rows by cohort dimensions, then slices each cohort into
    tenure bands of `tenure_band_size` rows and renders each band
    as a natural-language paragraph.

    Key design choices:
    - Tenure range appears FIRST in the chunk text so it dominates
      the embedding signal (FIX 4 from original debugging).
    - Band size = 10 keeps chunks specific enough for precise
      retrieval on tenure-based queries (FIX 5).
    """
    group_cols = [c for c in DIMENSION_COLS if c in df.columns]

    chunks      = []
    chunk_index = 0

    for _, group_df in df.groupby(group_cols, sort=False):
        group_df  = group_df.sort_values("tenure").reset_index(drop=True)
        first_row = group_df.iloc[0]

        cohort_tag = (
            f"{first_row['project_type']} {first_row['project_sub_type']} | "
            f"{first_row['report_sub_type']} | "
            f"Customer: {first_row['customer_type']} | "
            f"Product: {first_row['product']} | "
            f"Channel: {first_row['channel']} | "
            f"Month: {first_row['reporting_month']} | "
            f"Adjusted: {'Yes' if first_row['max_tenure_adj_prediction'] == 1 else 'No'}"
        )

        total_rows = len(group_df)
        band_start = 0

        while band_start < total_rows:
            band_end   = min(band_start + tenure_band_size, total_rows)
            band_df    = group_df.iloc[band_start:band_end]
            tenure_min = int(band_df["tenure"].iloc[0])
            tenure_max = int(band_df["tenure"].iloc[-1])

            row_sentences = []
            for _, row in band_df.iterrows():
                sentence = (
                    f"Tenure {int(row['tenure'])}: "
                    f"active customers = {row['total_customer']:.2f}, "
                    f"total churn count = {row['total_churn']:.2f}, "
                    f"churn rate = {row['churn'] * 100:.4f}%, "
                    f"survival rate = {row['survival'] * 100:.4f}%"
                )
                if row["new_customer"] > 0:
                    sentence += f", new customers (gross adds) = {int(row['new_customer'])}"
                row_sentences.append(sentence + ".")

            chunk_text = (
                f"Tenures {tenure_min} to {tenure_max} | {cohort_tag}:\n"
                + " ".join(row_sentences)
            )
            source_tag = (
                f"{filepath} | "
                f"{first_row['project_type']}|{first_row['product']}|"
                f"{first_row['channel']}|{first_row['customer_type']}|"
                f"tenures {tenure_min}-{tenure_max}"
            )

            chunks.append(Chunk(
                text=chunk_text,
                source=source_tag,
                chunk_index=chunk_index
            ))
            chunk_index += 1
            band_start   = band_end

    print(f"[Chunker] Generated {len(chunks)} chunks from CSV.")
    return chunks
