"""Shared data-cleaning step applied to every imported/seeded dataset."""
import pandas as pd

# Columns dropped on every import: constant (no variance) or pure identifiers.
DROP_COLUMNS = [
    "EmployeeCount",    # tất cả đều bằng 1
    "StandardHours",    # tất cả đều bằng 80
    "Over18",           # tất cả đều là 'Y'
    "EmployeeNumber",   # mã ID nhân viên, không có ý nghĩa phân loại
]


def clean_dataframe(df):
    """Clean a raw DataFrame in place-ish and return (cleaned_df, report).

    Steps: drop unnecessary columns, strip text junk, fill missing values.
    Rows are never merged (no dedup) so distinct records are preserved.
    """
    rows_before = len(df)

    # 1. Drop unnecessary columns (only those present).
    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)

    # 2. Handle junk: trim whitespace, treat blank strings as missing.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().replace("", pd.NA)

    # Drop rows that are entirely empty.
    df = df.dropna(how="all")

    # 3. Fill missing: numeric -> median, categorical -> mode.
    missing_filled = 0
    for col in df.columns:
        na_count = int(df[col].isna().sum())
        if not na_count:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")
        missing_filled += na_count

    report = {
        "dropped_columns": dropped,
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_after": len(df.columns),
        "missing_filled": missing_filled,
    }
    return df, report
