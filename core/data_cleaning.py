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
    """Clean a raw DataFrame and return (cleaned_df, report).

    Ordered stages, each recorded in report["steps"]:
    drop dead cols -> coerce dtypes -> standardize text -> drop near-constant
    -> dedup rows -> IQR outlier cap -> fill missing.
    Legacy report keys are preserved for the import view and seed command.
    """
    rows_before = len(df)
    steps = []

    # 1. Drop configured dead columns.
    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
    steps.append({"step": "drop_dead", "detail": "Bỏ cột định danh/hằng số",
                  "columns": dropped})

    # 2. Coerce object columns that are fully numeric-looking -> numeric.
    coerced = []
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum() and df[col].notna().any():
            df[col] = converted
            coerced.append(col)
    steps.append({"step": "coerce_dtypes", "detail": "Ép kiểu cột số",
                  "columns": coerced})

    # 3. Standardize categorical text: trim, collapse whitespace, blank -> NA.
    for col in df.select_dtypes(include="object").columns:
        df[col] = (df[col].str.strip()
                          .str.replace(r"\s+", " ", regex=True)
                          .replace("", pd.NA))
    df = df.dropna(how="all")
    steps.append({"step": "standardize_text",
                  "detail": "Chuẩn hóa khoảng trắng cột chữ"})

    # 4. Drop near-constant columns (nunique <= 1, or one value > 99% of rows).
    near_constant = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=True)
        if df[col].nunique(dropna=True) <= 1:
            near_constant.append(col)
        elif not counts.empty and counts.iloc[0] / len(df) > 0.99:
            near_constant.append(col)
    if near_constant:
        df = df.drop(columns=near_constant)
    dropped = dropped + near_constant
    steps.append({"step": "drop_near_constant",
                  "detail": "Bỏ cột gần như không đổi", "columns": near_constant})

    # 5. Drop full-row duplicates.
    before_dedup = len(df)
    df = df.drop_duplicates()
    steps.append({"step": "dedup", "detail": "Bỏ dòng trùng lặp hoàn toàn",
                  "rows_removed": before_dedup - len(df)})

    # 6. IQR outlier capping (winsorize) for numeric columns.
    values_capped = 0
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        capped = ((df[col] < low) | (df[col] > high)).sum()
        values_capped += int(capped)
        df[col] = df[col].clip(lower=low, upper=high)
    steps.append({"step": "outlier_cap", "detail": "Chặn ngoại lai theo IQR",
                  "values_capped": values_capped})

    # 7. Fill missing: numeric -> median, categorical -> mode.
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
    steps.append({"step": "fill_missing", "detail": "Điền khuyết thiếu",
                  "missing_filled": missing_filled})

    report = {
        "dropped_columns": dropped,
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_after": len(df.columns),
        "missing_filled": missing_filled,
        "steps": steps,
    }
    return df, report
