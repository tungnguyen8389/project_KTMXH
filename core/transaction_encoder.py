"""Encode a cleaned HR DataFrame into Apriori transactions.

Numeric columns become semantic bins (readable labels), categorical columns
become Col=Value tokens. pandas is allowed here; the mining engine is not
touched by this module.
"""
import pandas as pd

# Attrition-focused subset; intersected with columns actually present.
CURATED_COLUMNS = [
    "OverTime", "JobSatisfaction", "JobRole", "MaritalStatus",
    "WorkLifeBalance", "Department", "Age", "MonthlyIncome",
    "YearsAtCompany", "Attrition",
]

# Fixed-threshold semantic bins for known numeric columns.
# Each rule: (label, low_inclusive_or_None, high_inclusive_or_None).
BIN_RULES = {
    "Age": [("Young (≤29)", None, 29), ("Middle (30–50)", 30, 50),
            ("Senior (≥51)", 51, None)],
    "DistanceFromHome": [("Near", None, 5), ("Medium", 6, 15), ("Far", 16, None)],
    "TotalWorkingYears": [("Junior", None, 4), ("Mid", 5, 15), ("Senior", 16, None)],
    "YearsAtCompany": [("New", None, 2), ("Established", 3, 10), ("Veteran", 11, None)],
}
# Columns binned by quantile (equal-frequency) with Low/Medium/High labels.
QCUT_LABELS = ["Low", "Medium", "High"]
QCUT_COLUMNS = {"MonthlyIncome"}
# Numeric columns with <= this many distinct values stay categorical.
ORDINAL_MAX = 6

# Human-readable labels for the IBM HR ordinal codes, so tokens read as
# WorkLifeBalance=Best instead of WorkLifeBalance=4. Unmapped codes fall back to
# the raw number; columns not listed here keep their numeric value unchanged.
_SATISFACTION = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
ORDINAL_LABELS = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor",
                  4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction": _SATISFACTION,
    "JobInvolvement": _SATISFACTION,
    "JobSatisfaction": _SATISFACTION,
    "RelationshipSatisfaction": _SATISFACTION,
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}


def _ordinal_label_tokens(numeric, col):
    """Map a known HR ordinal column's integer codes to readable labels.

    Non-integer or unmapped values degrade to their raw string, so a stray float
    never raises and an unexpected code still shows.
    """
    mapping = ORDINAL_LABELS[col]

    def lab(v):
        if pd.isna(v):
            return pd.NA
        if float(v).is_integer():
            return mapping.get(int(v), str(int(v)))
        return str(v)

    return numeric.apply(lab).radd(f"{col}=")


def _label_by_rules(value, rules):
    for label, lo, hi in rules:
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return label
    return rules[-1][0]  # clamp out-of-range into the last bin


def _rule_tokens(numeric, col):
    labels = numeric.apply(lambda v: _label_by_rules(v, BIN_RULES[col])
                           if pd.notna(v) else pd.NA)
    return labels.radd(f"{col}="), [lbl for lbl, _, _ in BIN_RULES[col]]


def _qcut_tokens(numeric, col, n_bins):
    """Quantile bins with readable labels; degrades gracefully, never raises."""
    try:
        binned = pd.qcut(numeric, q=n_bins, labels=QCUT_LABELS, duplicates="drop")
    except ValueError:
        try:
            binned = pd.qcut(numeric, q=n_bins, duplicates="drop").astype("string")
        except ValueError:
            return None, None  # too few distinct values -> caller uses categorical
    tokens = binned.astype("string").radd(f"{col}=")
    labels = list(dict.fromkeys(binned.dropna().astype(str)))
    return tokens, labels


def _cat_tokens(series, col):
    return series.astype("string").radd(f"{col}=")


def encode_transactions(df, columns=None, target="Attrition", n_bins=3):
    columns = columns if columns is not None else CURATED_COLUMNS
    used = [c for c in columns if c in df.columns]
    if len(used) < 2:
        raise ValueError(
            "Cần ít nhất 2 cột hợp lệ để khai phá luật kết hợp, "
            f"nhưng chỉ có {len(used)}."
        )

    tokens_per_col = {}   # col -> pandas Series of string tokens
    bins_report = {}

    for col in used:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().sum() == series.notna().sum() and series.notna().any()

        if not is_numeric:
            tokens_per_col[col] = _cat_tokens(series, col)
            continue

        if col in BIN_RULES:                       # fixed-threshold semantic bins
            tokens_per_col[col], bins_report[col] = _rule_tokens(numeric, col)
        elif col in QCUT_COLUMNS:                  # quantile Low/Medium/High
            toks, labels = _qcut_tokens(numeric, col, n_bins)
            if toks is None:
                tokens_per_col[col] = _cat_tokens(series, col)
            else:
                tokens_per_col[col], bins_report[col] = toks, labels
        elif numeric.nunique(dropna=True) <= ORDINAL_MAX:   # ordinal -> categorical
            if col in ORDINAL_LABELS:                       # readable HR labels
                tokens_per_col[col] = _ordinal_label_tokens(numeric, col)
            else:
                try:
                    tokens_per_col[col] = _cat_tokens(numeric.astype("Int64"), col)
                except (TypeError, ValueError):
                    # fractional ordinal-like column -> keep original values as-is
                    tokens_per_col[col] = _cat_tokens(series, col)
        else:                                      # generic numeric -> quantile
            toks, labels = _qcut_tokens(numeric, col, n_bins)
            if toks is None:
                tokens_per_col[col] = _cat_tokens(series, col)
            else:
                tokens_per_col[col], bins_report[col] = toks, labels

    transactions = []
    vocab = set()
    for i in range(len(df)):
        items = []
        for col in used:
            tok = tokens_per_col[col].iloc[i]
            if pd.notna(tok) and not tok.endswith("="):
                items.append(str(tok))
                vocab.add(str(tok))
        transactions.append({"tid": f"T{i + 1}", "items": items})

    report = {
        "columns_used": used,
        "bins": bins_report,
        "vocab_size": len(vocab),
        "num_transactions": len(transactions),
    }
    return transactions, report
