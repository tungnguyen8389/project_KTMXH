# HR Data Cleaning → Apriori Association Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the imported IBM HR dataset into transactions (clean → semantic-bin → tokenize) and mine frequent itemsets + association rules with a fully hand-coded Apriori, surfaced through a button in the existing Apriori tab.

**Architecture:** Extend the existing shared cleaner into a reported multi-stage pipeline; add a pandas-based transaction encoder that maps a curated attrition-focused column subset to `Col=Label` tokens (semantic bins for known HR columns, qcut fallback otherwise); expose a GET endpoint that encodes the single stored `Dataset`; wire an "use imported HR data" button in the Apriori tab that fills the existing transactions textarea. The mining engine already exists and only loses its one library helper (`itertools.combinations`).

**Tech Stack:** Django + Django REST Framework, pandas/numpy (cleaning + encoding only), vanilla JS frontend, Django `TestCase` unit tests.

**Spec:** `docs/superpowers/specs/2026-09-26-hr-apriori-pipeline-design.md`

## Global Constraints

- **Mining stays 100% manual:** `core/algorithms/association.py` must import zero third-party or stdlib algorithm helpers. `itertools` must not appear in the file after Task 1.
- **pandas allowed only in cleaning + encoding:** `core/data_cleaning.py` and `core/transaction_encoder.py` may use pandas/numpy. The mining engine may not.
- **`clean_dataframe(df)` keeps its contract:** signature `-> (df, report)`; report must still contain keys `dropped_columns`, `rows_before`, `rows_after`, `columns_after`, `missing_filled`. A new `steps` key is added, not a replacement.
- **All user-facing error messages are Vietnamese**, matching existing views (e.g. `'Danh sách giao dịch transactions không được rỗng!'`).
- **Semantic bin labels** (locked decision): known HR numeric columns use fixed-threshold readable labels; other numerics use qcut labeled `Low|Medium|High`.
- **Test run command:** activate the shared venv first, then `python manage.py test core.tests -v 2` (or a specific `core.tests.ClassName.method`).

## Review Focus

- **Non-HR / partial dataset:** curated columns absent → encoder silently intersects with present columns; fewer than 2 usable columns → 400 with a clear Vietnamese reason. (Covered in Task 3 + Task 4.)
- **JSONField numeric round-trip:** `data_json` may store numbers as strings; the encoder must coerce numeric-looking columns before binning or every value lands in one bin. (Covered in Task 3.)
- **Age/threshold boundary values:** age exactly 30 and 50 must fall in `Middle` (rule: Young `<30`, Middle `30–50` inclusive, Senior `>50`); off-by-one here silently mislabels people. (Covered in Task 3.)
- **qcut duplicate edges:** a skewed numeric column (e.g. `MonthlyIncome`) can produce identical quantile edges → `qcut` raises; must pass `duplicates="drop"` and not crash. (Covered in Task 3.)
- **No dataset imported:** GET before any CSV import → 400 Vietnamese message, never a 500. (Covered in Task 4.)

---

## File Structure

- `core/algorithms/association.py` — **modify.** Replace `itertools.combinations` with a module-level hand-coded `_combinations(iterable, r)` generator. No behavior change.
- `core/data_cleaning.py` — **modify.** Extend `clean_dataframe` into ordered stages, each appending to `report["steps"]`. Keep existing keys.
- `core/transaction_encoder.py` — **create.** `encode_transactions(df, columns=None, target="Attrition", n_bins=3) -> (transactions, report)` plus a `BIN_RULES` config dict.
- `core/views.py` — **modify.** Add `EncodeTransactionsAPIView` (GET).
- `core/urls.py` — **modify.** Import + register `encode-transactions/`.
- `static/js/api.js` — **modify.** Add `API.get(endpoint)`.
- `templates/components/tab_apriori.html` — **modify.** Add prefill button + encoder-report box above the textarea.
- `static/js/ui_apriori.js` — **modify.** Add button handler → GET → fill textarea + render report.
- `core/tests.py` — **modify.** Add `AprioriCombinationsTest`, `DataCleaningTest`, `TransactionEncoderTest`, `EncodeEndpointTest`.

---

### Task 1: Hand-code combinations in the Apriori engine

Removes the last library helper from the mining algorithm. Isolated refactor guarded by a test that pins the generator's output to the old `itertools.combinations` behavior.

**Files:**
- Modify: `core/algorithms/association.py` (line 1 `import itertools`; line 122 `itertools.combinations`)
- Test: `core/tests.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: module-level `_combinations(iterable, r)` — a generator yielding tuples of length `r` in the same order as `itertools.combinations`. `AprioriEngine.run_apriori(transactions, min_supp_pct, min_conf_pct)` return shape unchanged.

- [ ] **Step 1: Write the failing test**

In `core/tests.py`, add at the bottom:

```python
from core.algorithms.association import AprioriEngine, _combinations


class AprioriCombinationsTest(TestCase):
    def test_combinations_matches_itertools(self):
        # WHY: the mining algorithm must generate rule antecedents itself,
        # with no stdlib helper, yet produce identical subsets — a drift here
        # would silently change which association rules exist.
        import itertools
        items = ["A", "B", "C", "D"]
        for r in range(1, len(items) + 1):
            self.assertEqual(
                list(_combinations(items, r)),
                list(itertools.combinations(items, r)),
            )

    def test_no_itertools_import(self):
        # WHY: "manual algorithm" is a course requirement, not a style note.
        import core.algorithms.association as assoc_mod
        with open(assoc_mod.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn("itertools", source)

    def test_run_apriori_still_works(self):
        tx = [
            {"tid": "T1", "items": ["A", "B"]},
            {"tid": "T2", "items": ["A", "B", "C"]},
            {"tid": "T3", "items": ["A", "C"]},
        ]
        ap = AprioriEngine.run_apriori(tx, min_supp_pct=50.0, min_conf_pct=50.0)
        self.assertEqual(ap["num_transactions"], 3)
        self.assertTrue(len(ap["valid_rules"]) > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python manage.py test core.tests.AprioriCombinationsTest -v 2`
Expected: FAIL — `ImportError: cannot import name '_combinations'`.

- [ ] **Step 3: Add the hand-coded generator and remove itertools**

In `core/algorithms/association.py`, delete line 1 (`import itertools`) and add at module top:

```python
def _combinations(iterable, r):
    """Hand-coded stand-in for itertools.combinations.

    Yields length-r tuples of items from `iterable` in lexicographic index
    order, identical to itertools.combinations. Used by rule generation so
    the mining algorithm relies on no library helper.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)
```

Then change the rule-subset loop (was line 122):

```python
                    for subset_A_tuple in _combinations(items_in_set, r):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python manage.py test core.tests.AprioriCombinationsTest -v 2`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add core/algorithms/association.py core/tests.py
git commit -m "refactor(apriori): hand-code combinations, drop itertools dependency"
```

---

### Task 2: Extend the data cleaner into a reported multi-stage pipeline

Turns the 3-step cleaner into the standard pipeline the spec describes, while keeping the exact report keys the import view and seed command already read.

**Files:**
- Modify: `core/data_cleaning.py`
- Test: `core/tests.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `clean_dataframe(df) -> (cleaned_df, report)`. `report` keeps keys `dropped_columns`, `rows_before`, `rows_after`, `columns_after`, `missing_filled` and gains `steps`: an ordered `list[dict]`, each `{"step": str, "detail": str, ...counts}`.

- [ ] **Step 1: Write the failing test**

In `core/tests.py` add:

```python
import pandas as pd
from core.data_cleaning import clean_dataframe


class DataCleaningTest(TestCase):
    def _crafted_df(self):
        # rows 1 & 2 identical (duplicate); "dead" near-constant; age has an
        # outlier (999) and a missing value.
        return pd.DataFrame([
            {"EmployeeNumber": 1, "Age": 30, "dead": "X", "Dept": "Sales"},
            {"EmployeeNumber": 1, "Age": 30, "dead": "X", "Dept": "Sales"},
            {"EmployeeNumber": 2, "Age": 999, "dead": "X", "Dept": "R&D"},
            {"EmployeeNumber": 3, "Age": None, "dead": "X", "Dept": "Sales"},
        ])

    def test_report_keeps_legacy_keys(self):
        # WHY: the import view and seed command read these exact keys; renaming
        # any of them breaks the CSV import UI silently.
        _, report = clean_dataframe(self._crafted_df())
        for key in ("dropped_columns", "rows_before", "rows_after",
                    "columns_after", "missing_filled"):
            self.assertIn(key, report)

    def test_drops_id_and_near_constant_columns(self):
        # WHY: EmployeeNumber is an ID and "dead" is >99% one value; both would
        # pollute Apriori with useless items.
        df, report = clean_dataframe(self._crafted_df())
        self.assertNotIn("EmployeeNumber", df.columns)
        self.assertNotIn("dead", df.columns)

    def test_dedup_and_outlier_and_fill_reported(self):
        # WHY: each stage must be auditable in the UI report, so counts must be
        # real, not zero placeholders.
        df, report = clean_dataframe(self._crafted_df())
        steps = {s["step"]: s for s in report["steps"]}
        self.assertIn("dedup", steps)
        self.assertEqual(steps["dedup"]["rows_removed"], 1)
        self.assertIn("outlier_cap", steps)
        self.assertGreaterEqual(steps["outlier_cap"]["values_capped"], 1)
        self.assertEqual(report["missing_filled"], 1)
        # outlier 999 must have been capped below itself
        self.assertLess(df["Age"].max(), 999)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python manage.py test core.tests.DataCleaningTest -v 2`
Expected: FAIL — `KeyError: 'dedup'` / `steps` absent.

- [ ] **Step 3: Rewrite `clean_dataframe` with ordered stages**

Replace the body of `clean_dataframe` in `core/data_cleaning.py` (keep `DROP_COLUMNS` and the module docstring):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python manage.py test core.tests.DataCleaningTest -v 2`
Expected: PASS (3 tests).

- [ ] **Step 5: Regression-check the import path still works**

Run: `python manage.py test core -v 2`
Expected: PASS (all existing + new tests). This confirms the preserved report keys didn't break anything.

- [ ] **Step 6: Commit**

```bash
git add core/data_cleaning.py core/tests.py
git commit -m "feat(cleaning): staged reported pipeline (dedup, coerce, IQR cap)"
```

---

### Task 3: Transaction encoder with semantic bins

Maps the cleaned dataset's curated column subset to Apriori transactions using readable semantic labels.

**Files:**
- Create: `core/transaction_encoder.py`
- Test: `core/tests.py`

**Interfaces:**
- Consumes: a pandas `DataFrame` (from `Dataset.data_json`).
- Produces:
  - `CURATED_COLUMNS: list[str]` — default subset.
  - `BIN_RULES: dict[str, list[tuple]]` — per-known-column `[(label, lo, hi)]` fixed thresholds.
  - `encode_transactions(df, columns=None, target="Attrition", n_bins=3) -> (transactions, report)` where `transactions` is `list[{"tid": str, "items": list[str]}]` and `report` is `{"columns_used": list[str], "bins": dict, "vocab_size": int, "num_transactions": int}`.

- [ ] **Step 1: Write the failing test**

In `core/tests.py` add:

```python
from core.transaction_encoder import encode_transactions, CURATED_COLUMNS


class TransactionEncoderTest(TestCase):
    def _hr_df(self):
        rows = []
        for i in range(9):
            rows.append({
                "Age": [25, 29, 30, 40, 50, 51, 60, 35, 45][i],
                "MonthlyIncome": str(1000 * (i + 1)),   # stored as string
                "OverTime": "Yes" if i % 2 else "No",
                "Attrition": "Yes" if i % 3 == 0 else "No",
            })
        return pd.DataFrame(rows)

    def test_age_boundaries_map_to_semantic_labels(self):
        # WHY: readable rules require exact bins; age 30 and 50 must be Middle,
        # off-by-one silently mislabels employees.
        tx, _ = encode_transactions(self._hr_df())
        items_by_tid = {t["tid"]: set(t["items"]) for t in tx}
        self.assertIn("Age=Young", items_by_tid["T1"])    # 25
        self.assertIn("Age=Middle", items_by_tid["T3"])   # 30 boundary
        self.assertIn("Age=Middle", items_by_tid["T5"])   # 50 boundary
        self.assertIn("Age=Senior", items_by_tid["T6"])   # 51

    def test_string_numeric_column_is_binned_not_one_bucket(self):
        # WHY: JSONField can store numbers as strings; without coercion every
        # MonthlyIncome lands in one bucket and yields no useful rules.
        tx, report = encode_transactions(self._hr_df())
        income_tokens = {it for t in tx for it in t["items"]
                         if it.startswith("MonthlyIncome=")}
        self.assertEqual(income_tokens,
                         {"MonthlyIncome=Low", "MonthlyIncome=Medium",
                          "MonthlyIncome=High"})

    def test_transaction_count_and_categorical_token(self):
        tx, report = encode_transactions(self._hr_df())
        self.assertEqual(report["num_transactions"], 9)
        self.assertEqual(len(tx), 9)
        self.assertTrue(any("OverTime=Yes" in t["items"] for t in tx))

    def test_missing_curated_columns_degrade_gracefully(self):
        # WHY: a non-HR dataset must not crash; usable columns are intersected.
        df = pd.DataFrame([{"OverTime": "Yes", "Attrition": "No"},
                           {"OverTime": "No", "Attrition": "Yes"}])
        tx, report = encode_transactions(df)
        self.assertEqual(set(report["columns_used"]), {"OverTime", "Attrition"})

    def test_too_few_usable_columns_raises(self):
        # WHY: with <2 columns there are no associations to mine; fail loudly.
        df = pd.DataFrame([{"OverTime": "Yes"}, {"OverTime": "No"}])
        with self.assertRaises(ValueError):
            encode_transactions(df)

    def test_low_cardinality_numeric_stays_categorical(self):
        # WHY: ordinals like JobSatisfaction (1..4) must read as JobSatisfaction=3,
        # not be blurred into Low/Medium/High quantile bins.
        df = pd.DataFrame([{"JobSatisfaction": v, "OverTime": "Yes", "Attrition": "No"}
                           for v in [1, 2, 3, 4, 1, 2, 3, 4]])
        tx, _ = encode_transactions(df)
        toks = {it for t in tx for it in t["items"] if it.startswith("JobSatisfaction=")}
        self.assertEqual(toks, {"JobSatisfaction=1", "JobSatisfaction=2",
                                "JobSatisfaction=3", "JobSatisfaction=4"})

    def test_skewed_numeric_does_not_crash(self):
        # WHY: a clustered high-cardinality numeric can yield duplicate quantile
        # edges; encoding must degrade gracefully, never raise.
        vals = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 500000]
        df = pd.DataFrame([{"MonthlyIncome": v, "OverTime": "Yes", "Attrition": "No"}
                           for v in vals])
        tx, report = encode_transactions(df)  # must not raise
        self.assertEqual(len(tx), 9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python manage.py test core.tests.TransactionEncoderTest -v 2`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.transaction_encoder'`.

- [ ] **Step 3: Create `core/transaction_encoder.py`**

```python
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
    "Age": [("Young", None, 29), ("Middle", 30, 50), ("Senior", 51, None)],
    "DistanceFromHome": [("Near", None, 5), ("Medium", 6, 15), ("Far", 16, None)],
    "TotalWorkingYears": [("Junior", None, 4), ("Mid", 5, 15), ("Senior", 16, None)],
    "YearsAtCompany": [("New", None, 2), ("Established", 3, 10), ("Veteran", 11, None)],
}
# Columns binned by quantile (equal-frequency) with Low/Medium/High labels.
QCUT_LABELS = ["Low", "Medium", "High"]
QCUT_COLUMNS = {"MonthlyIncome"}
# Numeric columns with <= this many distinct values stay categorical (ordinals
# like JobSatisfaction/WorkLifeBalance read better as Col=3 than as bins).
ORDINAL_MAX = 6


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
            tokens_per_col[col] = _cat_tokens(numeric.astype("Int64"), col)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python manage.py test core.tests.TransactionEncoderTest -v 2`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add core/transaction_encoder.py core/tests.py
git commit -m "feat(encoder): semantic-bin HR transaction encoder"
```

---

### Task 4: `GET /api/encode-transactions/` endpoint

Encodes the single stored dataset on demand for the Apriori tab.

**Files:**
- Modify: `core/views.py` (add class after `AprioriAPIView`, ~line 186; add import)
- Modify: `core/urls.py`
- Test: `core/tests.py`

**Interfaces:**
- Consumes: `Dataset.objects.first()` (model at `core/models.py:3`, field `data_json` is a list of row dicts); `encode_transactions` from Task 3.
- Produces: `GET /api/encode-transactions/` → 200 `{"transactions": [...], "report": {...}}`; 400 with Vietnamese `{"error": ...}` on no dataset or too-few columns.

- [ ] **Step 1: Write the failing test**

In `core/tests.py` add:

```python
from core.models import Dataset


class EncodeEndpointTest(TestCase):
    def test_no_dataset_returns_400(self):
        # WHY: user hits the button before importing; must be a clean 400, not 500.
        resp = self.client.get("/api/encode-transactions/")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_returns_transactions_matching_row_count(self):
        rows = [{"Age": 25, "OverTime": "Yes", "Attrition": "No"},
                {"Age": 55, "OverTime": "No", "Attrition": "Yes"},
                {"Age": 40, "OverTime": "Yes", "Attrition": "Yes"}]
        Dataset.objects.create(name="HR", category="APRIORI", data_json=rows)
        resp = self.client.get("/api/encode-transactions/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["transactions"]), 3)
        self.assertEqual(body["report"]["num_transactions"], 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python manage.py test core.tests.EncodeEndpointTest -v 2`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Add the view**

In `core/views.py`, near the other imports (after `from .algorithms.association import AprioriEngine`, line 15) add:

```python
import pandas as pd
from .models import Dataset
from .transaction_encoder import encode_transactions
```

(If `Dataset` / `pd` are already imported at the top, do not duplicate — check first.)

After the `AprioriAPIView` class (line 186) add:

```python
class EncodeTransactionsAPIView(APIView):
    """[TV3] Encode the stored HR dataset into Apriori transactions (GET)."""
    def get(self, request):
        dataset = Dataset.objects.first()
        if dataset is None:
            return Response({'error': 'Chưa có dữ liệu, hãy nhập CSV trước.'},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            df = pd.DataFrame(dataset.data_json)
            transactions, report = encode_transactions(df)
            return Response({'transactions': transactions, 'report': report},
                            status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
```

- [ ] **Step 4: Register the route**

In `core/urls.py`, add `EncodeTransactionsAPIView` to the `from .views import (...)` block and add to `urlpatterns`:

```python
    path('encode-transactions/', EncodeTransactionsAPIView.as_view(), name='api_encode_transactions'),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python manage.py test core.tests.EncodeEndpointTest -v 2`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add core/views.py core/urls.py core/tests.py
git commit -m "feat(api): GET /api/encode-transactions/ for HR Apriori prefill"
```

---

### Task 5: Frontend — HR prefill button + encoder report

Wires the button that turns the imported data into transactions in the existing textarea. No JS test framework exists in this repo, so this task is verified by a scripted manual smoke check.

**Files:**
- Modify: `static/js/api.js` (add `get` to the `API` object, after the `post` method, ~line 24)
- Modify: `templates/components/tab_apriori.html` (above the `<textarea id="ap-json-input">`, ~line 19)
- Modify: `static/js/ui_apriori.js` (add handler inside the existing `DOMContentLoaded`, ~line 4)

**Interfaces:**
- Consumes: `GET /api/encode-transactions/` from Task 4.
- Produces: `API.get(endpoint)` returning parsed JSON; button `#btn-encode-hr`; report container `#ap-encode-report`.

- [ ] **Step 1: Add `API.get`**

In `static/js/api.js`, inside the `API` object after the `post` method's closing `},` (line 24) add:

```javascript
    async get(endpoint) {
        try {
            const response = await fetch(`/api/${endpoint}/`, {
                method: 'GET',
                headers: { 'X-CSRFToken': this.getCsrfToken() }
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Lỗi khi gọi API backend!');
            }
            return result;
        } catch (err) {
            alert(`Lỗi API (${endpoint}): ${err.message}`);
            throw err;
        }
    },
```

- [ ] **Step 2: Add the button + report box to the template**

In `templates/components/tab_apriori.html`, replace the `<div class="form-group">` that wraps the textarea label (line 19) so the button and report box sit directly above the textarea. Insert before line 19:

```html
                <div class="form-group">
                    <button id="btn-encode-hr" type="button" class="btn btn-secondary btn-block">
                        Dùng dữ liệu đã nhập (HR)
                    </button>
                    <div id="ap-encode-report" class="mt-2"></div>
                </div>
```

- [ ] **Step 3: Add the button handler**

In `static/js/ui_apriori.js`, inside the existing `DOMContentLoaded` callback (after line 3's opening), add before the `btnRunApriori` block:

```javascript
    const btnEncodeHr = document.getElementById('btn-encode-hr');
    if (btnEncodeHr) {
        btnEncodeHr.addEventListener('click', async () => {
            try {
                const res = await API.get('encode-transactions');
                document.getElementById('ap-json-input').value =
                    JSON.stringify(res.transactions, null, 2);
                // Prefill sensible defaults for the real HR dataset.
                document.getElementById('ap-minsupp').value = 10;
                document.getElementById('ap-minconf').value = 50;
                const r = res.report;
                const binLines = Object.entries(r.bins || {})
                    .map(([col, labels]) => `<li><strong>${col}</strong>: ${labels.join(', ')}</li>`)
                    .join('');
                document.getElementById('ap-encode-report').innerHTML = `
                    <div class="step-card">
                        <p>Đã mã hóa <strong>${r.num_transactions}</strong> giao dịch,
                           <strong>${r.vocab_size}</strong> mục (items) từ
                           ${r.columns_used.length} cột.</p>
                        <ul>${binLines}</ul>
                    </div>`;
            } catch (e) {
                console.error(e);
            }
        });
    }
```

- [ ] **Step 4: Manual smoke verification**

Run the server and drive the flow (evidence required before marking done):

```bash
python manage.py runserver 127.0.0.1:8010 &
sleep 3
# seed a dataset if the app has a seed command, else import a CSV via the UI
curl -s http://127.0.0.1:8010/api/encode-transactions/ | python -c "import sys,json; d=json.load(sys.stdin); print('tx:', len(d['transactions']), 'vocab:', d['report']['vocab_size'])"
```

Expected: prints a non-zero `tx:` count and `vocab:` size. Then in a browser open the Apriori tab, click **Dùng dữ liệu đã nhập (HR)**, confirm the textarea fills with `Age=…`, `OverTime=…` tokens and the report box shows bin labels, then click **Khai Thác** and confirm rules render. Stop the server (`kill %1`).

- [ ] **Step 5: Commit**

```bash
git add static/js/api.js static/js/ui_apriori.js templates/components/tab_apriori.html
git commit -m "feat(ui): HR prefill button + encoder report in Apriori tab"
```

---

## Final verification

- [ ] Run the whole suite: `python manage.py test core -v 2` → all PASS.
- [ ] Confirm `itertools` is absent from `core/algorithms/association.py`.
- [ ] Confirm `clean_dataframe` report still carries the five legacy keys plus `steps`.
- [ ] Manual smoke of the Apriori tab prefill + mining flow passed with real data.
