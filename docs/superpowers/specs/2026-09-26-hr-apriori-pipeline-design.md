# HR Data Cleaning → Apriori Pipeline — Design

**Date**: 2026-09-26
**Status**: Approved, ready for implementation planning

## Goal

Take the real imported dataset (IBM HR Employee Attrition, ~1470 rows × 31 cols,
mixed numeric + categorical), run a realistic ("chuẩn thực tế") data-cleaning
pipeline on it, then mine **frequent itemsets** and **association rules** from it
using a hand-coded Apriori algorithm.

Course context (KTMXH): the result must read as a real-world workflow, and the
mining algorithm must be implemented manually.

## Decisions (locked)

- **Encoding scope**: Attrition-focused subset of columns, not all 31.
- **Numeric binning**: semantic labels. Known HR columns use fixed domain
  thresholds with readable labels (e.g. `Age=Young|Middle|Senior`); other numerics
  fall back to quantile bins labeled `Low|Medium|High`. Readable rules over
  balanced support.
- **Cleaning depth**: standard pipeline (dedup, dtype coercion, categorical
  standardization, IQR outlier capping, near-constant drop) on top of the current
  drop-cols + trim + fill.
- **UX**: a button in the Apriori tab encodes the imported data into transactions
  and fills the existing JSON textarea; the user then runs mining as today.
- **Library constraint**:
  - Data cleaning (§1) and the transaction encoder (§2) **may use pandas**.
  - The **frequent-itemset + association-rule mining algorithm must be manual**
    — no third-party or convenience library. `itertools.combinations` is
    replaced with a hand-written generator so the mining code uses zero library
    helpers.
  - Everything else (CSV I/O in the import view and seed command) may keep pandas.

## Components

### §1 · Cleaning pipeline — extend `core/data_cleaning.py`

`clean_dataframe(df)` keeps its signature `-> (df, report)` and its existing
report keys (`dropped_columns`, `rows_before`, `rows_after`, `columns_after`,
`missing_filled`) so the import view and seed command keep working. It gains an
ordered `steps` list in the report, one entry per stage with a human-readable
label + counts.

Ordered stages:

1. **Drop configured dead columns** — existing `DROP_COLUMNS`.
2. **Coerce dtypes** — object columns that are fully numeric-looking → numeric.
3. **Standardize categorical text** — strip, collapse internal whitespace, treat
   blank string as missing. (Conservative: do not alter meaningful values like
   `Travel_Rarely`.)
4. **Drop near-constant columns** — dynamic: `nunique <= 1`, or a single value
   covers > 99% of rows. Catches dead columns beyond the hardcoded four.
5. **Drop duplicate rows** — full-row duplicates only.
6. **IQR outlier capping** — for each numeric column, winsorize values to
   `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`. Report count of values capped.
7. **Fill missing** — numeric → median, categorical → mode (existing behavior).

Each stage appends `{step, detail, ...counts}` to `report["steps"]`.

### §2 · Transaction encoder — new `core/transaction_encoder.py`

```
encode_transactions(df, columns=None, target="Attrition", n_bins=3)
    -> (transactions, report)
```

- **Curated subset** (default `columns`), intersected with columns actually
  present so a non-HR dataset degrades gracefully:
  `OverTime, JobSatisfaction, JobRole, MaritalStatus, WorkLifeBalance,
  Department, Age, MonthlyIncome, YearsAtCompany, Attrition`.
- **Numeric column** → semantic bins, token format `Col=Label`:
  - **Known HR columns** use fixed domain thresholds with readable labels:
    - `Age`: `Young` (<30), `Middle` (30–50), `Senior` (>50)
    - `MonthlyIncome`: `pandas.qcut` into `Low|Medium|High`
    - `DistanceFromHome`: `Near` (≤5), `Medium` (6–15), `Far` (>15)
    - `TotalWorkingYears`: `Junior` (<5), `Mid` (5–15), `Senior` (>15)
    - `YearsAtCompany`: `New` (<3), `Established` (3–10), `Veteran` (>10)
  - **Other numerics** → `pandas.qcut(..., duplicates="drop")` labeled
    `Low|Medium|High`.
  - A numeric column with few distinct values (`nunique <= 6`, e.g. ordinals
    like `WorkLifeBalance`, `JobSatisfaction`, both 1–4) is treated as
    categorical (`Col=3`) instead of binned.
  - Bin definitions live in one small config dict so labels stay auditable.
- **Categorical column** → token `Col=Value`.
- Each row → `{"tid": "T{i+1}", "items": [tokens…]}`.
- **Guard**: if fewer than 2 usable columns remain → raise a clear error.
- **Report**: per-column bin edges (for numeric), token vocabulary size, and
  transaction count.

Encoding runs on the already-cleaned `data_json`, so cleaning and encoding stay
separate concerns.

### §3 · Endpoint — `GET /api/encode-transactions/`

New `EncodeTransactionsAPIView`:

1. Load the single stored `Dataset` (`Dataset.objects.first()`).
2. Rebuild a DataFrame from `dataset.data_json`.
3. Run `encode_transactions`.
4. Return `{"transactions": [...], "report": {...}}`.

Errors: no dataset → 400 "Chưa có dữ liệu, hãy nhập CSV trước."; fewer than 2
usable columns → 400 with the reason. Pure deterministic transform — no model
call.

Register the route in `core/urls.py`.

### §4 · Frontend

- `templates/components/tab_apriori.html`: add a **"Dùng dữ liệu đã nhập (HR)"**
  button above the transactions textarea, plus a small box to render the encoder
  report (bin edges + vocab size + tx count).
- `static/js/ui_apriori.js`: button handler → `GET /api/encode-transactions/` →
  fill `#ap-json-input` with pretty-printed transactions JSON, render the report
  box. The existing "Khai Thác Tập Phổ Biến & Luật" run path is unchanged.
- `static/js/api.js`: register the new endpoint (JSON GET, no body).

### Δ · Apriori mining stays fully manual — `core/algorithms/association.py`

The engine is already hand-coded (candidate join, bitvector support counting,
confidence, lift). Only change: replace `itertools.combinations` in rule-subset
generation with a hand-written combinations generator, so the mining algorithm
uses no library helper at all. No behavioral change.

## Data flow

```
CSV import → clean_dataframe (standard pipeline) → Dataset.data_json
                                                         │
Apriori tab: click "Dùng dữ liệu đã nhập"                │
   → GET /api/encode-transactions/ (quantile bin + tokenize subset)
   → transactions JSON fills textarea
   → click "Khai Thác" → AprioriEngine (manual) → frequent itemsets F_k + rules
```

## Error handling

- No dataset imported → 400, Vietnamese message.
- Curated column absent → silently skipped (subset ∩ present columns).
- < 2 usable columns → 400 with reason.
- `qcut` failure / too-few unique values → column treated as categorical instead.

## Testing (tests encode intent, not just behavior)

- **Encoder**: known HR columns map to their exact semantic labels (e.g. age 25 →
  `Age=Young`, 40 → `Age=Middle`) — *why*: rules must read in human terms, not
  raw edges; a few-unique numeric column is treated categorical, not binned; token
  string format is exact; fallback qcut column produces `Low|Medium|High`.
- **Cleaning**: a crafted DataFrame with duplicate rows + a near-constant column
  + an outlier + missing values → each stage's report count is correct.
- **Endpoint**: with a seeded dataset, `GET` returns a transaction count equal to
  the dataset row count.
- **Apriori**: hand-written combinations generator produces the same subsets as
  the previous `itertools.combinations` output (guards the refactor).

## Files touched

- `core/data_cleaning.py` — extend (pandas OK)
- `core/transaction_encoder.py` — new (pandas OK)
- `core/algorithms/association.py` — remove `itertools`, hand-code combinations
- `core/views.py` — new `EncodeTransactionsAPIView`
- `core/urls.py` — route
- `templates/components/tab_apriori.html` — button + report box
- `static/js/ui_apriori.js` — handler
- `static/js/api.js` — endpoint
- tests — encoder, cleaning, endpoint, apriori
