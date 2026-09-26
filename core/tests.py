from django.test import TestCase
from core.algorithms.preprocessing import PreprocessingEngine
from core.algorithms.rough_set import RoughSetEngine
from core.algorithms.reduct import ReductEngine
from core.algorithms.kmeans import KMeansEngine
from core.algorithms.association import AprioriEngine
from core.algorithms.classification import ClassificationEngine


class AlgorithmTestCase(TestCase):
    """
    Automated verification unit tests for all 5 algorithm groups
    """

    def test_preprocessing(self):
        data = [{'score': 4.5}, {'score': 6.0}, {'score': 9.0}]
        minmax = PreprocessingEngine.min_max_normalize(data, 'score', 0.0, 1.0)
        self.assertEqual(minmax['min_val'], 4.5)
        self.assertEqual(minmax['max_val'], 9.0)

        zscore = PreprocessingEngine.z_score_normalize(data, 'score')
        self.assertIn('mean', zscore)
        self.assertIn('std', zscore)

    def test_rough_set_and_reduct(self):
        data = [
            {"id": "x1", "a": 1, "b": 1, "c": 0, "d": 1, "e": 1},
            {"id": "x2", "a": 1, "b": 0, "c": 0, "d": 1, "e": 1},
            {"id": "x3", "a": 0, "b": 0, "c": 0, "d": 0, "e": 0},
            {"id": "x4", "a": 1, "b": 1, "c": 0, "d": 0, "e": 0}
        ]
        rs = RoughSetEngine.analyze_rough_set(data, ['a', 'b', 'c', 'd'], 'e')
        self.assertIn('dependency_k', rs)
        self.assertIn('positive_region', rs)

        red = ReductEngine.compute_reducts(data, ['a', 'b', 'c', 'd'], 'e')
        self.assertIn('minimal_reducts', red)
        self.assertIn('matrix_n_x_n', red)

    def test_kmeans(self):
        points = [
            {"id": "P1", "x": 2.0, "y": 10.0},
            {"id": "P2", "x": 2.0, "y": 5.0},
            {"id": "P3", "x": 8.0, "y": 4.0},
            {"id": "P4", "x": 5.0, "y": 8.0}
        ]
        km = KMeansEngine.run_kmeans(points, k=2, max_iter=5)
        self.assertEqual(km['k'], 2)
        self.assertTrue(len(km['iterations']) > 0)

    def test_apriori(self):
        tx = [
            {"tid": "T1", "items": ["A", "B"]},
            {"tid": "T2", "items": ["A", "B", "C"]},
            {"tid": "T3", "items": ["A", "C"]}
        ]
        ap = AprioriEngine.run_apriori(tx, min_supp_pct=50.0, min_conf_pct=50.0)
        self.assertEqual(ap['num_transactions'], 3)
        self.assertTrue(len(ap['itemset_steps']) > 0)

    def test_id3_and_naive_bayes(self):
        data = [
            {"Outlook": "Sunny", "Wind": "Weak", "Play": "No"},
            {"Outlook": "Sunny", "Wind": "Strong", "Play": "No"},
            {"Outlook": "Overcast", "Wind": "Weak", "Play": "Yes"},
            {"Outlook": "Rain", "Wind": "Weak", "Play": "Yes"}
        ]
        id3 = ClassificationEngine.run_id3(data, ['Outlook', 'Wind'], 'Play')
        self.assertIn('mermaid_graph', id3)

        nb = ClassificationEngine.run_naive_bayes(data, ['Outlook', 'Wind'], 'Play', {'Outlook': 'Sunny', 'Wind': 'Weak'}, use_laplace=True)
        self.assertIn('predicted_class', nb)


from core.algorithms.association import _combinations


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


import pandas as pd
from core.data_cleaning import clean_dataframe


class DataCleaningTest(TestCase):
    def _crafted_df(self):
        # rows 1 & 2 identical (duplicate); "dead" near-constant; Age has enough
        # normal values that 999 is a genuine IQR outlier, plus one missing.
        return pd.DataFrame([
            {"EmployeeNumber": 1, "Age": 30, "dead": "X", "Dept": "Sales"},
            {"EmployeeNumber": 1, "Age": 30, "dead": "X", "Dept": "Sales"},
            {"EmployeeNumber": 2, "Age": 25, "dead": "X", "Dept": "R&D"},
            {"EmployeeNumber": 3, "Age": 35, "dead": "X", "Dept": "Sales"},
            {"EmployeeNumber": 4, "Age": 28, "dead": "X", "Dept": "R&D"},
            {"EmployeeNumber": 5, "Age": 40, "dead": "X", "Dept": "Sales"},
            {"EmployeeNumber": 6, "Age": 999, "dead": "X", "Dept": "R&D"},
            {"EmployeeNumber": 7, "Age": None, "dead": "X", "Dept": "Sales"},
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

    def test_low_cardinality_fractional_numeric_does_not_crash(self):
        # WHY: a non-HR ordinal-like float column (<=6 distinct, not configured)
        # must degrade gracefully — the Int64 cast raises on fractions and would
        # otherwise take down the whole encode (violates graceful degradation).
        df = pd.DataFrame([{"Rating": v, "Grp": "A" if i % 2 else "B"}
                           for i, v in enumerate([1.5, 2.5, 3.5, 1.5, 2.5, 3.5])])
        tx, report = encode_transactions(df, columns=["Rating", "Grp"])  # must not raise
        toks = {it for t in tx for it in t["items"] if it.startswith("Rating=")}
        self.assertTrue(toks)


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


class RuleFilterTest(TestCase):
    def _attr_tx(self):
        return [
            {"tid": "T1", "items": ["OverTime=Yes", "Attrition=Yes"]},
            {"tid": "T2", "items": ["OverTime=Yes", "Attrition=Yes"]},
            {"tid": "T3", "items": ["OverTime=No", "Attrition=No"]},
            {"tid": "T4", "items": ["OverTime=No", "Attrition=No"]},
        ]

    def test_max_len_caps_itemset_size(self):
        # WHY: HR cannot act on 5-condition rules; mining must stop at k=max_len.
        tx = [{"tid": f"T{i}", "items": ["A", "B", "C", "D"]} for i in range(4)]
        res = AprioriEngine.run_apriori(tx, 50, 50, max_len=2)
        self.assertEqual(max(s["k"] for s in res["itemset_steps"]), 2)

    def test_min_lift_filters_low_lift_rules(self):
        # WHY: lift<=1 means no positive correlation; those rules are noise.
        tx = [{"tid": f"T{i}", "items": ["A", "B", "C"]} for i in range(4)]
        hi = AprioriEngine.run_apriori(tx, 50, 50, min_lift=1.5)
        lo = AprioriEngine.run_apriori(tx, 50, 50, min_lift=0.0)
        self.assertEqual(len(hi["valid_rules"]), 0)
        self.assertGreater(len(lo["valid_rules"]), 0)

    def test_target_yes_only_keeps_attrition_yes_consequent(self):
        # WHY: the business question is who leaves; consequent must be exactly
        # Attrition=Yes, not the reverse direction or Attrition=No.
        res = AprioriEngine.run_apriori(self._attr_tx(), 50, 50, target_mode="yes")
        self.assertGreaterEqual(len(res["valid_rules"]), 1)
        for r in res["valid_rules"]:
            self.assertEqual(r["rhs"], ["Attrition=Yes"])

    def test_target_attrition_keeps_yes_and_no(self):
        res = AprioriEngine.run_apriori(self._attr_tx(), 50, 50, target_mode="attrition")
        self.assertTrue(all(any(i.startswith("Attrition=") for i in r["rhs"])
                            for r in res["valid_rules"]))
        rhs_vals = {i for r in res["valid_rules"] for i in r["rhs"]}
        self.assertIn("Attrition=No", rhs_vals)

    def test_target_all_is_superset_of_attrition(self):
        # WHY: 'all' must not silently drop the attrition-target rules.
        all_res = AprioriEngine.run_apriori(self._attr_tx(), 50, 50, target_mode="all")
        attr_res = AprioriEngine.run_apriori(self._attr_tx(), 50, 50, target_mode="attrition")
        self.assertGreater(len(all_res["valid_rules"]), len(attr_res["valid_rules"]))

    def test_defaults_preserve_unfiltered_behavior(self):
        # WHY: existing callers pass no new args; defaults must not filter.
        res = AprioriEngine.run_apriori(self._attr_tx(), 50, 50)
        self.assertGreater(len(res["valid_rules"]), 0)
