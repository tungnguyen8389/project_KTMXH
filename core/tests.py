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
