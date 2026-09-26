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

    # ---------- Weather 14-row dataset (slide Bai5 s.18) ----------
    WEATHER_14 = [
        {"Outlook": "Sunny",    "Temp": "Hot",  "Humidity": "High",   "Wind": "Weak",   "Play": "No"},
        {"Outlook": "Sunny",    "Temp": "Hot",  "Humidity": "High",   "Wind": "Strong", "Play": "No"},
        {"Outlook": "Overcast", "Temp": "Hot",  "Humidity": "High",   "Wind": "Weak",   "Play": "Yes"},
        {"Outlook": "Rain",     "Temp": "Mild", "Humidity": "High",   "Wind": "Weak",   "Play": "Yes"},
        {"Outlook": "Rain",     "Temp": "Cool", "Humidity": "Normal", "Wind": "Weak",   "Play": "Yes"},
        {"Outlook": "Rain",     "Temp": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
        {"Outlook": "Overcast", "Temp": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
        {"Outlook": "Sunny",    "Temp": "Mild", "Humidity": "High",   "Wind": "Weak",   "Play": "No"},
        {"Outlook": "Sunny",    "Temp": "Cool", "Humidity": "Normal", "Wind": "Weak",   "Play": "Yes"},
        {"Outlook": "Rain",     "Temp": "Mild", "Humidity": "Normal", "Wind": "Weak",   "Play": "Yes"},
        {"Outlook": "Sunny",    "Temp": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
        {"Outlook": "Overcast", "Temp": "Mild", "Humidity": "High",   "Wind": "Strong", "Play": "Yes"},
        {"Outlook": "Overcast", "Temp": "Hot",  "Humidity": "Normal", "Wind": "Weak",   "Play": "Yes"},
        {"Outlook": "Rain",     "Temp": "Mild", "Humidity": "High",   "Wind": "Strong", "Play": "No"},
    ]

    def test_id3_weather_binary_gain(self):
        attrs = ["Outlook", "Temp", "Humidity", "Wind"]
        res = ClassificationEngine.run_id3(self.WEATHER_14, attrs, "Play", criterion="gain")
        self.assertEqual(res["tree_structure"]["label"], "Outlook")
        gain_outlook = res["steps"][0]["attr_details"]["Outlook"]["gain"]
        self.assertAlmostEqual(gain_outlook, 0.2467, places=2)
        overcast_child = next(
            c["child_node"] for c in res["tree_structure"]["children"] if c["branch_value"] == "Overcast"
        )
        self.assertTrue(overcast_child["is_leaf"])
        self.assertEqual(overcast_child["label"], "Yes")

    def test_id3_multiclass_no_class_dropped(self):
        data = [
            {"F1": "a", "F2": "x", "T": "A"},
            {"F1": "a", "F2": "y", "T": "A"},
            {"F1": "b", "F2": "x", "T": "B"},
            {"F1": "b", "F2": "y", "T": "B"},
            {"F1": "c", "F2": "x", "T": "C"},
            {"F1": "c", "F2": "y", "T": "C"},
        ]
        res = ClassificationEngine.run_id3(data, ["F1", "F2"], "T")
        self.assertEqual(set(res["classes"]), {"A", "B", "C"})
        labels = set()

        def collect(node):
            if node["is_leaf"]:
                labels.add(node["label"])
            else:
                for ch in node["children"]:
                    collect(ch["child_node"])

        collect(res["tree_structure"])
        self.assertEqual(labels, {"A", "B", "C"})

    def test_id3_gini_runs(self):
        attrs = ["Outlook", "Temp", "Humidity", "Wind"]
        res = ClassificationEngine.run_id3(self.WEATHER_14, attrs, "Play", criterion="gini")
        self.assertEqual(res["criterion"], "gini")
        self.assertFalse(res["tree_structure"]["is_leaf"])
        self.assertIn("gini_split", res["steps"][0]["attr_details"]["Outlook"])

    def test_id3_quinlan_selects_pure_attr(self):
        # F1 pure theo class (a→A, b→B, c→C: 3 vector đơn vị); F2 hoàn toàn không phân biệt.
        data = [
            {"F1": "a", "F2": "x", "T": "A"},
            {"F1": "a", "F2": "y", "T": "A"},
            {"F1": "b", "F2": "x", "T": "B"},
            {"F1": "b", "F2": "y", "T": "B"},
            {"F1": "c", "F2": "x", "T": "C"},
            {"F1": "c", "F2": "y", "T": "C"},
        ]
        res = ClassificationEngine.run_id3(data, ["F1", "F2"], "T", criterion="quinlan")
        self.assertEqual(res["tree_structure"]["label"], "F1")
        self.assertEqual(res["steps"][0]["attr_details"]["F1"]["unit_count"], 3)

    # ---------- Naive Bayes (slide Bai5.1 s.7-13, dataset 9 dòng) ----------
    NB_9 = [
        {"ThoiTiet": "Nang", "NhietDo": "Nong", "DoAm": "Cao",       "Gio": "Yeu",  "DiChoi": "No"},
        {"ThoiTiet": "Nang", "NhietDo": "Nong", "DoAm": "Cao",       "Gio": "Manh", "DiChoi": "No"},
        {"ThoiTiet": "Uam",  "NhietDo": "Nong", "DoAm": "Cao",       "Gio": "Manh", "DiChoi": "Yes"},
        {"ThoiTiet": "Mua",  "NhietDo": "Mat",  "DoAm": "Cao",       "Gio": "Yeu",  "DiChoi": "Yes"},
        {"ThoiTiet": "Mua",  "NhietDo": "Lanh", "DoAm": "Cao",       "Gio": "Manh", "DiChoi": "No"},
        {"ThoiTiet": "Mua",  "NhietDo": "Lanh", "DoAm": "BinhThuong", "Gio": "Manh", "DiChoi": "No"},
        {"ThoiTiet": "Uam",  "NhietDo": "Lanh", "DoAm": "BinhThuong", "Gio": "Yeu",  "DiChoi": "Yes"},
        {"ThoiTiet": "Nang", "NhietDo": "Mat",  "DoAm": "Cao",       "Gio": "Yeu",  "DiChoi": "No"},
        {"ThoiTiet": "Nang", "NhietDo": "Lanh", "DoAm": "BinhThuong", "Gio": "Yeu",  "DiChoi": "Yes"},
    ]

    def test_nb_weather_no_laplace(self):
        res = ClassificationEngine.run_naive_bayes(
            self.NB_9,
            ["ThoiTiet", "NhietDo"],
            "DiChoi",
            {"ThoiTiet": "Nang", "NhietDo": "Nong"},
            use_laplace=False,
        )
        # Slide 10: P(Nang|Yes)=1/4=0.25, P(Nang|No)=3/5=0.6.
        self.assertAlmostEqual(res["likelihood_table"]["ThoiTiet"]["Nang"]["Yes"], 0.25, places=3)
        self.assertAlmostEqual(res["likelihood_table"]["ThoiTiet"]["Nang"]["No"], 0.6, places=3)
        # Slide 12: P(Yes|Nang,Nong)=0.028, P(No|Nang,Nong)=0.133 (unnormalized).
        self.assertAlmostEqual(res["posterior_scores"]["Yes"], 0.0278, places=3)
        self.assertAlmostEqual(res["posterior_scores"]["No"], 0.1333, places=3)
        self.assertEqual(res["predicted_class"], "No")

    def test_nb_laplace_no_zero(self):
        res = ClassificationEngine.run_naive_bayes(
            self.NB_9,
            ["ThoiTiet", "NhietDo", "DoAm", "Gio"],
            "DiChoi",
            {"ThoiTiet": "Uam", "NhietDo": "Mat", "DoAm": "BinhThuong", "Gio": "Yeu"},
            use_laplace=True,
        )
        for attr, val_map in res["likelihood_table"].items():
            for val, class_map in val_map.items():
                for cls, prob in class_map.items():
                    self.assertGreater(prob, 0.0, f"{attr}={val}|{cls} bị 0 dù bật Laplace")
