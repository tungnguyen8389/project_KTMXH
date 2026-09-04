import pandas as pd
import numpy as np

class PreprocessingEngine:
    """
    [TV1] Preprocessing Engine for Min-Max and Z-Score Normalization
    Provides transparent step-by-step mathematical calculation steps.
    """

    @staticmethod
    def min_max_normalize(data_list, feature_name, new_min=0.0, new_max=1.0):
        """
        Min-Max Normalization Formula:
        v' = ((v - min_A) / (max_A - min_A)) * (new_max_A - new_min_A) + new_min_A
        """
        values = [float(item[feature_name]) for item in data_list]
        min_v = min(values)
        max_v = max(values)
        range_v = max_v - min_v if max_v != min_v else 1.0
        new_range = new_max - new_min

        steps = []
        result_data = []

        for i, item in enumerate(data_list):
            raw_v = float(item[feature_name])
            norm_v = ((raw_v - min_v) / range_v) * new_range + new_min
            norm_v = round(norm_v, 4)

            step_detail = (
                f"Dòng {i+1} (v = {raw_v}): "
                f"v' = [({raw_v} - {min_v}) / ({max_v} - {min_v})] \\times ({new_max} - {new_min}) + {new_min} "
                f"= ({raw_v - min_v:.4f} / {range_v:.4f}) \\times {new_range} + {new_min} = {norm_v}"
            )
            steps.append(step_detail)

            row_copy = dict(item)
            row_copy[f"{feature_name}_minmax"] = norm_v
            result_data.append(row_copy)

        katex_formula = r"v' = \frac{v - \min_A}{\max_A - \min_A} \times (new\_max_A - new\_min_A) + new\_min_A"

        return {
            "feature_name": feature_name,
            "min_val": min_v,
            "max_val": max_v,
            "new_min": new_min,
            "new_max": new_max,
            "formula_katex": katex_formula,
            "steps": steps,
            "result_data": result_data
        }

    @staticmethod
    def z_score_normalize(data_list, feature_name):
        """
        Z-Score Normalization Formula:
        v' = (v - mean_A) / std_A
        """
        values = [float(item[feature_name]) for item in data_list]
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=0))  # Population standard deviation
        if std_v == 0:
            std_v = 1.0

        steps = []
        result_data = []

        for i, item in enumerate(data_list):
            raw_v = float(item[feature_name])
            norm_v = (raw_v - mean_v) / std_v
            norm_v = round(norm_v, 4)

            step_detail = (
                f"Dòng {i+1} (v = {raw_v}): "
                f"v' = \\frac{{{raw_v} - {mean_v:.4f}}}{{{std_v:.4f}}} "
                f"= \\frac{{{raw_v - mean_v:.4f}}}{{{std_v:.4f}}} = {norm_v}"
            )
            steps.append(step_detail)

            row_copy = dict(item)
            row_copy[f"{feature_name}_zscore"] = norm_v
            result_data.append(row_copy)

        katex_formula = r"v' = \frac{v - \bar{A}}{\sigma_A}"

        return {
            "feature_name": feature_name,
            "mean": round(mean_v, 4),
            "std": round(std_v, 4),
            "formula_katex": katex_formula,
            "steps": steps,
            "result_data": result_data
        }
