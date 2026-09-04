import pandas as pd
import numpy as np

class RoughSetEngine:
    """
    [TV1] Rough Set Engine
    Calculates IND relation, Lower/Upper Approximations, Boundary Region,
    Accuracy of Approximation, Positive Region, and Dependency Degree k.
    """

    @staticmethod
    def analyze_rough_set(data_list, condition_attrs, decision_attr):
        """
        Input:
          data_list: list of dicts (rows with 'id' or object index, attributes, decision)
          condition_attrs: list of strings (e.g. ['a', 'b', 'c'])
          decision_attr: string (e.g. 'd')
        """
        df = pd.DataFrame(data_list)
        if 'id' not in df.columns:
            df['id'] = [f"x{i+1}" for i in range(len(df))]

        universe = list(df['id'])
        u_size = len(universe)

        # 1. Compute Equivalence Classes U/IND(B)
        grouped_b = df.groupby(condition_attrs)
        ind_b_classes = []
        ind_b_dict = {}

        for _, group in grouped_b:
            objs = list(group['id'])
            ind_b_classes.append(objs)
            for obj in objs:
                ind_b_dict[obj] = objs

        # 2. Compute Decision Classes U/IND(D)
        grouped_d = df.groupby(decision_attr)
        decision_classes = {}
        for d_val, group in grouped_d:
            decision_classes[str(d_val)] = list(group['id'])

        # 3. For each decision class X in U/D, compute Lower, Upper, BN, Accuracy
        approx_results = {}
        all_lower_union = set()

        for d_val, x_set in decision_classes.items():
            x_set_set = set(x_set)
            lower = []
            upper = []

            for eq_cls in ind_b_classes:
                eq_set = set(eq_cls)
                # Lower: [x]_B is subset of X
                if eq_set.issubset(x_set_set):
                    lower.extend(eq_cls)
                # Upper: [x]_B intersects X
                if len(eq_set.intersection(x_set_set)) > 0:
                    upper.extend(eq_cls)

            lower = sorted(list(set(lower)))
            upper = sorted(list(set(upper)))
            boundary = sorted(list(set(upper) - set(lower)))
            accuracy = round(len(lower) / len(upper), 4) if len(upper) > 0 else 1.0

            all_lower_union.update(lower)

            approx_results[d_val] = {
                "decision_val": d_val,
                "target_set_X": sorted(x_set),
                "lower_approx": lower,
                "upper_approx": upper,
                "boundary_region": boundary,
                "accuracy": accuracy,
                "katex_lower": rf"B_{{\underline{{A}}}}(X_{{{d_val}}}) = \{{{', '.join(lower)}\}}",
                "katex_upper": rf"B^{{\overline{{A}}}}(X_{{{d_val}}}) = \{{{', '.join(upper)}\}}",
                "katex_bn": rf"BN_B(X_{{{d_val}}}) = \{{{', '.join(boundary)}\}}",
                "katex_accuracy": rf"\alpha_B(X_{{{d_val}}}) = \frac{{|B_{{\underline{{A}}}}(X_{{{d_val}}})|}}{{|B^{{\overline{{A}}}}(X_{{{d_val}}})|}} = \frac{{{len(lower)}}}{{{len(upper)}}} = {accuracy}"
            }

        # 4. Compute Positive Region POS_B(D) and Dependency Degree k
        pos_b_d = sorted(list(all_lower_union))
        dependency_k = round(len(pos_b_d) / u_size, 4) if u_size > 0 else 0.0

        katex_pos = rf"POS_B(D) = \{{{', '.join(pos_b_d)}\}}"
        katex_k = rf"k = \gamma_B(D) = \frac{{|POS_B(D)|}}{{|U|}} = \frac{{{len(pos_b_d)}}}{{{u_size}}} = {dependency_k}"

        return {
            "universe": universe,
            "condition_attrs": condition_attrs,
            "decision_attr": decision_attr,
            "ind_b_classes": ind_b_classes,
            "decision_classes": decision_classes,
            "approximations": approx_results,
            "positive_region": pos_b_d,
            "dependency_k": dependency_k,
            "katex_pos": katex_pos,
            "katex_k": katex_k
        }
