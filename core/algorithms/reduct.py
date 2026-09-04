import pandas as pd
import itertools

class ReductEngine:
    """
    [TV1] Reduct & Discriminiability Matrix Engine
    Constructs n x n Discriminiability Matrix, proposition logic Boolean expansion,
    absorption reduction, Minimal Reducts, and Core Attributes.
    """

    @staticmethod
    def compute_reducts(data_list, condition_attrs, decision_attr):
        df = pd.DataFrame(data_list)
        if 'id' not in df.columns:
            df['id'] = [f"x{i+1}" for i in range(len(df))]

        objects = list(df['id'])
        n = len(df)

        # 1. Construct Discriminiability Matrix (n x n)
        matrix = []
        non_empty_clauses = []

        for i in range(n):
            row_matrix = []
            for j in range(n):
                if i <= j:
                    row_matrix.append("-")
                else:
                    d_i = df.iloc[i][decision_attr]
                    d_j = df.iloc[j][decision_attr]
                    if d_i != d_j:
                        diff_attrs = []
                        for attr in condition_attrs:
                            if df.iloc[i][attr] != df.iloc[j][attr]:
                                diff_attrs.append(attr)
                        diff_attrs = sorted(diff_attrs)
                        row_matrix.append(", ".join(diff_attrs) if diff_attrs else "Ø")
                        if diff_attrs:
                            clause = set(diff_attrs)
                            if clause not in non_empty_clauses:
                                non_empty_clauses.append(clause)
                    else:
                        row_matrix.append("Ø")
            matrix.append(row_matrix)

        # 2. Convert non-empty clauses to Boolean formula representation
        # f_M = (a v b) ^ (a v c) ^ ...
        boolean_terms = ["(" + " ∨ ".join(sorted(list(clause))) + ")" for clause in non_empty_clauses]
        formula_str = " ∧ ".join(boolean_terms) if boolean_terms else "1"

        # 3. Compute Reducts via set expansion & absorption
        # A reduct is a minimal subset R of C such that R intersects every non-empty clause in non_empty_clauses
        minimal_reducts = []
        c_set = set(condition_attrs)

        # Iterate subsets of C by increasing size
        for k in range(1, len(condition_attrs) + 1):
            for subset in itertools.combinations(condition_attrs, k):
                sub_set = set(subset)
                # Check if sub_set intersects every non-empty clause in non_empty_clauses
                is_valid = True
                for clause in non_empty_clauses:
                    if not sub_set.intersection(clause):
                        is_valid = False
                        break
                if is_valid:
                    # Check if sub_set is minimal (no existing reduct is a subset of this sub_set)
                    is_minimal = True
                    for red in minimal_reducts:
                        if set(red).issubset(sub_set):
                            is_minimal = False
                            break
                    if is_minimal:
                        minimal_reducts.append(sorted(list(subset)))

        # 4. Core attributes = Intersection of all minimal reducts
        if minimal_reducts:
            core_set = set(minimal_reducts[0])
            for red in minimal_reducts[1:]:
                core_set = core_set.intersection(set(red))
            core_attrs = sorted(list(core_set))
        else:
            core_attrs = []

        return {
            "objects": objects,
            "condition_attrs": condition_attrs,
            "decision_attr": decision_attr,
            "matrix_n_x_n": matrix,
            "boolean_formula": formula_str,
            "clauses": [sorted(list(c)) for c in non_empty_clauses],
            "minimal_reducts": minimal_reducts,
            "core_attributes": core_attrs,
            "katex_formula": rf"f_M = {formula_str.replace('∨', r'\vee').replace('∧', r'\wedge')}" if formula_str != "1" else r"f_M = 1",
            "katex_core": rf"CORE(C) = \bigcap RED(C) = \{{{', '.join(core_attrs) if core_attrs else r'\emptyset'}\}}"
        }
