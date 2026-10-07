import pandas as pd
import itertools

class ReductEngine:
    """
    [TV1] Reduct & Discriminiability Matrix Engine
    Constructs n x n Discriminiability Matrix, proposition logic Boolean expansion,
    absorption reduction, Minimal Reducts, and Core Attributes.
    """

    MAX_CONDITION_ATTRS = 12
    MAX_MATRIX_OBJECTS = 60

    @staticmethod
    def compute_reducts(data_list, condition_attrs, decision_attr):
        if len(condition_attrs) > ReductEngine.MAX_CONDITION_ATTRS:
            raise ValueError(
                f"Vui lòng chọn tối đa {ReductEngine.MAX_CONDITION_ATTRS} thuộc tính điều kiện "
                f"để tránh bùng nổ tổ hợp 2^|C| (hiện có {len(condition_attrs)})."
            )
        df = pd.DataFrame(data_list)
        if 'id' not in df.columns:
            df['id'] = [f"x{i+1}" for i in range(len(df))]

        n = len(df)
        matrix_truncated = n > ReductEngine.MAX_MATRIX_OBJECTS
        non_empty_clauses = []
        matrix = []

        def add_clause(diff_attrs):
            clause = set(diff_attrs)
            if clause not in non_empty_clauses:
                non_empty_clauses.append(clause)

        if not matrix_truncated:
            # 1. Discernibility matrix (n x n), shown step by step in the UI.
            objects = list(df['id'])
            for i in range(n):
                row_matrix = []
                for j in range(n):
                    if i == j:
                        row_matrix.append("Ø")  # an object vs itself: nothing differs (slide shows ∅ on the diagonal)
                    elif i < j:
                        row_matrix.append("-")  # symmetric half, not shown
                    else:
                        d_i = df.iloc[i][decision_attr]
                        d_j = df.iloc[j][decision_attr]
                        if d_i != d_j:
                            diff_attrs = sorted(
                                a for a in condition_attrs
                                if df.iloc[i][a] != df.iloc[j][a])
                            row_matrix.append(", ".join(diff_attrs) if diff_attrs else "Ø")
                            if diff_attrs:
                                add_clause(diff_attrs)
                        else:
                            row_matrix.append("Ø")
                matrix.append(row_matrix)
        else:
            # 1'. Large tables: a clause only depends on the pair of distinct
            # (condition values, decision) rows, so scan those instead of all
            # n^2 object pairs. Same clause set, orders of magnitude faster.
            objects = []
            distinct = df[condition_attrs + [decision_attr]].drop_duplicates().values.tolist()
            k = len(condition_attrs)
            for i in range(len(distinct)):
                for j in range(i):
                    if distinct[i][k] != distinct[j][k]:
                        diff_attrs = sorted(
                            condition_attrs[c] for c in range(k)
                            if distinct[i][c] != distinct[j][c])
                        if diff_attrs:
                            add_clause(diff_attrs)

        # 2. Convert non-empty clauses to Boolean formula representation
        # f_M = (a v b) ^ (a v c) ^ ...
        boolean_terms = ["(" + " ∨ ".join(sorted(list(clause))) + ")" for clause in non_empty_clauses]
        formula_str = " ∧ ".join(boolean_terms) if boolean_terms else "1"

        # 2b. CORE computed directly from the matrix: an attribute is indispensable
        # (belongs to every reduct) iff it appears alone in some c_ij (singleton
        # clause) — cheaper than waiting for the full reduct search below, and
        # matches how the theorem is taught (slide: single-attribute cells -> Core).
        core_from_matrix = sorted({next(iter(clause)) for clause in non_empty_clauses if len(clause) == 1})

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
            "matrix_truncated": matrix_truncated,
            "num_objects": n,
            "boolean_formula": formula_str,
            "clauses": [sorted(list(c)) for c in non_empty_clauses],
            "minimal_reducts": minimal_reducts,
            "core_attributes": core_attrs,
            "core_from_matrix": core_from_matrix,
            "katex_formula": rf"f_M = {formula_str.replace('∨', r'\vee').replace('∧', r'\wedge')}" if formula_str != "1" else r"f_M = 1",
            "katex_core": rf"CORE(C) = \bigcap RED(C) = \{{{', '.join(core_attrs) if core_attrs else r'\emptyset'}\}}"
        }
