import math
import pandas as pd
import numpy as np

class ClassificationEngine:
    """
    [TV3] Classification Engine: ID3 Decision Tree & Naive Bayes Classifier
    Calculates Entropy I(p,n), Expected Entropy E(A), Gain(A), builds Mermaid.js TD graph,
    and Naive Bayes conditional probabilities with optional Laplace smoothing.
    """

    # ------------------ ID3 DECISION TREE ------------------
    @staticmethod
    def _entropy_from_counts(counts_dict, total):
        if total <= 0:
            return 0.0
        h = 0.0
        for v in counts_dict.values():
            if v > 0:
                p = v / total
                h -= p * math.log2(p)
        return h

    @classmethod
    def run_id3(cls, data_list, condition_attrs, target_attr, criterion='gain'):
        """
        Build ID3 tree using one of two selection criteria from the course slides:
          - 'gain'    : Information Gain (Slide 40, entropy-based, multi-class).
          - 'quinlan' : Quinlan unit-vector heuristic (Slide 27, 33); pick attribute producing
                        the most pure (single-class) child subsets, tie-broken by Information Gain.
        """
        if criterion not in ('gain', 'quinlan'):
            criterion = 'gain'

        df = pd.DataFrame(data_list)
        classes = sorted(str(c) for c in df[target_attr].unique())

        steps = []
        node_counter = [0]

        def counts_of(sub_df):
            vc = sub_df[target_attr].astype(str).value_counts().to_dict()
            return {c: int(vc.get(c, 0)) for c in classes}

        def build_tree(sub_df, curr_attrs, parent_label="Root"):
            node_counter[0] += 1
            node_id = f"node_{node_counter[0]}"
            total = len(sub_df)
            cc = counts_of(sub_df)

            # Pure node: only one class present.
            nonzero = [c for c, v in cc.items() if v > 0]
            if len(nonzero) == 1:
                return {"id": node_id, "label": nonzero[0], "is_leaf": True, "type": "leaf", "count": total}
            if not curr_attrs or total == 0:
                majority = max(cc, key=cc.get) if cc else "?"
                return {"id": node_id, "label": majority, "is_leaf": True, "type": "leaf", "count": total}

            parent_entropy = cls._entropy_from_counts(cc, total)

            attr_details = {}
            scores = {}  # higher = better

            for attr in curr_attrs:
                e_A = 0.0       # weighted entropy of children
                unit_count = 0  # # of pure child subsets (Quinlan)
                sub_details = []

                for val, gdf in sub_df.groupby(attr):
                    val_total = len(gdf)
                    val_cc = counts_of(gdf)
                    val_ent = cls._entropy_from_counts(val_cc, val_total)
                    e_A += (val_total / total) * val_ent
                    if val_total > 0 and max(val_cc.values()) == val_total:
                        unit_count += 1
                    sub_details.append({
                        "val": str(val),
                        "counts": val_cc,
                        "total": val_total,
                        "entropy": round(val_ent, 4),
                    })

                gain = parent_entropy - e_A
                attr_details[attr] = {
                    "expected_entropy_E": round(e_A, 4),
                    "gain": round(gain, 4),
                    "unit_count": unit_count,
                    "sub_details": sub_details,
                }

                if criterion == 'quinlan':
                    # Prefer more unit-vector branches; tie-break with Information Gain.
                    scores[attr] = (unit_count, gain)
                else:
                    scores[attr] = (gain,)

            best_attr = max(scores, key=scores.get)

            steps.append({
                "parent_label": parent_label,
                "node_samples": total,
                "class_counts": cc,
                "info": round(parent_entropy, 4),
                "attr_details": attr_details,
                "selected_best_attr": best_attr,
            })

            children = []
            remaining_attrs = [a for a in curr_attrs if a != best_attr]
            for val in sorted(sub_df[best_attr].unique(), key=str):
                child_df = sub_df[sub_df[best_attr] == val]
                child_tree = build_tree(child_df, remaining_attrs, parent_label=f"{best_attr}={val}")
                children.append({"branch_value": str(val), "child_node": child_tree})

            return {"id": node_id, "label": best_attr, "is_leaf": False, "children": children}

        tree_structure = build_tree(df, condition_attrs)

        # Generate Mermaid.js graph string (graph TD).
        # Labels from real HR data can contain '&', '"', etc. which crash the
        # Mermaid parser; escape them as HTML entities inside "..." wrappers.
        def _esc(s):
            return (str(s)
                    .replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

        mermaid_lines = ["graph TD"]

        def generate_mermaid(node):
            label = _esc(node["label"])
            if node["is_leaf"]:
                mermaid_lines.append(f'    {node["id"]}[["{label}"]]')
            else:
                mermaid_lines.append(f'    {node["id"]}{{"{label}"}}')
                for child_edge in node.get("children", []):
                    branch_val = _esc(child_edge["branch_value"])
                    child_node = child_edge["child_node"]
                    generate_mermaid(child_node)
                    mermaid_lines.append(f'    {node["id"]} -->|"{branch_val}"| {child_node["id"]}')

        generate_mermaid(tree_structure)

        return {
            "target_attr": target_attr,
            "criterion": criterion,
            "classes": classes,
            "total_samples": len(df),
            "steps": steps,
            "tree_structure": tree_structure,
            "mermaid_graph": "\n".join(mermaid_lines),
        }

    # ------------------ NAIVE BAYES ------------------
    @staticmethod
    def run_naive_bayes(data_list, condition_attrs, target_attr, test_instance, use_laplace=False):
        """
        test_instance: dict of feature values e.g. {'Outlook': 'Sunny', 'Temp': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
        use_laplace: boolean (Laplace smoothing)
        """
        df = pd.DataFrame(data_list)
        total_samples = len(df)
        classes = sorted(list(df[target_attr].unique()))

        class_priors = {}
        class_counts = {}

        for c in classes:
            cnt = len(df[df[target_attr] == c])
            class_counts[c] = cnt
            class_priors[c] = cnt / total_samples

        conditional_probs = {}
        posterior_scores = {}
        katex_steps = []

        for c in classes:
            c_df = df[df[target_attr] == c]
            c_count = class_counts[c]
            prod_prob = class_priors[c]
            cond_details = {}
            katex_terms = [f"P({c}) = \\frac{{{c_count}}}{{{total_samples}}}"]

            for attr in condition_attrs:
                val = test_instance.get(attr)
                match_count = len(c_df[c_df[attr] == val])

                if use_laplace:
                    num_vocab = len(df[attr].unique())
                    prob = (match_count + 1) / (c_count + num_vocab)
                    katex_terms.append(f"P({attr}={val} \\mid {c}) = \\frac{{{match_count} + 1}}{{{c_count} + {num_vocab}}} = {prob:.4f}")
                else:
                    prob = match_count / c_count if c_count > 0 else 0.0
                    katex_terms.append(f"P({attr}={val} \\mid {c}) = \\frac{{{match_count}}}{{{c_count}}} = {prob:.4f}")

                cond_details[attr] = {
                    "val": val,
                    "match_count": match_count,
                    "prob": round(prob, 4)
                }
                prod_prob *= prob

            conditional_probs[str(c)] = cond_details
            posterior_scores[str(c)] = prod_prob

            katex_steps.append({
                "class_val": str(c),
                "prior": round(class_priors[c], 4),
                "katex_terms": katex_terms,
                "unnormalized_posterior": round(prod_prob, 6)
            })

        # Normalize probabilities
        total_score = sum(posterior_scores.values())
        normalized_posteriors = {}
        for c, score in posterior_scores.items():
            normalized_posteriors[c] = round(score / total_score, 4) if total_score > 0 else 0.0

        predicted_class = max(normalized_posteriors, key=normalized_posteriors.get)

        return {
            "test_instance": test_instance,
            "use_laplace": use_laplace,
            "class_priors": {str(k): round(v, 4) for k, v in class_priors.items()},
            "katex_steps": katex_steps,
            "posterior_scores": {str(k): round(v, 6) for k, v in posterior_scores.items()},
            "normalized_posteriors": normalized_posteriors,
            "predicted_class": predicted_class
        }
