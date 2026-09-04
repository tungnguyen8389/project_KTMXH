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
    def _entropy(p, n):
        total = p + n
        if total == 0 or p == 0 or n == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return - (p_ratio * math.log2(p_ratio) + n_ratio * math.log2(n_ratio))

    @classmethod
    def run_id3(cls, data_list, condition_attrs, target_attr):
        df = pd.DataFrame(data_list)
        target_vals = sorted(list(df[target_attr].unique()))
        
        # Binary target assumed e.g. Yes/No or Play/Don't Play
        pos_val = target_vals[0] if len(target_vals) > 0 else "Yes"
        neg_val = target_vals[1] if len(target_vals) > 1 else "No"

        steps = []
        node_counter = [0]

        def build_tree(sub_df, curr_attrs, parent_label="Root"):
            node_counter[0] += 1
            node_id = f"node_{node_counter[0]}"

            p = len(sub_df[sub_df[target_attr] == pos_val])
            n = len(sub_df[sub_df[target_attr] == neg_val])
            total = len(sub_df)

            if p == total:
                return {"id": node_id, "label": f"{pos_val}", "is_leaf": True, "type": "leaf", "count": total}
            if n == total:
                return {"id": node_id, "label": f"{neg_val}", "is_leaf": True, "type": "leaf", "count": total}
            if not curr_attrs:
                majority = pos_val if p >= n else neg_val
                return {"id": node_id, "label": f"{majority}", "is_leaf": True, "type": "leaf", "count": total}

            total_info = cls._entropy(p, n)

            gains = {}
            attr_details = {}

            for attr in curr_attrs:
                e_A = 0.0
                val_counts = sub_df.groupby([attr, target_attr]).size().unstack(fill_value=0)
                
                sub_details = []
                for val, row in val_counts.iterrows():
                    val_p = row.get(pos_val, 0)
                    val_n = row.get(neg_val, 0)
                    val_total = val_p + val_n
                    val_ent = cls._entropy(val_p, val_n)
                    e_A += (val_total / total) * val_ent
                    sub_details.append({
                        "val": str(val),
                        "p": int(val_p),
                        "n": int(val_n),
                        "entropy": round(val_ent, 4)
                    })

                gain = total_info - e_A
                gains[attr] = round(gain, 4)
                attr_details[attr] = {
                    "expected_entropy_E": round(e_A, 4),
                    "gain": round(gain, 4),
                    "sub_details": sub_details
                }

            best_attr = max(gains, key=gains.get)

            steps.append({
                "parent_label": parent_label,
                "node_samples": total,
                "p_count": p,
                "n_count": n,
                "info_p_n": round(total_info, 4),
                "attr_details": attr_details,
                "selected_best_attr": best_attr,
                "max_gain": gains[best_attr]
            })

            children = []
            remaining_attrs = [a for a in curr_attrs if a != best_attr]

            for val in sorted(sub_df[best_attr].unique()):
                child_df = sub_df[sub_df[best_attr] == val]
                child_tree = build_tree(child_df, remaining_attrs, parent_label=f"{best_attr}={val}")
                children.append({
                    "branch_value": str(val),
                    "child_node": child_tree
                })

            return {
                "id": node_id,
                "label": best_attr,
                "is_leaf": False,
                "children": children
            }

        tree_structure = build_tree(df, condition_attrs)

        # Generate Mermaid.js graph string (graph TD)
        mermaid_lines = ["graph TD"]

        def generate_mermaid(node):
            if node["is_leaf"]:
                mermaid_lines.append(f'    {node["id"]}[["{node["label"]}"]]')
            else:
                mermaid_lines.append(f'    {node["id"]}{{"{node["label"]}"}}')
                for child_edge in node.get("children", []):
                    branch_val = child_edge["branch_value"]
                    child_node = child_edge["child_node"]
                    generate_mermaid(child_node)
                    mermaid_lines.append(f'    {node["id"]} -->|"{branch_val}"| {child_node["id"]}')

        generate_mermaid(tree_structure)
        mermaid_graph = "\n".join(mermaid_lines)

        return {
            "target_attr": target_attr,
            "pos_val": pos_val,
            "neg_val": neg_val,
            "total_samples": len(df),
            "steps": steps,
            "tree_structure": tree_structure,
            "mermaid_graph": mermaid_graph
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
