import math
import pandas as pd


class ClassificationEngine:
    """
    [TV3] Classification Engine: ID3 Decision Tree & Naive Bayes Classifier.

    ID3 hỗ trợ đa lớp theo công thức entropy tổng quát (slide Bai5 s.40):
        I(s1,...,sm) = - sum_i (si/s) * log2(si/s)
    Bổ sung Gini index (s.39) và biến thể Quinlan chọn theo số "vector đơn vị" (s.27, s.33).

    Naive Bayes: argmax_C P(C) * prod_j P(xj | C), có Laplace smoothing tùy chọn (Bai5.1).
    """

    # ------------------ ID3 DECISION TREE ------------------
    @staticmethod
    def _entropy_multiclass(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for c in counts.values():
            if c <= 0:
                continue
            p = c / total
            ent -= p * math.log2(p)
        return ent

    @classmethod
    def _entropy(cls, p, n):
        # Backward-compat wrapper cho binary.
        return cls._entropy_multiclass({"+": p, "-": n})

    @staticmethod
    def _gini(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return 1.0 - sum((c / total) ** 2 for c in counts.values())

    @staticmethod
    def _quinlan_unit_count(sub_df, attr, target_attr):
        # Số giá trị của `attr` mà tại đó toàn bộ mẫu cùng 1 class (leaf thuần).
        grouped = sub_df.groupby(attr)[target_attr].nunique()
        return int((grouped == 1).sum())

    @classmethod
    def run_id3(cls, data_list, condition_attrs, target_attr, criterion="gain"):
        """
        criterion: "gain" (mặc định, Information Gain) | "gini" | "quinlan".
        Giữ chữ ký cũ; các key `pos_val` / `neg_val` / `p_count` / `n_count` được giữ khi
        target đúng 2 lớp để tương thích view/template hiện tại.
        """
        if criterion not in {"gain", "gini", "quinlan"}:
            criterion = "gain"

        df = pd.DataFrame(data_list)
        classes = sorted(df[target_attr].unique().tolist(), key=str)
        is_binary = len(classes) == 2
        pos_val = classes[0] if len(classes) >= 1 else None
        neg_val = classes[1] if len(classes) >= 2 else None

        steps = []
        node_counter = [0]

        def class_counts_of(sub_df):
            vc = sub_df[target_attr].value_counts().to_dict()
            return {c: int(vc.get(c, 0)) for c in classes}

        def leaf_node(node_id, label, total, cc):
            return {
                "id": node_id,
                "label": str(label),
                "is_leaf": True,
                "type": "leaf",
                "count": total,
                "class_counts": cc,
            }

        def build_tree(sub_df, curr_attrs, parent_label="Root"):
            node_counter[0] += 1
            node_id = f"node_{node_counter[0]}"

            total = len(sub_df)
            cc = class_counts_of(sub_df)

            # Điều kiện dừng (slide 26).
            non_zero = [c for c, v in cc.items() if v > 0]
            if total == 0:
                return leaf_node(node_id, "∅", 0, cc)
            if len(non_zero) == 1:
                return leaf_node(node_id, non_zero[0], total, cc)
            if not curr_attrs:
                majority = max(cc, key=cc.get)
                return leaf_node(node_id, majority, total, cc)

            total_info = cls._entropy_multiclass(cc)
            total_gini = cls._gini(cc)

            attr_details = {}
            for attr in curr_attrs:
                e_A = 0.0
                gini_split = 0.0
                sub_details = []
                for val in sorted(sub_df[attr].unique(), key=str):
                    val_df = sub_df[sub_df[attr] == val]
                    val_cc = class_counts_of(val_df)
                    val_total = sum(val_cc.values())
                    val_ent = cls._entropy_multiclass(val_cc)
                    val_gini = cls._gini(val_cc)
                    weight = val_total / total if total else 0.0
                    e_A += weight * val_ent
                    gini_split += weight * val_gini

                    sd = {
                        "val": str(val),
                        "class_counts": val_cc,
                        "entropy": round(val_ent, 4),
                        "gini": round(val_gini, 4),
                    }
                    if is_binary:
                        sd["p"] = int(val_cc.get(pos_val, 0))
                        sd["n"] = int(val_cc.get(neg_val, 0))
                    sub_details.append(sd)

                gain = total_info - e_A
                unit_count = cls._quinlan_unit_count(sub_df, attr, target_attr)
                attr_details[attr] = {
                    "expected_entropy_E": round(e_A, 4),
                    "gain": round(gain, 4),
                    "gini_split": round(gini_split, 4),
                    "unit_count": unit_count,
                    "sub_details": sub_details,
                }

            if criterion == "gini":
                best_attr = min(attr_details, key=lambda a: attr_details[a]["gini_split"])
            elif criterion == "quinlan":
                best_attr = max(
                    attr_details,
                    key=lambda a: (attr_details[a]["unit_count"], attr_details[a]["gain"]),
                )
            else:
                best_attr = max(attr_details, key=lambda a: attr_details[a]["gain"])

            step_entry = {
                "parent_label": parent_label,
                "node_samples": total,
                "class_counts": cc,
                "info": round(total_info, 4),
                "gini": round(total_gini, 4),
                "attr_details": attr_details,
                "selected_best_attr": best_attr,
                "max_gain": attr_details[best_attr]["gain"],
                "criterion": criterion,
            }
            if is_binary:
                step_entry["p_count"] = int(cc.get(pos_val, 0))
                step_entry["n_count"] = int(cc.get(neg_val, 0))
                step_entry["info_p_n"] = round(total_info, 4)
            steps.append(step_entry)

            children = []
            remaining_attrs = [a for a in curr_attrs if a != best_attr]
            for val in sorted(sub_df[best_attr].unique(), key=str):
                child_df = sub_df[sub_df[best_attr] == val]
                child_tree = build_tree(
                    child_df, remaining_attrs, parent_label=f"{best_attr}={val}"
                )
                children.append({"branch_value": str(val), "child_node": child_tree})

            return {
                "id": node_id,
                "label": best_attr,
                "is_leaf": False,
                "children": children,
            }

        tree_structure = build_tree(df, condition_attrs)

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
                    mermaid_lines.append(
                        f'    {node["id"]} -->|"{branch_val}"| {child_node["id"]}'
                    )

        generate_mermaid(tree_structure)
        mermaid_graph = "\n".join(mermaid_lines)

        result = {
            "target_attr": target_attr,
            "classes": [str(c) for c in classes],
            "criterion": criterion,
            "total_samples": len(df),
            "steps": steps,
            "tree_structure": tree_structure,
            "mermaid_graph": mermaid_graph,
        }
        if is_binary:
            result["pos_val"] = str(pos_val)
            result["neg_val"] = str(neg_val)
        return result

    # ------------------ NAIVE BAYES ------------------
    @staticmethod
    def run_naive_bayes(data_list, condition_attrs, target_attr, test_instance, use_laplace=False):
        """
        test_instance: dict giá trị đặc trưng, ví dụ {'Outlook':'Sunny','Temp':'Cool',...}.
        use_laplace: bật smoothing (num+1)/(c_count+|V|) theo từng attr.
        """
        df = pd.DataFrame(data_list)
        total_samples = len(df)
        classes = sorted(df[target_attr].unique().tolist(), key=str)

        class_priors = {}
        class_counts = {}
        for c in classes:
            cnt = int(len(df[df[target_attr] == c]))
            class_counts[c] = cnt
            class_priors[c] = cnt / total_samples if total_samples else 0.0

        # Bảng likelihood P(attr=val | class) theo format slide 15 Bai5.1.
        likelihood_table = {}
        for attr in condition_attrs:
            vocab = sorted(df[attr].unique().tolist(), key=str)
            num_vocab = len(vocab)
            likelihood_table[attr] = {}
            for val in vocab:
                likelihood_table[attr][str(val)] = {}
                for c in classes:
                    c_count = class_counts[c]
                    match = int(len(df[(df[target_attr] == c) & (df[attr] == val)]))
                    if use_laplace:
                        prob = (match + 1) / (c_count + num_vocab) if (c_count + num_vocab) else 0.0
                    else:
                        prob = match / c_count if c_count > 0 else 0.0
                    likelihood_table[attr][str(val)][str(c)] = round(prob, 4)

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
                match_count = int(len(c_df[c_df[attr] == val]))

                if use_laplace:
                    num_vocab = len(df[attr].unique())
                    denom = c_count + num_vocab
                    prob = (match_count + 1) / denom if denom else 0.0
                    katex_terms.append(
                        f"P({attr}={val} \\mid {c}) = "
                        f"\\frac{{{match_count} + 1}}{{{c_count} + {num_vocab}}} = {prob:.4f}"
                    )
                else:
                    prob = match_count / c_count if c_count > 0 else 0.0
                    katex_terms.append(
                        f"P({attr}={val} \\mid {c}) = "
                        f"\\frac{{{match_count}}}{{{c_count}}} = {prob:.4f}"
                    )

                cond_details[attr] = {
                    "val": val,
                    "match_count": match_count,
                    "prob": round(prob, 4),
                }
                prod_prob *= prob

            conditional_probs[str(c)] = cond_details
            posterior_scores[str(c)] = prod_prob

            katex_steps.append({
                "class_val": str(c),
                "prior": round(class_priors[c], 4),
                "katex_terms": katex_terms,
                "unnormalized_posterior": round(prod_prob, 6),
            })

        total_score = sum(posterior_scores.values())
        normalized_posteriors = {}
        for c, score in posterior_scores.items():
            normalized_posteriors[str(c)] = round(score / total_score, 4) if total_score > 0 else 0.0

        predicted_class = max(posterior_scores, key=posterior_scores.get) if posterior_scores else None

        return {
            "test_instance": test_instance,
            "use_laplace": use_laplace,
            "classes": [str(c) for c in classes],
            "class_priors": {str(k): round(v, 4) for k, v in class_priors.items()},
            "likelihood_table": likelihood_table,
            "conditional_probs": conditional_probs,
            "katex_steps": katex_steps,
            "posterior_scores": {str(k): round(v, 6) for k, v in posterior_scores.items()},
            "normalized_posteriors": normalized_posteriors,
            "predicted_class": str(predicted_class) if predicted_class is not None else None,
        }
