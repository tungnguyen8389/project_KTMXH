import itertools

class AprioriEngine:
    """
    [TV2] Apriori Association Rule Mining Engine
    Tracks bitvectors v(X), Candidate itemsets C_k, Frequent itemsets F_k,
    minsupp %, minconf %, and association rules with Support, Confidence, and Lift.
    """

    @staticmethod
    def run_apriori(transactions, min_supp_pct=50.0, min_conf_pct=70.0):
        """
        transactions: list of lists or dicts [{'tid': 'T1', 'items': ['A', 'B', 'E']}, ...]
        min_supp_pct: float (e.g. 50.0%)
        min_conf_pct: float (e.g. 70.0%)
        """
        # Parse transaction database
        tid_list = []
        raw_tx_items = []
        all_unique_items = set()

        for i, tx in enumerate(transactions):
            tid = tx.get('tid', f"T{i+1}")
            items = set(tx.get('items', []))
            tid_list.append(tid)
            raw_tx_items.append(items)
            all_unique_items.update(items)

        sorted_all_items = sorted(list(all_unique_items))
        n_tx = len(raw_tx_items)
        min_supp_count = (min_supp_pct / 100.0) * n_tx

        # 1. Compute Bitvectors v(X) for each item across transactions
        bitvectors = {}
        for item in sorted_all_items:
            bv = [1 if item in tx else 0 for tx in raw_tx_items]
            bitvectors[item] = bv

        itemset_steps = []
        frequent_itemsets = {} # map frozenset -> support count

        # Level k = 1
        c1 = [frozenset([item]) for item in sorted_all_items]
        f1 = {}
        c1_details = []

        for itemset in c1:
            item = list(itemset)[0]
            supp_cnt = sum(bitvectors[item])
            supp_pct = (supp_cnt / n_tx) * 100.0
            is_freq = supp_cnt >= min_supp_count
            c1_details.append({
                "itemset": sorted(list(itemset)),
                "support_count": supp_cnt,
                "support_pct": round(supp_pct, 2),
                "is_frequent": is_freq
            })
            if is_freq:
                f1[itemset] = supp_cnt
                frequent_itemsets[itemset] = supp_cnt

        itemset_steps.append({
            "k": 1,
            "candidates_C_k": c1_details,
            "frequent_F_k": [{"itemset": sorted(list(k)), "support_count": v, "support_pct": round((v/n_tx)*100, 2)} for k, v in f1.items()]
        })

        current_f = f1
        k = 2

        while current_f:
            prev_fsets = list(current_f.keys())
            # Candidate generation C_k by joining F_{k-1}
            candidate_ck = set()
            for i in range(len(prev_fsets)):
                for j in range(i + 1, len(prev_fsets)):
                    union_set = prev_fsets[i].union(prev_fsets[j])
                    if len(union_set) == k:
                        candidate_ck.add(union_set)

            if not candidate_ck:
                break

            ck_details = []
            next_f = {}

            for c_set in sorted(list(candidate_ck), key=lambda x: sorted(list(x))):
                # Count support by checking transactions
                supp_cnt = sum(1 for tx in raw_tx_items if c_set.issubset(tx))
                supp_pct = (supp_cnt / n_tx) * 100.0
                is_freq = supp_cnt >= min_supp_count

                ck_details.append({
                    "itemset": sorted(list(c_set)),
                    "support_count": supp_cnt,
                    "support_pct": round(supp_pct, 2),
                    "is_frequent": is_freq
                })

                if is_freq:
                    next_f[c_set] = supp_cnt
                    frequent_itemsets[c_set] = supp_cnt

            itemset_steps.append({
                "k": k,
                "candidates_C_k": ck_details,
                "frequent_F_k": [{"itemset": sorted(list(k_set)), "support_count": v, "support_pct": round((v/n_tx)*100, 2)} for k_set, v in next_f.items()]
            })

            current_f = next_f
            k += 1

        # 2. Rule Generation from Frequent Itemsets (size >= 2)
        generated_rules = []
        min_conf = min_conf_pct / 100.0

        for f_set, supp_AB in frequent_itemsets.items():
            if len(f_set) >= 2:
                items_in_set = list(f_set)
                # Generate non-empty proper subsets A
                for r in range(1, len(items_in_set)):
                    for subset_A_tuple in itertools.combinations(items_in_set, r):
                        subset_A = frozenset(subset_A_tuple)
                        subset_B = f_set - subset_A
                        supp_A = frequent_itemsets.get(subset_A, 0)
                        supp_B = frequent_itemsets.get(subset_B, 0)

                        if supp_A > 0:
                            conf = supp_AB / supp_A
                            conf_pct = round(conf * 100.0, 2)
                            lift = round(supp_AB / (supp_A * (supp_B / n_tx)), 2) if supp_B > 0 else 0.0
                            is_valid_rule = conf >= min_conf

                            generated_rules.append({
                                "rule_str": f"{', '.join(sorted(list(subset_A)))} ➔ {', '.join(sorted(list(subset_B)))}",
                                "lhs": sorted(list(subset_A)),
                                "rhs": sorted(list(subset_B)),
                                "support_AB_count": supp_AB,
                                "support_A_count": supp_A,
                                "support_pct": round((supp_AB / n_tx) * 100.0, 2),
                                "confidence_pct": conf_pct,
                                "lift": lift,
                                "is_valid": is_valid_rule
                            })

        valid_rules = [r for r in generated_rules if r['is_valid']]

        return {
            "num_transactions": n_tx,
            "min_supp_pct": min_supp_pct,
            "min_conf_pct": min_conf_pct,
            "all_items": sorted_all_items,
            "bitvectors": {item: bv for item, bv in bitvectors.items()},
            "itemset_steps": itemset_steps,
            "generated_rules": generated_rules,
            "valid_rules": valid_rules
        }
