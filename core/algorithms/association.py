def _combinations(iterable, r):
    """Hand-coded k-subset generator (no library helpers).

    Yields length-r tuples of items from `iterable` in lexicographic index
    order. Used by rule generation so the mining algorithm relies on no
    third-party or stdlib combinatorics helper at all.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


class AprioriEngine:
    """
    [TV2] Apriori Association Rule Mining Engine
    Tracks bitvectors v(X), Candidate itemsets C_k, Frequent itemsets F_k,
    minsupp %, minconf %, and association rules with Support, Confidence, and Lift.
    """

    @staticmethod
    def _vector_product(v_s, v_t):
        """Representation-vector product (x) per slide: z_k = min(s_k, t_k).

        v(S (union) T) = v(S) (x) v(T); the 1-count of the result is the joined
        itemset's support (SPV(v(S)) = SP(S)).
        """
        return [min(a, b) for a, b in zip(v_s, v_t)]

    @staticmethod
    def run_apriori(transactions, min_supp_pct=50.0, min_conf_pct=70.0,
                    max_len=None, min_lift=0.0, target_mode='all',
                    analysis_mode='class'):
        """
        transactions: list of lists or dicts [{'tid': 'T1', 'items': ['A', 'B', 'E']}, ...]
        min_supp_pct: float (e.g. 50.0%)
        min_conf_pct: float (e.g. 70.0%)
        max_len: int or None — cap itemset size k (rule-length control); None = no cap.
        min_lift: float — keep only rules with lift >= this (noise control).
        target_mode: 'all' | 'attrition' | 'yes' — filter rules by consequent:
            'all' keeps every rule; 'attrition' keeps rules whose consequent
            contains an Attrition= token; 'yes' keeps only consequent == [Attrition=Yes].
        analysis_mode: 'class' | 'segment'.
            'class' (default): mine the full database; support/confidence/lift are
                measured over all records (faithful to the slide's SP=|rho(S)|/|O|).
            'segment': restrict the database to records matching target_mode
                (yes -> Attrition=Yes rows) and drop the Attrition tokens, so mining
                describes the common traits WITHIN that segment. target_mode no
                longer filters rules (nothing left to target).
        """
        # Segment mode: keep only records in the chosen class, strip the class
        # tokens (constant within the segment, so noise), and disable the rule-
        # consequent filter. Everything downstream then runs on this sub-database.
        if analysis_mode == 'segment':
            def _in_segment(item_set):
                if target_mode == 'yes':
                    return 'Attrition=Yes' in item_set
                if target_mode == 'attrition':
                    return any(i.startswith('Attrition=') for i in item_set)
                return True

            reduced = []
            for i, tx in enumerate(transactions):
                items = set(tx.get('items', []))
                if _in_segment(items):
                    kept = [i2 for i2 in items if not i2.startswith('Attrition=')]
                    reduced.append({'tid': tx.get('tid', f"T{i+1}"), 'items': kept})
            transactions = reduced
            target_mode = 'all'
            if not transactions:
                raise ValueError("Không có nhân sự nào khớp phân khúc đã chọn.")

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
        # v(X) representation-vector for every FREQUENT itemset, keyed by frozenset.
        # Reused so k>=2 support comes from the (x) product, not a re-scan.
        itemset_vectors = {}

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
                itemset_vectors[itemset] = bitvectors[item]

        itemset_steps.append({
            "k": 1,
            "candidates_C_k": c1_details,
            "frequent_F_k": [{"itemset": sorted(list(k)), "support_count": v, "support_pct": round((v/n_tx)*100, 2)} for k, v in f1.items()]
        })

        current_f = f1
        k = 2

        while current_f and (max_len is None or k <= max_len):
            prev_fsets = list(current_f.keys())
            prev_set = current_f  # membership test for the prune step
            # Buoc ket hop (join): C_k = F_{k-1} join F_{k-1}. Remember one parent
            # pair per candidate so its representation vector can be built via (x).
            candidate_parents = {}
            for i in range(len(prev_fsets)):
                for j in range(i + 1, len(prev_fsets)):
                    union_set = prev_fsets[i].union(prev_fsets[j])
                    if len(union_set) == k and union_set not in candidate_parents:
                        candidate_parents[union_set] = (prev_fsets[i], prev_fsets[j])

            # Buoc rut gon (prune): drop any candidate having a (k-1)-subset that
            # is not frequent — it cannot be part of a frequent k-itemset.
            candidate_ck = set()
            for c_set in candidate_parents:
                subsets_ok = all(
                    frozenset(sub) in prev_set
                    for sub in _combinations(sorted(c_set), k - 1)
                )
                if subsets_ok:
                    candidate_ck.add(c_set)

            if not candidate_ck:
                break

            ck_details = []
            next_f = {}

            for c_set in sorted(list(candidate_ck), key=lambda x: sorted(list(x))):
                # Support via representation-vector product (x): v(c) = v(p1) (x) v(p2),
                # supp = number of 1s in v(c)  (SPV(v(S)) = SP(S)).
                p1, p2 = candidate_parents[c_set]
                c_vec = AprioriEngine._vector_product(itemset_vectors[p1],
                                                      itemset_vectors[p2])
                supp_cnt = sum(c_vec)
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
                    itemset_vectors[c_set] = c_vec

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
                    for subset_A_tuple in _combinations(items_in_set, r):
                        subset_A = frozenset(subset_A_tuple)
                        subset_B = f_set - subset_A
                        supp_A = frequent_itemsets.get(subset_A, 0)
                        supp_B = frequent_itemsets.get(subset_B, 0)

                        if supp_A > 0:
                            conf = supp_AB / supp_A
                            conf_pct = round(conf * 100.0, 2)
                            lift = round(supp_AB / (supp_A * (supp_B / n_tx)), 2) if supp_B > 0 else 0.0
                            is_valid_rule = conf >= min_conf and lift >= min_lift

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

        # Target-oriented filter: keep only rules whose consequent matches the goal.
        def _match_target(rule):
            if target_mode == 'all':
                return True
            rhs = rule['rhs']
            if target_mode == 'attrition':
                return any(i.startswith('Attrition=') for i in rhs)
            if target_mode == 'yes':
                return rhs == ['Attrition=Yes']
            return True

        valid_rules = [r for r in valid_rules if _match_target(r)]

        # 3. Maximal frequent itemsets (tap pho bien toi dai — slide 18):
        # M is maximal if no other frequent itemset is a proper superset of it.
        all_fsets = list(frequent_itemsets.keys())
        maximal_itemsets = []
        for s in all_fsets:
            if not any(s < other for other in all_fsets):
                cnt = frequent_itemsets[s]
                maximal_itemsets.append({
                    "itemset": sorted(list(s)),
                    "support_count": cnt,
                    "support_pct": round((cnt / n_tx) * 100.0, 2),
                })
        maximal_itemsets.sort(key=lambda m: (len(m["itemset"]), m["itemset"]))

        return {
            "num_transactions": n_tx,
            "min_supp_pct": min_supp_pct,
            "min_conf_pct": min_conf_pct,
            "all_items": sorted_all_items,
            "bitvectors": {item: bv for item, bv in bitvectors.items()},
            "itemset_steps": itemset_steps,
            "maximal_itemsets": maximal_itemsets,
            "generated_rules": generated_rules,
            "valid_rules": valid_rules
        }
