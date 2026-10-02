/**
 * [TV4] UI Script for Apriori Itemsets C_k, F_k, Bitvectors, and Rule Cards
 */
// Slide datasets (Bai2_TapPhoBienVaLuatKetHop) for checking results by hand.
// Params are in %, min_lift 0 so no slide rule is hidden by the lift filter.
const AP_PRESETS = {
    slide: {
        params: { supp: 40, conf: 67, lift: 0 },
        tx: { o1: ['i1', 'i2', 'i3'], o2: ['i2', 'i3', 'i4'], o3: ['i2', 'i3', 'i4'],
              o4: ['i1', 'i2', 'i3'], o5: ['i3', 'i4'] }
    },
    bt1: {
        params: { supp: 30, conf: 100, lift: 0 },
        tx: { o1: ['d1', 'd3', 'd4'], o2: ['d1', 'd3', 'd4'], o3: ['d3', 'd5'],
              o4: ['d4', 'd5'], o5: ['d2', 'd3', 'd5'] }
    },
    bt2: {
        params: { supp: 30, conf: 100, lift: 0 },
        tx: { o1: ['i1', 'i3', 'i4', 'i6'], o2: ['i1', 'i3', 'i6'], o3: ['i3', 'i5', 'i6'],
              o4: ['i1', 'i2', 'i4', 'i5'], o5: ['i2', 'i4', 'i6'], o6: ['i1', 'i2', 'i4', 'i5', 'i6'] }
    }
};
let apIsDbMode = true;

document.addEventListener('DOMContentLoaded', () => {
    const btnModeDb = document.getElementById('ap-mode-db');
    const btnModeCustom = document.getElementById('ap-mode-custom');
    const customControls = document.getElementById('ap-custom-controls');
    const presetSelect = document.getElementById('ap-preset');
    const jsonInput = document.getElementById('ap-json-input');

    const applyPreset = (key) => {
        const p = AP_PRESETS[key];
        if (!p) return;
        const lines = Object.entries(p.tx).map(
            ([tid, items]) => `  {"tid": "${tid}", "items": ${JSON.stringify(items)}}`);
        jsonInput.value = '[\n' + lines.join(',\n') + '\n]';
        document.getElementById('ap-minsupp').value = p.params.supp;
        document.getElementById('ap-minconf').value = p.params.conf;
        document.getElementById('ap-minlift').value = p.params.lift;
    };

    if (btnModeDb && btnModeCustom) {
        btnModeDb.addEventListener('click', () => {
            apIsDbMode = true;
            btnModeDb.classList.add('active', 'btn-primary');
            btnModeDb.classList.remove('btn-secondary');
            btnModeCustom.classList.remove('active', 'btn-primary');
            btnModeCustom.classList.add('btn-secondary');
            customControls.style.display = 'none';
        });

        btnModeCustom.addEventListener('click', () => {
            apIsDbMode = false;
            btnModeCustom.classList.add('active', 'btn-primary');
            btnModeCustom.classList.remove('btn-secondary');
            btnModeDb.classList.remove('active', 'btn-primary');
            btnModeDb.classList.add('btn-secondary');
            customControls.style.display = 'block';
            if (!jsonInput.value.trim()) applyPreset(presetSelect.value);
        });
        presetSelect.addEventListener('change', () => applyPreset(presetSelect.value));
    }

    const btnRunApriori = document.getElementById('btn-run-apriori');
    if (btnRunApriori) {
        btnRunApriori.addEventListener('click', async () => {
            const minSupp = parseFloat(document.getElementById('ap-minsupp').value);
            const minConf = parseFloat(document.getElementById('ap-minconf').value);
            const minLift = parseFloat(document.getElementById('ap-minlift').value);

            try {
                // Only the data source differs between modes; mining and
                // rendering below are identical for both.
                let transactions;
                if (apIsDbMode) {
                    // DB mode: encode the stored HR dataset, then run Apriori.
                    const enc = await API.get('encode-transactions');
                    transactions = enc.transactions;
                } else {
                    try {
                        transactions = JSON.parse(jsonInput.value);
                    } catch (parseErr) {
                        throw new Error('JSON không hợp lệ: ' + parseErr.message);
                    }
                }
                const payload = {
                    transactions,
                    min_supp: minSupp, min_conf: minConf, min_lift: minLift,
                    include_vectors: true
                };
                const res = await API.post('apriori', payload);
                const rowLabel = 'Giao dịch';
                const tids = res.tids || [];

                // 1. Binary table — one ROW per transaction, one column per item,
                //    so it scrolls vertically instead of showing hundreds of T
                //    columns ("theo hàng"). Whole dataset, no filtering.
                const bitvectorsBox = document.getElementById('ap-bitvectors-box');
                const items = Object.keys(res.bitvectors);
                const keepRows = [];
                for (let i = 0; i < res.num_transactions; i++) keepRows.push(i);
                const shownRows = keepRows;

                // Collapse binary Yes/No attributes to a single "=Yes" column: a 0
                // there already means "No", so a separate "=No" column is redundant.
                // Multi-valued attributes (e.g. WorkLifeBalance=1..4) keep every column.
                const valuesByAttr = {};
                items.forEach(it => {
                    const eq = it.indexOf('=');
                    if (eq === -1) return;
                    const attr = it.slice(0, eq), val = it.slice(eq + 1);
                    (valuesByAttr[attr] = valuesByAttr[attr] || new Set()).add(val);
                });
                const isRedundantNo = (it) => {
                    const eq = it.indexOf('=');
                    if (eq === -1) return false;
                    const attr = it.slice(0, eq), val = it.slice(eq + 1);
                    const vals = valuesByAttr[attr];
                    return val === 'No' && vals && vals.size === 2
                        && vals.has('Yes') && vals.has('No');
                };

                const displayItems = items.filter(it => !isRedundantNo(it));

                let bvHtml = '<h4>1. Bảng Nhị Phân v(X)</h4>';
                bvHtml += `<p class="placeholder-text">${keepRows.length} ${rowLabel.toLowerCase()} × ${displayItems.length} mục.</p>`;
                const thSticky = 'position:sticky;top:0;background:#f1f5f9;z-index:2';
                bvHtml += '<div style="max-height:480px;overflow:auto;border:1px solid #e2e8f0;border-radius:8px">';
                bvHtml += '<table style="margin:0;width:max-content;border-collapse:separate;border-spacing:0;overflow:visible;border:0"><thead><tr>';
                bvHtml += `<th style="${thSticky};left:0;z-index:3">${rowLabel}</th>`;
                displayItems.forEach(it => bvHtml += `<th style="${thSticky}">${it}</th>`);
                bvHtml += '</tr></thead><tbody>';
                shownRows.forEach(i => {
                    bvHtml += `<tr><td style="position:sticky;left:0;background:#fff;font-weight:bold">${tids[i] || 'T' + (i + 1)}</td>`;
                    displayItems.forEach(it => {
                        const val = res.bitvectors[it][i];
                        bvHtml += `<td style="${val === 1 ? 'color:#34d399; font-weight:bold;' : 'color:#94a3b8'}">${val}</td>`;
                    });
                    bvHtml += '</tr>';
                });
                bvHtml += '</tbody></table></div>';
                bitvectorsBox.innerHTML = bvHtml;

                // 1b. Candidate trace C_k: v(S), SP and verdict per candidate,
                //     laid out like slides 30-35, one tab per level k. Pruned
                //     candidates are listed too: the slide evaluates them, the
                //     engine drops them via the prune step (slide 14).
                const candidatesBox = document.getElementById('ap-candidates-box');
                const n = res.num_transactions;
                const fmtSet = (arr) => '{' + arr.join(', ') + '}';
                const fmtVec = (bits) => '(' + (bits || '').split('').join(', ') + ')';
                const tidRange = tids.length <= 10
                    ? tids.join(', ') : `${tids[0]} … ${tids[tids.length - 1]}`;
                const ckLevels = res.itemset_steps.filter(
                    s => s.candidates_C_k.length || (s.pruned_C_k || []).length);
                let cTabs = '<div class="fk-tabs">';
                let cPanels = '';
                ckLevels.forEach((s, idx) => {
                    const active = idx === 0 ? ' active' : '';
                    const nPruned = (s.pruned_C_k || []).length;
                    cTabs += `<button type="button" class="ck-tab${active}" data-ck="${idx}">C_${s.k} (${s.candidates_C_k.length}${nPruned ? ' + ' + nPruned + ' bị loại' : ''})</button>`;
                    let rows = '';
                    s.candidates_C_k.forEach(c => {
                        const sp = (c.support_count / n).toFixed(2);
                        const verdict = c.is_frequent
                            ? '<span class="badge badge-green">Tập phổ biến</span>'
                            : '<span class="badge" style="background:#fee2e2;color:#991b1b">Không phổ biến</span>';
                        rows += `<tr><td>${fmtSet(c.itemset)}</td>`
                            + `<td><div style="max-width:420px;overflow-x:auto;white-space:nowrap"><code>${fmtVec(c.vector)}</code></div></td>`
                            + `<td>${c.support_count}/${n} = ${sp}</td><td>${verdict}</td></tr>`;
                    });
                    (s.pruned_C_k || []).forEach(p => {
                        const why = p.infrequent_subsets.map(fmtSet).join(', ');
                        rows += `<tr style="color:#94a3b8"><td>${fmtSet(p.itemset)}</td>`
                            + '<td colspan="2">— không tính (bước rút gọn Apriori)</td>'
                            + `<td>Bị loại: ${why} ∉ F<sub>${s.k - 1}</sub></td></tr>`;
                    });
                    cPanels += `<div class="ck-panel fk-panel${active}" data-ck="${idx}">`
                        + '<div style="max-height:480px;overflow:auto">'
                        + '<table><thead><tr><th>Tập S</th><th>v(S)</th><th>SP(S)</th><th>Kết quả</th></tr></thead>'
                        + `<tbody>${rows}</tbody></table></div></div>`;
                });
                cTabs += '</div>';
                candidatesBox.innerHTML =
                    '<h4>2. Các bước tính C<sub>k</sub> / F<sub>k</sub> (đối chiếu slide)</h4>'
                    + `<p class="placeholder-text">v(S) theo thứ tự giao dịch (${tidRange}); SP(S) = số thành phần 1 / ${n}; ngưỡng minsupp = ${res.min_supp_pct}%.</p>`
                    + cTabs + cPanels;
                candidatesBox.querySelectorAll('.ck-tab').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const id = btn.dataset.ck;
                        candidatesBox.querySelectorAll('.ck-tab').forEach(
                            b => b.classList.toggle('active', b.dataset.ck === id));
                        candidatesBox.querySelectorAll('.ck-panel').forEach(
                            p => p.classList.toggle('active', p.dataset.ck === id));
                    });
                });

                // 3. Frequent itemsets across every level k, one tab per k so the
                //    lists stay short. Cards that are also maximal get a chip; at a
                //    single top level they'd all be maximal, so all levels are shown.
                const itemsetsBox = document.getElementById('ap-itemsets-box');
                const levels = res.itemset_steps.filter(s => (s.frequent_F_k || []).length > 0);
                const maximalKeys = new Set(
                    (res.maximal_itemsets || []).map(m => m.itemset.join('|')));

                const buildCard = (c) => {
                    const lines = c.itemset.map(it => `<li>${it}</li>`).join('');
                    const isMax = maximalKeys.has(c.itemset.join('|'));
                    const maxChip = isMax
                        ? '<span class="badge" style="background:#fbbf24;color:#78350f">★ Tối đại</span>'
                        : '';
                    const cardStyle = isMax ? ' style="border:2px solid #fbbf24"' : '';
                    return `<div class="rule-card"${cardStyle}>
                        <ul style="margin:0 0 8px;padding-left:18px">${lines}</ul>
                        <div class="rule-meta">
                            <span class="badge badge-blue">Support: ${c.support_count}</span>
                            <span class="badge badge-green">${c.support_pct}%</span>
                            ${maxChip}
                        </div>
                    </div>`;
                };

                if (levels.length === 0) {
                    itemsetsBox.innerHTML = '<h4 class="mt-2">Tập Phổ Biến</h4>'
                        + '<p class="placeholder-text">Không có tập phổ biến.</p>';
                } else {
                    const total = levels.reduce((a, s) => a + s.frequent_F_k.length, 0);
                    let tabsBar = '<div class="fk-tabs">';
                    let panels = '';
                    levels.forEach((s, idx) => {
                        const active = idx === 0 ? ' active' : '';
                        tabsBar += `<button type="button" class="fk-tab${active}" data-fk="${idx}">F_${s.k} (${s.frequent_F_k.length})</button>`;
                        const grid = '<div class="scroll-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">'
                            + s.frequent_F_k.map(buildCard).join('') + '</div>';
                        panels += `<div class="fk-panel${active}" data-fk="${idx}">${grid}</div>`;
                    });
                    tabsBar += '</div>';
                    itemsetsBox.innerHTML =
                        `<h4 class="mt-2">Tập Phổ Biến theo mức k (tổng ${total})</h4>`
                        + tabsBar + panels;
                    itemsetsBox.querySelectorAll('.fk-tab').forEach(btn => {
                        btn.addEventListener('click', () => {
                            const id = btn.dataset.fk;
                            itemsetsBox.querySelectorAll('.fk-tab').forEach(
                                b => b.classList.toggle('active', b.dataset.fk === id));
                            itemsetsBox.querySelectorAll('.fk-panel').forEach(
                                p => p.classList.toggle('active', p.dataset.fk === id));
                        });
                    });
                }

                // 4. Maximal frequent itemsets (tập phổ biến tối đại), tabbed by size.
                const maximalBox = document.getElementById('ap-maximal-box');
                const maximal = res.maximal_itemsets || [];
                if (maximal.length === 0) {
                    maximalBox.innerHTML = '<h4 class="mt-2">Tập Phổ Biến Tối Đại</h4>'
                        + '<p class="placeholder-text">Không có tập phổ biến tối đại.</p>';
                } else {
                    // Group by itemset size k -> one tab per size.
                    const bySize = {};
                    maximal.forEach(m => {
                        (bySize[m.itemset.length] = bySize[m.itemset.length] || []).push(m);
                    });
                    const sizes = Object.keys(bySize).map(Number).sort((a, b) => a - b);
                    let mxTabs = '<div class="fk-tabs">';
                    let mxPanels = '';
                    sizes.forEach((sz, idx) => {
                        const active = idx === 0 ? ' active' : '';
                        mxTabs += `<button type="button" class="mx-tab${active}" data-mx="${idx}">k=${sz} (${bySize[sz].length})</button>`;
                        const grid = '<div class="scroll-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">'
                            + bySize[sz].map(m => {
                                const lines = m.itemset.map(it => `<li>${it}</li>`).join('');
                                return `<div class="rule-card" style="border:2px solid #fbbf24">
                                    <ul style="margin:0 0 8px;padding-left:18px">${lines}</ul>
                                    <div class="rule-meta">
                                        <span class="badge badge-blue">Support: ${m.support_count}</span>
                                        <span class="badge badge-green">${m.support_pct}%</span>
                                    </div>
                                </div>`;
                            }).join('') + '</div>';
                        mxPanels += `<div class="mx-panel${active}" data-mx="${idx}">${grid}</div>`;
                    });
                    mxTabs += '</div>';
                    maximalBox.innerHTML =
                        `<h4 class="mt-2">Tập Phổ Biến Tối Đại (${maximal.length})</h4>`
                        + mxTabs + mxPanels;
                    maximalBox.querySelectorAll('.mx-tab').forEach(btn => {
                        btn.addEventListener('click', () => {
                            const id = btn.dataset.mx;
                            maximalBox.querySelectorAll('.mx-tab').forEach(
                                b => b.classList.toggle('active', b.dataset.mx === id));
                            maximalBox.querySelectorAll('.mx-panel').forEach(
                                p => p.classList.toggle('active', p.dataset.mx === id));
                        });
                    });
                }

                // 3. Render Rule Cards
                const rulesBox = document.getElementById('ap-rules-cards');
                if (res.valid_rules.length === 0) {
                    rulesBox.innerHTML = '<p class="placeholder-text">Không tìm thấy luật nào thỏa mãn MinConfidence.</p>';
                } else {
                    const chips = (arr) => arr
                        .map(it => `<span class="cond-chip">${it}</span>`).join('');
                    rulesBox.innerHTML = res.valid_rules.map(rule => `
                        <div class="rule-card">
                            <div class="rule-flow">
                                <div class="rule-side">
                                    <span class="rule-label">NẾU</span>
                                    <div class="chip-row">${chips(rule.lhs)}</div>
                                </div>
                                <div class="rule-arrow">➔</div>
                                <div class="rule-side">
                                    <span class="rule-label rule-label--then">THÌ</span>
                                    <div class="chip-row">${chips(rule.rhs)}</div>
                                </div>
                            </div>
                            <div class="rule-meta">
                                <span class="badge badge-blue">Support: ${rule.support_pct}%</span>
                                <span class="badge badge-green">Confidence: ${rule.confidence_pct}%</span>
                                <span class="badge badge-purple">Lift: ${rule.lift}</span>
                                ${maximalKeys.has([...rule.lhs, ...rule.rhs].sort().join('|'))
                                    ? '<span class="badge" style="background:#fbbf24;color:#78350f">★ Từ tập tối đại</span>'
                                    : ''}
                            </div>
                        </div>
                    `).join('');
                }

            } catch (e) {
                console.error(e);
                const msg = (e && e.message) ? e.message : 'Không chạy được thuật toán Apriori.';
                document.getElementById('ap-bitvectors-box').innerHTML =
                    `<p class="placeholder-text">${msg}</p>`;
                document.getElementById('ap-candidates-box').innerHTML = '';
            }
        });
    }
});
