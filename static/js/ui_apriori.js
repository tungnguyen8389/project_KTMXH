/**
 * [TV4] UI Script for Apriori Itemsets C_k, F_k, Bitvectors, and Rule Cards
 */
document.addEventListener('DOMContentLoaded', () => {
    const btnEncodeHr = document.getElementById('btn-encode-hr');
    if (btnEncodeHr) {
        btnEncodeHr.addEventListener('click', async () => {
            try {
                const res = await API.get('encode-transactions');
                document.getElementById('ap-json-input').value =
                    JSON.stringify(res.transactions, null, 2);
                // Prefill sensible defaults for the real HR dataset. Attrition=Yes
                // is a minority class, so confidence must stay modest to surface
                // meaningful leave-risk rules; lift>1.2 keeps only positive links.
                document.getElementById('ap-minsupp').value = 5;
                document.getElementById('ap-minconf').value = 30;
                document.getElementById('ap-minlift').value = 1.2;
                document.getElementById('ap-maxlen').value = 3;
                document.getElementById('ap-target').value = 'yes';
                const r = res.report;
                const binLines = Object.entries(r.bins || {})
                    .map(([col, labels]) => `<li><strong>${col}</strong>: ${labels.join(', ')}</li>`)
                    .join('');
                document.getElementById('ap-encode-report').innerHTML = `
                    <div class="step-card">
                        <p>Đã mã hóa <strong>${r.num_transactions}</strong> giao dịch,
                           <strong>${r.vocab_size}</strong> mục (items) từ
                           ${r.columns_used.length} cột.</p>
                        <ul>${binLines}</ul>
                    </div>`;
            } catch (e) {
                console.error(e);
            }
        });
    }

    const btnRunApriori = document.getElementById('btn-run-apriori');
    if (btnRunApriori) {
        btnRunApriori.addEventListener('click', async () => {
            const minSupp = parseFloat(document.getElementById('ap-minsupp').value);
            const minConf = parseFloat(document.getElementById('ap-minconf').value);
            const minLift = parseFloat(document.getElementById('ap-minlift').value);
            const maxLen = parseInt(document.getElementById('ap-maxlen').value, 10);
            const targetMode = document.getElementById('ap-target').value;
            const jsonRaw = document.getElementById('ap-json-input').value;

            try {
                const transactions = JSON.parse(jsonRaw);
                const payload = {
                    transactions, min_supp: minSupp, min_conf: minConf,
                    min_lift: minLift, max_len: maxLen, target_mode: targetMode
                };
                const res = await API.post('apriori', payload);

                // 1. Binary table — one ROW per (target-matched) transaction, one
                //    column per item, so it scrolls vertically instead of showing
                //    hundreds of T columns ("theo hàng").
                const bitvectorsBox = document.getElementById('ap-bitvectors-box');
                const items = Object.keys(res.bitvectors);
                let targetTokens = [];
                if (targetMode === 'yes') targetTokens = ['Attrition=Yes'];
                else if (targetMode === 'attrition')
                    targetTokens = items.filter(k => k.startsWith('Attrition='));
                const keepRows = [];
                for (let i = 0; i < res.num_transactions; i++) {
                    if (targetTokens.length === 0 ||
                        targetTokens.some(t => res.bitvectors[t] && res.bitvectors[t][i] === 1)) {
                        keepRows.push(i);
                    }
                }
                const ROW_CAP = 50;
                const shownRows = keepRows.slice(0, ROW_CAP);

                let bvHtml = '<h4>1. Bảng Nhị Phân v(X)</h4>';
                bvHtml += `<p class="placeholder-text">${keepRows.length} giao dịch khớp mục tiêu × ${items.length} mục${keepRows.length > ROW_CAP ? ` (hiển thị ${ROW_CAP} hàng đầu)` : ''}.</p>`;
                const thSticky = 'position:sticky;top:0;background:#f1f5f9;z-index:2';
                bvHtml += '<div style="max-height:480px;overflow:auto;border:1px solid #e2e8f0;border-radius:8px">';
                bvHtml += '<table style="margin:0;width:max-content;border-collapse:separate;border-spacing:0;overflow:visible;border:0"><thead><tr>';
                bvHtml += `<th style="${thSticky};left:0;z-index:3">Giao dịch</th>`;
                items.forEach(it => bvHtml += `<th style="${thSticky}">${it}</th>`);
                bvHtml += '</tr></thead><tbody>';
                shownRows.forEach(i => {
                    bvHtml += `<tr><td style="position:sticky;left:0;background:#fff;font-weight:bold">T${i + 1}</td>`;
                    items.forEach(it => {
                        const val = res.bitvectors[it][i];
                        bvHtml += `<td style="${val === 1 ? 'color:#34d399; font-weight:bold;' : 'color:#94a3b8'}">${val}</td>`;
                    });
                    bvHtml += '</tr>';
                });
                bvHtml += '</tbody></table></div>';
                bitvectorsBox.innerHTML = bvHtml;

                // 3. Frequent-itemset cards (F_k) for the entered k — 3 per row,
                //    each card lists the itemset values line by line.
                const itemsetsBox = document.getElementById('ap-itemsets-box');
                let shownSteps = res.itemset_steps.filter(s => s.k === maxLen);
                if (shownSteps.length === 0) shownSteps = res.itemset_steps.slice(-1);
                const kLabel = shownSteps.length ? shownSteps[shownSteps.length - 1].k : maxLen;
                const freqSets = shownSteps.reduce((acc, s) => acc.concat(s.frequent_F_k || []), []);

                let isHtml = `<h4 class="mt-2">Tập Phổ Biến F_${kLabel} (${freqSets.length})</h4>`;
                if (freqSets.length === 0) {
                    isHtml += '<p class="placeholder-text">Không có tập phổ biến ở mức k này.</p>';
                } else {
                    isHtml += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">';
                    freqSets.forEach(c => {
                        const lines = c.itemset.map(it => `<li>${it}</li>`).join('');
                        isHtml += `<div class="rule-card">
                            <ul style="margin:0 0 8px;padding-left:18px">${lines}</ul>
                            <div class="rule-meta">
                                <span class="badge badge-blue">Support: ${c.support_count}</span>
                                <span class="badge badge-green">${c.support_pct}%</span>
                            </div>
                        </div>`;
                    });
                    isHtml += '</div>';
                }
                itemsetsBox.innerHTML = isHtml;

                // 3. Render Rule Cards
                const rulesBox = document.getElementById('ap-rules-cards');
                if (res.valid_rules.length === 0) {
                    rulesBox.innerHTML = '<p class="placeholder-text">Không tìm thấy luật nào thỏa mãn MinConfidence.</p>';
                } else {
                    rulesBox.innerHTML = res.valid_rules.map(rule => `
                        <div class="rule-card">
                            <div class="rule-title">Luật: ${rule.rule_str}</div>
                            <div class="rule-meta">
                                <span class="badge badge-blue">Support: ${rule.support_pct}%</span>
                                <span class="badge badge-green">Confidence: ${rule.confidence_pct}%</span>
                                <span class="badge badge-purple">Lift: ${rule.lift}</span>
                            </div>
                        </div>
                    `).join('');
                }

            } catch (e) {
                console.error(e);
            }
        });
    }
});
