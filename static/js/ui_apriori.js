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

                // 1. Render v(X) support — only items that appear in the filtered
                //    rules; the per-transaction T columns are collapsed to a
                //    Support count so large datasets stay readable ("filter cho T").
                const bitvectorsBox = document.getElementById('ap-bitvectors-box');
                const ruleItems = new Set();
                res.valid_rules.forEach(r => {
                    (r.lhs || []).forEach(i => ruleItems.add(i));
                    (r.rhs || []).forEach(i => ruleItems.add(i));
                });
                const bvEntries = Object.entries(res.bitvectors)
                    .filter(([item]) => ruleItems.size === 0 || ruleItems.has(item));
                let bvHtml = '<h4>1. Support v(X) các Mục (Item)</h4>';
                if (ruleItems.size > 0) {
                    bvHtml += `<p class="placeholder-text">Chỉ hiển thị ${bvEntries.length} mục xuất hiện trong luật đã lọc (ẩn ${res.num_transactions} cột giao dịch T).</p>`;
                }
                bvHtml += '<table><thead><tr><th>Mục (Item)</th><th>Support Count</th><th>Support %</th></tr></thead><tbody>';
                bvEntries.forEach(([item, bv]) => {
                    const suppCnt = bv.reduce((a, b) => a + b, 0);
                    const suppPct = ((suppCnt / res.num_transactions) * 100).toFixed(2);
                    bvHtml += `<tr><td><strong>${item}</strong></td><td>${suppCnt}</td><td>${suppPct}%</td></tr>`;
                });
                bvHtml += '</tbody></table>';
                bitvectorsBox.innerHTML = bvHtml;

                // 2. Render Candidates C_k / Frequent F_k — only the table for the
                //    entered k (Max Length); fall back to the largest k reached.
                const itemsetsBox = document.getElementById('ap-itemsets-box');
                let shownSteps = res.itemset_steps.filter(s => s.k === maxLen);
                if (shownSteps.length === 0) shownSteps = res.itemset_steps.slice(-1);
                const kLabel = shownSteps.length ? shownSteps[shownSteps.length - 1].k : maxLen;
                let isHtml = '<h4 class="mt-3">2. Tập Ứng Viên C_k & Tập Phổ Biến F_k (k = ' + kLabel + ')</h4>';

                shownSteps.forEach(step => {
                    isHtml += `<div class="step-card my-2">
                        <h5>Vòng k = ${step.k}:</h5>
                        <p><strong>C_${step.k} (${step.candidates_C_k.length} tập ứng viên):</strong></p>
                        <div class="table-responsive mb-2">
                            <table><thead><tr><th>Tập Mục (Itemset)</th><th>Support Count</th><th>Support %</th><th>Trạng Thái</th></tr></thead><tbody>`;

                    step.candidates_C_k.forEach(c => {
                        isHtml += `<tr>
                            <td>{ ${c.itemset.join(', ')} }</td>
                            <td>${c.support_count}</td>
                            <td>${c.support_pct}%</td>
                            <td>${c.is_frequent ? '<span class="badge badge-green">Phổ biến (F_'+step.k+')</span>' : '<span class="badge" style="background:rgba(251,113,133,0.2); color:#fb7185">Loại (Pruned)</span>'}</td>
                        </tr>`;
                    });
                    isHtml += `</tbody></table></div></div>`;
                });

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
