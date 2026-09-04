/**
 * [TV4] UI Script for Apriori Itemsets C_k, F_k, Bitvectors, and Rule Cards
 */
document.addEventListener('DOMContentLoaded', () => {
    const btnRunApriori = document.getElementById('btn-run-apriori');
    if (btnRunApriori) {
        btnRunApriori.addEventListener('click', async () => {
            const minSupp = parseFloat(document.getElementById('ap-minsupp').value);
            const minConf = parseFloat(document.getElementById('ap-minconf').value);
            const jsonRaw = document.getElementById('ap-json-input').value;

            try {
                const transactions = JSON.parse(jsonRaw);
                const payload = { transactions, min_supp: minSupp, min_conf: minConf };
                const res = await API.post('apriori', payload);

                // 1. Render Transaction Bitvectors v(X)
                const bitvectorsBox = document.getElementById('ap-bitvectors-box');
                let bvHtml = '<h4>1. Ma Trận Vector Bít v(X) Theo Thuộc Tính</h4><table><thead><tr><th>Mục (Item)</th>';
                for (let i = 1; i <= res.num_transactions; i++) bvHtml += `<th>T${i}</th>`;
                bvHtml += '<th>Support Count</th></tr></thead><tbody>';

                Object.entries(res.bitvectors).forEach(([item, bv]) => {
                    const suppCnt = bv.reduce((a, b) => a + b, 0);
                    bvHtml += `<tr><td><strong>${item}</strong></td>`;
                    bv.forEach(val => {
                        bvHtml += `<td style="${val === 1 ? 'color:#34d399; font-weight:bold;' : 'color:#94a3b8'}">${val}</td>`;
                    });
                    bvHtml += `<td><strong>${suppCnt}</strong></td></tr>`;
                });
                bvHtml += '</tbody></table>';
                bitvectorsBox.innerHTML = bvHtml;

                // 2. Render Candidates C_k and Frequent F_k
                const itemsetsBox = document.getElementById('ap-itemsets-box');
                let isHtml = '<h4 class="mt-3">2. Danh Sách Tập Ứng Viên C_k & Tập Phổ Biến F_k</h4>';

                res.itemset_steps.forEach(step => {
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
