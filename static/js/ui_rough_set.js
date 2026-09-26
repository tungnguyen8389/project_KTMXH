/**
 * [TV4] UI Script for Preprocessing, Rough Set, and Reduct Visualizations
 */
document.addEventListener('DOMContentLoaded', () => {
    // ---------------- 1. ROUGH SET EVENT LISTENER ----------------
    const btnRunRS = document.getElementById('btn-run-rs');
    if (btnRunRS) {
        btnRunRS.addEventListener('click', async () => {
            const condAttrs = document.getElementById('rs-cond-attrs').value.split(',').map(s => s.trim());
            const decAttr = document.getElementById('rs-dec-attr').value.trim();
            const jsonRaw = document.getElementById('rs-json-input').value;

            try {
                const data = JSON.parse(jsonRaw);
                const payload = { condition_attrs: condAttrs, decision_attr: decAttr, data };
                const res = await API.post('rough-set', payload);

                const katexBox = document.getElementById('rs-katex-output');
                let katexHtml = `<h3>1. Các lớp tương đương U/IND(B):</h3>`;
                katexHtml += `<p>$$U/IND(B) = \\{ ${res.ind_b_classes.map(c => '\\{' + c.join(', ') + '\\}').join(', ')} \\}$$</p>`;

                katexHtml += `<h3 class="mt-3">2. Xấp xỉ Trên/Dưới & Miền Ranh Giới:</h3>`;
                Object.values(res.approximations).forEach(app => {
                    katexHtml += `<div class="step-card my-2">
                        <p><strong>Lớp Quyết định X = ${app.decision_val}:</strong></p>
                        <p>$$${app.katex_lower}$$</p>
                        <p>$$${app.katex_upper}$$</p>
                        <p>$$${app.katex_bn}$$</p>
                        <p>$$${app.katex_accuracy}$$</p>
                    </div>`;
                });

                katexHtml += `<h3 class="mt-3">3. Miền Dương & Hệ số phụ thuộc k:</h3>`;
                katexHtml += `<p>$$${res.katex_pos}$$</p>`;
                katexHtml += `<p>$$${res.katex_k}$$</p>`;

                katexBox.innerHTML = katexHtml;
                if (window.renderMathInElement) {
                    renderMathInElement(katexBox, { delimiters: [{ left: '$$', right: '$$', display: true }] });
                }
            } catch (e) {
                console.error(e);
            }
        });
    }

    // ---------------- 3. REDUCT EVENT LISTENER ----------------
    const btnRunReduct = document.getElementById('btn-run-reduct');
    if (btnRunReduct) {
        btnRunReduct.addEventListener('click', async () => {
            const condAttrs = document.getElementById('rs-cond-attrs').value.split(',').map(s => s.trim());
            const decAttr = document.getElementById('rs-dec-attr').value.trim();
            const jsonRaw = document.getElementById('rs-json-input').value;

            try {
                const data = JSON.parse(jsonRaw);
                const payload = { condition_attrs: condAttrs, decision_attr: decAttr, data };
                const res = await API.post('reduct', payload);

                const katexBox = document.getElementById('rs-katex-output');
                let html = `<h3>Ma Trận Phân Biệt $n \\times n$ & Rút Gọn Boole</h3>`;
                html += `<p>$$${res.katex_formula}$$</p>`;
                html += `<p><strong>Các Tập Rút Gọn Tối Thiểu RED(C):</strong></p>`;
                html += `<ul>${res.minimal_reducts.map(r => `<li>$$RED = \\{ ${r.join(', ')} \\}$$</li>`).join('')}</ul>`;
                html += `<p class="mt-2">$$${res.katex_core}$$</p>`;

                katexBox.innerHTML = html;

                // Render n x n matrix table
                const matrixBox = document.getElementById('rs-matrix-output');
                let mHtml = '<h4>Ma Trận Phân Biệt M = (c_ij)</h4><table><thead><tr><th></th>';
                res.objects.forEach(obj => { mHtml += `<th>${obj}</th>`; });
                mHtml += '</tr></thead><tbody>';

                res.objects.forEach((obj, i) => {
                    mHtml += `<tr><th>${obj}</th>`;
                    res.matrix_n_x_n[i].forEach(cell => {
                        mHtml += `<td>${cell}</td>`;
                    });
                    mHtml += '</tr>';
                });
                mHtml += '</tbody></table>';
                matrixBox.innerHTML = mHtml;

                if (window.renderMathInElement) {
                    renderMathInElement(katexBox, { delimiters: [{ left: '$$', right: '$$', display: true }] });
                }
            } catch (e) {
                console.error(e);
            }
        });
    }
});
