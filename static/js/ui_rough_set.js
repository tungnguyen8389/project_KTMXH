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

                const kPct = Math.round(res.dependency_k * 100);
                const kBadge = res.dependency_k >= 0.9 ? 'badge-green' : (res.dependency_k >= 0.5 ? 'badge-blue' : 'badge-purple');

                const katexBox = document.getElementById('rs-katex-output');
                let katexHtml = `<div class="formula-box" style="text-align:left;">
                    <div class="rule-meta" style="margin-bottom:0.75rem;">
                        <span class="badge badge-blue">|U| = ${res.universe.length}</span>
                        <span class="badge badge-blue">|U/IND(B)| = ${res.ind_b_classes.length}</span>
                        <span class="badge ${kBadge}">k = γ(B,D) = ${res.dependency_k} (${kPct}%)</span>
                    </div>
                    $$${res.katex_k}$$
                </div>`;

                katexHtml += `<h3>1. Các lớp tương đương U/IND(B)</h3>`;
                katexHtml += `<div class="chip-row" style="margin:0.5rem 0 1rem;">${res.ind_b_classes.map(c =>
                    `<span class="cond-chip">{ ${c.join(', ')} }</span>`).join('')}</div>`;

                katexHtml += `<h3 style="margin-top:1.25rem;">2. Xấp xỉ Trên/Dưới & Miền Ranh Giới</h3>`;
                Object.values(res.approximations).forEach(app => {
                    const accBadge = app.accuracy >= 0.9 ? 'badge-green' : (app.accuracy >= 0.5 ? 'badge-blue' : 'badge-purple');
                    katexHtml += `<div class="step-card" style="margin:0.75rem 0;">
                        <div class="rule-meta" style="margin-bottom:0.5rem;">
                            <strong>Lớp Quyết định X = ${app.decision_val}</strong>
                            <span class="badge ${accBadge}">α = ${app.accuracy}</span>
                        </div>
                        <div class="chip-row" style="margin-bottom:0.5rem;">
                            ${app.lower_approx.map(o => `<span class="cond-chip">${o}</span>`).join('')}
                            ${app.boundary_region.length ? `<span class="rule-arrow" style="font-size:1rem;">+</span>` : ''}
                            ${app.boundary_region.map(o => `<span class="cond-chip" style="background:#fff7ed;border-color:#fed7aa;color:#c2410c;">${o}</span>`).join('')}
                        </div>
                        <p>$$${app.katex_lower}$$</p>
                        <p>$$${app.katex_upper}$$</p>
                        <p>$$${app.katex_bn}$$</p>
                        <p>$$${app.katex_accuracy}$$</p>
                    </div>`;
                });

                katexHtml += `<h3 style="margin-top:1.25rem;">3. Miền Dương & Hệ số phụ thuộc k</h3>`;
                katexHtml += `<div class="chip-row" style="margin:0.5rem 0;">${res.positive_region.map(o => `<span class="cond-chip">${o}</span>`).join('')}</div>`;
                katexHtml += `<p>$$${res.katex_pos}$$</p>`;

                katexBox.innerHTML = katexHtml;
                if (window.renderMathInElement) {
                    renderMathInElement(katexBox, { delimiters: [{ left: '$$', right: '$$', display: true }] });
                }
            } catch (e) {
                console.error(e);
            }
        });
    }

    // ---------------- 2. HR PREFILL (Attrition dataset from DB) ----------------
    const btnPrefillRS = document.getElementById('btn-prefill-rs-hr');
    if (btnPrefillRS) {
        btnPrefillRS.addEventListener('click', async () => {
            try {
                const res = await API.get('roughset-data');
                const rpt = res.report || {};
                const target = rpt.target_attr || 'Attrition';
                const defaultAttrs = rpt.default_selected_attrs || [];
                const pool = rpt.available_attrs || [];

                document.getElementById('rs-cond-attrs').value = defaultAttrs.join(', ');
                document.getElementById('rs-dec-attr').value = target;
                document.getElementById('rs-json-input').value = JSON.stringify(res.data, null, 2);

                const info = document.getElementById('rs-prefill-report');
                if (info) {
                    const dist = Object.entries(rpt.class_distribution || {})
                        .map(([k, v]) => `${k}=${v}`).join(', ');
                    info.innerHTML =
                        `Đã nạp <strong>${rpt.num_records}</strong> bản ghi · nhãn <strong>${target}</strong> (${dist})<br>` +
                        `Thuộc tính có thể chọn: <code>${pool.join(', ')}</code><br>` +
                        `Đang chọn mặc định <strong>${defaultAttrs.length}</strong> thuộc tính (tối đa 12 thuộc tính điều kiện cho Reduct để tránh bùng nổ tổ hợp).`;
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

                const coreSet = new Set(res.core_attributes || []);
                const coreMatches = res.core_from_matrix &&
                    JSON.stringify([...res.core_from_matrix].sort()) === JSON.stringify([...(res.core_attributes || [])].sort());
                const chipFor = attr => coreSet.has(attr)
                    ? `<span class="cond-chip" style="background:#f3e8ff;border-color:#e9d5ff;color:#7c3aed;font-weight:700;">${attr}</span>`
                    : `<span class="cond-chip">${attr}</span>`;

                const katexBox = document.getElementById('rs-katex-output');
                let html = `<h3>Ma Trận Phân Biệt & Hàm Phân Biệt Boole</h3>`;
                html += `<p>$$${res.katex_formula}$$</p>`;

                html += `<h3 style="margin-top:1.25rem;">Các Tập Rút Gọn Tối Thiểu RED(C)</h3>`;
                html += `<div class="rules-grid" style="margin-top:0.5rem;">`;
                html += res.minimal_reducts.map((r, i) => `
                    <div class="rule-card">
                        <div class="rule-title">Reduct #${i + 1}</div>
                        <div class="chip-row">${r.map(chipFor).join('')}</div>
                    </div>`).join('');
                html += `</div>`;

                html += `<div class="formula-box" style="text-align:left;margin-top:1.25rem;">
                    <div class="rule-meta" style="margin-bottom:0.5rem;">
                        <span class="badge badge-purple">CORE(C) = ⋂ RED(C)</span>
                        ${res.core_from_matrix ? `<span class="badge ${coreMatches ? 'badge-green' : 'badge-blue'}">
                            Đối chiếu ma trận: ${coreMatches ? 'khớp' : 'khác'}</span>` : ''}
                    </div>
                    <div class="chip-row">${(res.core_attributes.length ? res.core_attributes : ['∅']).map(chipFor).join('')}</div>
                    <p style="margin-top:0.5rem;">$$${res.katex_core}$$</p>
                </div>`;

                katexBox.innerHTML = html;

                // Render n x n matrix table — highlight singleton (Core) cells.
                const matrixBox = document.getElementById('rs-matrix-output');
                let mHtml = '<h4>Ma Trận Phân Biệt M = (c_ij)</h4><table><thead><tr><th></th>';
                res.objects.forEach(obj => { mHtml += `<th>${obj}</th>`; });
                mHtml += '</tr></thead><tbody>';

                res.objects.forEach((obj, i) => {
                    mHtml += `<tr><th>${obj}</th>`;
                    res.matrix_n_x_n[i].forEach(cell => {
                        const isSingletonCore = !cell.includes(',') && coreSet.has(cell);
                        const style = isSingletonCore ? ' style="background:#f3e8ff;color:#7c3aed;font-weight:700;"' : '';
                        mHtml += `<td${style}>${cell}</td>`;
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
