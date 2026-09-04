/**
 * [TV4] UI Script for ID3 Decision Tree (Mermaid.js) & Naive Bayes Breakdown
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Mermaid.js for light theme
    if (window.mermaid) {
        mermaid.initialize({ startOnLoad: false, theme: 'default' });
    }

    // ---------------- 1. ID3 DECISION TREE ----------------
    const btnRunID3 = document.getElementById('btn-run-id3');
    if (btnRunID3) {
        btnRunID3.addEventListener('click', async () => {
            const condAttrs = document.getElementById('cl-cond-attrs').value.split(',').map(s => s.trim());
            const targetAttr = document.getElementById('cl-target-attr').value.trim();
            const jsonRaw = document.getElementById('cl-json-input').value;

            try {
                const data = JSON.parse(jsonRaw);
                const payload = { condition_attrs: condAttrs, target_attr: targetAttr, data };
                const res = await API.post('id3', payload);

                // Render Mermaid Diagram
                const mermaidContainer = document.getElementById('cl-mermaid-container');
                mermaidContainer.innerHTML = `<div class="mermaid">${res.mermaid_graph}</div>`;
                if (window.mermaid) {
                    mermaid.run({ nodes: mermaidContainer.querySelectorAll('.mermaid') });
                }

                // Render Gain calculation steps
                const stepsOutput = document.getElementById('cl-steps-output');
                let html = `<h4>📊 Chi Tiết Tính Entropy & Information Gain theo Cấp Nút</h4>`;
                res.steps.forEach((step, idx) => {
                    html += `<div class="step-card my-2">
                        <p><strong>Nút Split ${idx+1} (${step.parent_label}):</strong> Mẫu = ${step.node_samples} (Pos = ${step.p_count}, Neg = ${step.n_count})</p>
                        <p>$$Entropy(S) = I(${step.p_count}, ${step.n_count}) = ${step.info_p_n}$$</p>
                        <ul class="mt-2">`;
                    Object.entries(step.attr_details).forEach(([attr, det]) => {
                        const isMax = attr === step.selected_best_attr;
                        html += `<li style="${isMax ? 'color:#4f46e5; font-weight:bold;' : ''}">
                            Gain(${attr}) = ${step.info_p_n} - ${det.expected_entropy_E} = ${det.gain}
                        </li>`;
                    });
                    html += `</ul><p class="mt-2">➔ <strong>Chọn thuộc tính chia: <span style="color:#059669">${step.selected_best_attr}</span> (Gain max = ${step.max_gain})</strong></p></div>`;
                });

                stepsOutput.innerHTML = html;
                if (window.renderMathInElement) {
                    renderMathInElement(stepsOutput, { delimiters: [{left: '$$', right: '$$', display: true}] });
                }

            } catch (e) {
                console.error(e);
            }
        });
    }

    // ---------------- 2. NAIVE BAYES ----------------
    const btnRunNB = document.getElementById('btn-run-nb');
    if (btnRunNB) {
        btnRunNB.addEventListener('click', async () => {
            const condAttrs = document.getElementById('cl-cond-attrs').value.split(',').map(s => s.trim());
            const targetAttr = document.getElementById('cl-target-attr').value.trim();
            const jsonRaw = document.getElementById('cl-json-input').value;
            const testRaw = document.getElementById('nb-test-instance').value;
            const useLaplace = document.getElementById('nb-laplace').checked;

            try {
                const data = JSON.parse(jsonRaw);
                const testInstance = JSON.parse(testRaw);
                const payload = {
                    condition_attrs: condAttrs,
                    target_attr: targetAttr,
                    data,
                    test_instance: testInstance,
                    use_laplace: useLaplace
                };
                const res = await API.post('naive-bayes', payload);

                const mermaidContainer = document.getElementById('cl-mermaid-container');
                mermaidContainer.innerHTML = `<div class="card p-3" style="width:100%; border:none; background:#ffffff;">
                    <h3>🎲 Dự Đoán Naive Bayes Classifier</h3>
                    <p class="mt-2"><strong>Mẫu thử nghiệm X:</strong> <code>${JSON.stringify(res.test_instance)}</code></p>
                    <p><strong>Hiệu chỉnh Laplace:</strong> ${res.use_laplace ? '<span style="color:#059669; font-weight:bold;">Đang bật</span>' : 'Tắt'}</p>
                    <div class="formula-box mt-3">
                        <p>Dự đoán Lớp Quyết Định: <strong style="color:#059669; font-size:1.4rem;">${res.predicted_class}</strong></p>
                    </div>
                </div>`;

                const stepsOutput = document.getElementById('cl-steps-output');
                let html = `<h4>📐 Bước Tính Xác Suất Hậu Kỳ P(C_i | X)</h4>`;
                res.katex_steps.forEach(step => {
                    html += `<div class="step-card my-2">
                        <p><strong>Lớp ${step.class_val}:</strong></p>
                        <ul>${step.katex_terms.map(t => `<li>$$${t}$$</li>`).join('')}</ul>
                        <p class="mt-2">$$P(${step.class_val}) \\times \\prod P(x_k \\mid ${step.class_val}) = ${step.unnormalized_posterior}$$</p>
                        <p><strong>Xác suất chuẩn hóa: <span style="color:#0284c7; font-weight:bold;">${(res.normalized_posteriors[step.class_val] * 100).toFixed(2)}%</span></strong></p>
                    </div>`;
                });

                stepsOutput.innerHTML = html;
                if (window.renderMathInElement) {
                    renderMathInElement(stepsOutput, { delimiters: [{left: '$$', right: '$$', display: true}] });
                }

            } catch (e) {
                console.error(e);
            }
        });
    }
});
