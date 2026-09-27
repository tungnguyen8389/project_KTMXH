/**
 * [TV4] UI Script for K-Means Chart.js 2D Scatter Plot, DB Integration, Step Playback & Cluster Profiles
 */
let kmeansChartInstance = null;
let kmeansAutoPlayTimer = null;
let currentKMeansResult = null;
let currentStepIndex = 0;
let isDbMode = true;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Setup Mode Toggles
    const btnModeDb = document.getElementById('km-mode-db');
    const btnModeCustom = document.getElementById('km-mode-custom');
    const dbControls = document.getElementById('km-db-controls');
    const customControls = document.getElementById('km-custom-controls');

    if (btnModeDb && btnModeCustom) {
        btnModeDb.addEventListener('click', () => {
            isDbMode = true;
            btnModeDb.classList.add('active', 'btn-primary');
            btnModeDb.classList.remove('btn-secondary');
            btnModeCustom.classList.remove('active', 'btn-primary');
            btnModeCustom.classList.add('btn-secondary');
            dbControls.style.display = 'block';
            customControls.style.display = 'none';
        });

        btnModeCustom.addEventListener('click', () => {
            isDbMode = false;
            btnModeCustom.classList.add('active', 'btn-primary');
            btnModeCustom.classList.remove('btn-secondary');
            btnModeDb.classList.remove('active', 'btn-primary');
            btnModeDb.classList.add('btn-secondary');
            dbControls.style.display = 'none';
            customControls.style.display = 'block';
        });
    }

    // 2. Setup Run Button
    const btnRunKMeans = document.getElementById('btn-run-kmeans');
    if (btnRunKMeans) {
        btnRunKMeans.addEventListener('click', async () => {
            const k = parseInt(document.getElementById('km-k').value) || 3;
            const maxIter = parseInt(document.getElementById('km-max-iter').value) || 15;
            const initMethod = document.getElementById('km-init-method').value || 'kmeans++';
            const normalize = document.getElementById('km-normalize') ? document.getElementById('km-normalize').checked : false;

            btnRunKMeans.disabled = true;
            btnRunKMeans.textContent = '⏳ Đang tính toán phân cụm...';

            try {
                let payload = {
                    k: k,
                    max_iter: maxIter,
                    init_method: initMethod,
                    normalize: normalize
                };

                if (isDbMode) {
                    payload.use_db = true;
                    payload.feature_x = document.getElementById('km-feature-x').value;
                    payload.feature_y = document.getElementById('km-feature-y').value;
                    payload.sample_size = parseInt(document.getElementById('km-sample-size').value);
                } else {
                    payload.use_db = false;
                    const jsonRaw = document.getElementById('km-points-input').value;
                    payload.points = JSON.parse(jsonRaw);
                }

                const res = await API.post('kmeans', payload);

                if (!res.iterations || res.iterations.length === 0) {
                    alert('Không có dữ liệu trả về từ thuật toán K-Means!');
                    return;
                }

                currentKMeansResult = res;
                currentStepIndex = 0;
                stopAutoPlay();

                // Show Results Card
                const resultsCard = document.getElementById('km-results-card');
                if (resultsCard) resultsCard.style.display = 'block';

                // Setup Subtitles & KPIs
                document.getElementById('km-chart-subtitle').textContent = 
                    `Đặc trưng: [X: ${res.feature_x_name || 'X'}] vs [Y: ${res.feature_y_name || 'Y'}] | Tổng ${res.total_points || res.point_labels.length} điểm dữ liệu`;

                renderKMeansKPIs(res);
                renderClusterProfiles(res);

                // Setup step playback buttons
                const stepControls = document.getElementById('km-step-controls');
                stepControls.innerHTML = res.iterations.map((iter, idx) => `
                    <button class="btn btn-secondary btn-sm km-step-btn ${idx === 0 ? 'active' : ''}" data-step="${idx}">
                        Lặp ${iter.iteration}
                    </button>
                `).join('');

                // Render first iteration
                renderKMeansStep(res, 0);

                // Step click listeners
                document.querySelectorAll('.km-step-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        stopAutoPlay();
                        const stepIdx = parseInt(btn.getAttribute('data-step'));
                        currentStepIndex = stepIdx;
                        setActiveStepButton(stepIdx);
                        renderKMeansStep(res, stepIdx);
                    });
                });

                // Scroll to result smoothly
                resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

            } catch (e) {
                console.error(e);
                alert('Lỗi phân cụm K-Means: ' + (e.message || e));
            } finally {
                btnRunKMeans.disabled = false;
                btnRunKMeans.textContent = '🚀 Chạy Phân Cụm K-Means';
            }
        });
    }

    // 3. Setup AutoPlay Button
    const btnAutoPlay = document.getElementById('km-btn-autoplay');
    if (btnAutoPlay) {
        btnAutoPlay.addEventListener('click', () => {
            if (kmeansAutoPlayTimer) {
                stopAutoPlay();
            } else {
                startAutoPlay();
            }
        });
    }
});

function setActiveStepButton(stepIdx) {
    document.querySelectorAll('.km-step-btn').forEach((b, idx) => {
        if (idx === stepIdx) {
            b.classList.add('active', 'btn-primary');
            b.classList.remove('btn-secondary');
        } else {
            b.classList.remove('active', 'btn-primary');
            b.classList.add('btn-secondary');
        }
    });
}

function startAutoPlay() {
    const btn = document.getElementById('km-btn-autoplay');
    if (btn) btn.textContent = '⏸ Tạm dừng';
    if (!currentKMeansResult) return;

    kmeansAutoPlayTimer = setInterval(() => {
        currentStepIndex = (currentStepIndex + 1) % currentKMeansResult.iterations.length;
        setActiveStepButton(currentStepIndex);
        renderKMeansStep(currentKMeansResult, currentStepIndex);
    }, 1200);
}

function stopAutoPlay() {
    if (kmeansAutoPlayTimer) {
        clearInterval(kmeansAutoPlayTimer);
        kmeansAutoPlayTimer = null;
        const btn = document.getElementById('km-btn-autoplay');
        if (btn) btn.textContent = '▶ Tự động phát';
    }
}

function renderKMeansKPIs(res) {
    const kpiBox = document.getElementById('km-kpis');
    if (!kpiBox) return;

    kpiBox.innerHTML = `
        <div style="background:#f1f5f9;padding:0.6rem 1rem;border-radius:8px;border-left:4px solid #0284c7;font-size:0.9rem;">
            <strong>Số cụm K:</strong> <span style="color:#0284c7;font-weight:bold;">${res.k}</span>
        </div>
        <div style="background:#f1f5f9;padding:0.6rem 1rem;border-radius:8px;border-left:4px solid #10b981;font-size:0.9rem;">
            <strong>Số vòng lặp:</strong> <span style="color:#10b981;font-weight:bold;">${res.total_iterations} bước</span>
        </div>
        <div style="background:#f1f5f9;padding:0.6rem 1rem;border-radius:8px;border-left:4px solid ${res.converged ? '#059669' : '#d97706'};font-size:0.9rem;">
            <strong>Hội tụ:</strong> <span style="font-weight:bold;">${res.converged ? '✅ Hoàn tất' : '⚠️ Đạt max_iter'}</span>
        </div>
        <div style="background:#f1f5f9;padding:0.6rem 1rem;border-radius:8px;border-left:4px solid #8b5cf6;font-size:0.9rem;">
            <strong>Khởi tạo:</strong> <span style="font-weight:bold;">${res.init_method}</span>
        </div>
        <div style="background:#f1f5f9;padding:0.6rem 1rem;border-radius:8px;border-left:4px solid #ec4899;font-size:0.9rem;">
            <strong>Tổng sai số nội cụm (WCSS):</strong> <span style="color:#ec4899;font-weight:bold;">${res.final_wcss}</span>
        </div>
    `;
}

function renderClusterProfiles(res) {
    const box = document.getElementById('km-cluster-profiles-box');
    if (!box || !res.cluster_profiles) return;

    const clusterColors = ['#0284c7', '#7c3aed', '#059669', '#d97706', '#e11d48', '#0d9488'];

    let html = `
        <h4 class="mb-2">📊 Hồ Sơ Phân Tích Cụm & Tỷ Lệ Nghỉ Việc (Cluster Profiles)</h4>
        <div class="table-responsive">
            <table class="table" style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
                        <th>Cụm</th>
                        <th>Số nhân viên</th>
                        <th>Tỷ lệ %</th>
                        <th>Tâm cụm [${res.feature_x_name || 'X'}, ${res.feature_y_name || 'Y'}]</th>
                        <th>Miền giá trị X</th>
                        <th>Miền giá trị Y</th>
                        <th>Nghỉ việc (Attrition)</th>
                    </tr>
                </thead>
                <tbody>
    `;

    res.cluster_profiles.forEach((p, idx) => {
        const color = clusterColors[idx % clusterColors.length];
        const attritionBadge = p.attrition_rate > 20
            ? `<span style="background:#fee2e2;color:#b91c1c;padding:0.2rem 0.6rem;border-radius:12px;font-weight:bold;">${p.attrition_count} NV (${p.attrition_rate}%) ⚠️</span>`
            : `<span style="background:#dcfce7;color:#15803d;padding:0.2rem 0.6rem;border-radius:12px;font-weight:bold;">${p.attrition_count} NV (${p.attrition_rate}%)</span>`;

        html += `
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td>
                    <span style="display:inline-block;width:12px;height:12px;background:${color};border-radius:50%;margin-right:6px;"></span>
                    <strong>${p.name}</strong>
                </td>
                <td><strong>${p.count}</strong></td>
                <td>${p.percentage}%</td>
                <td><code style="background:#f1f5f9;padding:0.2rem 0.4rem;border-radius:4px;">(${p.centroid_x}, ${p.centroid_y})</code></td>
                <td>${p.min_x} &rarr; ${p.max_x} (TB: ${p.mean_x})</td>
                <td>${p.min_y} &rarr; ${p.max_y} (TB: ${p.mean_y})</td>
                <td>${attritionBadge}</td>
            </tr>
        `;
    });

    html += `</tbody></table></div>`;
    box.innerHTML = html;
}

function renderKMeansStep(res, stepIdx) {
    const iter = res.iterations[stepIdx];
    const canvas = document.getElementById('kmeansChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const clusterColors = ['#0284c7', '#7c3aed', '#059669', '#d97706', '#e11d48', '#0d9488'];

    // 1. Group points by assigned cluster for Chart.js datasets
    const datasets = [];
    const numClusters = res.k;

    for (let c = 0; c < numClusters; c++) {
        const clusterPoints = [];
        iter.cluster_assignments.forEach((assigned, ptIdx) => {
            if (assigned === c) {
                const meta = iter.metadata ? iter.metadata[ptIdx] : {};
                clusterPoints.push({
                    x: iter.points_x[ptIdx],
                    y: iter.points_y[ptIdx],
                    label: iter.point_labels[ptIdx],
                    attrition: meta.attrition || '',
                    job_role: meta.job_role || '',
                    department: meta.department || ''
                });
            }
        });

        datasets.push({
            label: `Cụm C${c+1} (${clusterPoints.length} điểm)`,
            data: clusterPoints,
            backgroundColor: clusterColors[c % clusterColors.length] + 'b3', // slight opacity
            borderColor: clusterColors[c % clusterColors.length],
            borderWidth: 1,
            pointRadius: clusterPoints.length > 300 ? 5 : 7,
            pointHoverRadius: 9
        });

        // Add Centroid point
        const centroidCoords = iter.centroids[c];
        datasets.push({
            label: `Tâm m${c+1} (${centroidCoords[0]}, ${centroidCoords[1]})`,
            data: [{ x: centroidCoords[0], y: centroidCoords[1], label: `Tâm m${c+1}` }],
            backgroundColor: '#ffffff',
            borderColor: clusterColors[c % clusterColors.length],
            borderWidth: 3,
            pointStyle: 'rectRot',
            pointRadius: 14,
            pointHoverRadius: 16
        });
    }

    if (kmeansChartInstance) {
        kmeansChartInstance.destroy();
    }

    const xLabel = res.feature_x_name || 'Trục X';
    const yLabel = res.feature_y_name || 'Trục Y';

    kmeansChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            plugins: {
                title: {
                    display: true,
                    text: `Vòng lặp ${iter.iteration} / ${res.total_iterations} ${res.converged && stepIdx === res.iterations.length - 1 ? '— (Đã hội tụ tối ưu! 🎉)' : ''}`,
                    color: '#0f172a',
                    font: { size: 15, weight: 'bold' }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    padding: 10,
                    callbacks: {
                        label: (ctx) => {
                            const raw = ctx.raw;
                            if (raw.label && raw.label.startsWith('Tâm')) {
                                return `🎯 ${raw.label}: [${xLabel}: ${raw.x}, ${yLabel}: ${raw.y}]`;
                            }
                            let text = `👤 ${raw.label}: (${raw.x}, ${raw.y})`;
                            if (raw.job_role) text += ` | ${raw.job_role}`;
                            if (raw.attrition) text += ` | Nghỉ việc: ${raw.attrition}`;
                            return text;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: xLabel, font: { weight: 'bold' } },
                    grid: { color: 'rgba(0, 0, 0, 0.06)' },
                    ticks: { color: '#475569' }
                },
                y: {
                    title: { display: true, text: yLabel, font: { weight: 'bold' } },
                    grid: { color: 'rgba(0, 0, 0, 0.06)' },
                    ticks: { color: '#475569' }
                }
            }
        }
    });

    // 2. Render Distance Matrix D^(t) and Partition Matrix U^(t) tables
    const tablesOutput = document.getElementById('km-tables-output');
    if (!tablesOutput) return;

    let html = `<div class="panel-grid" style="grid-template-columns: 1fr 1fr;gap:1.5rem;">`;

    // Limit displayed rows in matrix if dataset is huge (>50) to prevent DOM freeze
    const maxMatrixRows = 50;
    const totalPoints = iter.point_labels.length;
    const isTruncated = totalPoints > maxMatrixRows;
    const displayIndices = isTruncated ? Array.from({length: maxMatrixRows}, (_, i) => i) : Array.from({length: totalPoints}, (_, i) => i);

    // Distance D_t table
    html += `<div>
        <h4>Bảng Khoảng Cách D^(${iter.iteration}) ${isTruncated ? `<small class="text-muted">(Hiển thị ${maxMatrixRows}/${totalPoints} điểm)</small>` : ''}</h4>
        <div style="max-height:360px;overflow:auto;border:1px solid #e2e8f0;border-radius:6px;">
        <table style="width:100%;font-size:0.88rem;margin:0;"><thead><tr style="position:sticky;top:0;background:#f8fafc;"><th>Điểm</th>`;
    for (let c = 0; c < numClusters; c++) html += `<th>d(x_i, m_${c+1})</th>`;
    html += `</tr></thead><tbody>`;

    displayIndices.forEach((i) => {
        const label = iter.point_labels[i];
        html += `<tr><td><strong>${label}</strong></td>`;
        for (let c = 0; c < numClusters; c++) {
            const isMin = iter.cluster_assignments[i] === c;
            html += `<td style="${isMin ? 'color:#0284c7; font-weight:bold; background:#f0f9ff;' : ''}">${iter.distance_matrix_D[i][c]}</td>`;
        }
        html += `</tr>`;
    });
    html += `</tbody></table></div></div>`;

    // Partition U_t table
    html += `<div>
        <h4>Ma Trận Phân Hoạch U^(${iter.iteration}) ${isTruncated ? `<small class="text-muted">(Hiển thị ${maxMatrixRows}/${totalPoints} điểm)</small>` : ''}</h4>
        <div style="max-height:360px;overflow:auto;border:1px solid #e2e8f0;border-radius:6px;">
        <table style="width:100%;font-size:0.88rem;margin:0;"><thead><tr style="position:sticky;top:0;background:#f8fafc;"><th>Điểm</th>`;
    for (let c = 0; c < numClusters; c++) html += `<th>u_{i${c+1}}</th>`;
    html += `</tr></thead><tbody>`;

    displayIndices.forEach((i) => {
        const label = iter.point_labels[i];
        html += `<tr><td><strong>${label}</strong></td>`;
        for (let c = 0; c < numClusters; c++) {
            const val = iter.partition_matrix_U[i][c];
            html += `<td style="${val === 1 ? 'color:#059669; font-weight:bold; background:#f0fdf4;' : ''}">${val}</td>`;
        }
        html += `</tr>`;
    });
    html += `</tbody></table></div></div></div>`;

    tablesOutput.innerHTML = html;
}

