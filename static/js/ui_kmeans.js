/**
 * [TV4] UI Script for K-Means Chart.js 2D Scatter Plot & D_t / U_t Matrices
 */
let kmeansChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    const btnRunKMeans = document.getElementById('btn-run-kmeans');
    if (btnRunKMeans) {
        btnRunKMeans.addEventListener('click', async () => {
            const k = parseInt(document.getElementById('km-k').value);
            const maxIter = parseInt(document.getElementById('km-max-iter').value);
            const jsonRaw = document.getElementById('km-points-input').value;

            try {
                const points = JSON.parse(jsonRaw);
                const payload = { points, k, max_iter: maxIter };
                const res = await API.post('kmeans', payload);

                if (!res.iterations || res.iterations.length === 0) return;

                // Setup step playback buttons
                const stepControls = document.getElementById('km-step-controls');
                stepControls.innerHTML = res.iterations.map((iter, idx) => `
                    <button class="btn btn-secondary btn-sm km-step-btn ${idx === 0 ? 'active' : ''}" data-step="${idx}">
                        Lặp ${iter.iteration}
                    </button>
                `).join('');

                // Render first iteration
                renderKMeansStep(res, 0);

                // Add step click listeners
                document.querySelectorAll('.km-step-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        document.querySelectorAll('.km-step-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        const stepIdx = parseInt(btn.getAttribute('data-step'));
                        renderKMeansStep(res, stepIdx);
                    });
                });

            } catch (e) {
                console.error(e);
            }
        });
    }
});

function renderKMeansStep(res, stepIdx) {
    const iter = res.iterations[stepIdx];
    const ctx = document.getElementById('kmeansChart').getContext('2d');

    const clusterColors = ['#0284c7', '#7c3aed', '#059669', '#d97706', '#e11d48'];

    // 1. Group points by assigned cluster for Chart.js datasets
    const datasets = [];
    const numClusters = res.k;

    for (let c = 0; c < numClusters; c++) {
        const clusterPoints = [];
        iter.cluster_assignments.forEach((assigned, ptIdx) => {
            if (assigned === c) {
                clusterPoints.push({
                    x: iter.points_x[ptIdx],
                    y: iter.points_y[ptIdx],
                    label: iter.point_labels[ptIdx]
                });
            }
        });

        datasets.push({
            label: `Cụm C${c+1}`,
            data: clusterPoints,
            backgroundColor: clusterColors[c % clusterColors.length],
            pointRadius: 8,
            pointHoverRadius: 10
        });

        // Add Centroid point
        const centroidCoords = iter.centroids[c];
        datasets.push({
            label: `Tâm m${c+1}`,
            data: [{ x: centroidCoords[0], y: centroidCoords[1], label: `m${c+1}` }],
            backgroundColor: '#ffffff',
            borderColor: clusterColors[c % clusterColors.length],
            borderWidth: 3,
            pointStyle: 'rectRot',
            pointRadius: 12,
            pointHoverRadius: 14
        });
    }

    if (kmeansChartInstance) {
        kmeansChartInstance.destroy();
    }

    kmeansChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Vòng lặp ${iter.iteration} / ${res.total_iterations} ${res.converged && stepIdx === res.iterations.length - 1 ? '(Hội tụ hoàn tất!)' : ''}`,
                    color: '#0f172a',
                    font: { size: 14, weight: 'bold' }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.raw.label}: (${ctx.raw.x}, ${ctx.raw.y})`
                    }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(0, 0, 0, 0.06)' }, ticks: { color: '#475569' } },
                y: { grid: { color: 'rgba(0, 0, 0, 0.06)' }, ticks: { color: '#475569' } }
            }
        }
    });

    // 2. Render Distance Matrix D^(t) and Partition Matrix U^(t) tables
    const tablesOutput = document.getElementById('km-tables-output');
    let html = `<div class="panel-grid" style="grid-template-columns: 1fr 1fr;">`;

    // Distance D_t table
    html += `<div><h4>Bảng Khoảng Cách D^(${iter.iteration})</h4><table><thead><tr><th>Điểm</th>`;
    for (let c = 0; c < numClusters; c++) html += `<th>d(x_i, m_${c+1})</th>`;
    html += `</tr></thead><tbody>`;

    iter.point_labels.forEach((label, i) => {
        html += `<tr><td><strong>${label}</strong></td>`;
        for (let c = 0; c < numClusters; c++) {
            const isMin = iter.cluster_assignments[i] === c;
            html += `<td style="${isMin ? 'color:#0284c7; font-weight:bold;' : ''}">${iter.distance_matrix_D[i][c]}</td>`;
        }
        html += `</tr>`;
    });
    html += `</tbody></table></div>`;

    // Partition U_t table
    html += `<div><h4>Ma Trận Phân Hoạch U^(${iter.iteration})</h4><table><thead><tr><th>Điểm</th>`;
    for (let c = 0; c < numClusters; c++) html += `<th>u_{i${c+1}}</th>`;
    html += `</tr></thead><tbody>`;

    iter.point_labels.forEach((label, i) => {
        html += `<tr><td><strong>${label}</strong></td>`;
        for (let c = 0; c < numClusters; c++) {
            const val = iter.partition_matrix_U[i][c];
            html += `<td style="${val === 1 ? 'color:#059669; font-weight:bold;' : ''}">${val}</td>`;
        }
        html += `</tr>`;
    });
    html += `</tbody></table></div></div>`;

    tablesOutput.innerHTML = html;
}
