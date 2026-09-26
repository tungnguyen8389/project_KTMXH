/**
 * Import UI: upload a CSV -> replace the single working dataset.
 * On success (and on reload if already imported), prefill every algorithm tab's
 * data input with the imported rows and unlock the algorithm tabs.
 */
document.addEventListener('DOMContentLoaded', () => {
    const IMPORT_FLAG = 'ktmxh_data_imported';
    // Data-input textareas of each algorithm tab.
    const DATA_TEXTAREAS = ['rs-json-input', 'km-points-input', 'ap-json-input', 'cl-json-input'];

    function applyImportedData(records) {
        const json = JSON.stringify(records);
        DATA_TEXTAREAS.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = json;
        });
        window.IMPORTED_DATASET = records;
    }

    // On reload: if data was already imported in this browser, load it into the tabs.
    let alreadyImported = false;
    try { alreadyImported = localStorage.getItem(IMPORT_FLAG) === '1'; } catch (e) { /* ignore */ }
    if (alreadyImported) {
        fetch('/api/datasets/')
            .then(r => (r.ok ? r.json() : []))
            .then(list => { if (Array.isArray(list) && list.length) applyImportedData(list[0].data_json); })
            .catch(() => { /* ignore */ });
    }

    const btn = document.getElementById('btn-run-import');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        const fileInput = document.getElementById('import-file');
        const summaryBox = document.getElementById('import-summary');
        const stepsBox = document.getElementById('import-steps-output');
        const tableBox = document.getElementById('import-table-output');

        if (!fileInput.files.length) {
            alert('Vui lòng chọn một tệp CSV!');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        btn.disabled = true;
        btn.textContent = 'Đang nhập...';
        try {
            const response = await fetch('/api/import/', {
                method: 'POST',
                headers: { 'X-CSRFToken': API.getCsrfToken() },
                body: formData
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Lỗi khi nhập dữ liệu!');
            }

            // Feed imported data into every algorithm tab, then unlock them.
            applyImportedData(result.data_json);
            if (typeof window.unlockDataTabs === 'function') {
                window.unlockDataTabs();
            }

            summaryBox.innerHTML =
                `<strong>Nhập thành công!</strong> ${result.row_count} dòng, ${result.columns.length} cột. ` +
                `Các tab thuật toán đã mở khoá và đang dùng bộ dữ liệu này.`;
            const cl = result.cleaning || {};
            const dropped = (cl.dropped_columns && cl.dropped_columns.length)
                ? cl.dropped_columns.join(', ') : 'không có';
            stepsBox.innerHTML =
                `<p><strong>Tên:</strong> ${result.name}</p>` +
                `<p><strong>Data Cleaning:</strong> đã bỏ cột [${escapeHtml(dropped)}], ` +
                `điền ${cl.missing_filled || 0} ô thiếu.</p>` +
                `<p><strong>Cột còn lại:</strong> ${escapeHtml(result.columns.join(', '))}</p>`;
            tableBox.innerHTML = renderPreview(result.columns, result.data_json.slice(0, 10));
        } catch (err) {
            alert(`Lỗi nhập dữ liệu: ${err.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Nhập Dữ Liệu';
        }
    });

    function renderPreview(columns, rows) {
        const head = columns.map(c => `<th>${escapeHtml(c)}</th>`).join('');
        const body = rows.map(row => {
            const cells = columns.map(c => `<td>${escapeHtml(row[c])}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
});
