/**
 * [TV4] API Client Helper for making AJAX requests to Django REST Framework endpoints
 */
const API = {
    async post(endpoint, data) {
        try {
            const response = await fetch(`/api/${endpoint}/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCsrfToken()
                },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Lỗi khi gọi API backend!');
            }
            return result;
        } catch (err) {
            alert(`Lỗi API (${endpoint}): ${err.message}`);
            throw err;
        }
    },

    getCsrfToken() {
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue || '';
    }
};

// Global Tab Switching Logic
document.addEventListener('DOMContentLoaded', () => {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const pageHeading = document.getElementById('page-heading');

    const titles = {
        'tab-roughset': '2. Tập thô & Rút gọn Thuộc tính (IND, Upper/Lower, Discriminiability Matrix)',
        'tab-kmeans': '3. Phân cụm K-Means (Euclidean, Distance D_t, Partition U_t)',
        'tab-apriori': '4. Luật kết hợp Apriori (Bitvector, Itemsets C_k/F_k, Rules)',
        'tab-classification': '5. Phân lớp (ID3 Decision Tree & Naive Bayes Laplace)',
        'tab-import': 'Nhập Dữ liệu từ tệp CSV'
    };

    // Gate: algorithm tabs stay locked until data is imported in this browser.
    const IMPORT_FLAG = 'ktmxh_data_imported';
    function setTabsLocked(locked) {
        navItems.forEach(item => {
            if (item.getAttribute('data-tab') === 'tab-import') return;
            item.classList.toggle('locked', locked);
        });
    }
    // Called by ui_import.js after a successful import.
    window.unlockDataTabs = function () {
        try { localStorage.setItem(IMPORT_FLAG, '1'); } catch (e) { /* ignore */ }
        setTabsLocked(false);
    };
    let alreadyImported = false;
    try { alreadyImported = localStorage.getItem(IMPORT_FLAG) === '1'; } catch (e) { /* ignore */ }
    setTabsLocked(!alreadyImported);

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            if (item.classList.contains('locked')) return;
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(nav => nav.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));

            item.classList.add('active');
            const targetPanel = document.getElementById(targetTab);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            if (pageHeading && titles[targetTab]) {
                pageHeading.textContent = titles[targetTab];
            }
        });
    });
});
