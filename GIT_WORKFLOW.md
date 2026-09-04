# 🚀 HUỚNG DẪN QUY TRÌNH LÀM VIỆC THEO NHÁNH GIT (GIT WORKFLOW)

Hệ thống được thiết kế hoàn chỉnh thành **Framework cơ bản** với sự phân chia nghiêm ngặt cho 4 thành viên phát triển song song trên 4 nhánh độc lập trước khi gộp (merge) vào nhánh `main`.

---

## 👥 PHÂN CHIA NHÁNH & NHIỆM VỤ CỦA 4 THÀNH VIÊN

| Thành viên | Tên Nhánh (Git Branch) | Tập File Phụ Trách (Cấm sửa file thành viên khác) |
|---|---|---|
| **Thành viên 1 (TV1)** | `feature/tv1-preprocessing-roughset` | `core/models.py`, `core/admin.py`, `core/algorithms/preprocessing.py`, `core/algorithms/rough_set.py`, `core/algorithms/reduct.py` |
| **Thành viên 2 (TV2)** | `feature/tv2-kmeans-apriori` | `core/algorithms/kmeans.py`, `core/algorithms/association.py` |
| **Thành viên 3 (TV3)** | `feature/tv3-classification-api` | `core/algorithms/classification.py`, `core/views.py`, `core/urls.py` |
| **Thành viên 4 (TV4)** | `feature/tv4-ui-visualization` | `templates/base.html`, `templates/index.html`, `static/css/style.css`, `static/js/api.js`, `static/js/ui_*.js` |

---

## 🛠️ QUY TRÌNH THỰC HIỆN TỪNG THÀNH VIÊN

### Bước 1: Khởi tạo repository & Checkout sang nhánh cá nhân

Dưới đây là lệnh mẫu cho từng thành viên:

#### 1️⃣ Thành viên 1:
```bash
git checkout main
git pull origin main
git checkout -b feature/tv1-preprocessing-roughset
```

#### 2️⃣ Thành viên 2:
```bash
git checkout main
git pull origin main
git checkout -b feature/tv2-kmeans-apriori
```

#### 3️⃣ Thành viên 3:
```bash
git checkout main
git pull origin main
git checkout -b feature/tv3-classification-api
```

#### 4️⃣ Thành viên 4:
```bash
git checkout main
git pull origin main
git checkout -b feature/tv4-ui-visualization
```

---

### Bước 2: Phát triển & Kiểm thử tự động trên nhánh cá nhân

Mỗi thành viên làm việc trong các file được phân công. Để kiểm tra hệ thống hoạt động:
```bash
# Kiểm tra hệ thống Django
python manage.py check

# Chạy Unit Test thuật toán
python manage.py test core

# Chạy server thử nghiệm
python manage.py runserver 8000
```

---

### Bước 3: Commit & Push nhánh cá nhân lên Remote Repository

```bash
git add .
git commit -m "feat(tvX): hoàn thiện nâng cấp thuật toán và giao diện"
git push -u origin feature/tvX-ten-nhanh
```

---

### Bước 4: Merge nhánh cá nhân vào nhánh main (Không bị xung đột)

Vì các tập file đã được module hóa hoàn toàn tách biệt:
```bash
# Chuyển về nhánh main
git checkout main
git pull origin main

# Merge nhánh tính năng
git merge feature/tvX-ten-nhanh

# Push nhánh main mới nhất lên server
git push origin main
```

---

## ⚡ HƯỚNG DẪN KHỞI CHẠY DỰ ÁN BAN ĐẦU
1. Cài đặt thư viện: `pip install django djangorestframework pandas numpy`
2. Khởi tạo Database: `python manage.py migrate`
3. Nạp dữ liệu mẫu 5 thuật toán: `python manage.py seed_datasets`
4. Chạy Unit Test: `python manage.py test core`
5. Khởi chạy Web Application: `python manage.py runserver 8000`
