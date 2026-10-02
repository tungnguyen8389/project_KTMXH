# KTMXH Visualizer — Hướng dẫn cài đặt và chạy

Ứng dụng web Django minh họa các thuật toán khai phá dữ liệu (Tiền xử lý, Tập thô / Reduct, K-Means, Apriori, ID3, Naive Bayes) trên bộ dữ liệu nhân sự HR (1.470 dòng).

## 1. Yêu cầu

| Thành phần | Phiên bản |
|---|---|
| Python | **>= 3.12** (Django 6.x yêu cầu) |
| pip | bản mới nhất |
| Trình duyệt | có kết nối Internet — giao diện tải Chart.js, Mermaid, KaTeX, Google Fonts từ CDN |
| MySQL | *tùy chọn* — mặc định dùng SQLite, không cần cài gì thêm |

Kiểm tra phiên bản Python:

```bash
python3 --version      # macOS / Linux
py --version           # Windows
```

## 2. Cài đặt nhanh (macOS / Linux)

```bash
bash setup.sh
```

Script sẽ: tạo virtualenv `~/.venv` → cài `requirements.txt` → `migrate` → nạp dữ liệu HR từ `data/init.sql` → tạo tài khoản admin `admin / admin123`.

Xong chuyển sang [mục 4](#4-chạy-ứng-dụng).

## 3. Cài đặt thủ công

### 3.1. Tạo và kích hoạt virtualenv

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Nếu PowerShell chặn script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3.2. Cài thư viện

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

### 3.3. Tạo bảng và nạp dữ liệu

```bash
python manage.py migrate
python manage.py seed_datasets
```

Kết quả mong đợi: `Bảng Employee hiện có 1470 dòng (nguồn: data/init.sql).`

> Nếu bỏ qua bước `seed_datasets`, dữ liệu cũng tự được nạp khi chạy `runserver` lần đầu (bảng Employee đang rỗng).
> Nạp lại từ đầu: `python manage.py seed_datasets --force`

### 3.4. (Tùy chọn) Tạo tài khoản admin

```bash
python manage.py createsuperuser
```

## 4. Chạy ứng dụng

```bash
python manage.py runserver
```

| Địa chỉ | Nội dung |
|---|---|
| http://127.0.0.1:8000/ | Giao diện chính (các tab thuật toán) |
| http://127.0.0.1:8000/admin/ | Trang quản trị Django |
| http://127.0.0.1:8000/api/ | REST API (`datasets/`, `preprocessing/`, `rough-set/`, `reduct/`, `kmeans/`, `apriori/`, `encode-transactions/`, `classification-data/`, `id3/`, `naive-bayes/`) |

Đổi cổng: `python manage.py runserver 8080`. Dừng server: `Ctrl + C`.

Mỗi lần mở terminal mới cần kích hoạt lại virtualenv (`source .venv/bin/activate`, hoặc `source ~/.venv/bin/activate` nếu đã dùng `setup.sh`).

## 5. Chạy test

```bash
python manage.py test core
```

## 6. Dùng MySQL thay cho SQLite (tùy chọn)

1. Cài driver: `pip install PyMySQL`
2. Tạo database (charset `utf8mb4`) và user trên MySQL.
3. Đặt biến môi trường rồi chạy như bình thường:

```bash
export DB_ENGINE=mysql
export DB_NAME=ktmxh DB_USER=ktmxh DB_PASSWORD=ktmxh DB_HOST=127.0.0.1 DB_PORT=3306

python manage.py migrate
python manage.py seed_datasets
python manage.py runserver
```

Các giá trị trên là mặc định trong `config/settings.py`; chỉ cần đặt biến nào khác mặc định. Windows PowerShell dùng `$env:DB_ENGINE="mysql"`.

## 7. Cấu trúc thư mục

```
config/            Cấu hình Django (settings, urls)
core/
  algorithms/      Cài đặt các thuật toán (không dùng thư viện ML)
  views.py         API endpoints
  seeding.py       Nạp data/init.sql vào bảng Employee
  tests.py         Bộ test
data/init.sql      Dữ liệu HR gốc (1.470 dòng)
templates/         Giao diện HTML (mỗi tab một file trong components/)
static/            CSS, JavaScript
docs/              Tài liệu đặc tả, kiến trúc, báo cáo
```

Chi tiết kiến trúc: [docs/KIEN_TRUC_HE_THONG.md](docs/KIEN_TRUC_HE_THONG.md).

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'django'` | Chưa kích hoạt virtualenv, hoặc chưa `pip install -r requirements.txt` |
| `ERROR: No matching distribution found for Django>=6.1` | Python cũ hơn 3.12 — cài Python mới hơn rồi tạo lại virtualenv |
| `no such table: core_employee` | Chưa chạy `python manage.py migrate` |
| Giao diện không có biểu đồ / công thức | Máy không có Internet nên không tải được thư viện từ CDN |
| `Error: That port is already in use.` | Chạy cổng khác: `python manage.py runserver 8080` |
| `django.core.exceptions.ImproperlyConfigured: Error loading MySQLdb module` | Đang đặt `DB_ENGINE=mysql` nhưng chưa `pip install PyMySQL` |

> **Lưu ý bảo mật:** `config/settings.py` đang để `DEBUG = True`, `ALLOWED_HOSTS = ['*']` và `SECRET_KEY` cố định — chỉ dùng cho môi trường học tập / chạy local, không triển khai công khai với cấu hình này.
