# TÀI LIỆU ĐẶC TẢ HỆ THỐNG (SYSTEM SPECIFICATION)
## DỰ ÁN: NỀN TẢNG TRỰC QUAN HÓA THUẬT TOÁN KHAI PHÁ DỮ LIỆU (KTMXH VISUALIZER)

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

### 1.1. Mục tiêu
Hệ thống **KTMXH Visualizer** là nền tảng web mã nguồn mở được thiết kế nhằm phục vụ giảng dạy, học tập và kiểm chứng thực nghiệm các thuật toán cốt lõi trong môn học **Khai Thác Dữ Liệu / Khai Thác Truyền Thông Mạng Xã Hội**.

Hệ thống cho phép:
- Trực quan hóa từng bước thực thi (step-by-step execution) của các thuật toán.
- Kết xuất công thức toán học chi tiết (LaTeX/KaTeX) cùng ma trận trung gian.
- Ứng dụng quy trình làm sạch và khai phá trên tập dữ liệu thực tế: **IBM HR Employee Attrition** (~1.470 bản ghi $\times$ 31 thuộc tính).

### 1.2. Công nghệ sử dụng (Tech Stack)
- **Backend Framework:** Django 5.x, Django REST Framework (DRF).
- **Ngôn ngữ:** Python 3.13+.
- **Thư viện tính toán:** Pandas, NumPy (chỉ dùng cho I/O, tiền xử lý và ma trận, thuật toán lõi cài đặt thủ công).
- **Cơ sở dữ liệu:** SQLite / MySQL.
- **Frontend:** HTML5 Semantic, Vanilla CSS (Modern Theme, Glassmorphism, CSS Grid/Flexbox), Vanilla JavaScript (ES6+ Modules).
- **Thư viện đồ họa & toán học:** [Chart.js](https://www.chartjs.org/) (2D Dynamic Scatter Plot), [KaTeX](https://katex.org/) (Hiển thị công thức toán học).

---

## 2. KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│   • Base Template + 4 Tab Components (Apriori, K-Means, RoughSet, Tree)│
│   • Client-side Controllers: api.js, ui_*.js                           │
│   • Renderers: KaTeX Math Formatter, Chart.js 2D Visualizer            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / JSON (REST API)
┌───────────────────────────────────▼────────────────────────────────────┐
│                          APPLICATION LAYER                             │
│   • Django Views & DRF API Controllers (core/views.py)                 │
│   • Data Pipeline: Data Cleaning & Transaction Encoder                 │
│   • Database Models: Employee, ExecutionHistory                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Python Call
┌───────────────────────────────────▼────────────────────────────────────┐
│                        CORE ALGORITHMS ENGINE                          │
│   ├── PreprocessingEngine (Min-Max, Z-Score)                           │
│   ├── RoughSetEngine (Lower/Upper Approx, Boundary, Pos, Degree k)     │
│   ├── ReductEngine (Discernibility Matrix, Boolean Reducts, Core)      │
│   ├── KMeansEngine (Step-by-step D_t, U_t, Centroids update)           │
│   ├── AprioriEngine (Manual k-subset, Bitvector support, Rules)        │
│   └── ClassificationEngine (ID3 Decision Tree, Naive Bayes Classifier) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ĐẶC TẢ DỮ LIỆU & PIPELINE TIỀN XỬ LÝ

### 3.1. Mô hình Dữ liệu (Data Models)
- **`Employee` (Bảng chính):** Lưu trữ 1.470 nhân viên từ bộ dữ liệu IBM HR với 31 trường thông tin phân loại và số:
  - *Biến mục tiêu:* `Attrition` (`Yes` / `No`).
  - *Biến phân loại:* `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`.
  - *Biến số/thang đo:* `Age`, `DailyRate`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `MonthlyRate`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `WorkLifeBalance`, `YearsAtCompany`, `NumCompaniesWorked`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.
- **`ExecutionHistory`:** Lưu trữ vết lịch sử thực thi các thuật toán và tham số đầu vào.

### 3.2. Quy trình làm sạch dữ liệu 7 bước (`core/data_cleaning.py`)
1. **Loại bỏ cột chết (Dead Columns):** Bỏ các cột vô nghĩa hoặc đồng nhất: `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`.
2. **Ép kiểu dữ liệu (Type Coercion):** Chuyển đổi các cột dạng chuỗi số sang số nguyên/thực.
3. **Chuẩn hóa chuỗi (Categorical Standardization):** Cắt khoảng trắng dư thừa, chuẩn hóa chữ hoa/thường.
4. **Loại bỏ cột gần như hằng số (Near-constant drop):** Bỏ các cột có $>99\%$ giá trị trùng lặp.
5. **Khử trùng lặp (Deduplication):** Loại bỏ các hàng trùng lặp hoàn toàn.
6. **Xử lý ngoại lai (IQR Outlier Capping / Winsorization):** Giới hạn giá trị số trong khoảng $[Q_1 - 1.5 \times \text{IQR},\, Q_3 + 1.5 \times \text{IQR}]$.
7. **Điền giá trị thiếu (Imputation):** Điền `Median` cho thuộc tính số, điền `Mode` (yếu vị) cho thuộc tính phân loại.

### 3.3. Bộ mã hóa giao dịch (`core/transaction_encoder.py`)
- Rút trích các thuộc tính quan trọng liên quan đến Attrition.
- Rời rạc hóa (Binning) các biến số thành nhãn ngữ nghĩa:
  - `Age`: `Young` (<30), `Middle` (30–50), `Senior` (>50).
  - `DistanceFromHome`: `Near` ($\le 5$), `Medium` (6–15), `Far` (>15).
  - `YearsAtCompany`: `New` (<3), `Established` (3–10), `Veteran` (>10).
  - Các biến liên tục khác: Sử dụng phân vị `pandas.qcut` chia 3 mức `Low`, `Medium`, `High`.
- Sinh chuỗi giao dịch dạng token: `{"tid": "T1", "items": ["OverTime=Yes", "Age=Young", "Attrition=Yes"]}`.

---

## 4. ĐẶC TẢ CHI TIẾT CÁC THUẬT TOÁN (CORE ALGORITHMS)

### 4.1. Lý thuyết Tập thô & Rút gọn thuộc tính (Rough Set & Reduct)
- **Tập tương đương:** $U/\text{IND}(B)$ gom cụm các đối tượng có cùng giá trị thuộc tính điều kiện $B$.
- **Xấp xỉ dưới (Lower Approximation):**
  $$B_{\underline{A}}(X) = \bigcup \{ [x]_B \mid [x]_B \subseteq X \}$$
- **Xấp xỉ trên (Upper Approximation):**
  $$B^{\overline{A}}(X) = \bigcup \{ [x]_B \mid [x]_B \cap X \neq \emptyset \}$$
- **Vùng biên (Boundary Region):**
  $$BN_B(X) = B^{\overline{A}}(X) \setminus B_{\underline{A}}(X)$$
- **Độ chính xác xấp xỉ:**
  $$\alpha_B(X) = \frac{|B_{\underline{A}}(X)|}{|B^{\overline{A}}(X)|}$$
- **Vùng dương và độ phụ thuộc thuộc tính:**
  $$POS_B(D) = \bigcup_{X \in U/D} B_{\underline{A}}(X), \quad k = \gamma_B(D) = \frac{|POS_B(D)|}{|U|}$$
- **Ma trận phân biệt $n \times n$ & Rút gọn tối tiểu:**
  - Phần tử $c_{ij} = \{ a \in C \mid a(x_i) \neq a(x_j) \}$ khi $d(x_i) \neq d(x_j)$.
  - Hàm phân biệt logic: $f_M = \bigwedge_{(i,j)} \left( \bigvee a \right)$.
  - Rút gọn tối tiểu $\text{RED}(C)$ và thuộc tính cốt lõi $\text{CORE}(C) = \bigcap \text{RED}(C)$.
  - **Cài đặt `ReductEngine`:** tìm các tập thuộc tính nhỏ nhất giao đủ mọi mệnh đề của $f_M$ (minimal hitting set, duyệt theo kích thước tăng dần) — tương đương toán học với rút gọn $f_M$ bằng luật hấp thụ/phân phối nhưng không bùng nổ số đơn thức trung gian.
  - **CORE suy trực tiếp từ ma trận:** thuộc tính xuất hiện đơn lẻ (mệnh đề 1 phần tử) trong một ô $c_{ij}$ — trả về ở trường `core_from_matrix`, luôn phải trùng `core_attributes` (giao các reduct).
  - **Giới hạn an toàn:** tối đa 12 thuộc tính điều kiện mỗi lần tính Reduct (độ phức tạp $2^{|C|}$), vượt ngưỡng trả lỗi 400.
  - **Bảng lớn:** với hơn 60 đối tượng, chỉ quét các dòng (giá trị điều kiện, quyết định) khác nhau thay vì mọi cặp đối tượng — cùng tập mệnh đề, nhanh hơn nhiều bậc (1.470 dòng ≈ 0,2 giây); ma trận $n \times n$ không được trả về (`matrix_truncated = true`), vẫn có đủ mệnh đề, Reduct và Core.

---

### 4.2. Phân cụm K-Means (K-Means Clustering)
- **Nguồn dữ liệu:** Hỗ trợ cả 2 chế độ:
  - *Dữ liệu Nhân sự từ Database (IBM HR Employee):* Tự động lấy trực tiếp từ bảng `Employee` với các cặp thuộc tính số tùy chọn (ví dụ: `Age` vs `MonthlyIncome`, `TotalWorkingYears` vs `MonthlyIncome`, `YearsAtCompany` vs `PercentSalaryHike`,...).
  - *Dữ liệu Điểm tùy chỉnh:* Nhập tọa độ 2D dưới dạng JSON.
- **Phương pháp khởi tạo tâm cụm ban đầu:**
  - `kmeans++`: Chọn tâm thông minh theo xác suất tỉ lệ với bình phương khoảng cách $D(x)^2$, giúp tăng tốc độ hội tụ và tránh nghiệm cục bộ.
  - `random`: Chọn ngẫu nhiên $k$ điểm với seed cố định.
  - `first_k`: Chọn $k$ điểm đầu tiên của tập dữ liệu.
- **Chuẩn hóa Min-Max (Feature Scaling):**
  - Tùy chọn chuẩn hóa các đặc trưng về $[0, 1]$ khi tính ma trận khoảng cách $D^{(t)}$, loại bỏ sự chi phối lệch của các thuộc tính có biên độ lớn (`MonthlyIncome` vs `Age`), sau đó tự động ánh xạ tâm cụm về thang đo gốc.
- **Khoảng cách Euclid từ điểm $x_i$ tới tâm cụm $m_j$:**
  $$d(x_i, m_j) = \sqrt{\sum_{l=1}^d (x_{il} - m_{jl})^2}$$
- **Ma trận khoảng cách $D^{(t)}$:** Kích thước $n \times k$.
- **Ma trận phân hoạch $U^{(t)}$:** 
  $$u_{ij} = \begin{cases} 1 & \text{nếu } j = \arg\min_{l} d(x_i, m_l) \\ 0 & \text{ngược lại} \end{cases}$$
- **Tổng sai số nội cụm (Within-Cluster Sum of Squares - WCSS / Inertia):**
  $$\text{WCSS} = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - m_j\|^2$$
- **Cập nhật tâm cụm $m_j^{(t+1)}$:**
  $$m_j^{(t+1)} = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$
- **Điều kiện dừng:** $\max_j \Vert m_j^{(t+1)} - m_j^{(t)} \Vert \le 10^{-4}$ hoặc đạt `max_iter`.
- **Hồ sơ Phân tích Cụm & Tỷ lệ Nghỉ việc (Cluster Profiles & Attrition Insights):**
  - Tự động thống kê số lượng nhân viên, tỷ lệ %, miền giá trị (Min, Max, Mean) và tỷ lệ nhân viên nghỉ việc (`Attrition = Yes` %) trong từng cụm sau khi hội tụ.


---

### 4.3. Khai phá Luật kết hợp (Apriori Algorithm)
- **Cài đặt:** Thủ công hoàn toàn (Manual hand-coded 100%), không dùng `itertools` hay thư viện ngoài.
- **Biểu diễn Bitvector:** $v(S \cup T) = v(S) \otimes v(T)$ với phép nhân vector $z_k = \min(s_k, t_k)$.
- **Tập mục phổ biến $F_k$:** Các tập có độ hỗ trợ $\text{Support}(X) \ge \text{minsupp}$.
- **Sinh luật kết hợp $X \Rightarrow Y$:**
  $$\text{Support}(X \Rightarrow Y) = P(X \cup Y) = \frac{|\sigma(X \cup Y)|}{|D|}$$
  $$\text{Confidence}(X \Rightarrow Y) = P(Y \mid X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$
  $$\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$$
- **Bộ lọc luật:** Hỗ trợ lọc luật đích `Attrition=Yes` và phân tích theo phân khúc (`Segment Mode`).

---

### 4.4. Phân lớp dữ liệu (Classification)
- **Cây quyết định ID3:**
  - Entropy tập mẫu $S$:
    $$I(p, n) = -\frac{p}{p+n}\log_2\left(\frac{p}{p+n}\right) - \frac{n}{p+n}\log_2\left(\frac{n}{p+n}\right)$$
  - Entropy kỳ vọng khi phân hoạch theo thuộc tính $A$:
    $$E(A) = \sum_{v \in \text{Values}(A)} \frac{p_v + n_v}{p + n} I(p_v, n_v)$$
  - Độ lợi thông tin (Information Gain):
    $$\text{Gain}(A) = I(p, n) - E(A)$$
- **Phân loại Naive Bayes:**
  - Xác suất hậu nghiệm:
    $$P(C_k \mid X) \propto P(C_k) \prod_{i=1}^d P(x_i \mid C_k)$$
  - Tích hợp kỹ thuật làm trơn Laplace:
    $$P(x_i = v \mid C_k) = \frac{N_{ik} + 1}{N_k + |V_i|}$$

---

## 5. ĐẶC TẢ API RESTFUL (API SPECIFICATIONS)

| Method | Endpoint | Chức năng | Tham số chính (Body/Query) |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/datasets/` | Lấy dữ liệu nhân sự đã nạp | *None* |
| `GET` | `/api/encode-transactions/` | Mã hóa dữ liệu HR sang giao dịch Apriori | `columns`, `n_bins` |
| `POST` | `/api/preprocessing/` | Chuẩn hóa Min-Max / Z-Score | `data`, `feature`, `type`, `new_min`, `new_max` |
| `POST` | `/api/rough-set/` | Tính xấp xỉ thô, vùng dương, độ phụ thuộc $k$ | `data`, `condition_attrs`, `decision_attr` |
| `POST` | `/api/reduct/` | Tính ma trận phân biệt & tập rút gọn tối tiểu | `data`, `condition_attrs`, `decision_attr` |
| `GET` | `/api/roughset-data/` | Nạp mẫu dữ liệu HR từ DB (đã rời rạc hoá bằng `encode_records`) cho tab Tập thô | `sample_size` (mặc định 30, `0` = toàn bộ), `seed` (mặc định 42) |
| `POST` | `/api/kmeans/` | Phân cụm K-Means từng vòng lặp | `points`, `k`, `max_iter`, `initial_centroids` |
| `POST` | `/api/apriori/` | Khai phá tập phổ biến & luật kết hợp | `transactions`, `min_supp`, `min_conf`, `min_lift`, `target_mode` |
| `POST` | `/api/id3/` | Xây dựng cây quyết định ID3 | `data`, `condition_attrs`, `target_attr` |
| `POST` | `/api/naive-bayes/` | Phân lớp xác suất Naive Bayes | `train_data`, `test_sample`, `target_attr`, `laplace` |

---

## 6. ĐẶC TẢ GIAO DIỆN & TRẢI NGHIỆM NGƯỜI DÙNG (FRONTEND UX/UI)

1. **Tab 1: Lý thuyết Tập thô & Rút gọn thuộc tính (`tab_roughset.html`)**
   - Bảng nhập dữ liệu mẫu (hoặc JSON).
   - Nút **"Nạp dữ liệu HR (Attrition) từ DB"** cùng ô *Số bản ghi lấy mẫu* (0 = toàn bộ 1.470): lấy mẫu ngẫu nhiên phân tầng theo `Attrition`, tự điền thuộc tính điều kiện, thuộc tính quyết định và JSON; sau đó chạy 2 nút Xấp xỉ / Reduct như thường.
   - Render trực tiếp công thức xấp xỉ trên, dưới, vùng biên, độ phụ thuộc $k$ bằng **KaTeX**.
   - Bảng ma trận phân biệt $n \times n$ có highlight các thuộc tính phân biệt; ô đơn lẻ (thuộc tính cốt lõi) được tô nổi bật. Bảng hơn 60 đối tượng chỉ hiển thị mệnh đề, Reduct và Core.
2. **Tab 2: Phân cụm K-Means (`tab_kmeans.html`)**
   - Khung nhập tọa độ 2D JSON, tùy chọn $k$ và `max_iter`.
   - Biểu đồ **2D Dynamic Scatter Plot** hiển thị màu cụm và tâm cụm xoay hình thoi.
   - Thanh điều khiển tua bước lặp (Step playback: Lặp 1, Lặp 2,...) kèm bảng ma trận khoảng cách $D^{(t)}$ và ma trận phân hoạch $U^{(t)}$.
3. **Tab 3: Khai phá Luật kết hợp Apriori (`tab_apriori.html`)**
   - Nút **"Dùng dữ liệu đã nhập (HR)"** tự động mã hóa 1.470 dòng thành giao dịch.
   - Bảng tập phổ biến $F_1, F_2, \dots, F_k$ kèm biểu diễn Bitvector.
   - Bảng luật kết hợp với bộ lọc Support, Confidence, Lift, Target Attrition.
4. **Tab 4: Phân lớp & Cây quyết định (`tab_classification.html`)**
   - Bảng tính Entropy, Gain chi tiết từng thuộc tính tại mỗi nút phân nhánh.
   - Biểu đồ cây phân nhánh trực quan.
   - Form dự đoán xác suất Naive Bayes cho mẫu mới.

---

## 7. KIỂM THỬ VÀ VẬN HÀNH (TESTING & OPERATIONS)

### 7.1. Chạy Unit Tests
Hệ thống bao gồm 58 bài test tự động bao phủ toàn diện các phân hệ:
```bash
python manage.py test
```
*Đối soát với bài giảng:* lớp `RoughSetLectureExamplesTest` (Tập thô & Rút gọn) chạy lại 4 ví dụ giải sẵn trong `Bai3_Reduct.pdf` — bảng Thi đậu (slide 5–19), bảng thời tiết (slide 30–34), bảng rám nắng (slide 35–43), bảng tuyển dụng (slide 25–28) — và yêu cầu lớp tương đương, xấp xỉ dưới/trên, độ chính xác, độ phụ thuộc $k$, ma trận phân biệt, hàm phân biệt, các Reduct và Core trùng đáp án của thầy. Riêng slide 34 ghi $k = 6/8 = 0.66$ là nhầm phép chia; test dùng giá trị đúng $6/8 = 0.75$.

*Kết quả:* `Ran 58 tests - OK`.

### 7.2. Khởi chạy Hệ thống
```bash
# Áp dụng migration
python manage.py migrate

# Khởi chạy máy chủ phát triển
python manage.py runserver 127.0.0.1:8000
```
Truy cập giao diện: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
