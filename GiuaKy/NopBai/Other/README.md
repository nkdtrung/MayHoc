# Product Classification and Clustering

- **Đề tài**: Product Classification and Clustering
- **Bộ môn / Học phần**: Machine Learning

---

## 1. Giới thiệu bài toán

Trong thương mại điện tử, việc tự động chuẩn hóa và phân loại hàng triệu tiêu đề sản phẩm do nhiều nhà bán lẻ (merchants) cung cấp vào đúng cấu trúc danh mục hàng hóa là bài toán then chốt nhằm tối ưu trải nghiệm tìm kiếm, so sánh giá và gợi ý sản phẩm.

Dự án tiếp cận bài toán qua hai hướng bổ trợ:
1. **Phân loại có giám sát (Supervised Multi-class Classification)**: Xây dựng pipeline học máy từ dữ liệu văn bản và ngữ cảnh nhà bán hàng để phân loại sản phẩm vào 10 danh mục chính xác.
2. **Gom cụm không giám sát (Unsupervised Clustering)**: Khám phá không gian biểu diễn ngữ nghĩa tiềm ẩn bằng kỹ thuật LSA (Latent Semantic Analysis) và thuật toán KMeans, từ đó đối chiếu mức độ phân tách tự nhiên của các cụm so với nhãn danh mục thực tế.

---

## 2. Dữ liệu thực nghiệm

Dự án sử dụng bộ dữ liệu **PriceRunner Dataset** từ kho lưu trữ **UCI Machine Learning Repository** (`data/pricerunner_aggregate.csv`):
- **Quy mô ban đầu**: 35.311 bản ghi, 7 thuộc tính (`Product ID`, `Product Title`, `Merchant ID`, `Cluster ID`, `Cluster Label`, `Category ID`, `Category Label`).
- **Tiền xử lý & làm sạch**:
  - Phát hiện và loại bỏ 4 dòng mâu thuẫn nhãn nghiêm trọng (cùng tiêu đề nhưng khác nhãn danh mục).
  - Loại bỏ 4.316 bản ghi tiêu đề trùng lặp hoàn toàn, bảo toàn 30.991 tiêu đề sản phẩm duy nhất.
- **Chiến lược chia dữ liệu chống rò rỉ (Zero Data Leakage)**:
  - Bắt buộc chia theo nhóm sản phẩm tương đương (`Cluster ID`) bằng `GroupShuffleSplit` theo tỉ lệ 70% Train (21.543 mẫu), 15% Validation (4.766 mẫu), 15% Test (4.682 mẫu).
  - Giao tập `Cluster ID` giữa cả 3 tập dữ liệu bằng 0 tuyệt đối.
  - Phân định rõ các thuộc tính cấm sử dụng làm đặc trưng: `Product ID`, `Category ID`, `Cluster ID`, `Cluster Label`.

---

## 3. Kiến trúc Pipeline

Toàn bộ quy trình từ nạp dữ liệu, làm sạch, trích xuất đặc trưng, huấn luyện mô hình, đánh giá chuyên sâu đến đóng gói tài nguyên được tổ chức khép kín và tự động:

![Sơ đồ Pipeline](figures/pipeline_diagram.png)

### Không gian đặc trưng (Feature Engineering)
- **Nhánh văn bản (Text)**: Chuẩn hóa chữ thường, lọc ký tự đặc biệt, trích xuất TF-IDF unigram & bigram (1-2), áp dụng sublinear term frequency, giới hạn 5.000 từ khóa đặc trưng nhất.
- **Nhánh danh mục (Categorical)**: Mã hóa One-Hot cho `Merchant ID` gồm 288 chiều với cấu hình `handle_unknown="ignore"`.
- **Nhánh số học (Numeric)**: Trích xuất độ dài ký tự (`char_length`), số từ (`word_count`), số token chứa chữ số (`digit_token_count`) và chuẩn hóa bằng `StandardScaler`.
- **Tổng hợp**: Kết hợp bằng ma trận thưa CSR (`scipy.sparse.hstack`) đạt tổng số 5.291 chiều đặc trưng, fit duy nhất trên tập Train.

---

## 4. Kết quả thực nghiệm

Toàn bộ kết quả thực nghiệm được ghi nhận tự động theo chuẩn 17 cột tại `experiments/experiment_log.csv`.

### 4.1. Bảng so sánh các mô hình phân loại (Classification)

| STT | Mô hình | Kỹ thuật / Siêu tham số | Macro-F1 (Val) | Accuracy (Val) | Macro-F1 (Test) | Accuracy (Test) | Thời gian huấn luyện |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| EXP_01 | DummyClassifier | `strategy="most_frequent"` | 0.0263 | 0.1517 | - | - | 0.01 s |
| EXP_02 | LogisticRegression | `C=1.0, max_iter=1000` | 0.9431 | 0.9431 | - | - | 2.45 s |
| EXP_03 | MultinomialNB | `alpha=1.0` | 0.9055 | 0.9089 | - | - | 0.01 s |
| EXP_04 | LinearSVC (Gốc) | `C=1.0, max_iter=2000` | 0.9479 | 0.9475 | - | - | 1.36 s |
| EXP_05 | RandomForest | `n_estimators=100, max_depth=30` | 0.8992 | 0.8919 | - | - | 1.85 s |
| **EXP_06** | **LinearSVC (Tuned)** | **`C=0.5` (GroupKFold 3-folds)** | **0.9482** | **0.9478** | **0.9488** | **0.9475** | **1.22 s** |
| EXP_07 | LinearSVC (Đối chứng) | Random Split không theo nhóm | 0.9477 | 0.9470 | - | - | 0.88 s |

### 4.2. Hiệu năng theo từng danh mục sản phẩm (Tập Test)
Mô hình tối ưu **LinearSVC (C=0.5)** đạt kết quả xuất sắc trên hầu hết các lớp:
- **Nhóm công nghệ / thông số đặc thù**: `CPUs` (F1: 99.80%), `Mobile Phones` (F1: 99.72%), `Digital Cameras` (F1: 99.57%), `TVs` (F1: 98.44%), `Dishwashers` (F1: 97.58%), `Microwaves` (F1: 96.95%), `Washing Machines` (F1: 96.06%).
- **Nhóm thách thức ngữ nghĩa (Bộ ba điện lạnh)**: `Fridge Freezers` (F1: 89.28%), `Fridges` (F1: 85.93%), `Freezers` (F1: 85.49%). Nhầm lẫn chủ yếu phát sinh do các sản phẩm dùng chung nhiều từ khóa miêu tả ngoại quan và công nghệ bảo quản lạnh.

### 4.3. Kết quả gom nhóm không giám sát (Clustering)

| STT | Thuật toán | Giảm chiều | Số cụm k | Silhouette Score | ARI | NMI | Thời gian chạy |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP_08** | **KMeans** | **TruncatedSVD (50 dims) + L2** | **10** | **0.2133** | **0.4725** | **0.6617** | **1.54 s** |

- KMeans tự động phân tách rất thuần khiết các danh mục công nghệ (`CPUs`, `Mobile Phones`, `Digital Cameras`, `TVs`).
- Ba danh mục điện lạnh bị gom chung vào một siêu cụm lớn, minh chứng rõ ràng cho giá trị định hướng của nhãn giám sát trong bài toán phân loại sản phẩm.

---

## 5. Cấu trúc thư mục

```text
ML-PCC_Group/
├── data/
│   └── pricerunner_aggregate.csv          # Tập dữ liệu gốc UCI PriceRunner (35.311 dòng)
├── docs/
│   ├── CONSTRAINTS.md                     # Ràng buộc kỹ thuật và quy trình chuẩn
│   ├── PROGRESS.md                        # Nhật ký tiến độ và số liệu báo cáo
│   └── TASK.md                            # Danh sách 6 task thực hiện
├── experiments/
│   └── experiment_log.csv                 # Bảng ghi nhận 8 thực nghiệm (17 cột chuẩn)
├── figures/                               # 10 biểu đồ độ phân giải cao 300 DPI
│   ├── eda_label_distribution.png
│   ├── eda_title_length_distribution.png
│   ├── eda_top_keywords_per_category.png
│   ├── eda_cluster_size_distribution.png
│   ├── eda_merchant_distribution.png
│   ├── eda_duplicated_titles.png
│   ├── eval_models_comparison.png
│   ├── eval_confusion_matrix.png
│   ├── eval_leakage_comparison.png
│   ├── cluster_elbow_silhouette.png
│   ├── cluster_2d_projection.png
│   └── pipeline_diagram.png
├── models/                                # Các artifact mô hình và transformer
│   ├── best_model.joblib                  # Mô hình LinearSVC tối ưu (414 KB)
│   ├── tfidf_vectorizer.joblib            # TF-IDF Vectorizer 5.000 từ khóa (191 KB)
│   ├── merchant_encoder.joblib            # OneHotEncoder cho Merchant ID (3 KB)
│   ├── numeric_scaler.joblib              # StandardScaler cho 3 feature số (1 KB)
│   └── label_encoder.joblib               # LabelEncoder cho 10 danh mục (1 KB)
├── notebooks/
│   └── product_classification_clustering.ipynb # Notebook thực thi toàn bộ 9 Section
├── requirements.txt                       # Danh sách thư viện và phiên bản chuẩn
├── .env.example                           # Tệp cấu hình biến môi trường mẫu
└── README.md                              # Tài liệu hướng dẫn dự án
```

---

## 6. Hướng dẫn cài đặt và chạy lại (Reproducibility)

### 6.1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

Khởi tạo tệp cấu hình môi trường từ bản mẫu:
```bash
cp .env.example .env
```
*(Trên Windows PowerShell: `Copy-Item .env.example .env`)*


### 6.2. Chạy lại toàn bộ quy trình
Mở notebook bằng Jupyter Lab / Jupyter Notebook / VS Code:
```bash
jupyter notebook notebooks/product_classification_clustering.ipynb
```
Chọn **Kernel -> Restart and Run All**. Toàn bộ 59 cells sẽ tự động thực thi tuần tự từ đầu đến cuối trong khoảng 1.5 - 2 phút, nạp dữ liệu, sinh biểu đồ trong `figures/`, huấn luyện mô hình, ghi log thực nghiệm và xuất các file mô hình vào `models/`.

### 6.3. Tái sử dụng mô hình độc lập (Inference)
Các artifact đã đóng gói có thể được nạp lại trong bất kỳ ứng dụng Python độc lập nào mà không cần chạy lại notebook:
```python
import joblib
from scipy.sparse import hstack
import pandas as pd

# Nạp tài nguyên
model = joblib.load("models/best_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
merchant_enc = joblib.load("models/merchant_encoder.joblib")
scaler = joblib.load("models/numeric_scaler.joblib")
label_enc = joblib.load("models/label_encoder.joblib")

# Dữ liệu sản phẩm mới
title = "apple iphone 13 pro max 128gb graphite"
merchant_id = "1"

# Trích xuất đặc trưng
feat_tfidf = tfidf.transform([title])
feat_merchant = merchant_enc.transform([[merchant_id]])
feat_num = scaler.transform(pd.DataFrame([[len(title), len(title.split()), sum(c.isdigit() for c in title.split())]],
                                        columns=["char_length", "word_count", "digit_token_count"]))

# Dự đoán
X = hstack([feat_tfidf, feat_merchant, feat_num]).tocsr()
pred_code = model.predict(X)
predicted_category = label_enc.inverse_transform(pred_code)[0]
print("Danh mục dự đoán:", predicted_category)
# Kết quả: Mobile Phones
```
