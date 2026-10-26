# Machine Learning — MayHoc

Kho lưu trữ bài tập, bài thực hành và đồ án học phần **Machine Learning** của **Ngô Kiều Đình Trung**. Nội dung gồm làm quen với PyTorch, khám phá và tiền xử lý dữ liệu, hồi quy, lựa chọn đặc trưng, phân lớp và gom cụm.

Repository lưu cả notebook, dữ liệu, hình minh họa, kết quả thực nghiệm và tài liệu nộp bài. Các thư mục `Tuan01` và `Tuan02` được giữ theo cấu trúc ban đầu; những bài tiếp theo được tổ chức theo lab, challenge và giai đoạn đồ án.

## Nội dung chính

| Phần | Nội dung | Điểm bắt đầu |
| --- | --- | --- |
| Tuần 1 / Lab01 | Tensor, indexing, reshape, phép toán, autograd và tối ưu tham số với PyTorch. | [Notebook Lab01](Tuan01/lab01.ipynb) |
| Tuần 2 / Lab02 | Bài thực hành House Price. | [Notebook Lab02](Tuan02/Lab02.ipynb) |
| Lab03 | EDA House Price: giá trị thiếu, phân phối, tương quan và ngoại lệ; kèm báo cáo và slide. | [Notebook EDA](Lab03/Lab03_Kaggle.ipynb) |
| Lab04 | Tiền xử lý và lựa chọn đặc trưng; dữ liệu gốc và dữ liệu đã xử lý. | [Notebook feature selection](Lab04/sourcecode/featureSelection.ipynb) |
| Lab05 — Phân lớp | Decision Tree, Random Forest, SVM, Naive Bayes và so sánh mô hình qua 11 notebook. | [Các notebook phân lớp](Lab05/NopBai/Lab05_ThuatToanPhanLop/notebooks) |
| Lab05 — EDA | Thống kê mô tả, xử lý dữ liệu, trực quan hóa và báo cáo EDA tự động qua 13 notebook. | [Các notebook EDA](Lab05/NopBai/Lba05-PhanTichKhamPhaDulieu/notebooks) |
| Challenge1 | House Prices: pipeline hồi quy, đánh giá mô hình và tạo submission; kèm hồ sơ nộp bài. | [Pipeline House Prices](Challenge1/NopBai/MC_Code_Challenge_01/HousePrices_Pipeline.ipynb) |
| Challenge2 | Phân loại thể loại âm nhạc: EDA, huấn luyện, cross-validation và submission. | [Notebook modeling](Challenge2/NopBai/Code/Challenge2/notebooks/02_modeling.ipynb) |
| Giữa kỳ | Product Classification and Clustering: TF-IDF, phân loại sản phẩm và gom cụm; kèm mô hình đã lưu. | [Notebook đồ án](GiuaKy/NopBai/ML-PCC_Group-main-Code/notebooks/product_classification_clustering.ipynb) |
| Cuối kỳ | Hiện mới có file README trống để giữ cấu trúc thư mục tài nguyên. | [Thư mục tài nguyên](CuoiKy/NopBai/OtherResouce) |

### Bài tập và đề cương

- [BT01 — Data Understanding](BT01_DataUnderstanding_Ng%C3%B4%20Ki%E1%BB%81u%20%C4%90%C3%ACnh%20Trung_3124411327.docx)
- [BT03 — Regression](BT03_Regression_Ng%C3%B4%20Ki%E1%BB%81u%20%C4%90%C3%ACnh%20Trung_3124411327.docx)
- [BT4 — Rule, kNN và Bayes](BT4_Rule_kNN_Bayes_Ng%C3%B4%20Ki%E1%BB%81u%20%C4%90%C3%ACnh%20Trung_3124411327.docx)
- [Đề cương nhóm 24 — bản Word](Nhom24_PhanLoaiGomNhomSanPham_Proposal.docx)
- [Đề cương nhóm 24 — bản PDF](Nhom24_PhanLoaiGomNhomSanPham_Proposal.pdf)
- [Đề cương Product Classification and Clustering — bản PDF](Nhom24_Product%20Classification%20and%20Clustering_Proposal.pdf)

## Cấu trúc thư mục

```text
MayHoc/
├── Tuan01/                         # Notebook PyTorch của tuần 1
├── Tuan02/                         # Notebook House Price của tuần 2
├── Lab03/                          # Notebook EDA, figures, báo cáo và slide
├── Lab04/
│   ├── Data/                       # Raw_data và trained_data
│   └── sourcecode/                 # Tiền xử lý và feature selection
├── Lab05/
│   └── NopBai/
│       ├── Lab05_ThuatToanPhanLop/  # data, notebooks, figures
│       └── Lba05-PhanTichKhamPhaDulieu/
│                                   # data, notebooks, figures, reports
├── Challenge1/NopBai/              # Code, báo cáo, poster, slide, minh chứng
├── Challenge2/NopBai/              # Code, báo cáo, poster, minh chứng
├── GiuaKy/NopBai/
│   ├── ML-PCC_Group-main-Code/     # data, notebooks, models
│   ├── Experiment Log/            # Nhật ký thực nghiệm
│   ├── figures/                   # Hình EDA và đánh giá
│   └── Other/                     # README, requirements, .env.example
├── CuoiKy/NopBai/OtherResouce/     # Thư mục tài nguyên cuối kỳ
├── BT*.docx                        # Báo cáo bài tập
├── Nhom24_*.docx / *.pdf            # Các phiên bản đề cương
├── .gitignore
└── README.md
```

Tên thư mục được giữ theo bản nộp, bao gồm `Lba05-PhanTichKhamPhaDulieu` và `OtherResouce`.

## Cài đặt và chạy notebook

### 1. Lấy mã nguồn và tạo môi trường

Các lệnh dưới đây dùng PowerShell trên Windows:

```powershell
git clone https://github.com/nkdtrung/MayHoc.git
cd MayHoc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel
```

### 2. Cài thư viện theo từng bài

Các bài sử dụng những nhóm thư viện khác nhau; chọn môi trường và bộ thư viện phù hợp với notebook cần chạy.

**Lab01 — PyTorch:**

```powershell
python -m pip install torch numpy
```

**Challenge1 — House Prices:**

```powershell
python -m pip install -r "Challenge1/NopBai/MC_Code_Challenge_01/requirements.txt"
```

**Đồ án giữa kỳ — Product Classification and Clustering:**

```powershell
python -m pip install -r "GiuaKy/NopBai/Other/requirements.txt"
```

Các lab và Challenge2 còn dùng pandas, NumPy, Matplotlib, Seaborn, scikit-learn và, tùy bài, SciPy, XGBoost hoặc LightGBM. Một số notebook EDA dùng thêm YData Profiling, Sweetviz hoặc AutoViz; xem các cell import/cài đặt của từng notebook để chuẩn bị thư viện tương ứng.

### 3. Chọn đúng thư mục làm việc

Notebook sử dụng đường dẫn dữ liệu tương đối. Nên mở Jupyter từ thư mục chứa notebook và chạy các cell theo thứ tự từ trên xuống dưới. Ví dụ với Challenge1:

```powershell
cd "Challenge1/NopBai/MC_Code_Challenge_01"
jupyter notebook HousePrices_Pipeline.ipynb
```

Với Challenge2, mở từ `Challenge2/NopBai/Code/Challenge2/notebooks` để các đường dẫn `../data` và `../outputs` trỏ đúng vị trí. Với đồ án giữa kỳ, có thể mở từ `GiuaKy/NopBai/ML-PCC_Group-main-Code/notebooks`; notebook tự xác định thư mục gốc của phần code.

Trong VS Code, chọn kernel thuộc môi trường đã cài thư viện và kiểm tra thư mục làm việc trước khi chạy. Chạy lại notebook có thể tạo hoặc cập nhật hình, mô hình, báo cáo và submission.

## Dữ liệu và kết quả

- **Lab03/Lab04:** dữ liệu và kết quả phân tích bài toán House Price; Lab04 lưu cả dữ liệu gốc và dữ liệu đã xử lý.
- **Lab05:** dữ liệu cho các bài phân lớp và EDA; kết quả nằm trong `figures/`, các báo cáo EDA tương tác nằm trong `reports/`. Tải báo cáo HTML về máy và mở bằng trình duyệt để xem đầy đủ.
- **Challenge1:** dữ liệu trong `Data house prices/`, biểu đồ trong `figures/`, dự đoán trong `submissions/` thuộc thư mục pipeline.
- **Challenge2:** dữ liệu trong `Code/Challenge2/data/`; các kết quả cross-validation, biểu đồ và submission nằm trong `Code/Challenge2/outputs/`.
- **Giữa kỳ:** dữ liệu và mô hình ở `ML-PCC_Group-main-Code/`; hình, nhật ký thực nghiệm và hồ sơ nộp bài được lưu ở các thư mục cùng cấp trong `NopBai/`. Notebook có thể tạo kết quả chạy mới trong thư mục code.

Dataset `covid-data.csv` có kích thước khoảng **98 MB** (93,83 MiB), nên việc clone repository có thể mất thêm thời gian. Kết quả đã lưu trong notebook và báo cáo phản ánh các lần chạy trước; chạy lại có thể cho kết quả khác tùy môi trường và cấu hình.

## Quy tắc lưu trữ

- Ưu tiên các file đã giải nén để đọc và theo dõi thay đổi trực tiếp trên GitHub.
- Loại `Trangbia.docx`, ZIP Lab03/Lab04 và sáu ZIP trùng hoàn toàn với nội dung đã giải nén.
- Giữ ZIP Lab01, Lab02 và ZIP nguồn Challenge1 vì có nội dung khác các file đang được theo dõi.
- Không theo dõi môi trường ảo, cache Python, checkpoint Jupyter và file `.env` cục bộ; `.env.example` được giữ làm cấu hình mẫu.

Chi tiết các đường dẫn bị loại nằm trong [.gitignore](.gitignore).
