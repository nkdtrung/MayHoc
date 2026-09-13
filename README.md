# Machine Learning | Tuần 1

## Lab 01: Làm quen với PyTorch

Repository này lưu trữ các bài thực hành trong học phần **Machine Learning**. Nội dung của tuần đầu tiên tập trung vào việc làm quen với PyTorch, nền tảng tính toán tensor và cơ chế tự động tính gradient được sử dụng trong quá trình huấn luyện mô hình học máy.

## Mục tiêu học tập

Sau khi hoàn thành bài lab, người học có thể:

- Khởi tạo và đọc các thuộc tính cơ bản của một tensor.
- Thực hiện indexing, slicing, reshape và flatten dữ liệu.
- Áp dụng các phép toán phần tử, nhân ma trận và các hàm thống kê trên tensor.
- Sử dụng `autograd` để tính đạo hàm tự động.
- Hiểu quy trình tối ưu tham số bằng gradient descent.
- Áp dụng PyTorch để tìm hệ số của đa thức và nghiệm gần đúng của một hệ phương trình.

## Nội dung bài lab

Notebook [Tuan01/lab01.ipynb](Tuan01/lab01.ipynb) gồm các phần thực hành sau:

| Phần | Nội dung |
| --- | --- |
| 1 | Tạo tensor cơ bản và kiểm tra `shape`, số chiều, kiểu dữ liệu |
| 2 | Khởi tạo tensor bằng `rand`, `randn`, `zeros`, `ones`, `eye` và `full` |
| 3 | Đọc thông tin tensor qua `shape`, `size()`, `ndim` và `dtype` |
| 4 | Indexing, slicing và truy xuất các phần tử theo từng chiều |
| 5 | Reshape, flatten và kiểm tra số lượng phần tử bằng `numel()` |
| 6 | Phép toán trên tensor và nhân ma trận |
| 7 | Tính trung bình, độ lệch chuẩn và tổng tích lũy theo từng chiều |
| 8 | Tính đạo hàm tự động với `torch.autograd` |
| 9 | Học hệ số đa thức bằng bộ tối ưu `NAdam` và hàm mất mát MSE |
| 10 | Giải gần đúng hệ phương trình tuyến tính bằng autograd và tối ưu lặp |
| 11 | Bài tập tổng hợp về reshape, phép toán tensor, đạo hàm và gradient descent |

## Công nghệ sử dụng

- Python 3.9 trở lên
- PyTorch
- NumPy
- Jupyter Notebook hoặc Visual Studio Code với Jupyter extension

## Cài đặt môi trường

Khuyến nghị sử dụng virtual environment để quản lý các thư viện của bài học:

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Cài đặt các thư viện cần thiết:

```bash
python -m pip install --upgrade pip
pip install torch numpy jupyter
```

## Cách chạy

1. Mở thư mục repository trong Visual Studio Code.
2. Chọn kernel Python thuộc môi trường `.venv`.
3. Mở [Tuan01/lab01.ipynb](Tuan01/lab01.ipynb).
4. Chạy lần lượt các cell từ trên xuống dưới để bảo đảm biến và kết quả được khởi tạo đúng thứ tự.

Hoặc khởi động Jupyter Notebook bằng terminal:

```bash
jupyter notebook
```

## Kết quả mong đợi

Notebook minh họa được các kết quả nền tảng sau:

- Tensor có thể được biểu diễn dưới nhiều shape và kiểu dữ liệu khác nhau.
- Các phép biến đổi shape không làm thay đổi tổng số phần tử.
- Kết quả của các phép tính theo chiều (`dim`) phụ thuộc vào trục được lựa chọn.
- Giá trị gradient do autograd tính khớp với đạo hàm giải tích trong các ví dụ đa thức.
- Gradient descent có thể học gần đúng các hệ số của đa thức từ dữ liệu mẫu.
- Các nghiệm tìm được của hệ phương trình cần được kiểm tra lại bằng cách thay vào từng phương trình.

## Cấu trúc thư mục

```text
MayHoc/
├── README.md
└── Tuan01/
	└── lab01.ipynb
```

## Ghi chú

Đây là bài thực hành nền tảng của tuần 1. Người học nên đọc kết quả in ra ở từng cell, đối chiếu kích thước tensor trước và sau mỗi phép biến đổi, đồng thời so sánh gradient tự động với đạo hàm tính thủ công để củng cố kiến thức.