# 📰 Fake News Detection

Đồ án cuối kỳ môn **Học Máy (2526II\_INT3405#\_3)** — Trường Đại học Công nghệ, Viện Trí tuệ Nhân tạo.

Hệ thống phát hiện tin giả tự động sử dụng TF-IDF kết hợp với các thuật toán phân lớp: Linear SVM (tự xây dựng & sklearn), Logistic Regression, Multinomial Naive Bayes.

---

## 👥 Thành viên nhóm

| Họ tên | MSSV | Vai trò |
|---|---|---|
| Lê Văn Sang | 2402xxxx | Phân tích đề bài & Dữ liệu |
| Nguyễn Sỹ Trường Sơn | 2402xxxx | Xây dựng mô hình SVM |
| Nguyễn Sỹ Quyền | 2402xxxx | So sánh & Phân tích mô hình |
| Phạm Quang Minh | 2402xxxx | Viết báo cáo |

---

## 📁 Cấu trúc project

```
.
├── FakeNewDetection.ipynb       # Notebook huấn luyện và đánh giá mô hình
├── app.py                       # Ứng dụng web Streamlit
├── svm_pipeline_model_1.jb      # Model đã huấn luyện (TF-IDF + LinearSVC pipeline)
├── True.csv                     # Dữ liệu tin thật (cần tải về, xem bên dưới)
├── Fake.csv                     # Dữ liệu tin giả (cần tải về, xem bên dưới)
└── README.md
```

---

## 📦 Cài đặt

### Yêu cầu
- Python 3.8+

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install streamlit scikit-learn joblib numpy pandas nltk
```

Sau đó tải dữ liệu NLTK cần thiết (chạy một lần):

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

---

## 📊 Dữ liệu

Bộ dữ liệu gồm hai file CSV (không đính kèm trong repo do dung lượng lớn):

- **`True.csv`** — ~21,000 bài báo thật từ Reuters, AP, AFP
- **`Fake.csv`** — ~23,000 bài báo giả được gán nhãn bởi chuyên gia

Tải về tại: [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Đặt cả hai file vào thư mục gốc của project trước khi chạy notebook.

---

## 🚀 Hướng dẫn sử dụng

### 1. Huấn luyện mô hình

Mở và chạy toàn bộ notebook:

```bash
jupyter notebook FakeNewDetection.ipynb
```

Notebook sẽ tự động lưu model tốt nhất ra file `svm_pipeline_model_1.jb`.

### 2. Chạy ứng dụng web

```bash
streamlit run app.py
```

Truy cập `http://localhost:8501`, dán nội dung bài báo vào ô nhập và nhấn **Check news**.

> **Lưu ý:** File `svm_pipeline_model_1.jb` phải nằm cùng thư mục với `app.py`.

---

## 🤖 Các mô hình

| Mô hình | Macro F1 | Ghi chú |
|---|---|---|
| LinearSVC (sklearn) | **0.9970** | Tốt nhất, được dùng trong app |
| Logistic Regression | 0.9920 | |
| Multinomial Naive Bayes | 0.9626 | |
| Linear SVM (from scratch) | 0.9492 | Mini-batch Gradient Descent |

---

## 🖥️ Tính năng ứng dụng

- Dự đoán 3 nhãn: **Fake News** / **Real News** / **Need Fact-checking**
- Hiển thị điểm độ tin cậy (Fake Probability Score)
- Giải thích kết quả qua các từ/cụm từ ảnh hưởng đến quyết định
- Xem văn bản sau tiền xử lý

---

## ⚙️ Pipeline xử lý

```
Văn bản thô
    → Tiền xử lý (lowercase, loại URL/email, lemmatization, stop words)
    → TF-IDF Vectorizer (max_features=20000, ngram=(1,2))
    → LinearSVC (C=1.0)
    → Sigmoid → Fake Probability Score
    → Tầng quyết định 3 nhãn
```

---

## 📄 Báo cáo

Xem file `FakeNewsDetection_Report.pdf` để biết chi tiết lý thuyết, thực nghiệm và phân tích kết quả.

---

## 📚 Tài liệu tham khảo chính

- Joachims, T. (1998). *Text categorization with SVMs*. ECML.
- Shu et al. (2017). *Fake News Detection on Social Media*. ACM SIGKDD.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
