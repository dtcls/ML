import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =========================
# 1. Cấu hình đường dẫn model
# =========================

MODEL_CANDIDATES = [
    Path("Main/svm_pipeline_model_1.jb")
]


# =========================
# 2. Tiền xử lý văn bản
# =========================

STOP_WORDS = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = str(text).lower()

    # Xóa link
    text = re.sub(r"http\S+|www\S+", "", text)

    # Xóa email
    text = re.sub(r"\S+@\S+", "", text)

    # Chỉ giữ chữ cái
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    # Xóa stopwords đơn giản
    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


# =========================
# 3. Load model
# =========================

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), path

    return None, None


svm_pipeline_model, loaded_path = load_model()


# =========================
# 4. Hàm lấy vectorizer và classifier cuối pipeline
# =========================

def get_vectorizer_and_classifier(pipeline_model):
    if not hasattr(pipeline_model, "named_steps"):
        return None, None

    tfidf = None

    for name, step in pipeline_model.named_steps.items():
        if hasattr(step, "get_feature_names_out") and hasattr(step, "transform"):
            tfidf = step
            break

    classifier = pipeline_model.steps[-1][1]

    return tfidf, classifier


# =========================
# 5. Hàm sigmoid
# =========================

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


# =========================
# 6. Tính fake_probability
# =========================

def get_fake_probability(pipeline_model, text):
    """
    Trả về xác suất/mức nghi ngờ tin giả.

    Quy ước trong bài:
    0 = Fake
    1 = Real

    Với Logistic Regression:
    - dùng predict_proba()

    Với LinearSVC:
    - không có predict_proba()
    - dùng decision_function()
    - đưa qua sigmoid để ước lượng điểm nghi ngờ
    """

    fake_label = 0

    # Trường hợp model có predict_proba, ví dụ LogisticRegression
    if hasattr(pipeline_model, "predict_proba"):
        probabilities = pipeline_model.predict_proba([text])[0]
        classes = list(pipeline_model.classes_)

        if fake_label in classes:
            fake_index = classes.index(fake_label)
            fake_probability = probabilities[fake_index]
            return float(fake_probability), "predict_proba"

    # Trường hợp LinearSVC
    if hasattr(pipeline_model, "decision_function"):
        decision_score = pipeline_model.decision_function([text])
        decision_score = float(np.ravel(decision_score)[0])

        _, classifier = get_vectorizer_and_classifier(pipeline_model)
        classes = list(classifier.classes_)

        # Với sklearn LinearSVC nhị phân:
        # decision_score > 0 nghiêng về classes[1]
        # decision_score < 0 nghiêng về classes[0]
        prob_class_1 = sigmoid(decision_score)

        if fake_label == classes[1]:
            fake_probability = prob_class_1
        else:
            fake_probability = 1 - prob_class_1

        return float(fake_probability), "decision_function"

    return None, None


# =========================
# 7. Tầng quyết định Fake / Real / Need Fact-checking
# =========================

def make_final_decision(fake_probability):
    """
    Tầng quyết định 3 nhãn:

    fake_probability >= 0.7  -> Fake News
    fake_probability <= 0.3  -> Real News
    còn lại                  -> Need Fact-checking
    """

    if fake_probability >= 0.7:
        return "Fake News"
    elif fake_probability <= 0.3:
        return "Real News"
    else:
        return "Need Fact-checking"


# =========================
# 8. Lấy các từ/cụm từ ảnh hưởng
# =========================

def get_important_words(pipeline_model, text, target_label, top_n=8):
    tfidf, classifier = get_vectorizer_and_classifier(pipeline_model)

    if tfidf is None or classifier is None:
        return []

    if not hasattr(classifier, "coef_"):
        return []

    # Vector hóa văn bản nhập vào
    X = tfidf.transform([text])

    # Lấy danh sách từ/cụm từ trong TF-IDF
    feature_names = tfidf.get_feature_names_out()

    # Với mô hình nhị phân, coef_[0]:
    # contribution dương -> nghiêng về classes[1], thường là Real = 1
    # contribution âm   -> nghiêng về classes[0], thường là Fake = 0
    coef = classifier.coef_[0]

    # Contribution = TF-IDF value * trọng số coef
    contributions = X.multiply(coef).toarray().ravel()

    # Chỉ xét những từ/cụm từ thật sự xuất hiện trong văn bản nhập vào
    nonzero_indices = X.nonzero()[1]

    classes = list(classifier.classes_)

    # Nếu cần tìm từ nghiêng về Real
    if target_label == classes[1]:
        candidates = [
            (feature_names[i], contributions[i])
            for i in nonzero_indices
            if contributions[i] > 0
        ]

    # Nếu cần tìm từ nghiêng về Fake
    else:
        candidates = [
            (feature_names[i], abs(contributions[i]))
            for i in nonzero_indices
            if contributions[i] < 0
        ]

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    return candidates[:top_n]


# =========================
# 9. Giao diện Streamlit
# =========================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.write("Nhập nội dung bài báo bên dưới để kiểm tra tin là Fake, Real hoặc cần kiểm chứng thêm.")

if svm_pipeline_model is None:
    st.error(
        "Không tìm thấy file mô hình. "
        "Bạn hãy chạy notebook và lưu model vào: svm_pipeline_model_1.jb"
    )
    st.stop()

new_input = st.text_area(
    "News Article:",
    height=220,
    placeholder="Nhập nội dung tin tức tại đây..."
)

if st.button("Check news"):
    if not new_input.strip():
        st.warning("Vui lòng nhập nội dung tin tức trước khi kiểm tra.")
        st.stop()

    # Tiền xử lý giống dữ liệu train
    cleaned_input = preprocess_text(new_input)

    if not cleaned_input.strip():
        st.warning("Sau khi tiền xử lý, văn bản không còn từ hợp lệ để phân tích.")
        st.stop()

    # Tính fake_probability
    fake_probability, probability_method = get_fake_probability(
        svm_pipeline_model,
        cleaned_input
    )

    if fake_probability is None:
        st.error("Không thể tính Fake News Risk Score cho model hiện tại.")
        st.stop()

    # Tầng quyết định 3 nhãn
    final_label = make_final_decision(fake_probability)

    # Lấy từ khóa nghiêng về Fake và Real
    fake_words = get_important_words(
        svm_pipeline_model,
        cleaned_input,
        target_label=0,
        top_n=8
    )

    real_words = get_important_words(
        svm_pipeline_model,
        cleaned_input,
        target_label=1,
        top_n=8
    )

    st.divider()

    # =========================
    # Hiển thị kết quả chính
    # =========================

    if final_label == "Fake News":
        st.error("Dự đoán: Fake News")
    elif final_label == "Real News":
        st.success("Dự đoán: Real News")
    else:
        st.warning("Dự đoán: Need Fact-checking")

    st.metric("Độ tin cậy", f"{fake_probability * 100:.1f}%")


    # =========================
    # Hiển thị từ/cụm từ giải thích
    # =========================

    if final_label == "Fake News":
        st.subheader("Các từ/cụm từ làm mô hình nghi ngờ tin giả:")

        if fake_words:
            for word, score in fake_words:
                st.write(f"- {word}")
        else:
            st.info("Không tìm thấy từ/cụm từ nổi bật nghiêng về Fake.")

    elif final_label == "Real News":
        st.subheader("Các từ/cụm từ làm mô hình nghiêng về tin thật:")

        if real_words:
            for word, score in real_words:
                st.write(f"- {word}")
        else:
            st.info("Không tìm thấy từ/cụm từ nổi bật nghiêng về Real.")

    else:

        st.write(
            "Mức nghi ngờ tin giả nằm trong khoảng chưa chắc chắn "
            "(từ 30% đến 70%) nên không kết luận là Fake hoặc Real"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Từ/cụm từ nghiêng về Fake:**")
            if fake_words:
                for word, score in fake_words[:5]:
                    st.write(f"- {word}")
            else:
                st.write("Không có từ nổi bật.")

        with col2:
            st.markdown("**Từ/cụm từ nghiêng về Real:**")
            if real_words:
                for word, score in real_words[:5]:
                    st.write(f"- {word}")
            else:
                st.write("Không có từ nổi bật.")

    with st.expander("Xem văn bản sau khi tiền xử lý"):
        st.write(cleaned_input)