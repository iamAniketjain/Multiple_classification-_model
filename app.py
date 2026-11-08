from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🧬", layout="wide")
st.title("Breast Cancer Classification — Model Comparison")
st.write(
    "Upload the held-out test CSV (including the `target` column), select a trained "
    "classifier, and evaluate it. This app loads saved models; it does not retrain them."
)
st.info("Target encoding: 0 = Malignant, 1 = Benign. The positive class for metrics and AUC is 1 (Benign).")

uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
model_name = st.selectbox("Choose a classification model", list(MODEL_FILES))
sample_file = st.checkbox("Use the bundled test_data.csv instead", value=False)

if uploaded is not None or sample_file:
    try:
        df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(ROOT / "test_data.csv")
        if "target" not in df.columns:
            st.error("The uploaded CSV must include the target column named `target`.")
            st.stop()
        model_path = MODEL_DIR / MODEL_FILES[model_name]
        if not model_path.exists():
            st.error(f"Model file not found: {model_path.name}")
            st.stop()
        model = joblib.load(model_path)
        y_true = df["target"].astype(int)
        X = df.drop(columns=["target"])
        expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if expected and set(X.columns) != set(expected):
            missing = sorted(set(expected) - set(X.columns))
            extra = sorted(set(X.columns) - set(expected))
            st.error(f"Feature columns do not match. Missing: {missing}; unexpected: {extra}")
            st.stop()
        if expected:
            X = X[expected]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        scores = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        st.subheader(f"Evaluation results: {model_name}")
        cols = st.columns(3)
        for col, (label, value) in zip(cols * 2, scores.items()):
            col.metric(label, f"{value:.4f}")
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks([0, 1], ["Malignant (0)", "Benign (1)"])
        ax.set_yticks([0, 1], ["Malignant (0)", "Benign (1)"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Confusion Matrix")
        for (i, j), value in __import__("numpy").ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        st.pyplot(fig)
        st.subheader("Classification report")
        report = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Malignant", "Benign"], zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
        st.subheader("Predictions preview")
        preview = X.head(10).copy()
        preview["Actual target"] = y_true.head(10).values
        preview["Predicted target"] = y_pred[:10]
        preview["Predicted class"] = ["Benign" if x == 1 else "Malignant" for x in y_pred[:10]]
        st.dataframe(preview, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not evaluate this file: {exc}")
else:
    st.caption("Upload test_data.csv or select the bundled test data option to see evaluation results.")
