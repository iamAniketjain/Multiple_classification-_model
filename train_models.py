"""Train five classifiers and save model artifacts plus test_data.csv."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "Decision Tree": __import__("sklearn").tree.DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, random_state=42))]),
}
files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / files[name])
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, probability),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted),
    })
X_test.assign(target=y_test).to_csv(ROOT / "test_data.csv", index=False)
pd.DataFrame(results).to_csv(ROOT / "model_metrics.csv", index=False)
print(pd.DataFrame(results).round(4).to_string(index=False))
print("Saved five models and test_data.csv.")
