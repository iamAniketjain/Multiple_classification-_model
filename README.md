# Machine Learning Assignment 2 — Breast Cancer Classification

> **GitHub:** [Multiple Classification Model](https://github.com/iamAniketjain/Multiple_classification-_model)  
> **Live Streamlit App:** [ismassingment2.streamlit.app](https://ismassingment2.streamlit.app/)

## a. Problem statement
Build and evaluate five classification algorithms on one common dataset, then provide an interactive Streamlit application that evaluates a saved model on a supplied test CSV. The application does not retrain on uploaded test data.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), provided by `sklearn.datasets.load_breast_cancer`.
- **Task:** Binary classification of breast tumours.
- **Instances:** 569.
- **Input features:** 30 numeric diagnostic features.
- **Target:** `target` (0 = Malignant, 1 = Benign).
- **Split:** 80% training / 20% testing, stratified, `random_state=42`.
- **Test file:** `test_data.csv` contains the held-out test features and target.

The dataset meets the assignment minimum of 500 instances and 12 features.

## c. GitHub Repository Link
[Multiple Classification Model — GitHub Repository](https://github.com/iamAniketjain/Multiple_classification-_model)

## d. Models used and evaluation

The five required classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. StandardScaler is included in pipelines for Logistic Regression, KNN, and SVM. AUC is calculated using the probability assigned to class 1 (Benign).

### Model comparison

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree          |     0.9211 | 0.9163 |      0.9565 |   0.9167 | 0.9362 | 0.8341 |
| KNN                    |     0.9561 | 0.9788 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |
| Naive Bayes            |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9950 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline. Standardization helps the optimizer handle features on different scales; its performance is competitive when the classes are well separated. |
| Decision Tree | Captures nonlinear decision boundaries and is easy to interpret. A depth limit of 5 controls complexity, although a single tree can be less stable than ensemble methods. |
| KNN | Classifies observations using nearby training samples. Standardization is important because distance-based methods are sensitive to feature scales; performance depends on the selected neighbour count. |
| Naive Bayes | Uses the Gaussian feature-independence assumption. It is fast and simple, but correlated diagnostic measurements can limit its performance. |
| Support Vector Machine | The RBF kernel can model nonlinear boundaries. Standardization is essential, and probability estimates are enabled for AUC calculation. |
| **Overall winner for this dataset** | **Logistic Regression** ranks highest by F1, with AUC and MCC used as tie-breakers in this comparison. |

*The values above are calculated from the fixed 80/20 stratified split and the model settings in `train_models.py`. Results can vary if the split or hyperparameters are changed.*

## Streamlit application
The application provides:
1. CSV upload for the held-out test data.
2. Dropdown selection of one of the five saved classifiers.
3. Accuracy, AUC, precision, recall, F1, and MCC.
4. Confusion matrix and classification report.
5. Prediction preview.

The uploaded CSV must contain all 30 feature columns and a `target` column. The app loads a saved model and evaluates it; it does not fit or retrain a model on the uploaded file.

## Repository structure

```text
ML_Assignment_2_Solution/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Re-train the models

```bash
python train_models.py
```

This recreates the five saved model files, `test_data.csv`, and `model_metrics.csv`.

## Deployment on Streamlit Community Cloud
1. Push this project folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **New app**, choose the repository and `main` branch.
4. Set the main file path to `app.py`, then deploy.
5. Live deployed application: [https://ismassingment2.streamlit.app/](https://ismassingment2.streamlit.app/).

## Project links
- **GitHub repository:** https://github.com/iamAniketjain/Multiple_classification-_model
- **Live Streamlit application:** https://ismassingment2.streamlit.app/

## Final submission checklist
- [x] GitHub repository created and project source code uploaded.
- [x] Streamlit application deployed.

## Academic integrity and reproducibility
Use this as a learning and implementation starting point. Run the code yourself in BITS Virtual Lab, inspect the outputs, customize the interface and analysis in your own words, maintain your own Git commit history, and verify the deployment before submitting. Do not submit links or screenshots that you have not personally created and checked.
