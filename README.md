# Cardiovascular Disease Prediction using Bagging Classifier

This project builds a machine learning model to predict the likelihood of cardiovascular disease using ensemble learning — specifically a **Bagging Classifier** with a **Decision Tree** as the base estimator.

---

## Dataset

The dataset used is `improved_cardiovascular_dataset.csv`, which contains anonymized patient data with various health indicators and a target variable `cardiovascular_disease` (0 or 1).

### Target:
- `cardiovascular_disease`: 
  - 0 = No disease
  - 1 = Presence of disease

---

## Data Preprocessing

- Drop `id` column if present
- Split into `features (X)` and `target (y)`
- Perform train-test split with stratification
- Scale features using **RobustScaler** (effective for outliers)

---

## Model Details

### Model Used:
- **Bagging Classifier** from Scikit-learn

### Parameters:
- `base_estimator`: DecisionTreeClassifier
- `n_estimators`: 100
- `max_samples`: 80% of training data
- `bootstrap`: True
- `n_jobs`: -1 (parallel processing)

### Evaluation Metrics:
- Accuracy
- Classification Report (Precision, Recall, F1)
- ROC AUC Score
- Confusion Matrix (heatmap)

---

## Evaluation Output

The model outputs:

- **Accuracy Score**
- **ROC AUC Score**
- **Classification Report**
- **Confusion Matrix Plot**

---

