# Titanic Passengers Survival Predictor (ML)

This project builds a **Machine Learning classification model** to predict whether a passenger **survived** the Titanic disaster. It includes a full ML workflow (preprocessing → pipeline → training → cross-validation → hyperparameter tuning → evaluation) and a **Gradio web app** for interactive predictions.

---

## Project Overview

- **Problem Type:** Binary Classification  
- **Target:** `survived` (0 = Not Survived, 1 = Survived)  
- **Model Output:** Prediction label + survival probability  
- **Deployment:** Gradio app (Hugging Face Spaces compatible)

---

## Features Used

User inputs (from Gradio):
- `pclass` (Passenger class)
- `sex`
- `age`
- `sibsp` (siblings/spouses)
- `parch` (parents/children)
- `fare`
- `embarked`

Engineered features:
- `family_size = sibsp + parch + 1`
- `alone` (1 if no family onboard else 0)

---

## Preprocessing Steps (5+)

This project performs and documents multiple preprocessing steps:
1. **Column selection / removal** (drop unnecessary or redundant columns)
2. **Missing value handling**
   - Numeric: Median imputation
   - Categorical: Most frequent imputation
3. **Outlier handling**
   - Clipping numeric features between **1st and 99th percentiles**
4. **Categorical encoding**
   - OneHotEncoding (`handle_unknown="ignore"`)
5. **Scaling**
   - StandardScaler for numeric features
6. **Feature engineering**
   - `family_size`, `alone`

---

## ML Pipeline

A single **scikit-learn Pipeline** is used to combine:
- `ColumnTransformer` preprocessing (numeric + categorical)
- Classifier (Random Forest)

This ensures consistent preprocessing during training and prediction.

---

## Model Selection (Primary Model)

**RandomForestClassifier** was selected because:
- Works well on **tabular datasets**
- Captures **non-linear relationships**
- Handles mixed feature types effectively after encoding
- Strong baseline model with good robustness

---

## Training, Validation & Tuning

- **Train/Test Split** with stratification
- **Cross-Validation:** Stratified K-Fold (reports mean ± std score)
- **Hyperparameter Tuning:** GridSearchCV to test multiple parameter combinations
- **Best Model:** Selected from GridSearchCV results and saved as `model.joblib`

---

## Evaluation Metrics

The final model is evaluated on the test set using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

## Gradio Web App

A user-friendly Gradio interface allows public users to enter passenger details and get:
- Predicted survival result
- Survival probability

---

## Project Structure

```bash
.
├── app.py                # Gradio app (loads model.joblib and predicts)
├── model.joblib          # Trained sklearn pipeline/model
├── requirements.txt      # Dependencies for local run + HF deployment
├── notebook.ipynb        # Training + CV + tuning + evaluation (Colab)
└── README.md             # This file
