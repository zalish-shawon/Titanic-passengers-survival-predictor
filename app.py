import numpy as np
import pandas as pd
import joblib
import gradio as gr

# MUST exist before loading model
def clip_outliers(X):
    """
    Works whether X is a pandas DataFrame or a numpy ndarray.
    Clips each feature between 1st and 99th percentile.
    """
    X_arr = np.asarray(X, dtype=float)
    X_out = X_arr.copy()

    if X_out.ndim == 2:
        for j in range(X_out.shape[1]):
            col = X_out[:, j]
            lo, hi = np.nanpercentile(col, [1, 99])
            X_out[:, j] = np.clip(col, lo, hi)
    else:
        lo, hi = np.nanpercentile(X_out, [1, 99])
        X_out = np.clip(X_out, lo, hi)

    return X_out


loaded = joblib.load("model.joblib")
model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded


def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    pclass = int(pclass)
    sibsp = int(sibsp)
    parch = int(parch)

    age = float(age) if age is not None else np.nan
    fare = float(fare) if fare is not None else np.nan

    family_size = sibsp + parch + 1
    alone_val = 1 if (sibsp + parch == 0) else 0

    x = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked,
        "family_size": family_size,
        "alone": alone_val,     # your pipeline expects this
        "is_alone": alone_val,  # extra (ok if unused)
    }])

    # If pipeline expects specific columns, ensure they're present
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in x.columns:
                x[col] = np.nan
        x = x[list(model.feature_names_in_)]

    proba = float(model.predict_proba(x)[0][1])
    pred = "Survived ✅" if proba >= 0.5 else "Not Survived ❌"
    return pred, proba


demo = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], value=3, label="Passenger Class (pclass)"),
        gr.Dropdown(["male", "female"], value="male", label="Sex"),
        gr.Number(value=25, label="Age"),
        gr.Number(value=0, label="Siblings/Spouses aboard (sibsp)"),
        gr.Number(value=0, label="Parents/Children aboard (parch)"),
        gr.Number(value=30, label="Fare"),
        gr.Dropdown(["S", "C", "Q"], value="S", label="Embarked"),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Survival Probability"),
    ],
    title="Titanic Passenger Survival Predictor",
    description="Predict survival using a trained sklearn Pipeline and Gradio."
)

if __name__ == "__main__":
    demo.launch()
