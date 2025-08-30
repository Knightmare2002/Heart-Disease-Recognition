# Heart Disease Predictor

> **Live app:** [https://heart-disease-recognition-3daedgxusiccpm6gvsifb4.streamlit.app/](https://heart-disease-recognition-3daedgxusiccpm6gvsifb4.streamlit.app/)

A Streamlit web app that predicts the presence of heart disease from 11 clinical features. It supports single‑patient prediction, bulk CSV prediction, and a simple model overview.

## Table of Contents

* [Live Demo](#live-demo)
* [Training Notebook](#training-notebook)
* [Features](#features)
* [Input Schema](#input-schema)
* [Models](#models)
* [Getting Started (Local)](#getting-started-local)
* [CSV Bulk Predict](#csv-bulk-predict)
* [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
* [Troubleshooting](#troubleshooting)

---

## Live Demo

Open the hosted app here:

* **[https://heart-disease-recognition-3daedgxusiccpm6gvsifb4.streamlit.app/](https://heart-disease-recognition-3daedgxusiccpm6gvsifb4.streamlit.app/)**

> The first load may take a few seconds due to cold‑start.

## Training Notebook

Model training and export are documented in **`main.ipynb`** (included in this repository). The notebook:

* loads/cleans the dataset and applies the same feature schema used by the app,
* trains the following models (often wrapped in `Pipeline` with preprocessing),
* evaluates them (accuracy/F1, etc.),
* saves the fitted artifacts with `joblib.dump(...)` as `.pkl` files consumed by the app.

## Features

* **Predict tab**: interactive form for one patient; on submit the app loads pre‑trained models and returns each model’s prediction.
* **Bulk Predict tab**: upload a CSV with the 11 features and get per‑row predictions appended as new columns; you can download the enriched CSV from the app.
* **Model information tab**: quick bar chart of example model accuracies.

## Input Schema

The app expects **11 features** with the following names and encodings:

| Feature          | Type / Encoding                                                                 |
| ---------------- | ------------------------------------------------------------------------------- |
| `Age`            | integer (years)                                                                 |
| `Sex`            | 0 = Male, 1 = Female                                                            |
| `ChestPainType`  | 0 = Typical Angina, 1 = Atypical Angina, 2 = Non‑Anginal Pain, 3 = Asymptomatic |
| `RestingBP`      | integer (mm Hg)                                                                 |
| `Cholesterol`    | integer (mg/dl)                                                                 |
| `FastingBS`      | 1 if > 120 mg/dl else 0                                                         |
| `RestingECG`     | 0 = Normal, 1 = ST, 2 = LVH                                                     |
| `MaxHR`          | integer (60–202)                                                                |
| `ExerciseAngina` | 1 = Yes, 0 = No                                                                 |
| `Oldpeak`        | float                                                                           |
| `ST_Slope`       | 0 = Upsloping, 1 = Flat, 2 = Downsloping                                        |

**CSV header example**

```
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
```

**Example rows (positive cases)**

```
62,0,3,158,264,1,1,112,1,2.6,2
57,0,2,145,263,0,2,109,1,1.8,1
```

## Models

The app loads these pre‑trained models from `.pkl` files placed alongside the app code:

* Decision Tree → `dt.pkl`
* Logistic Regression → `lr.pkl`
* Random Forest → `rf.pkl`
* Support Vector Machine → `svm.pkl`
* XGBoost → `xgb.pkl`

Each model predicts independently; the app shows the result per model.

> Tip: Save entire preprocessing + estimator as a single `Pipeline` before exporting, so inference is consistent.

## Getting Started (Local)

### Prerequisites

* **Python 3.11** (pinned for cloud deploys via `runtime.txt`)

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

### Example `requirements.txt`

```
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
imbalanced-learn==0.12.2
xgboost==2.0.3
matplotlib==3.8.4
joblib==1.3.2
openpyxl>=3.1.2
```

## CSV Bulk Predict

1. Prepare a CSV with the **exact 11 columns** listed in [Input Schema](#input-schema).
2. Go to **Bulk Predict** → upload your CSV.
3. The app encodes any textual values to numeric (if needed), runs all models, and appends a new column per model to the uploaded data.
4. Click **Download predictions CSV** (and/or Excel) to save the enriched file.

## Deployment (Streamlit Community Cloud)

1. Push your code, `requirements.txt`, `.pkl` models, and `runtime.txt` (with `python-3.11`) to a Git repository.
2. In Streamlit Community Cloud → **Create app** → choose **Repository**, **Branch** (can be *any*, not just `main`), and entry file (e.g., `app.py`).
3. Deploy. You can switch branches from the app settings later.

### Notes for cloud

* GPU is not available; for XGBoost use `tree_method="hist"`.
* Use relative paths for loading models.
* Store any credentials in Streamlit **Secrets**.

## Troubleshooting

* **Feature names mismatch**: Ensure the input DataFrame at inference has the **same column names and order** used during training (or use the model’s `feature_names_in_` to enforce order).
* **`could not convert string to float: 'M'`**: Your CSV has textual categories; either let the app encode them or save the model pipeline with its own encoder.
* **Build errors on deploy**: Pin Python via `runtime.txt` and use compatible versions as in the example requirements; avoid GPU‑only options in XGBoost on cloud.
