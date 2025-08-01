# ❤️ Heart Disease Recognition using Machine Learning
## 📌 Project Overview
This project focuses on predicting the presence of heart disease using multiple machine learning models. The pipeline integrates data preprocessing, feature engineering, and model evaluation on datasets from Cleveland, Hungarian, Switzerland, and VA heart disease studies.

The final solution combines traditional ML models and neural networks, with an ensemble approach to maximize prediction performance.

## 📂 Dataset
The project utilizes the Heart Disease datasets from multiple sources:

- Cleveland

- Hungarian

- Switzerland

- VA

Key details:

- Total samples (train): 720

- Total samples (test): 200

- Number of features: 23 (including missingness indicators)

- Target: Binary classification → 0 (no heart disease) vs 1 (presence of heart disease)

Preprocessing:

- Missing Value Imputation using Iterative Imputer.

- Missingness Indicators created for imputed columns.

- SMOTE applied to balance classes.

- Feature scaling with StandardScaler.

## 🔄 Data Preprocessing Pipeline

- Data Cleaning: Replaced missing values ? with NaN.

- Iterative Imputation: Estimated missing values using other features.

- Class Balancing: Applied SMOTE for oversampling the minority class.

- Feature Engineering: Added missingness flags.

- Standardization: Scaled features before feeding them into models.

## 🤖 Models Implemented
The project evaluates several models:

1. Support Vector Machine (SVM)

    - Baseline

    - With SMOTE

    - With GridSearchCV tuning

2. Logistic Regression

    - Baseline with class balancing

3. XGBoost

    - Baseline

    - With SMOTE

    - With RandomizedSearchCV and GPU acceleration

4. Neural Networks (MLP)

    - Baseline MLP

    - Tuned with KerasTuner Hyperband

5. Ensemble Soft Voting

    - Combination of Logistic Regression, XGBoost, and Neural Network

    - Optimal weights tuned via grid search