# DiaLLM-BD

# DiaLLM-BD: LLM-Assisted Type-2 Diabetes Prediction System

DiaLLM-BD is a research-based prototype that combines machine learning and an LLM-style chatbot interface for Type-2 diabetes risk prediction and patient education. The system uses a tabular clinical diabetes dataset to train multiple machine learning models and selects the best-performing model for prediction. After prediction, the application generates a patient-friendly explanation and provides a diabetes education chatbot for general awareness.

This project is designed as a healthcare decision-support prototype, not as a replacement for professional medical diagnosis or treatment.

---

## Project Objective

The main objective of this project is to develop an AI-assisted diabetes prediction system that can:

- Predict Type-2 diabetes risk from patient clinical information.
- Train and compare multiple machine learning models.
- Handle missing values and class imbalance.
- Save the best trained model for deployment.
- Provide an interactive Streamlit web interface.
- Generate simple LLM-style explanations of model predictions.
- Provide a basic diabetes education chatbot.

---

## Key Features

- Type-2 diabetes prediction using machine learning.
- Automatic target column detection.
- Missing value handling using imputation.
- Medical zero-value correction for features such as glucose, BMI, insulin, and blood pressure.
- Train-validation-test split.
- SMOTE-based class imbalance handling.
- Multiple model training:
  - Logistic Regression
  - Support Vector Machine
  - Random Forest
  - Extra Trees
  - Gradient Boosting
- Best model selection based on validation performance.
- Threshold tuning for improved prediction.
- Streamlit-based web application.
- LLM-style explanation module.
- Diabetes education chatbot.
- Safe medical disclaimer included.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit
- Matplotlib
- OpenPyXL

---

## Project Structure

```text
diabetes_llm_project/
│
├── data/
│   └── diabetes_bangladesh.csv
│
├── models/
│   ├── diabetes_prediction_model.pkl
│   ├── best_threshold.pkl
│   ├── feature_columns.pkl
│   └── feature_metadata.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
