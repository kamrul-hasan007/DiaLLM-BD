import joblib
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "diabetes_prediction_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
FEATURE_METADATA_PATH = MODEL_DIR / "feature_metadata.pkl"


st.set_page_config(
    page_title="DiaLLM-BD",
    page_icon="🩺",
    layout="wide"
)


@st.cache_resource
def load_prediction_model():
    missing_files = []

    if not MODEL_PATH.exists():
        missing_files.append("models/diabetes_prediction_model.pkl")

    if not THRESHOLD_PATH.exists():
        missing_files.append("models/best_threshold.pkl")

    if not FEATURE_COLUMNS_PATH.exists():
        missing_files.append("models/feature_columns.pkl")

    if not FEATURE_METADATA_PATH.exists():
        missing_files.append("models/feature_metadata.pkl")

    if missing_files:
        st.error("Model files are missing. Train the model first.")
        st.write("Open VS Code terminal and run:")
        st.code("python train_model.py", language="bash")
        st.write("Missing files:")
        for file in missing_files:
            st.write(f"- {file}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    feature_metadata = joblib.load(FEATURE_METADATA_PATH)

    return model, threshold, feature_columns, feature_metadata


def predict_probability(model, input_df):
    return model.predict_proba(input_df)[:, 1][0]


def generate_llm_style_explanation(prediction, probability, threshold, patient_data):
    if prediction == 1:
        risk_status = "higher diabetes risk"
    else:
        risk_status = "lower diabetes risk"

    details = []
    for key, value in patient_data.items():
        details.append(f"{key}: {value}")

    patient_summary = ", ".join(details)

    explanation = (
        f"The model predicts {risk_status}. "
        f"The predicted probability is {probability:.2f}. "
        f"The decision threshold is {threshold:.2f}. "
        f"This result was generated from the entered patient information: {patient_summary}. "
        f"This is an AI-based decision-support output only and should not be used as a final medical diagnosis."
    )

    return explanation


def diabetes_chatbot_answer(question, prediction_text, probability):
    q = question.lower()

    intro = (
        f"Current model result: {prediction_text} "
        f"with probability {probability:.2f}. "
    )

    if "food" in q or "eat" in q or "diet" in q or "meal" in q:
        return (
            intro
            + "For general diabetes awareness, patients are usually advised to reduce sugary drinks, control refined carbohydrates, "
            + "eat more vegetables and fiber-rich foods, and follow a balanced diet. "
            + "A doctor or registered dietitian should provide a personal diet plan."
        )

    if "exercise" in q or "walk" in q or "physical" in q:
        return (
            intro
            + "Regular physical activity may help with blood sugar control. "
            + "Walking, light cardio, and supervised exercise can be useful depending on the patient's condition. "
            + "Medical advice is recommended before starting a new routine."
        )

    if "symptom" in q or "sign" in q:
        return (
            intro
            + "Common diabetes-related symptoms may include unusual thirst, frequent urination, tiredness, blurred vision, "
            + "and slow wound healing. A qualified doctor should confirm the condition using clinical tests."
        )

    if "medicine" in q or "tablet" in q or "insulin" in q or "dose" in q:
        return (
            intro
            + "Medication or insulin decisions must be made only by a qualified doctor. "
            + "This system cannot prescribe, stop, or change medicine."
        )

    if "test" in q or "diagnosis" in q or "check" in q:
        return (
            intro
            + "Diabetes is commonly confirmed using medical tests such as fasting blood glucose, HbA1c, "
            + "or oral glucose tolerance test. This AI system should only support awareness, not replace diagnosis."
        )

    return (
        intro
        + "Diabetes risk depends on multiple clinical factors such as glucose level, BMI, blood pressure, age, insulin-related values, "
        + "and family history. Please consult a qualified healthcare professional for proper medical guidance."
    )


model, threshold, feature_columns, feature_metadata = load_prediction_model()


st.title("🩺 DiaLLM-BD")
st.subheader("LLM-Assisted Type-2 Diabetes Prediction System")

st.warning(
    "This is a research prototype only. It does not provide final medical diagnosis or treatment."
)

tab1, tab2, tab3 = st.tabs(
    ["Patient Input", "Prediction Explanation", "Diabetes Chatbot"]
)


with tab1:
    st.header("Enter Patient Information")

    patient_input = {}

    numeric_features = feature_metadata.get("numeric_features", [])
    categorical_features = feature_metadata.get("categorical_features", [])

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        for index, feature in enumerate(feature_columns):
            active_col = col1 if index % 2 == 0 else col2

            with active_col:
                if feature in numeric_features:
                    default_value = feature_metadata["numeric_defaults"].get(feature, 0.0)

                    patient_input[feature] = st.number_input(
                        label=feature,
                        value=float(default_value),
                        step=0.1
                    )

                elif feature in categorical_features:
                    options = feature_metadata["categorical_options"].get(feature, ["Unknown"])
                    default_value = feature_metadata["categorical_defaults"].get(feature, options[0])

                    if default_value in options:
                        default_index = options.index(default_value)
                    else:
                        default_index = 0

                    patient_input[feature] = st.selectbox(
                        label=feature,
                        options=options,
                        index=default_index
                    )

                else:
                    patient_input[feature] = st.text_input(
                        label=feature,
                        value=""
                    )

        submitted = st.form_submit_button("Predict Diabetes Risk")

    if submitted:
        input_df = pd.DataFrame([patient_input])

        probability = predict_probability(model, input_df)
        prediction = int(probability >= threshold)

        st.session_state["patient_input"] = patient_input
        st.session_state["probability"] = probability
        st.session_state["prediction"] = prediction

        if prediction == 1:
            st.error(f"Higher Diabetes Risk Detected | Probability: {probability:.2f}")
        else:
            st.success(f"Lower Diabetes Risk Detected | Probability: {probability:.2f}")


with tab2:
    st.header("Prediction Explanation")

    if "prediction" not in st.session_state:
        st.info("Please enter patient information and click Predict Diabetes Risk first.")
    else:
        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]
        patient_input = st.session_state["patient_input"]

        if prediction == 1:
            st.error("Prediction: Higher Diabetes Risk")
            prediction_text = "Higher diabetes risk"
        else:
            st.success("Prediction: Lower Diabetes Risk")
            prediction_text = "Lower diabetes risk"

        st.metric("Predicted Diabetes Probability", f"{probability:.2f}")
        st.metric("Decision Threshold", f"{threshold:.2f}")

        explanation = generate_llm_style_explanation(
            prediction=prediction,
            probability=probability,
            threshold=threshold,
            patient_data=patient_input
        )

        st.info(explanation)

        st.write(
            "Clinical note: A healthcare professional should confirm diabetes using proper medical tests such as fasting blood glucose, HbA1c, or oral glucose tolerance test."
        )


with tab3:
    st.header("Diabetes Education Chatbot")

    question = st.text_input(
        "Ask a diabetes-related question:",
        placeholder="Example: What food should a diabetic patient avoid?"
    )

    if st.button("Ask Chatbot"):
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            if "prediction" in st.session_state:
                probability = st.session_state["probability"]
                prediction = st.session_state["prediction"]
                prediction_text = "Higher diabetes risk" if prediction == 1 else "Lower diabetes risk"
            else:
                probability = 0.0
                prediction_text = "No prediction has been made yet"

            answer = diabetes_chatbot_answer(
                question=question,
                prediction_text=prediction_text,
                probability=probability
            )

            st.success(answer)


st.markdown("---")
st.caption("DiaLLM-BD: Machine Learning + LLM-style patient education prototype.")