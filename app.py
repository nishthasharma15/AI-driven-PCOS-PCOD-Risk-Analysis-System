import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# Load Models
# ================================

clinical_model = joblib.load("pcos_rf_model.pkl")
basic_model = joblib.load("pcos_basic_model.pkl")

# ================================
# App Config
# ================================

st.set_page_config(page_title="PCOS Risk Analysis System", layout="centered")

st.title("🩺 PCOS / PCOD Risk Analysis System")
st.write(
    "This tool estimates **PCOS/PCOD risk** using machine learning. "
    "It supports **early screening (Basic mode)** and **clinical assessment (Clinical mode)**.\n\n"
    "**⚠️ Not a medical diagnosis.**"
)

st.divider()

# ================================
# Mode Selection
# ================================

mode = st.radio(
    "Choose Assessment Type",
    ["Basic (No Lab Tests)", "Clinical (With Lab Tests)"]
)

st.divider()

# ================================
# Basic Inputs (Always Visible)
# ================================

st.subheader("Patient Details")

age = st.number_input("Age (years)", 15, 50, 25)
weight = st.number_input("Weight (kg)", 30.0, 120.0, 55.0)
height = st.number_input("Height (cm)", 130.0, 190.0, 160.0)

bmi = round(weight / ((height / 100) ** 2), 2)

cycle_length = st.number_input("Menstrual Cycle Length (days)", 15, 60, 28)
cycle_irregular = st.selectbox("Cycle Regularity", ["Regular", "Irregular"])

weight_gain = st.selectbox("Recent Weight Gain", ["No", "Yes"])
hair_growth = st.selectbox("Excess Hair Growth", ["No", "Yes"])
pimples = st.selectbox("Frequent Pimples / Acne", ["No", "Yes"])
fast_food = st.selectbox("Fast Food Consumption", ["No", "Yes"])
exercise = st.selectbox("Regular Exercise", ["No", "Yes"])

# ================================
# Clinical Inputs (Only if Clinical Mode)
# ================================

if mode == "Clinical (With Lab Tests)":
    st.subheader("Clinical / Lab Parameters")

    follicle_l = st.number_input("Follicle Count (Left Ovary)", 0, 30, 6)
    follicle_r = st.number_input("Follicle Count (Right Ovary)", 0, 30, 6)

    amh = st.number_input("AMH Level (ng/mL)", 0.0, 20.0, 2.5)
    lh = st.number_input("LH Level (mIU/mL)", 0.0, 30.0, 5.0)
    fsh = st.number_input("FSH Level (mIU/mL)", 0.0, 30.0, 5.0)

    fsh_lh_ratio = fsh / lh if lh != 0 else 0

st.divider()

# ================================
# Prediction + Result + Recommendations
# ================================

if st.button("Analyze PCOS Risk"):

    # -------- BASIC MODE --------
    if mode == "Basic (No Lab Tests)":
        input_data = {
            "Age (yrs)": age,
            "BMI": bmi,
            "Cycle(R/I)": 1 if cycle_irregular == "Irregular" else 0,
            "Cycle length(days)": cycle_length,
            "Weight gain(Y/N)": 1 if weight_gain == "Yes" else 0,
            "hair growth(Y/N)": 1 if hair_growth == "Yes" else 0,
            "Pimples(Y/N)": 1 if pimples == "Yes" else 0,
            "Fast food (Y/N)": 1 if fast_food == "Yes" else 0,
            "Reg.Exercise(Y/N)": 1 if exercise == "Yes" else 0,
        }

        input_df = pd.DataFrame([input_data])
        probability = basic_model.predict_proba(input_df)[0][1]

    # -------- CLINICAL MODE --------
    else:
        input_data = {
            "Age (yrs)": age,
            "Weight (Kg)": weight,
            "Height(Cm)": height,
            "BMI": bmi,
            "Cycle(R/I)": 1 if cycle_irregular == "Irregular" else 0,
            "Cycle length(days)": cycle_length,
            "Weight gain(Y/N)": 1 if weight_gain == "Yes" else 0,
            "hair growth(Y/N)": 1 if hair_growth == "Yes" else 0,
            "Pimples(Y/N)": 1 if pimples == "Yes" else 0,
            "Fast food (Y/N)": 1 if fast_food == "Yes" else 0,
            "Reg.Exercise(Y/N)": 1 if exercise == "Yes" else 0,
            "Follicle No. (L)": follicle_l,
            "Follicle No. (R)": follicle_r,
            "AMH(ng/mL)": amh,
            "LH(mIU/mL)": lh,
            "FSH(mIU/mL)": fsh,
            "FSH/LH": fsh_lh_ratio,
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(
            columns=clinical_model.feature_names_in_,
            fill_value=0
        )

        probability = clinical_model.predict_proba(input_df)[0][1]

    # ================================
    # Risk Categorization
    # ================================

    if probability < 0.30:
        risk = "Low Risk"
        color = "green"
    elif probability < 0.60:
        risk = "Moderate Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # ================================
    # Result
    # ================================

    st.subheader("🧪 Result")
    st.markdown(f"**PCOS Probability:** `{probability:.2f}`")
    st.markdown(f"**Risk Level:** :{color}[{risk}]")

    # ================================
    # Recommendations (NOW SAFE)
    # ================================

    st.subheader("🩺 Recommendations")

    if risk == "Low Risk":
        st.success(
            "🟢 **Low PCOS Risk**\n\n"
            "• Maintain a healthy lifestyle\n"
            "• Monitor menstrual regularity\n"
            "• No immediate medical consultation required"
        )

    elif risk == "Moderate Risk":
        st.warning(
            "🟠 **Moderate PCOS Risk**\n\n"
            "• Lifestyle changes recommended\n"
            "• Consider consulting a gynecologist if symptoms persist"
        )

    else:
        st.error(
            "🔴 **High PCOS Risk**\n\n"
            "• Strongly recommended to consult a gynecologist\n"
            "• Early diagnosis helps prevent long-term complications"
        )

        st.info(
            "This result is intended for **screening and awareness only**. "
            "Please consult a healthcare professional for diagnosis."
        )