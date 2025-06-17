import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# === Load trained model ===
model = joblib.load("models/cancer_model.pkl")

# === Feature Names ===
feature_names = [
    "Age", "Gender", "BMI", "Smoking", "GeneticRisk",
    "PhysicalActivity", "AlcoholIntake", "CancerHistory"
]

# === Background Image Function ===
def set_bg_image(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_base64}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Apply background
set_bg_image("images/background_image.jpg")

# === PDF Report Generator ===
def generate_pdf(data: dict, diagnosis: str, confidence: float):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "Cancer Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 90, f"Prediction Result: {diagnosis}")
    c.drawString(40, height - 110, f"Confidence: {confidence:.2f}%")

    y = height - 150
    for k, v in data.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 20

    # Suggestions
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Health Recommendation:")
    y -= 20
    c.setFont("Helvetica", 11)

    if diagnosis == "Cancer":
        suggestions = [
            "→ Please consult a specialist for further screening.",
            "→ Early diagnosis can lead to better outcomes.",
            "→ Maintain a healthy lifestyle and follow up regularly."
        ]
    else:
        suggestions = [
            "→ No signs detected, keep up with regular checkups.",
            "→ Continue a healthy lifestyle (exercise, avoid smoking/alcohol)."
        ]

    for tip in suggestions:
        c.drawString(50, y, tip)
        y -= 18

    c.save()
    buffer.seek(0)
    return buffer

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", [
    "Home", 
    "Cancer Prediction", 
    "Learn More About Cancer", 
    "About"
])

# === Home Page ===
if page == "Home":
    st.title("Welcome to the Cancer Prediction App")
    st.write("""
        This app uses a trained machine learning model to predict the likelihood of cancer 
        based on patient data. Use the sidebar to navigate and make predictions or learn more about the project.
    """)

# === Cancer Prediction Page ===
elif page == "Cancer Prediction":
    st.title("🩺 Cancer Prediction")
    st.subheader("Enter patient details")

    default_values = {
        "Age": 45,
        "Gender": 1,
        "BMI": 26.0,
        "Smoking": 0,
        "GeneticRisk": 1,
        "PhysicalActivity": 4.5,
        "AlcoholIntake": 2.5,
        "CancerHistory": 0
    }

    types = {
        "Age": int,
        "Gender": int,
        "BMI": float,
        "Smoking": int,
        "GeneticRisk": int,
        "PhysicalActivity": float,
        "AlcoholIntake": float,
        "CancerHistory": int
    }

    user_input = []
    for feature in feature_names:
        val_type = types[feature]
        val = st.number_input(
            f"{feature}", 
            value=val_type(default_values[feature]), 
            step=1 if val_type == int else 0.1
        )
        user_input.append(val)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_names)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][prediction]
        label = "🛑 Cancer Detected" if prediction == 1 else "✅ No Cancer"

        st.success(f"**Prediction**: {label}")
        st.info(f"**Confidence**: {probability * 100:.2f}%")

        report = input_df.copy()
        report["Prediction"] = label
        report["Confidence (%)"] = round(probability * 100, 2)
        st.subheader("📋 Prediction Summary")
        st.dataframe(report)

        st.subheader("🔍 Cancer Likelihood")
        fig, ax = plt.subplots()
        ax.bar(["No Cancer", "Cancer"], model.predict_proba(input_df)[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        pdf_buffer = generate_pdf(
            dict(zip(feature_names, user_input)),
            label.replace("🛑 ", "").replace("✅ ", ""),
            probability * 100
        )

        st.download_button(
            label="📥 Download Prediction Report (PDF)",
            data=pdf_buffer,
            file_name="cancer_prediction_report.pdf",
            mime="application/pdf"
        )

# === Learn More About Cancer Page ===
elif page == "Learn More About Cancer":
    st.title("📘 Learn More About Cancer")

    st.subheader("🔎 Common Causes of Cancer")
    st.write("""
    - Smoking and tobacco use  
    - Obesity and unhealthy diet  
    - Genetic mutations and family history  
    - Exposure to radiation or harmful chemicals  
    - Lack of physical activity  
    - Alcohol consumption  
    """)

    st.subheader("💡 How to Deal With or Reduce Risk")
    st.write("""
    - Quit smoking and avoid secondhand smoke  
    - Maintain a healthy diet and weight  
    - Regular exercise and physical activity  
    - Routine medical checkups  
    - Limit alcohol consumption  
    - Protect yourself from the sun  
    """)

    st.subheader("🌍 Cancer Impact in Bangladesh")
    st.write("""
    - Cancer is a growing public health concern in Bangladesh.  
    - An estimated 150,000+ new cases are reported each year.  
    - Major challenges include late diagnosis and limited access to advanced treatment.  
    - Awareness, screening programs, and healthcare infrastructure are improving.  
    """)

    st.subheader("📊 Stages of Cancer")
    st.write("""
    - **Stage 0**: Abnormal cells, not yet cancer.  
    - **Stage I**: Small and localized tumor.  
    - **Stage II & III**: Larger tumors, possible spread to lymph nodes.  
    - **Stage IV**: Metastatic cancer, spread to other organs.  
    """)

    st.subheader("🏥 Top Hospitals in Bangladesh")
    st.write("""
    - National Institute of Cancer Research & Hospital (NICRH), Dhaka  
    - Delta Medical College & Hospital  
    - United Hospital, Dhaka  
    - Square Hospitals Ltd  
    - Evercare Hospital, Dhaka  
    """)

    st.subheader("✈️ Popular International Hospitals for Treatment")
    st.write("""
    - Tata Memorial Hospital, Mumbai, India  
    - Apollo Hospitals, Chennai, India  
    - Mount Elizabeth Hospital, Singapore  
    - Bumrungrad International Hospital, Thailand  
    - MD Anderson Cancer Center, Texas, USA  
    - Mayo Clinic, Rochester, USA  
    """)

# === About Page ===
elif page == "About":
    st.title("About the App")
    st.subheader("Created by:")
    st.write("### Rayhan Mahmud Ansari")
    st.write("Dept. of CSE, Sylhet Engineering College")
    st.write("Email: rayhan_mahmud@sec.ac.bd")

    st.write("### Nurul Islam Opu")
    st.write("Dept. of CSE, Sylhet Engineering College")
    st.write("Email: nurulislamopu1@gmail.com")

    st.write("""
        This app predicts the presence of cancer based on various patient metrics like age, BMI,
        alcohol intake, genetic risk factors, etc. The model was trained using supervised learning techniques.

        **Features:**
        - Clean user interface with background image  
        - Instant predictions with default values  
        - Visualization and downloadable PDF report  
        - Educational section about cancer and resources  
    """)
    st.write("### Model Version: 1.0.3")
