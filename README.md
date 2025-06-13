# 🧬 Cancer Prediction ML Project

This project is a machine learning-powered web application built with **Streamlit** that predicts the likelihood of a patient being diagnosed with cancer based on medical and lifestyle data.

The app is fully interactive, supports instant predictions, downloadable **PDF reports**, a beautiful UI, and is ready for deployment.

---

## 🚀 Features

- 🎯 Predicts cancer based on 8 key patient features
- 📋 Clean user-friendly UI built with **Streamlit**
- 📈 Displays probability visualizations
- 📄 Downloadable **PDF report** with:
  - Prediction result
  - Confidence score
  - Patient input summary
  - Health recommendations
  - Creator signature

- 🎨 Custom background image and page navigation
- 🔎 Multiple models trained; best selected automatically

---

## 📊 Input Features

- `Age`
- `Gender` (0 = Female, 1 = Male)
- `BMI`
- `Smoking` (0/1)
- `GeneticRisk` (0–2)
- `PhysicalActivity`
- `AlcoholIntake`
- `CancerHistory` (0/1)

---

## 📂 Folder Structure
ML_Lab_Project_Cancer_Prediction/
├── app/                  # Streamlit application
│   └── app.py            # Main application logic
├── data/                 # Datasets
│   ├── The_Cancer_data_V2.csv
│   ├── X_train.csv
│   └── X_test.csv
├── images/               # Assets
│   └── background_image.jpg
├── models/               # Serialized models
│   └── cancer_model.pkl
├── src/                  # Development scripts
│   ├── eda.ipynb         # Exploratory analysis
│   ├── model_training.ipynb
│   └── model_training.py # Training pipeline
├── requirements.txt      # Dependencies
└── README.md             # Project documentation



