# 🦷 Dental Patient Analytics & Recommendation System

An end-to-end machine learning project that segments dental patients and provides data-driven procedure recommendations to support dental office staff and improve patient engagement.

---

## 🚀 Project Overview

This project simulates a real-world dental analytics solution by combining:

- **Patient segmentation (clustering)**
- **Procedure recommendation (classification)**
- **Interactive dashboard (Streamlit)**

The system analyzes patient behavior and suggests likely next procedures based on historical patterns.

---

## 🎯 Objectives

- Segment patients into meaningful groups
- Identify behavioral patterns across segments
- Recommend next procedures using ML
- Provide actionable insights for dental staff
- Improve patient engagement and clinic revenue

---

## 📊 Key Features

### 🧠 Patient Segmentation
- Model: **KMeans Clustering**
- Segments:
  - High Value Patients
  - Cosmetic Patients
  - Preventive Care Patients
  - Low Engagement Patients

---

### 🤖 Procedure Recommendation
- Model: **Random Forest Classifier**
- Predicts: `next_procedure`
- Outputs:
  - Recommended procedure
  - Top 3 procedure probabilities

---

### 📈 Dashboard (Streamlit)

#### 1. Patient Recommendation
- Enter patient details
- Get:
  - Segment classification
  - Recommended procedure
  - Top 3 probabilities
  - Segment insights & staff actions

#### 2. Segment Insights
- Summary statistics per segment
- Business interpretation
- Recommended actions

#### 3. Dataset Overview
- Sample data
- Procedure distribution

---

## 📌 Segment Definitions

| Segment | Description | Recommended Action |
|--------|------------|------------------|
| High Value Patients | High spend, multiple procedures | Focus on retention & long-term planning |
| Cosmetic Patients | Interested in aesthetics | Promote whitening & cosmetic services |
| Preventive Care Patients | Routine maintenance | Maintain engagement with hygiene plans |
| Low Engagement Patients | Low visit frequency | Re-engage with reminders |

---

## 🧮 Methodology

### Segmentation
```text
KMeans clustering on behavioral + financial features

Recommendation
Random Forest classification using patient features + segment

Project Structure
project/
│
├── data/
│   ├── dental_dataset.csv
│   └── dental_dataset_with_segments.csv
│
├── src/
│   ├── generate_dental_dataset.py
│   ├── train_segmentation_model.py
│   └── train_recommendation_model.py
│
├── outputs/
│   ├── dental_segmentation_model.joblib
│   ├── dental_recommendation_model.joblib
│   └── dental_segmentation_plot.png
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md

How to Run
1. Install dependencies
pip install -r requirements.txt
2. Generate dataset
python src/generate_dental_dataset.py
3. Train models
python src/train_segmentation_model.py
python src/train_recommendation_model.py
4. Run dashboard
python -m streamlit run app/streamlit_app.py

Example Output
Segment: High Value Patients
Recommended Procedure: Crown
Confidence:
Crown: 65%
Filling: 20%
Cleaning: 15%

Business Impact

This system helps dental clinics:

Improve patient retention
Increase procedure uptake
Identify high-value patients
Optimize treatment planning
Support staff decision-making

Disclaimer

This tool provides decision-support insights only.
Final clinical decisions must be made by licensed dental professionals.

Technologies Used
Python
Pandas / NumPy
Scikit-learn
Streamlit
Matplotlib

Project Summary

Built a machine learning-based dental analytics system that segments patients and predicts likely next procedures, enabling data-driven engagement strategies and improving operational decision-making.

Author
Feda Bashbishi

