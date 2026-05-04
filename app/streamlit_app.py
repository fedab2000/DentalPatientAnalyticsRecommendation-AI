import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Dental Patient Analytics", layout="wide")

segmentation_model = joblib.load("outputs/dental_segmentation_model.joblib")
recommendation_model = joblib.load("outputs/dental_recommendation_model.joblib")
df = pd.read_csv("data/dental_dataset_with_segments.csv")

segment_info = {
    "High Value Patients": {
        "description": "Patients with high total spend and multiple procedures.",
        "pattern": "Frequent treatments and higher-complexity procedures such as crowns or root canals.",
        "action": "Focus on retention, personalized care plans, and long-term treatment scheduling."
    },
    "Cosmetic Patients": {
        "description": "Patients with interest in aesthetic or elective procedures.",
        "pattern": "Higher whitening activity or interest in appearance-focused services.",
        "action": "Promote cosmetic services, whitening packages, and smile enhancement consultations."
    },
    "Low Engagement Patients": {
        "description": "Patients with low visit frequency and fewer completed procedures.",
        "pattern": "Few visits per year and lower procedure volume.",
        "action": "Send recall reminders, emphasize preventive care, and use re-engagement campaigns."
    },
    "Preventive Care Patients": {
        "description": "Patients focused on routine maintenance and preventive dental care.",
        "pattern": "Consistent cleanings and regular check-ups.",
        "action": "Maintain engagement with hygiene plans, routine follow-ups, and preventive care messaging."
    }
}

st.title("Dental Patient Segmentation & Procedure Recommendation Dashboard")

st.write(
    "This dashboard segments dental patients based on historical behavior and "
    "provides procedure propensity insights to support staff and dentists."
)

tab1, tab2, tab3 = st.tabs([
    "Patient Recommendation",
    "Segment Insights",
    "Dataset Overview"
])

features_for_segmentation = [
    "age",
    "visits_per_year",
    "cleaning_count",
    "filling_count",
    "root_canal_count",
    "crown_count",
    "whitening_count",
    "orthodontics_flag",
    "total_procedures",
    "total_spend",
    "avg_spend_per_procedure"
]

recommendation_features = [
    "age",
    "gender",
    "visits_per_year",
    "cleaning_count",
    "filling_count",
    "root_canal_count",
    "crown_count",
    "whitening_count",
    "orthodontics_flag",
    "total_procedures",
    "total_spend",
    "avg_spend_per_procedure",
    "segment_label"
]


def assign_segment_label(row):
    if row["total_spend"] > 4000:
        return "High Value Patients"
    elif row["whitening_count"] > 0.5:
        return "Cosmetic Patients"
    elif row["visits_per_year"] <= 2:
        return "Low Engagement Patients"
    else:
        return "Preventive Care Patients"


with tab1:
    st.header("Patient Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=40)
        gender = st.selectbox("Gender", ["Female", "Male"])
        visits_per_year = st.number_input("Visits Per Year", min_value=1, max_value=10, value=3)
        cleaning_count = st.number_input("Cleaning Count", min_value=0, max_value=20, value=2)
        filling_count = st.number_input("Filling Count", min_value=0, max_value=20, value=1)
        root_canal_count = st.number_input("Root Canal Count", min_value=0, max_value=10, value=0)

    with col2:
        crown_count = st.number_input("Crown Count", min_value=0, max_value=10, value=0)
        whitening_count = st.number_input("Whitening Count", min_value=0, max_value=10, value=0)
        orthodontics_flag = st.selectbox("Orthodontics History", [0, 1])
        total_spend = st.number_input("Total Spend", min_value=0, max_value=50000, value=1000)
        total_procedures = st.number_input("Total Procedures", min_value=1, max_value=100, value=4)

    avg_spend_per_procedure = total_spend / max(total_procedures, 1)

    patient_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "visits_per_year": visits_per_year,
        "cleaning_count": cleaning_count,
        "filling_count": filling_count,
        "root_canal_count": root_canal_count,
        "crown_count": crown_count,
        "whitening_count": whitening_count,
        "orthodontics_flag": orthodontics_flag,
        "total_procedures": total_procedures,
        "total_spend": total_spend,
        "avg_spend_per_procedure": avg_spend_per_procedure
    }])

    if st.button("Generate Patient Insight"):
        segment_number = segmentation_model.predict(patient_data[features_for_segmentation])[0]

        patient_row = patient_data.iloc[0]
        segment_label = assign_segment_label(patient_row)

        recommendation_input = patient_data.copy()
        recommendation_input["segment_label"] = segment_label

        predicted_procedure = recommendation_model.predict(
            recommendation_input[recommendation_features]
        )[0]

        probabilities = recommendation_model.predict_proba(
            recommendation_input[recommendation_features]
        )[0]

        classes = recommendation_model.classes_

        top3 = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        st.subheader("Patient Insight Results")

        st.write(f"**Segment Number:** {segment_number}")
        st.write(f"**Segment Label:** {segment_label}")

        info = segment_info.get(segment_label)

        if info:
            st.subheader("Segment Insight")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Typical Pattern:** {info['pattern']}")
            st.write(f"**Recommended Staff Action:** {info['action']}")

        st.write(f"**Recommended Next Procedure:** {predicted_procedure}")
        st.write(f"**Average Spend Per Procedure:** ${avg_spend_per_procedure:,.2f}")

        st.subheader("Top 3 Procedure Propensity Scores")

        top3_df = pd.DataFrame(top3, columns=["Procedure", "Probability"])
        st.dataframe(top3_df, use_container_width=True)
        st.bar_chart(top3_df.set_index("Procedure")["Probability"])

        st.info(
            "These outputs are decision-support insights based on historical patterns. "
            "Final treatment decisions should always be made by a licensed dental professional."
        )


with tab2:
    st.header("Segment Insights")

    segment_summary = df.groupby("segment_label").agg(
        patients=("patient_id", "count"),
        avg_age=("age", "mean"),
        avg_visits_per_year=("visits_per_year", "mean"),
        avg_total_procedures=("total_procedures", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_spend_per_procedure=("avg_spend_per_procedure", "mean")
    ).reset_index()

    st.subheader("Segment Summary")
    st.dataframe(segment_summary.round(2), use_container_width=True)

    st.subheader("Segment Descriptions")

    for segment, info in segment_info.items():
        st.markdown(f"### {segment}")
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Pattern:** {info['pattern']}")
        st.write(f"**Recommended Action:** {info['action']}")

    st.subheader("Average Total Spend by Segment")
    st.bar_chart(segment_summary.set_index("segment_label")["avg_total_spend"])

    st.subheader("Average Visits Per Year by Segment")
    st.bar_chart(segment_summary.set_index("segment_label")["avg_visits_per_year"])


with tab3:
    st.header("Dataset Overview")

    st.write("Sample of synthetic dental patient dataset:")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Procedure Distribution")

    procedure_counts = df["next_procedure"].value_counts().reset_index()
    procedure_counts.columns = ["Procedure", "Count"]

    st.dataframe(procedure_counts, use_container_width=True)
    st.bar_chart(procedure_counts.set_index("Procedure")["Count"])

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit | Dental Patient Analytics Project")