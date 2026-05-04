import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def train_recommendation_model():
    df = pd.read_csv("data/dental_dataset_with_segments.csv")

    target = "next_procedure"

    features = [
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

    X = df[features]
    y = df[target]

    categorical_cols = ["gender", "segment_label"]
    numerical_cols = [col for col in features if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== Recommendation Model Evaluation ===")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print(classification_report(y_test, predictions, zero_division=0))

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/dental_recommendation_model.joblib")

    print("\n✅ Recommendation model saved to outputs/dental_recommendation_model.joblib")

    return model


if __name__ == "__main__":
    model = train_recommendation_model()

    sample = pd.DataFrame([{
        "age": 45,
        "gender": "Female",
        "visits_per_year": 3,
        "cleaning_count": 2,
        "filling_count": 1,
        "root_canal_count": 0,
        "crown_count": 1,
        "whitening_count": 0,
        "orthodontics_flag": 0,
        "total_procedures": 4,
        "total_spend": 1550,
        "avg_spend_per_procedure": 387.50,
        "segment_label": "Preventive Care Patients"
    }])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.classes_

    top3 = sorted(
        zip(classes, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    print("\nSample recommended procedure:", prediction)
    print("Top 3 recommendations:")
    for procedure, prob in top3:
        print(f"{procedure}: {prob:.2%}")