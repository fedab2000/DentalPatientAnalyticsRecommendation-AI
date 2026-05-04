import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def train_segmentation_model():
    df = pd.read_csv("data/dental_dataset.csv")

    features = [
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

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df["segment"] = clusters

    score = silhouette_score(X_scaled, clusters)

    print("Segmentation model trained.")
    print("Silhouette Score:", round(score, 3))

    print("\nSegment Summary:")
    segment_summary = df.groupby("segment")[features].mean().round(2)
    print(segment_summary)

    # -----------------------------
    # Label segments
    # -----------------------------
    raw_summary = df.groupby("segment")[features].mean()

    segment_labels = {}

    for seg in raw_summary.index:
        row = raw_summary.loc[seg]

        if row["total_spend"] > 4000:
            segment_labels[seg] = "High Value Patients"
        elif row["whitening_count"] > 0.5:
            segment_labels[seg] = "Cosmetic Patients"
        elif row["visits_per_year"] <= 2:
            segment_labels[seg] = "Low Engagement Patients"
        else:
            segment_labels[seg] = "Preventive Care Patients"

    df["segment_label"] = df["segment"].map(segment_labels)

    print("\nSegment Labels:")
    for k, v in segment_labels.items():
        print(f"Segment {k} → {v}")

    # -----------------------------
    # PCA visualization
    # -----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    plt.figure(figsize=(8, 6))

    for segment in sorted(df["segment"].unique()):
        subset = df[df["segment"] == segment]
        label = segment_labels[segment]
        plt.scatter(
            subset["pca1"],
            subset["pca2"],
            label=f"{segment}: {label}",
            alpha=0.6
        )

    plt.title("Dental Patient Segmentation")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    os.makedirs("outputs", exist_ok=True)
    plot_path = "outputs/dental_segmentation_plot.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"\nSegmentation plot saved to: {plot_path}")

    # -----------------------------
    # Save model and labeled dataset
    # -----------------------------
    pipeline = Pipeline([
        ("scaler", scaler),
        ("kmeans", kmeans)
    ])

    joblib.dump(pipeline, "outputs/dental_segmentation_model.joblib")

    df.to_csv("data/dental_dataset_with_segments.csv", index=False)

    print("\nSaved:")
    print("outputs/dental_segmentation_model.joblib")
    print("data/dental_dataset_with_segments.csv")


if __name__ == "__main__":
    train_segmentation_model()