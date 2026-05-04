import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 10000

data = pd.DataFrame({
    "patient_id": range(1, n + 1),
    "age": np.random.randint(18, 80, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "visits_per_year": np.random.randint(1, 6, n)
})

# Procedure counts
data["cleaning_count"] = np.random.poisson(2, n)
data["filling_count"] = np.random.poisson(1, n)
data["root_canal_count"] = np.random.binomial(1, 0.1, n)
data["crown_count"] = np.random.binomial(2, 0.2, n)
data["whitening_count"] = np.random.binomial(1, 0.15, n)
data["orthodontics_flag"] = np.random.binomial(1, 0.1, n)

# Total procedures
data["total_procedures"] = (
    data["cleaning_count"]
    + data["filling_count"]
    + data["root_canal_count"]
    + data["crown_count"]
    + data["whitening_count"]
    + data["orthodontics_flag"]
)

# Pricing assumptions
costs = {
    "cleaning": 150,
    "filling": 250,
    "root_canal": 1200,
    "crown": 1000,
    "whitening": 400,
    "orthodontics": 3000
}

data["total_spend"] = (
    data["cleaning_count"] * costs["cleaning"]
    + data["filling_count"] * costs["filling"]
    + data["root_canal_count"] * costs["root_canal"]
    + data["crown_count"] * costs["crown"]
    + data["whitening_count"] * costs["whitening"]
    + data["orthodontics_flag"] * costs["orthodontics"]
)

data["avg_spend_per_procedure"] = data["total_spend"] / (data["total_procedures"] + 1)

# Synthetic next procedure logic
def assign_next_procedure(row):
    if row["cleaning_count"] < 2:
        return "Cleaning"
    elif row["filling_count"] > 1:
        return "Crown"
    elif row["age"] > 50:
        return "Root Canal"
    elif row["whitening_count"] == 0:
        return "Whitening"
    else:
        return np.random.choice(["Cleaning", "Filling", "Whitening"])

data["next_procedure"] = data.apply(assign_next_procedure, axis=1)

os.makedirs("data", exist_ok=True)
data.to_csv("data/dental_dataset.csv", index=False)

print("✅ Dental dataset created: data/dental_dataset.csv")