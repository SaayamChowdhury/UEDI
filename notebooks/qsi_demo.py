# ==============================
# QSI Demo Script (Python)
# ==============================

# Cell 1: Markdown
# # QSI Demo
# This script demonstrates computing **Quality of Student Experience (QSI)** composites using synthetic data.

# Cell 2: Imports
import pandas as pd
import matplotlib.pyplot as plt
from qsi import qsi_pipeline

# Cell 3: Load synthetic QSI data
df = pd.read_csv("../examples/qsi_synthetic.csv")
print(df.head())

# Cell 4: Define domain and SEM columns
domain_cols = [
    "Emotional Health",
    "Motivation",
    "Identity & Belonging",
    "Burnout Risk",
    "Cognitive Function",
    "Engagement",
]

sem_cols = [
    "Emotional_sem",
    "Motivation_sem",
    "Identity_sem",
    "Burnout_sem",
    "Cognitive_sem",
    "Engagement_sem",
]

# Cell 5: Compute QSI
result = qsi_pipeline(
    df,
    domain_cols,
    sem_cols=sem_cols,
    strata_cols=["grade_band", "language"],
    outcome_col="GPA",
    bootstrap_iters=50,
    pv_draws=3,
    random_state=42
)

# Cell 6: View bootstrap summary
print("Bootstrap summary:")
print(result["bootstrap_summary"])

# Cell 7: Plot bootstrap distributions
plt.hist(result["bootstrap_raw"]["theory_means"], bins=15, alpha=0.5, label="Theory")
plt.hist(result["bootstrap_raw"]["data_means"], bins=15, alpha=0.5, label="Data")
plt.hist(result["bootstrap_raw"]["decor_means"], bins=15, alpha=0.5, label="Decorrelated")
plt.legend()
plt.title("Bootstrap QSI Distribution")
plt.xlabel("QSI")
plt.ylabel("Frequency")
plt.show()
