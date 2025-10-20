import pandas as pd
from qsi import qsi_pipeline

# Load synthetic QSI data
df = pd.read_csv("qsi_synthetic.csv")

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

# Compute QSI pipeline
result = qsi_pipeline(
    df,
    domain_cols,
    sem_cols=sem_cols,
    strata_cols=["grade_band", "language"],
    outcome_col="GPA",
    bootstrap_iters=50,  # smaller for demo
    pv_draws=3,
    random_state=42
)

# Display bootstrap summary
print("Bootstrap summary:")
print(result["bootstrap_summary"])

# Display EDE keys
print("\nEDE summary keys:")
print(list(result["ede"].keys()))
