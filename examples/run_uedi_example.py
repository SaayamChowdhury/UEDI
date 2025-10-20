import pandas as pd
from uedi import compute_uedi

# Load synthetic UEDI indicators
indicator_df = pd.read_csv("uedi_synthetic.csv")

# Compute UEDI with 100 bootstrap iterations for speed
result = compute_uedi(
    indicator_df,
    n_boot=100,
    random_state=42
)

# Display summary
print("UEDI results:")
print(result[['uedi_mean', 'uedi_p2.5', 'uedi_p97.5']])
print("\nDomain scores (mean across boots):")
print(result['domain_scores'])
