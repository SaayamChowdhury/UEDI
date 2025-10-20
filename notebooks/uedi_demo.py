# ==============================
# UEDI Demo Notebook (fixed)
# ==============================

# Cell 1: Markdown
# # UEDI Demo
# This notebook demonstrates how to compute the **Urban/Education Development Index (UEDI)** using synthetic data.

# Cell 2: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from uedi import compute_uedi

# Cell 3: Load data
indicator_df = pd.read_csv("../examples/uedi_synthetic.csv")
print(indicator_df.head())

# Cell 4: Compute UEDI
result = compute_uedi(indicator_df, n_boot=100, random_state=42)
print(result[['uedi_mean','uedi_p2.5','uedi_p97.5']])

# Cell 5: Display domain scores
print(result['domain_scores'])

# Cell 6: Plot bootstrapped UEDI distributions (optional)
# Flatten bootstrap samples for plotting
all_samples = [val for samples in result['uedi_samples'] for val in samples]

plt.hist(all_samples, bins=20, color='skyblue', edgecolor='black')
plt.title("Bootstrapped UEDI Distribution")
plt.xlabel("UEDI")
plt.ylabel("Frequency")
plt.show()
