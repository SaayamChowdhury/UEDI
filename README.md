# Index Codes: UEDI & QSI

This repository contains Python implementations for computing two multidomain indices:

- **UEDI (Universal Education Development Index)** – evaluates equity and development across jurisdictions, adjusted for inequalities.
- **QSI (Quantitative Scoring Index)** – measures student psychological, engagement, and academic experiences using psychometric data.

---

## 🧩 Features

- Domain harmonization (logit for proportions, log for monetary, z-scoring)
- Factor extraction (PCA for UEDI domains)
- Equity adjustment (Atkinson index & group gap penalties)
- Multiple weighting schemes (theory-driven, data-driven, decorrelated)
- Bootstrap-based uncertainty estimates
- Plausible value generation for psychometric domains (QSI)
- Outputs in intuitive 0–100 scales
- Optional appendices for detailed documentation, formulas, and example derivations

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saayamchowdhury/uedi-qsi-indices.git
cd uedi-qsi-indices


