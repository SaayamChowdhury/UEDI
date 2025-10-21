# Index Codes: UEDI & QSI

This repository accompanies the research paper on multidomain educational indices and provides Python implementations for two main indices:

- **UEDI (Universal Education Development Index)** – evaluates education and development across jurisdictions, adjusted for inequalities.
- **QSI (Quantitative Scoring Index)** – measures student psychological, engagement, and academic experiences using psychometric data.

The purpose of this repository is to allow researchers and policymakers to **reproduce results**, **explore synthetic examples**, and **apply these indices to their own datasets**.

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

## 📂 Repository Structure
```
uedi-qsi-indices/
├─ examples/ # Example scripts & synthetic datasets
│ ├─ uedi_synthetic.csv
│ ├─ qsi_synthetic.csv
│ ├─ run_uedi_example.py
│ └─ run_qsi_example.py
├─ notebooks/ # Interactive Jupyter notebooks
│ ├─ uedi_demo.ipynb
│ ├─ qsi_demo.ipynb
│ └─ index_comparison.ipynb
├─ appendices/ # Optional detailed documentation, formulas, derivations
├─ uedi.py # UEDI implementation
├─ qsi.py # QSI implementation
├─ requirements.txt
└─ README.md
```
---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saayamchowdhury/uedi-qsi-indices.git
cd uedi-qsi-indices
```
Install Dependencies: 
```
pip install -r requirements.txt
```
---

## 🏃 Usage

Run UEDI Examples:
```
python examples/run_uedi_example.py
```
Run QSI Examples:
```
python examples/run_qsi_example.py
```
Compare UEDI and QSI
```
python examples/run_index_comparison.py
```

---

## 📊 Reproduce Paper Figures/Tables

1. Use the provided synthetic datasets in examples/ or your own data.
2. Run the example scripts or notebooks to compute indices.
3. Use the bootstrapped outputs for uncertainty intervals and plotting.
4. Follow notebook visualizations for side-by-side comparisons, domain scores, and equity adjustments.

---

## 📧 Contact

For questions or collaborations, reach out at: saayamchowdhury326@gmail.com

---




