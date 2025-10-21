### Appendix D. Default Modeling and Scoring Specifications

---

#### D1. Data Quality Checks
Admissible ranges; missingness patterns; straightlining flags; response-time outliers; balanced keying where feasible; latent acquiescence factor for multi-item self-reports.

---

#### D2. Estimation
- **Categorical indicators:** WLSMV.  
- **Mixed indicators:** MLR with cluster-robust SEs and sampling weights.  
- **Missing data:** FIML under MAR for ML/MLR; pairwise-present for WLSMV; MI sensitivity for MNAR as resources allow.  
- **Planned missingness:** Matrix sampling with ML/Bayesian IRT for item blocks.

---

#### D3. Reliability and Dimensionality
- Report **ω** (and **ωₕ** for bifactor considerations).  
- Retain factors via **parallel analysis** / **MAP**; confirm via **CFA**.  
- Avoid rigid cutoffs, but report **CFI/TLI**, **RMSEA**, and **SRMR**.

---

#### D4. Invariance and DIF
- **Invariance ladder:** configural → metric → scalar.  
  - Rules: ΔCFI ≤ .01, ΔRMSEA ≤ .015, plus theoretical justification.  
- If scalar fails: apply **partial-scalar** or **alignment** (with R² diagnostics); otherwise suppress cross-group mean comparisons.  
- **DIF:** use **ordinal logistic regression** (McFadden ΔR²) and **IRT LR tests**; revise or drop items with meaningful DIF.

---

#### D5. Latent Scoring
- **Ordinal items:** graded response (2PL) with **EAP scores** and **posterior SEs**.  
- **CFA scales:** regression-based factor scores with associated SEs.  
- **Large-scale assessment:** generate **5–10 plausible values per construct**; combine using **Rubin’s rules**.

---

#### D6. Scaling and Linking
- **Standardization:** within *grade-band × language* (mean 0, SD 1).  
- **Display transform:** S = 50 + 10z, bounded to [0, 100] for dashboards; estimation on z-scale.  
- **Linking across years:** via **anchor items** + **Stocking–Lord/Haebara (IRT)** or **multi-group CFA anchors**; report **linking error**.

---

