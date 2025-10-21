### Appendix E. Composite (QSI) Defaults

---

#### E1. Domains
Emotional Health; Motivation; Identity & Belonging; Burnout Risk; Cognitive Function; Engagement.

---

#### E2. Weighting Schemes

| **Scheme** | **Description** | **Notes** |
|-------------|------------------|------------|
| **Theory-fixed (default)** | Motivation 0.24; Engagement 0.24; Emotional Health 0.20; Cognitive Function 0.16; Identity & Belonging 0.10; Burnout Risk 0.06 | Fixed, conceptually grounded balance |
| **Data-driven** | Standardized SEM coefficients predicting GPA/attendance/progression, ridge-regularized | Use bootstrap CIs for uncertainty |
| **Decorrelated** | \( w \propto Σ^{-1}1 \), normalized; shrinkage estimator when *n* is small | Reduces redundancy across domains |

---

#### E3. Uncertainty
Composite SEs incorporate **score SEs** (via delta method or plausible values) and **design effects** (weights, clustering).  
Report **confidence intervals (CIs)** for all composite aggregates.

---

#### E4. Inequality Sensitivity (Optional Release)
Compute **Atkinson A(ε)** for ε ∈ {0.25, 0.5, 1.0} on a positively shifted scale.  
Report **EDE** (equally distributed equivalent) values alongside means.  
If applied, publish the adjustment coefficient λ in:

\[
QSI_{\text{adj}} = QSI_{\text{mean}} \times (1 - λA)
\]

---

