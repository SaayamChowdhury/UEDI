### Appendix G. Example Syntax Skeletons

---

#### G1. IRT Graded Response (Pseudo-Code)
- Specify item families per domain.  
- Estimate **2PL graded response** model.  
- Extract **EAP scores** and **standard errors (SEs)**.  
- Generate **10 plausible values (PVs)** with background conditioning (grade, sex, SES).

---

#### G2. SEM Model 1 (Distress → EF → Performance)
- **Latent variables:**  
  - Distress =~ PHQ / GAD / RCADS  
  - EF =~ Flanker + DigitSpan + BRIEF (include rater method factor)  
  - Perf =~ test scores + GPA
- **Paths:**  
  - Distress → EF  
  - EF → Perf  
  - Distress → Perf (optional)
- **Indirect effects:** bootstrap; Level-2 school climate on EF/Perf via MSEM.

---

#### G3. Invariance Testing
- **Configural:** free loadings and thresholds across groups.  
- **Metric:** equal loadings across groups.  
- **Scalar:** equal loadings + thresholds; if ΔCFI > .01, relax to **partial-scalar** or use **alignment**; log diagnostics.
