### Appendix J. Weight Construction, Replicate Weights, and Effective Sample Size

---

#### J.1 Base and Adjusted Weights
- **Base weight:** \( w_0 = \frac{1}{\pi} \), where π is the selection probability.  
- **Nonresponse adjustment:** via response-homogeneity classes.  
- **Post-stratification:** to enrollment margins (grade × sex × region).  
- **Final weight:** \( w_{\text{final}} = w_0 \times a_{\text{nr}} \times a_{\text{ps}} \).

---

#### J.2 Replicate Weights (BRR with Fay’s ρ = 0.5)
- Create **80 replicate weights** \( w^{(r)} \) using the stratified half-sample scheme with Fay’s perturbation.  
- Publish **generator seed** and the **stratum–PSU map** for reproducibility.

---
