import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional

def standardize_within_strata(df: pd.DataFrame, domain_cols: List[str], strata_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Return a copy of df with domain cols converted to z-scores within each stratum (mean=0, sd=1)."""
    out = df.copy()
    if strata_cols:
        grouped = out.groupby(strata_cols, dropna=False)
    else:
        # single global stratum
        for d in domain_cols:
            if d == "Motivation":
                data[d] = rng.normal(0.5, 1.0, size=n)
            elif d == "Engagement":
                data[d] = rng.normal(1.0, 1.0, size=n)
            elif d == "Burnout Risk":
                data[d] = rng.normal(-0.8, 1.2, size=n)
            else:
                data[d] = rng.normal(loc=0.0, scale=1.0, size=n)
    for _, group in grouped:
        means = group[domain_cols].mean(axis=0)
        sds = group[domain_cols].std(axis=0, ddof=0).replace(0, np.nan)  # avoid divide by zero
        idx = group.index
        z = (group[domain_cols] - means) / sds
        # where sd==0, set z to 0 (everyone identical)
        z = z.fillna(0)
        out.loc[idx, domain_cols] = z.values
    return out


def display_map(z_scores: pd.Series) -> pd.Series:
    """Monotone mapping Sd = 50 + 10*z bounded to [0,100]."""
    s = 50.0 + 10.0 * z_scores
    s = s.clip(0.0, 100.0)
    return s


def theory_fixed_weights(domain_cols: List[str]) -> np.ndarray:
    """Return theory-fixed weights in the order of domain_cols.
    Presumed order: Emotional Health, Motivation, Identity & Belonging, Burnout Risk, Cognitive Function, Engagement
    If domain_cols order differs, caller should pass correct ordering.
    """
    # weights from specification: higher on Motivation and Engagement; moderate on Emotional Health and Cognitive Function;
    # smaller on Identity & Belonging and Burnout.
    mapping = {
        0: 0.15,  # Emotional Health
        1: 0.25,  # Motivation
        2: 0.08,  # Identity & Belonging
        3: 0.07,  # Burnout Risk
        4: 0.20,  # Cognitive Function
        5: 0.25,  # Engagement
    }
    n = len(domain_cols)
    w = np.array([mapping.get(i, 1.0) for i in range(n)], dtype=float)
    w = w / w.sum()
    return w


def data_driven_weights(df: pd.DataFrame, domain_cols: List[str], outcome_col: str, alphas: Optional[List[float]] = None) -> np.ndarray:
    """Compute standardized, regularized regression coefficients (ridge) predicting outcome from domains.
    Returns nonnegative normalized weights proportional to absolute standardized coefficients.
    """
    X = df[domain_cols].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=float)
    # standardize predictors and outcome so coefficients are comparable
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    if alphas is None:
        alphas = np.logspace(-3, 3, 21)
    model = RidgeCV(alphas=alphas, store_cv_values=False)
    model.fit(Xs, ys)
    coefs = model.coef_.ravel()
    # use absolute standardized coefs
    abs_coefs = np.abs(coefs)
    if abs_coefs.sum() == 0:
        # fallback to equal weights
        w = np.ones(len(domain_cols)) / len(domain_cols)
    else:
        w = abs_coefs / abs_coefs.sum()
    return w


def decorrelated_weights(df: pd.DataFrame, domain_cols: List[str]) -> np.ndarray:
    """Compute inverse-covariance weights: w ∝ Σ^{-1} 1, with Ledoit-Wolf shrinkage covariance estimator."""
    X = df[domain_cols].to_numpy(dtype=float)
    # center the data
    Xc = X - np.nanmean(X, axis=0)
    # estimate covariance with shrinkage
    lw = LedoitWolf().fit(Xc)
    cov = lw.covariance_
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    one = np.ones((len(domain_cols),))
    w_raw = inv.dot(one)
    # make nonnegative and normalize
    w_raw = np.maximum(w_raw, 0.0)
    if w_raw.sum() == 0:
        w = np.ones(len(domain_cols)) / len(domain_cols)
    else:
        w = w_raw / w_raw.sum()
    return w


def compute_qsi_scores(df: pd.DataFrame, domain_cols: List[str], strata_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute standardized domain z-scores and mapped Sd,i display scores in [0,100].
    Returns df with columns: {domain}_z and {domain}_S.
    """
    zdf = standardize_within_strata(df, domain_cols, strata_cols)
    out = df.copy()
    for col in domain_cols:
        out[f"{col}_z"] = zdf[col]
        out[f"{col}_S"] = display_map(zdf[col])
    return out


def composite_from_weights(scores_df: pd.DataFrame, domain_S_cols: List[str], weights: np.ndarray, comp_name: str = "QSI") -> pd.DataFrame:
    """Compute composite QSI = sum_d w_d * S_d. weights should sum to 1."""
    if len(weights) != len(domain_S_cols):
        raise ValueError("weights length must match domain_S_cols")
    out = scores_df.copy()
    Smat = scores_df[domain_S_cols].to_numpy(dtype=float)
    comp = Smat.dot(weights)
    out[comp_name] = comp
    return out


def generate_plausible_values(df: pd.DataFrame, domain_z_cols: List[str], sem_cols: List[str], n_draws: int = 10, random_state: Optional[int] = None) -> List[pd.DataFrame]:
    """Generate plausible values (latent z) using domain SEMs: z_pv = z + N(0, sem).
    Returns list of dataframes with domain_z_cols replaced by draws. """
    rng = np.random.default_rng(random_state)
    draws = []
    for d in range(n_draws):
        df_draw = df.copy()
        noise = np.zeros_like(df[domain_z_cols].to_numpy(dtype=float))
        for j, sem_col in enumerate(sem_cols):
            sem = df[sem_col].to_numpy(dtype=float)
            # if sem missing, treat as 0
            sem = np.nan_to_num(sem, 0.0)
            noise[:, j] = rng.normal(loc=0.0, scale=sem)
        df_draw[domain_z_cols] = df[domain_z_cols].to_numpy(dtype=float) + noise
        draws.append(df_draw)
    return draws


def bootstrap_composite(df: pd.DataFrame,
                        domain_cols: List[str],
                        sem_cols: Optional[List[str]],
                        strata_cols: Optional[List[str]],
                        outcome_col: Optional[str],
                        n_boot: int = 500,
                        n_pv: int = 5,
                        random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Compute bootstrap distribution of composite under three weighting schemes.
    Returns dict of arrays (n_boot,) for mean composite under each scheme and per-boot raw composites if needed.
    """
    rng = np.random.default_rng(random_state)
    n = len(df)
    results = {"theory_means": [], "data_means": [], "decor_means": []}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)  # bootstrap sample indices with replacement
        sample = df.iloc[idx].reset_index(drop=True)
        # standardize within strata using sample's strata groups
        scores = compute_qsi_scores(sample, domain_cols, strata_cols)
        # domain S column names in same order
        S_cols = [f"{c}_S" for c in domain_cols]
        # weights: theory fixed
        w_theory = theory_fixed_weights(domain_cols)
        # data-driven: needs outcome; if missing, fallback to theory
        if outcome_col and outcome_col in sample.columns:
            try:
                w_data = data_driven_weights(sample, domain_cols, outcome_col)
            except Exception:
                w_data = w_theory
        else:
            w_data = w_theory
        # decorrelated
        try:
            w_decorr = decorrelated_weights(sample, domain_cols)
        except Exception:
            w_decorr = w_theory
        # plausible values: if sems provided, perturb z and recompute S and composite across n_pv draws
        if sem_cols:
            pv_draws = generate_plausible_values(scores, [f"{c}_z" for c in domain_cols], sem_cols, n_draws=n_pv, random_state=rng.integers(0, 2**32 - 1))
            comp_means = {"theory": [], "data": [], "decor": []}
            for pv in pv_draws:
                # map pv z to S
                for c in domain_cols:
                    pv[f"{c}_S"] = display_map(pv[f"{c}_z"])
                comp_means["theory"].append(composite_from_weights(pv, [f"{c}_S" for c in domain_cols], w_theory)["QSI"].mean())
                comp_means["data"].append(composite_from_weights(pv, [f"{c}_S" for c in domain_cols], w_data)["QSI"].mean())
                comp_means["decor"].append(composite_from_weights(pv, [f"{c}_S" for c in domain_cols], w_decorr)["QSI"].mean())
            results["theory_means"].append(np.mean(comp_means["theory"]))
            results["data_means"].append(np.mean(comp_means["data"]))
            results["decor_means"].append(np.mean(comp_means["decor"]))
        else:
            # single point estimate per bootstrap
            results["theory_means"].append(composite_from_weights(scores, S_cols, w_theory)["QSI"].mean())
            results["data_means"].append(composite_from_weights(scores, S_cols, w_data)["QSI"].mean())
            results["decor_means"].append(composite_from_weights(scores, S_cols, w_decorr)["QSI"].mean())
    # convert to arrays
    for k in list(results.keys()):
        results[k] = np.array(results[k])
    return results


def atkinson_index(values: np.ndarray, eps: float) -> float:
    """Compute Atkinson index A_e for positive values.
    For eps != 1: A = 1 - ( (1/n * sum(x^(1-e)) )^(1/(1-e)) ) / mean(x)
    For eps == 1: A = 1 - (geometric mean) / mean(x)
    """
    x = np.asarray(values, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    mean_x = x.mean()
    if mean_x == 0:
        return 0.0
    n = x.size
    if eps == 1.0:
        gm = np.exp(np.log(x).mean())
        A = 1.0 - (gm / mean_x)
    else:
        k = 1.0 - eps
        y = (x ** k).mean()
        denom = y ** (1.0 / k)
        A = 1.0 - (denom / mean_x)
    return float(np.clip(A, 0.0, 1.0))


def qsi_pipeline(df: pd.DataFrame,
                 domain_cols: List[str],
                 sem_cols: Optional[List[str]] = None,
                 strata_cols: Optional[List[str]] = None,
                 outcome_col: Optional[str] = None,
                 suppress_n: int = 50,
                 bootstrap_iters: int = 500,
                 pv_draws: int = 5,
                 random_state: Optional[int] = None) -> Dict:
    """High-level pipeline returning composites for three weighting schemes, bootstrap CIs, and optional Atkinson EDEs.
    Returns a dict with:
      - scores_df: dataframe with domain z and S columns and composite columns for each scheme
      - weights: dict of weight arrays
      - bootstrap: dict of bootstrap mean arrays
      - ede: dict of Atkinson/EDE summaries for epsilons
    """
    n = len(df)
    # optional: introduce band-specific shifts to Motivation for added variation (acts on a copy)
    if "grade_band" in df.columns and "Motivation" in df.columns:
        df = df.copy()
        offsets = {"K-2": 0.5, "3-5": 0.2, "6-8": -0.3, "9-12": -0.7}
        for band, off in offsets.items():
            mask = df["grade_band"] == band
            if mask.any():
                df.loc[mask, "Motivation"] = df.loc[mask, "Motivation"].astype(float) + off
    if n < suppress_n:
        raise ValueError(f"Suppressed: sample size n={n} < suppress_n={suppress_n}")
    # compute z and S
    scores = compute_qsi_scores(df, domain_cols, strata_cols)
    S_cols = [f"{c}_S" for c in domain_cols]
    # weights
    w_theory = theory_fixed_weights(domain_cols)
    if outcome_col and outcome_col in df.columns:
        try:
            w_data = data_driven_weights(df, domain_cols, outcome_col)
        except Exception:
            w_data = w_theory
    else:
        w_data = w_theory
    try:
        w_decorr = decorrelated_weights(df, domain_cols)
    except Exception:
        w_decorr = w_theory
    weights = {"theory": w_theory, "data": w_data, "decorrelated": w_decorr}
    # compute composites
    scores = composite_from_weights(scores, S_cols, w_theory, comp_name="QSI_theory")
    scores = composite_from_weights(scores, S_cols, w_data, comp_name="QSI_data")
    scores = composite_from_weights(scores, S_cols, w_decorr, comp_name="QSI_decor")
    # bootstrap uncertainty
    boot = bootstrap_composite(df, domain_cols, sem_cols, strata_cols, outcome_col, n_boot=bootstrap_iters, n_pv=pv_draws, random_state=random_state)
    # compute point estimates and CIs
    summary = {}
    for key, arr in [("theory", boot["theory_means"]), ("data", boot["data_means"]), ("decorrelated", boot["decor_means"])]:
        mean = arr.mean()
        ci = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
        summary[key] = {"mean": float(mean), "95ci": (float(ci[0]), float(ci[1]))}
    # inequality-aware EDEs on shifted scale
    ede = {}
    for scheme in ["QSI_theory", "QSI_data", "QSI_decor"]:
        vals = scores[scheme].to_numpy(dtype=float)
        mean_val = float(np.nanmean(vals))
        edes = {}
        for eps in [0.25, 0.5, 1.0]:
            c = 1e-3  # small positive shift
            X = vals + c
            A = atkinson_index(X, eps)
            EDE = mean_val * (1.0 - A)
            edes[f"eps_{eps}"] = {"Atkinson": float(A), "EDE": float(EDE)}
        ede[scheme] = edes
    return {
        "scores_df": scores,
        "weights": weights,
        "bootstrap_summary": summary,
        "ede": ede,
        "bootstrap_raw": boot,
    }

def grouped_bootstrap_summary(df: pd.DataFrame,
                              domain_cols: List[str],
                              group_cols: List[str],
                              sem_cols: Optional[List[str]] = None,
                              strata_cols: Optional[List[str]] = None,
                              outcome_col: Optional[str] = None,
                              weighting: str = "theory",
                              n_boot: int = 500,
                              n_pv: int = 5,
                              random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Compute bootstrap summaries (mean, 2.5th, 97.5th percentiles) of the composite
    within each group defined by group_cols. Returns a tidy DataFrame with columns
    ['uedi_mean', 'uedi_p2.5', 'uedi_p97.5'] + group_cols.

    weighting: one of "theory", "data", "decorrelated" to select which bootstrap
    results to summarize.
    """
    if weighting not in {"theory", "data", "decorrelated"}:
        raise ValueError("weighting must be one of 'theory', 'data', 'decorrelated'")

    key_map = {"theory": "theory_means", "data": "data_means", "decorrelated": "decor_means"}
    rng = np.random.default_rng(random_state)

    records = []
    groups = df.groupby(group_cols, dropna=False)
    print("Computing QSI (this will take a short while)...")
    for name, group in groups:
        # normalize group name to tuple for consistent unpacking
        tup = name if isinstance(name, tuple) else (name,)
        try:
            boot = bootstrap_composite(
                group,
                domain_cols,
                sem_cols,
                strata_cols,
                outcome_col,
                n_boot=n_boot,
                n_pv=n_pv,
                random_state=rng.integers(0, 2**32 - 1),
            )
            arr = boot[key_map[weighting]]
            rec = {
                "uedi_mean": float(np.nanmean(arr)),
                "uedi_p2.5": float(np.nanpercentile(arr, 2.5)),
                "uedi_p97.5": float(np.nanpercentile(arr, 97.5)),
            }
        except Exception:
            # on failure, fill with NaNs
            rec = {"uedi_mean": np.nan, "uedi_p2.5": np.nan, "uedi_p97.5": np.nan}

        # attach group key values (preserve order of group_cols)
        for col, val in zip(group_cols, tup):
            rec[col] = val
        records.append(rec)

    out_df = pd.DataFrame.from_records(records)
    # desired column order: metrics first, then group columns
    cols = ["uedi_mean", "uedi_p2.5", "uedi_p97.5"] + group_cols
    out_df = out_df[cols]

    # print in a compact, easy-to-read style similar to the example
    # format floats with 6 decimal places
    with pd.option_context("display.float_format", lambda x: f"{x:.6f}"):
        print(out_df.to_string(index=False))

    return out_df

# Example minimal usage (to be removed/adapted when integrating):
# import pandas as pd
# df = pd.read_csv("your_data.csv")
# domain_cols = ["Emotional", "Motivation", "Identity", "Burnout", "Cognitive", "Engagement"]
# sem_cols = ["Emotional_sem", "Motivation_sem", "Identity_sem", "Burnout_sem", "Cognitive_sem", "Engagement_sem"]
# out = qsi_pipeline(df, domain_cols, sem_cols=sem_cols, strata_cols=["grade_band", "language"], outcome_col="GPA")
# print(out["bootstrap_summary"])

if __name__ == "__main__":
    # Minimal runnable demo with synthetic data so the script produces output when executed.
    n = 200
    rng = np.random.default_rng(123)

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

    # generate synthetic domain z-like scores (not already standardized)
    data = {}
    for d in domain_cols:
        data[d] = rng.normal(loc=0.0, scale=1.0, size=n)
    # plausible measurement error standard errors per domain
    for s in sem_cols:
        data[s] = rng.uniform(0.1, 0.4, size=n)
    # outcome related to Motivation and Engagement plus noise
    data["GPA"] = 0.4 * data["Motivation"] + 0.4 * data["Engagement"] + rng.normal(0.0, 1.0, size=n)
    # strata columns
    data["grade_band"] = rng.choice(["K-2", "3-5", "6-8", "9-12"], size=n)
    data["language"] = rng.choice(["EN", "ES"], size=n)

    df_demo = pd.DataFrame(data)

    out = qsi_pipeline(
        df_demo,
        domain_cols,
        sem_cols=sem_cols,
        strata_cols=["grade_band", "language"],
        outcome_col="GPA",
        suppress_n=50,
        bootstrap_iters=100,
        pv_draws=3,
        random_state=123,
    )

    print("Bootstrap summary:")
    print(out["bootstrap_summary"])
    print("\nEDE summary keys:")
    print(list(out["ede"].keys()))
