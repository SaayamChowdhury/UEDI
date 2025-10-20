from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import zscore, norm
import itertools

"""
uedi.py

Compute a Universal Education Development Index (UEDI) for jurisdiction x academic-year
based on a configurable set of indicators and minimal metadata.

This is a practical, self-contained implementation approximating the methodology
you provided:
- indicator harmonization (logit for proportions, log for monetary, z-scoring)
- domain factor extraction (first principal component per domain)
- equity adjustment (Atkinson index from grouped shares + group gap penalties)
- aggregation via geometric mean
- uncertainty via nonparametric bootstrap

Inputs:
- indicator_df: pandas.DataFrame with columns
    ['jurisdiction','year','domain','indicator','value']
- group_dist_df (optional): DataFrame with columns
    ['jurisdiction','year','group','outcome_value','share']  (for Atkinson)
- group_gaps_df (optional): DataFrame with columns
    ['jurisdiction','year','group_dimension','gap']  (signed gaps Δg,jt)
- metadata (optional): dict mapping indicator -> {'type': 'prop'|'monetary'|'scale'|'within_system',
                                                'winsorize': True|False}
- weights (optional): dict mapping domain -> weight. If None, equal weights used.
- n_boot (optional): number of bootstrap draws for uncertainty (default 500)

Outputs:
- DataFrame per jurisdiction-year with columns:
    ['jurisdiction','year','uedi_mean','uedi_p2.5','uedi_p97.5',
     'uedi_samples' (list of samples), 'domain_scores' (dict of mean domain scores)]

Note: This is a pragmatic implementation for aggregated inputs. For full psychometric/Bayesian
specifications you'd replace PCA-with-projection by hierarchical Bayesian factor models
and MCMC sampling (e.g., via PyMC or Stan).
"""


# ---------- utilities ----------

def safe_logit(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def winsorize_series(s: pd.Series, p=0.01):
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lo, hi)

def atkinson_index_from_grouped(values: np.ndarray, shares: np.ndarray, eta: float = 0.5):
    """
    Compute Atkinson index from grouped outcomes (values) and population shares.
    values: outcome per group (non-negative, e.g., incomes or attainment)
    shares: population shares summing to 1
    eta: inequality aversion (>0, !=1). If eta==1 use limit form.
    Returns Atkinson index in [0,1).
    """
    values = np.asarray(values, dtype=float)
    shares = np.asarray(shares, dtype=float)
    # ensure non-negative and positive mean
    values = np.maximum(values, 0.0)
    shares = shares / shares.sum()
    mu = np.dot(shares, values)
    if mu <= 0:
        return 0.0
    if np.isclose(eta, 1.0):
        # limit case
        g = np.exp(np.dot(shares, np.log(np.maximum(values, 1e-12))) - np.log(mu))
        return 1.0 - g
    else:
        term = np.dot(shares, (values / mu) ** (1 - eta))
        A = 1.0 - term ** (1 / (1 - eta))
        return float(np.clip(A, 0.0, 1.0 - 1e-12))

def to_0_100_by_normal_cdf(arr: np.ndarray):
    """Map arbritrary scores to 0-100 by z-score -> standard normal CDF -> *100"""
    z = zscore(arr, nan_policy='omit')
    # zscore can produce nan if constant; handle:
    if np.all(np.isnan(z)):
        # return 50 for all
        return np.full_like(arr, 50.0, dtype=float)
    z = np.where(np.isnan(z), 0.0, z)
    p = norm.cdf(z)
    return (p * 100.0)

# ---------- core pipeline ----------

def compute_domain_factors(ind_df: pd.DataFrame, metadata: Optional[Dict] = None):
    """
    Input: ind_df rows = observations (jurisdiction-year), columns:
        'jurisdiction','year','domain','indicator','value'
    Returns:
        domain_scores: DataFrame with index ['jurisdiction','year'] and columns for each domain (0-100)
    """
    md = metadata or {}
    # copy
    df = ind_df.copy()

    # apply per-indicator harmonization
    out_rows = []
    for (ind,), grp in df.groupby(['indicator']):
        col = grp['value'].astype(float).copy()
        m = md.get(ind, {})
        typ = m.get('type', 'scale')
        if m.get('winsorize', False):
            col = winsorize_series(col, p=0.01)
        if typ == 'prop':
            # assume in [0,1] or [0,100]
            if col.max() > 1.5:
                col = col / 100.0
            col = safe_logit(np.clip(col, 1e-6, 1 - 1e-6))
        elif typ == 'monetary':
            # log transform
            col = np.log(np.maximum(col, 1e-6))
        elif typ == 'within_system':
            # treat as scale but note: ideally standardized within system; here we leave for z-scoring later
            col = col
        else:
            col = col

        temp = grp[['jurisdiction','year','domain','indicator']].copy()
        temp['hval'] = col.values
        out_rows.append(temp)

    harmonized = pd.concat(out_rows, ignore_index=True)

    # pivot to wide by indicator to run PCA per domain
    harmonized['obs_id'] = harmonized['jurisdiction'].astype(str) + "||" + harmonized['year'].astype(str)
    wide = harmonized.pivot_table(index='obs_id', columns='indicator', values='hval', aggfunc='first')
    # keep mapping of obs_id -> jurisdiction, year
    obs_map = harmonized[['obs_id','jurisdiction','year']].drop_duplicates().set_index('obs_id')

    # per-domain factor extraction: for each domain, take its indicators and compute first PC score
    domain_scores = {}
    for domain, dom_grp in harmonized.groupby('domain'):
        inds = dom_grp['indicator'].unique().tolist()
        sub = wide[inds].copy()
        # simple mean-impute missing cells for PCA input
        sub = sub.fillna(sub.mean(axis=0))
        # center
        X = sub.values
        # subtract column means
        Xc = X - np.nanmean(X, axis=0, keepdims=True)
        # SVD for first PC
        try:
            U, s, VT = np.linalg.svd(Xc, full_matrices=False)
            pc1 = U[:, 0] * s[0]  # first principal component scores (relative scale)
        except Exception:
            # fallback: column-average
            pc1 = np.nanmean(X, axis=1)
        # map pc1 to 0-100
        mapped = to_0_100_by_normal_cdf(pc1)
        domain_scores[domain] = pd.Series(mapped, index=sub.index)

    # assemble DataFrame
    ds = pd.DataFrame(domain_scores)
    ds = ds.join(obs_map)
    ds = ds.reset_index().set_index(['jurisdiction','year']).drop(columns=['obs_id'], errors='ignore')
    # ensure all seven domains present (missing domains -> NaN)
    return ds

def apply_equity_adjustment(domain_df: pd.DataFrame,
                            group_dist_df: Optional[pd.DataFrame],
                            group_gaps_df: Optional[pd.DataFrame],
                            eta: float = 0.5):
    """
    domain_df: index ['jurisdiction','year'] columns = domains with values in 0-100
    group_dist_df: optional DataFrame with columns ['jurisdiction','year','group','outcome_value','share']
        used to compute Atkinson Ajt(eta)
    group_gaps_df: optional DataFrame with columns ['jurisdiction','year','group_dimension','gap']
        gap is signed difference Δg,jt in same 0-100 units (e.g., male-female gap)
    Returns:
        adjusted_df: same shape as domain_df with Access & Learning adjusted
        also returns penalties Pjt as a DataFrame
    """
    domains = domain_df.columns.tolist()
    idx = domain_df.index
    P = pd.Series(1.0, index=idx, name='Pjt')

    # compute Atkinson if grouped distribution available
    A = pd.Series(0.0, index=idx, name='Atkinson')
    if group_dist_df is not None:
        g = group_dist_df.copy()
        # iterate per jurisdiction-year
        for (jur, yr), grp in g.groupby(['jurisdiction','year']):
            values = grp['outcome_value'].values.astype(float)
            shares = grp['share'].values.astype(float)
            shares = shares / shares.sum()
            a = atkinson_index_from_grouped(values, shares, eta=eta)
            if (jur, yr) in A.index:
                A.loc[(jur, yr)] = a
    # compute group gap penalties
    # estimate s_g per group dimension as global std of gap values if not provided
    penalties = pd.DataFrame(1.0, index=idx, columns=[])  # will add columns for each dimension
    if group_gaps_df is not None:
        gg = group_gaps_df.copy()
        # pivot to (jur,year) x group_dimension
        pivot = gg.pivot_table(index=['jurisdiction','year'], columns='group_dimension', values='gap', aggfunc='first')
        # compute s_g as std across all non-na gaps for each dimension
        s_g = pivot.std(axis=0, skipna=True).replace(0, np.nan).fillna(pivot.abs().mean(axis=0).replace(0, 1.0))
        # compute p_g,jt = exp(-|Δg| / s_g)
        pen = pd.DataFrame(index=idx, columns=pivot.columns, dtype=float)
        for col in pivot.columns:
            sg = float(s_g.loc[col]) if col in s_g.index else 1.0
            arr = pivot[col]
            arr = arr.reindex(idx)  # align
            pg = np.exp(-np.abs(arr.fillna(0.0)) / sg)
            pen[col] = pg
        # product across dimensions
        pen_prod = pen.prod(axis=1)
        penalties = pen_prod
    else:
        penalties = pd.Series(1.0, index=idx)

    # composite penalty Pjt = (1 - Ajt) * prod_g p_g,jt
    P = (1.0 - A) * penalties
    P = P.clip(lower=0.0, upper=1.0)

    adjusted = domain_df.copy()
    # domains to adjust:
    for dom in ['Access & Participation', 'Learning & Skills']:
        if dom in adjusted.columns:
            # multiply domain score (0-100) by Pjt
            adjusted[dom] = (adjusted[dom].astype(float).fillna(0.0) * P).astype(float)
    return adjusted, P, A

def geometric_mean_of_domains(domain_row: pd.Series, weights: Optional[Dict] = None, eps=1e-9):
    """
    domain_row: Series of domain scores in [0,100]
    weights: dict domain->weight (sum to 1). If None equal weights used.
    Returns scalar UEDI in [0,100]
    Implementation uses continuous geometric mean: exp(sum w_d * ln S_d)
    To avoid log(0), we clip at small epsilon.
    """
    S = domain_row.astype(float).values
    domains = domain_row.index.tolist()
    if weights is None:
        w = np.repeat(1.0 / len(S), len(S))
    else:
        w = np.array([weights.get(d, 0.0) for d in domains], dtype=float)
        if w.sum() == 0:
            w = np.repeat(1.0 / len(S), len(S))
        else:
            w = w / w.sum()
    S_clipped = np.clip(S, eps, 100.0)
    # geometric mean in original 0-100 scale; to maintain scale, we compute on (S_clipped/100) then *100
    gm = np.exp(np.dot(w, np.log(S_clipped / 100.0))) * 100.0
    return float(gm)

def compute_uedi(indicator_df: pd.DataFrame,
                 metadata: Optional[Dict] = None,
                 group_dist_df: Optional[pd.DataFrame] = None,
                 group_gaps_df: Optional[pd.DataFrame] = None,
                 weights: Optional[Dict] = None,
                 n_boot: int = 500,
                 eta: float = 0.5,
                 random_state: Optional[int] = None):
    """
    Main function to compute UEDI with bootstrap uncertainty.
    Returns a DataFrame indexed by jurisdiction, year with columns uedi_mean, uedi_p2.5, uedi_p97.5,
    and domain mean scores.
    """
    rnd = np.random.RandomState(random_state)
    # baseline domain scores
    base_domain_df = compute_domain_factors(indicator_df, metadata=metadata)
    # do bootstrapped resampling of indicator values to propagate uncertainty
    samples = {}  # key (jur,year) -> list of uedi samples
    domain_samples = {}  # per domain mean across boots
    # prepare grouped list of indicator observations for resampling
    ind_rows = indicator_df.copy()
    ind_rows = ind_rows.reset_index(drop=True)
    n = ind_rows.shape[0]
    for b in range(n_boot):
        # sample rows with replacement at indicator-observation level (nonparametric bootstrap)
        samp_idx = rnd.randint(0, n, size=n)
        samp_df = ind_rows.iloc[samp_idx].reset_index(drop=True)
        # recompute domain factors
        try:
            dom_scores = compute_domain_factors(samp_df, metadata=metadata)
        except Exception:
            # fallback to base
            dom_scores = base_domain_df.copy()
        # equity adjustment
        adj, P, A = apply_equity_adjustment(dom_scores, group_dist_df, group_gaps_df, eta=eta)
        # compute UEDI per observation
        for idx in adj.index:
            row = adj.loc[idx]
            u = geometric_mean_of_domains(row, weights=weights)
            samples.setdefault(idx, []).append(u)
            # store domain sample means
            for d in adj.columns:
                domain_samples.setdefault((idx, d), []).append(float(row[d] if not pd.isna(row[d]) else 0.0))

    # summarize
    out_rows = []
    for idx in base_domain_df.index:
        s = np.array(samples.get(idx, [geometric_mean_of_domains(base_domain_df.loc[idx], weights=weights)]))
        mean = float(s.mean())
        p2 = float(np.percentile(s, 2.5))
        p97 = float(np.percentile(s, 97.5))
        # compute domain means across boots
        dom_means = {}
        for d in base_domain_df.columns:
            arr = np.array(domain_samples.get((idx, d), [base_domain_df.loc[idx, d]]))
            dom_means[d] = float(np.mean(arr))
        out_rows.append({
            'jurisdiction': idx[0],
            'year': idx[1],
            'uedi_mean': mean,
            'uedi_p2.5': p2,
            'uedi_p97.5': p97,
            'uedi_samples': samples.get(idx, []),
            'domain_scores': dom_means
        })
    out_df = pd.DataFrame(out_rows).set_index(['jurisdiction','year']).sort_index()
    return out_df


# ---------- example usage ----------
if __name__ == "__main__":
    # Minimal runnable example with synthetic data:
    jurisdictions = ['A','B','C']
    years = [2020,2021]
    domains = [
        'Access & Participation','Learning & Skills','Psychological Climate & Student Wellbeing (QSI)',
        'Equity & Inclusion','Resources & Conditions','System Quality & Governance',
        'Transitions & Economic Linkages'
    ]
    # create synthetic indicators: 3 indicators per domain
    rows = []
    for jur, yr, dom in itertools.product(jurisdictions, years, domains):
        for k in range(3):
            ind = f"{dom[:10]}_ind{k+1}"
            val = np.random.rand() * 100
            rows.append({'jurisdiction': jur, 'year': yr, 'domain': dom, 'indicator': ind, 'value': val})
    indicator_df = pd.DataFrame(rows)

    # synthetic grouped distribution for Atkinson (3 groups)
    gd_rows = []
    for jur, yr in itertools.product(jurisdictions, years):
        shares = np.array([0.5, 0.3, 0.2])
        outcomes = np.array([60 + np.random.randn()*5, 50 + np.random.randn()*5, 40 + np.random.randn()*5])
        for g, s, v in zip(['g1','g2','g3'], shares, outcomes):
            gd_rows.append({'jurisdiction': jur, 'year': yr, 'group': g, 'outcome_value': float(max(v,0.1)), 'share': float(s)})
    group_dist_df = pd.DataFrame(gd_rows)

    # synthetic group gaps (dimensions: sex, SES)
    gg_rows = []
    for jur, yr in itertools.product(jurisdictions, years):
        gg_rows.append({'jurisdiction': jur, 'year': yr, 'group_dimension': 'sex', 'gap': float(np.random.randn()*5)})
        gg_rows.append({'jurisdiction': jur, 'year': yr, 'group_dimension': 'SES', 'gap': float(np.random.randn()*8)})
    group_gaps_df = pd.DataFrame(gg_rows)

    print("Computing UEDI (this will take a short while)...")
    res = compute_uedi(indicator_df, group_dist_df=group_dist_df, group_gaps_df=group_gaps_df, n_boot=200, random_state=1)
    print(res[['uedi_mean','uedi_p2.5','uedi_p97.5']])