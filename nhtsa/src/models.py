"""Competing-risks survival models, one per recall route.

Why survival and not a yes/no classifier: about 98.5% of cases have no
recall yet. A classifier calls them all 'no'; most are really 'not yet'.
Why two models: the routes behave differently, and one yes/no label
mixes them together. Each route is a cause-specific Cox model with the
other route treated as censored. Air bags are a separate stratum because
the proportional-hazards check flagged them.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

KEY_VARS = ["complaint_volume_log", "crash_or_fire", "vehicle_age"]


def covariates(g, airbag_as_stratum=True):
    X = pd.DataFrame({
        "complaint_volume_log": np.log1p(g["n_early"]),
        "crash_or_fire": ((g["crash"] > 0) | (g["fire"] > 0)).astype(float),
        "vehicle_age": g["age"],
    }, index=g.index)
    comps = [c for c in g["top_comp"].value_counts().head(6).index if c != "AIR BAGS"]
    for comp in comps:
        X["comp_" + comp.lower().replace(" ", "_")[:14]] = (g["top_comp"] == comp).astype(int)
    X["airbag"] = (g["top_comp"] == "AIR BAGS").astype(int)
    return X


def fit_route(g, route_code, stratify=True):
    d = covariates(g)
    d["dur"] = g["dur"].values
    d["event"] = (g["route"] == route_code).astype(int).values
    m = CoxPHFitter(penalizer=0.1)
    if stratify:
        m.fit(d, duration_col="dur", event_col="event", strata=["airbag"])
    else:
        m.fit(d, duration_col="dur", event_col="event")
    return m, d


def hr_table(m, events):
    s = m.summary
    rows = {v: {"hr": round(float(s.loc[v, "exp(coef)"]), 3),
                "lo": round(float(s.loc[v, "exp(coef) lower 95%"]), 3),
                "hi": round(float(s.loc[v, "exp(coef) upper 95%"]), 3),
                "p": round(float(s.loc[v, "p"]), 4)} for v in KEY_VARS}
    return {"rows": rows, "concordance": round(float(m.concordance_index_), 3), "events": int(events)}


def plain_percent(t):
    """Turn hazard ratios into plain % changes for the slide."""
    r = t["rows"]; ln2 = np.log(2)
    return {"double_complaints": round(float((r["complaint_volume_log"]["hr"] ** ln2 - 1) * 100), 1),
            "vehicle_5yrs_older": round((r["vehicle_age"]["hr"] ** 5 - 1) * 100, 1),
            "crash_or_fire": round((r["crash_or_fire"]["hr"] - 1) * 100, 1)}


def ph_check(g, route_code):
    """Proportional-hazards test on the UNstratified model (the check that flagged air bags)."""
    m, d = fit_route(g, route_code, stratify=False)
    res = proportional_hazard_test(m, d, time_transform="rank")
    return {k: round(float(v), 4) for k, v in res.summary["p"].items()}


def bootstrap_hr(g, route_code, n_boot=150, seed=1):
    rng = np.random.default_rng(seed)
    keep = {"complaint_volume_log": [], "vehicle_age": []}
    for _ in range(n_boot):
        s = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
        try:
            m, _ = fit_route(s, route_code)
            for k in keep:
                keep[k].append(float(m.summary.loc[k, "exp(coef)"]))
        except Exception:
            pass
    return {k: {"mean": round(float(np.mean(v)), 3), "lo": round(float(np.percentile(v, 2.5)), 3),
                "hi": round(float(np.percentile(v, 97.5)), 3), "n": len(v)} for k, v in keep.items()}
