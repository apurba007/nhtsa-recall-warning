"""How often owners complained BEFORE a recall, by route.

Counted once per recall campaign: one campaign can cover dozens of vehicle
lines, and counting each line separately lets a few huge campaigns swamp
the rate. Defect recalls only: compliance recalls (FMVSS set) are labels
and paperwork that owners don't complain about.
"""
import numpy as np


def warning_by_campaign(complaints, recalls, start=2016, end=2019):
    first_complaint = complaints.groupby("KEY")["received"].min()
    r = recalls[recalls["recall_date"].dt.year.between(start, end)
                & recalls["FMVSS"].isna()].drop_duplicates(["CAMPNO", "KEY"]).copy()
    r["prior"] = r["KEY"].map(first_complaint) < r["recall_date"]
    r["airbag"] = r["KEY"].str.split("|").str[3].str.startswith("AIR BAGS")
    camp = r.groupby("CAMPNO").agg(route=("INFLUENCED_BY", "first"),
                                   prior=("prior", "max"), airbag=("airbag", "max"))
    return camp[camp["route"].isin(["MFR", "ODI"])]


def _boot_rate(p, rng, n_boot):
    return [rng.choice(p, len(p)).mean() for _ in range(n_boot)]


def summarize(camp, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {"overall": {"n": int(len(camp)), "n_prior": int(camp["prior"].sum()),
                       "rate": round(float(camp["prior"].mean() * 100), 1)}}
    for route in ["ODI", "MFR"]:
        p = camp.loc[camp["route"] == route, "prior"].values.astype(float)
        b = _boot_rate(p, rng, n_boot)
        out[route] = {"n": int(len(p)), "rate": round(p.mean() * 100, 1),
                      "lo": round(np.percentile(b, 2.5) * 100, 1),
                      "hi": round(np.percentile(b, 97.5) * 100, 1)}
    gaps = []
    rng = np.random.default_rng(seed)        # the gap gets its own fixed stream
    for _ in range(n_boot):
        s = camp.sample(len(camp), replace=True, random_state=int(rng.integers(1e9)))
        m = s.groupby("route")["prior"].mean()
        gaps.append((m["ODI"] - m["MFR"]) * 100)
    out["gap"] = {"mean": round(float(np.mean(gaps)), 1), "lo": round(float(np.percentile(gaps, 2.5)), 1),
                  "hi": round(float(np.percentile(gaps, 97.5)), 1)}
    return out
