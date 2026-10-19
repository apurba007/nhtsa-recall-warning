"""Two checks added for the final presentation.

1. late_complaints(): is the 19% warning rate on NHTSA-pushed recalls just
   owners reacting to an investigation or news story that came shortly
   before the recall? The files carry no investigation-open date, so this
   uses a buffer: a complaint only counts as a warning if it arrived at
   least N months before the recall. If the 19% were mostly reaction
   complaints, it would collapse as N grows.

2. monthly_list(): a backtest of the recommended next step. Each month,
   put a vehicle-and-part on the list the first time it collects X
   complaints inside six months. Then look 36 months ahead: how often did
   a listed vehicle get a defect recall, how much earlier than the recall
   was it listed, and how long is the list each month?
"""
import numpy as np
import pandas as pd

from src.warning import summarize

MONTH = pd.Timedelta(days=30.44)


def late_complaints(complaints, recalls, buffers=(0, 3, 6, 12, 24), start=2016, end=2019, n_boot=2000):
    c = complaints[["KEY", "received"]].dropna().sort_values("received")
    r = recalls[recalls["recall_date"].dt.year.between(start, end)
                & recalls["FMVSS"].isna()].drop_duplicates(["CAMPNO", "KEY"]).copy()
    r["airbag"] = r["KEY"].str.split("|").str[3].str.startswith("AIR BAGS")
    first = c.groupby("KEY")["received"].min()
    out = {}
    for n in buffers:
        cutoff = r["recall_date"] - n * MONTH
        r["prior"] = r["KEY"].map(first) < cutoff
        camp = r.groupby("CAMPNO").agg(route=("INFLUENCED_BY", "first"),
                                       prior=("prior", "max"), airbag=("airbag", "max"))
        camp = camp[camp["route"].isin(["MFR", "ODI"])]
        out[str(n)] = summarize(camp, n_boot=n_boot)
    return out


def _flag_dates(c, x, window_days):
    """First date each case reaches x complaints inside a rolling window."""
    if x == 1:
        return c.groupby("KEY")["received"].min()
    back = c.groupby("KEY")["received"].shift(x - 1)
    hit = (c["received"] - back).dt.days <= window_days
    return c[hit].groupby("KEY")["received"].min()


def monthly_list(complaints, recalls, thresholds=(1, 3, 5, 10), window_days=182, horizon=36,
                 start="2015-01-01", end="2019-12-31"):
    c = complaints[["KEY", "received"]].dropna().sort_values(["KEY", "received"])
    defect = recalls[recalls["FMVSS"].isna() & recalls["INFLUENCED_BY"].isin(["MFR", "ODI"])]
    rd = defect[["KEY", "recall_date", "INFLUENCED_BY"]].drop_duplicates()
    months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44
    out = {}
    for x in thresholds:
        f = _flag_dates(c, x, window_days)
        f = f[(f >= start) & (f <= end)].rename("flag").reset_index()
        m = f.merge(rd, on="KEY", how="left")
        already = m[m["recall_date"] <= m["flag"]]["KEY"].unique()
        m = m[~m["KEY"].isin(already)]
        f = f[~f["KEY"].isin(already)]
        ahead = m[(m["recall_date"] > m["flag"]) & (m["recall_date"] <= m["flag"] + horizon * MONTH)]
        ahead = ahead.sort_values("recall_date").drop_duplicates("KEY")
        lead = (ahead["recall_date"] - ahead["flag"]).dt.days / 30.44
        out[str(x)] = {"listed": int(len(f)), "per_month": round(len(f) / months, 1),
                       "recalled_36mo": int(len(ahead)),
                       "hit_rate": round(len(ahead) / len(f) * 100, 2),
                       "median_lead_months": round(float(lead.median()), 1),
                       "nhtsa_pushed_share": round(float((ahead["INFLUENCED_BY"] == "ODI").mean() * 100), 1)}
    base = out["1"]["hit_rate"]
    for x in out:
        out[x]["lift"] = round(out[x]["hit_rate"] / base, 1)
    # of the cases that did get a recall within 36 months of their first complaint,
    # how many would each list have caught before the recall?
    all_hits = out["1"]["recalled_36mo"]
    for x in thresholds:
        out[str(x)]["share_of_recalls_caught"] = round(out[str(x)]["recalled_36mo"] / all_hits * 100, 1)
    return out
