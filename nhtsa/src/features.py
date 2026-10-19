"""One row per make | model | year | component ("group" or "case").

Every feature comes from complaints in the FIRST SIX MONTHS after the
group's first complaint. A fixed window stops features from growing with
the outcome: a group that took 40 months to recall had 40 months to
collect complaints, so using every pre-recall complaint would leak the answer.
"""
import numpy as np
import pandas as pd

WINDOW_MONTHS = 6
CENSOR_DATE = pd.Timestamp("2026-08-28")   # last recall date in the file


def early_window(complaints):
    c = complaints.copy()
    c["first"] = c.groupby("KEY")["received"].transform("min")
    c["months_in"] = (c["received"] - c["first"]).dt.days / 30.44
    return c[c["months_in"] <= WINDOW_MONTHS]


def build_groups(complaints, recalls):
    early = early_window(complaints)
    num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
    g = early.groupby("KEY").agg(
        first=("first", "first"),
        n_early=("CMPLID", "size"),
        crash=("CRASH", lambda s: (s == "Y").mean()),
        fire=("FIRE", lambda s: (s == "Y").mean()),
        injury=("INJURED", lambda s: (num(s) > 0).mean()),
        death=("DEATHS", lambda s: float((num(s) > 0).max())),
        narr_len=("narr_len", "mean"),
        model_year=("yr", "first"),
        make=("make_up", "first"),
        top_comp=("top_comp", "first"),
    )
    g["age"] = (g["first"].dt.year - g["model_year"]).clip(-2, 25)

    # first recall on each group, and who drove it
    first_rcl = (recalls.sort_values("recall_date").drop_duplicates("KEY")
                 .set_index("KEY")[["recall_date", "INFLUENCED_BY", "CAMPNO"]])
    g = g.join(first_rcl)

    g["t_event"] = (g["recall_date"] - g["first"]).dt.days / 30.44
    g["recalled"] = g["recall_date"].notna() & (g["t_event"] > 0)
    # route: 0 = no recall yet (still waiting), 1 = company, 2 = NHTSA pushed it (ODI)
    g["route"] = np.where(~g["recalled"], 0, np.where(g["INFLUENCED_BY"] == "ODI", 2, 1))
    g["dur"] = np.where(g["recalled"], g["t_event"], (CENSOR_DATE - g["first"]).dt.days / 30.44)
    g = g[g["dur"] > 0.5].copy()
    g["recalled_36mo"] = (g["recalled"] & (g["t_event"] <= 36)).astype(int)
    return g


def early_text(complaints, max_chars=6000):
    """All early-window narratives for each group, joined, for the text model."""
    early = early_window(complaints)
    return early.groupby("KEY")["CDESCR"].apply(lambda s: " ".join(s.fillna("nan").astype(str))[:max_chars])
