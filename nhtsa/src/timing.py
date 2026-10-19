"""How long recalls take, and how long unrecalled cases have been waiting."""
import numpy as np
from scipy.stats import mannwhitneyu


def timing_by_route(g, drop_airbags=False):
    r = g[g["recalled"]]
    if drop_airbags:
        r = r[r["top_comp"] != "AIR BAGS"]
    a = r.loc[r["route"] == 1, "t_event"]; b = r.loc[r["route"] == 2, "t_event"]
    return {"MFR": {"n": int(len(a)), "median": round(float(a.median()), 1)},
            "ODI": {"n": int(len(b)), "median": round(float(b.median()), 1)},
            "p": round(float(mannwhitneyu(a, b).pvalue), 3)}


def cumulative_share(g, months=120):
    """Share of each route's recalls that had arrived by each month (0..months)."""
    r = g[g["recalled"]]
    out = {}
    for code, name in [(1, "MFR"), (2, "ODI")]:
        t = np.sort(r.loc[r["route"] == code, "t_event"].values)
        out[name] = [round(float((t <= m).mean() * 100), 1) for m in range(months + 1)]
    return out


def waiting_by_entry_year(g):
    """For cases with no recall yet: median years waited, by year of first complaint."""
    g = g.assign(entry=g["first"].dt.year)
    w = g[g["route"] == 0].groupby("entry")["dur"].median() / 12
    share = g.groupby("entry")["route"].apply(lambda s: (s == 0).mean() * 100)
    return {"years": [int(y) for y in w.index], "median_years": [round(float(v), 1) for v in w.values],
            "share_waiting": [round(float(share[y]), 1) for y in w.index]}


def old_route_chart(g, months=120):
    """The Week 3 chart I later replaced: each route as a share of ALL cases.
    Its line height reflects how many recalls each route has, not speed."""
    from lifelines import AalenJohansenFitter
    out = {}
    for code, name in [(1, "MFR"), (2, "ODI")]:
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(g["dur"], g["route"], event_of_interest=code)
        cd = aj.cumulative_density_.iloc[:, 0]
        out[name] = [round(float(cd[cd.index <= m].iloc[-1] * 100), 3) for m in range(months + 1)]
    return out
