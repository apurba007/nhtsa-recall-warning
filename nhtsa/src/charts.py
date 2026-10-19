"""PNG versions of every chart in the deck. Headlines are the claims each chart supports."""
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, NAVY, SLATE = "#c00000", "#1f4e79", "#7f97b2"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path, title):
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.86]); fig.savefig(path, dpi=150); plt.close(fig)


def complaints_per_year(per_year, path):
    fig, ax = plt.subplots(figsize=(7, 3.7)); y = list(per_year); v = list(per_year.values())
    ax.bar(y, v, color=SLATE, width=0.65)
    for a, b in zip(y, v): ax.text(a, b + 1500, f"{b/1000:.0f}k", ha="center", fontsize=9)
    ax.set_ylabel("Vehicle complaints received"); ax.set_xticks(y)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _save(fig, path, f"Owners send NHTSA about {np.mean(v)/1000:.0f},000 vehicle safety complaints a year")


def warning_rate(w, path):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    rows = [("NHTSA pushed it", w["ODI"], RED), ("Company decided on its own", w["MFR"], NAVY)]
    for i, (lab, r, col) in enumerate(rows):
        ax.bar(i, r["rate"], color=col, width=0.55)
        ax.errorbar(i, r["rate"], yerr=[[r["rate"] - r["lo"]], [r["hi"] - r["rate"]]], color="#333", capsize=6)
        ax.text(i, r["hi"] + 1.2, f"{r['rate']:.0f}%", ha="center", fontsize=14, fontweight="bold", color=col)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"{l}\n({r['n']:,} recalls)" for l, r, _ in rows])
    ax.set_ylim(0, 30); ax.set_yticks([0, 10, 20, 30]); ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.set_ylabel("Recalls where owners\ncomplained first")
    _save(fig, path, "That warning shows up most on the recalls NHTSA has to force \u2014 owners\n"
                     "complained first on 1 in 5 of those, and 1 in 14 that companies started")


def case_study(k, path):
    fig, ax = plt.subplots(figsize=(7, 3.7)); ax.spines["left"].set_visible(False)
    ax.bar(k["years"], k["counts"], color=RED, width=0.62)
    for a, b in zip(k["years"], k["counts"]): ax.text(a, b + 0.15, str(b), ha="center")
    ax.set_yticks([]); ax.set_ylabel("Owner complaints per year"); ax.set_xticks(k["years"])
    _save(fig, path, f"2012 Kia Soul engine: {k['complaints_before_recall']} owners complained over "
                     f"{k['months']/12:.1f} years\nbefore the NHTSA-pushed recall ({k['recall_date']})")


def years_waited(wt, path):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    ax.bar(wt["years"], wt["median_years"], color=SLATE, width=0.65)
    for a, b in zip(wt["years"], wt["median_years"]): ax.text(a, b + 0.25, f"{b:.0f}", ha="center", fontsize=9)
    ax.set_xticks(wt["years"]); ax.set_ylabel("Years waited so far,\nno recall yet")
    ax.set_xlabel("Year of the first owner complaint")
    _save(fig, path, "Most cases aren\u2019t \u201cno recall\u201d \u2014 they\u2019re \u201cnot yet\u201d")


def speed_factors(plain, path):
    labs = ["Early complaints double", "Vehicle 5 years older", "Crash or fire reported"]
    keys = ["double_complaints", "vehicle_5yrs_older", "crash_or_fire"]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for off, route, col, lab in [(-w/2, "company", NAVY, "Company decided on its own"), (w/2, "nhtsa", RED, "NHTSA pushed it")]:
        v = [plain[route][k] for k in keys]; ax.bar(x + off, v, w, color=col, label=lab)
        for xi, vi in zip(x + off, v): ax.text(xi, vi + (0.4 if vi >= 0 else -1.0), f"{vi:+.1f}%", ha="center", fontsize=9)
    ax.axhline(0, color="#666", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(-9, 10)
    ax.set_ylabel("Change in how fast\na recall comes (%)"); ax.legend(frameon=False, fontsize=9)
    _save(fig, path, "More early complaints mean a faster recall on both routes \u2014\nbut carmakers move slower on older vehicles")


def recall_timing(cdf, tm, path):
    fig, ax = plt.subplots(figsize=(7, 3.7)); yrs = np.arange(121) / 12
    for name, col, lab in [("MFR", NAVY, "Company decided on its own"), ("ODI", RED, "NHTSA pushed it")]:
        ax.plot(yrs, cdf[name], color=col, lw=2.4, label=f"{lab} ({tm[name]['n']:,} recalls)")
    ax.axhline(50, color="#aaa", ls="--", lw=0.8)
    ax.set_xlim(0, 10); ax.set_ylim(0, 100); ax.set_ylabel("Share of that route\u2019s recalls (%)")
    ax.set_xlabel("Years from the first owner complaint to the recall"); ax.legend(frameon=False, fontsize=9)
    _save(fig, path, "Once a recall comes, it comes about two years after the first complaint\n\u2014 whichever route it takes")


def old_route_chart_png(old, path):
    fig, ax = plt.subplots(figsize=(7, 3.7)); yrs = np.arange(121) / 12
    ax.plot(yrs, old["MFR"], color=NAVY, lw=2.2, label="Company decided on its own")
    ax.plot(yrs, old["ODI"], color=RED, lw=2.2, label="NHTSA pushed it")
    ax.set_ylabel("Share of ALL cases recalled (%)"); ax.set_xlabel("Years since the first complaint")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path, "Appendix A10: the old chart showed how many recalls\neach route had, not how fast they came")
