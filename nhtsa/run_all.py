"""Rebuild every number and chart in the final deck from the raw NHTSA files.

Usage:
    python run_all.py                        
    python run_all.py --fast                   

"""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from src.load import load_complaints, load_recalls
from src.features import build_groups, early_text
from src.warning import warning_by_campaign, summarize
from src.timing import timing_by_route, cumulative_share, waiting_by_entry_year, old_route_chart
from src.models import fit_route, hr_table, plain_percent, ph_check, bootstrap_hr
from src.cases import case_timeline
from src import charts

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FAST = "--fast" in sys.argv
DATA_DIR = r"/Users/oxy/Desktop/MSDS/prac 1/Data"
data = Path(args[0] if args else DATA_DIR); out = Path("output"); out.mkdir(exist_ok=True)
T0 = time.time(); R = {}
step = lambda s: print(f"\n[{(time.time()-T0)/60:5.1f} min] {s}")

step("1. Loading the raw files")
complaints, c_counts = load_complaints([data / "COMPLAINTS_RECEIVED_2010-2014.txt",
                                        data / "COMPLAINTS_RECEIVED_2015-2019.txt"])
recalls, r_counts = load_recalls(data / "FLAT_RCL_POST_2010.txt")
per_year = complaints.groupby(complaints["received"].dt.year).size()
R["data"] = {**c_counts, **r_counts, "per_year": {int(k): int(v) for k, v in per_year.items()}}
print(f"  vehicle complaints with a valid model year: {c_counts['vehicle_complaints_valid_year']:,}")

step("2. Building one row per vehicle-and-part case (first-six-month window)")
g = build_groups(complaints, recalls)
R["data"]["cases"] = int(len(g))
R["data"]["still_waiting_pct"] = round(float((g["route"] == 0).mean() * 100), 1)
print(f"  {len(g):,} cases | company recalls {(g.route==1).sum():,} | NHTSA-pushed {(g.route==2).sum():,} "
      f"| still waiting {R['data']['still_waiting_pct']}%")

step("3. How often owners complained first, per recall campaign (2,000 bootstrap resamples)")
camp = warning_by_campaign(complaints, recalls)
R["warning"] = summarize(camp)
R["warning"]["no_airbags"] = summarize(camp[~camp["airbag"]])
w = R["warning"]
print(f"  overall {w['overall']['rate']}% ({w['overall']['n_prior']} of {w['overall']['n']})")
print(f"  NHTSA-pushed {w['ODI']['rate']}% [{w['ODI']['lo']}, {w['ODI']['hi']}] | "
      f"company {w['MFR']['rate']}% [{w['MFR']['lo']}, {w['MFR']['hi']}] | gap {w['gap']['mean']} [{w['gap']['lo']}, {w['gap']['hi']}]")
print(f"  without air bags: NHTSA {w['no_airbags']['ODI']['rate']}% | company {w['no_airbags']['MFR']['rate']}%")

step("4. Timing: months from first complaint to recall")
R["timing"] = timing_by_route(g); R["timing"]["no_airbags"] = timing_by_route(g, drop_airbags=True)
R["timing"]["cdf"] = cumulative_share(g); R["waiting"] = waiting_by_entry_year(g)
t = R["timing"]
print(f"  median: company {t['MFR']['median']} | NHTSA {t['ODI']['median']} (p={t['p']})")
print(f"  without air bags: company {t['no_airbags']['MFR']['median']} | NHTSA {t['no_airbags']['ODI']['median']} (p={t['no_airbags']['p']})")

step("5. Cause-specific Cox models, air bags as a stratum")
R["cox"], R["plain"] = {}, {}
for code, name in [(1, "company"), (2, "nhtsa")]:
    m, _ = fit_route(g, code)
    R["cox"][name] = hr_table(m, (g["route"] == code).sum()); R["plain"][name] = plain_percent(R["cox"][name])
    rows = R["cox"][name]["rows"]
    print(f"  {name:8s} concordance {R['cox'][name]['concordance']} | " + " | ".join(
        f"{k} {v['hr']} (p={v['p']})" for k, v in rows.items()))
    print(f"           plain: {R['plain'][name]}")

step("6. Proportional-hazards check (unstratified models)")
R["ph_check"] = {name: ph_check(g, code) for code, name in [(1, "company"), (2, "nhtsa")]}
for name, res in R["ph_check"].items():
    print(f"  {name}: flagged at p<0.01 -> {[k for k, p in res.items() if p < 0.01] or 'none'}")

step("7. Kia Soul case study")
R["kia"] = case_timeline(complaints, recalls, "KIA|SOUL|2012|ENGINE")
print(f"  {R['kia']['complaints_before_recall']} complaints, {R['kia']['first_complaint']} to recall "
      f"{R['kia']['recall_date']} ({R['kia']['months']} months), campaign {R['kia']['campaign']}, {R['kia']['route']}")

step("8. The old Week 3 chart, for appendix A10")
R["old_chart"] = old_route_chart(g)

if not FAST:
    step("9. Bootstrap the hazard ratios (150 resamples per route, the slowest step)")
    R["bootstrap_hr"] = {name: bootstrap_hr(g, code) for code, name in [(1, "company"), (2, "nhtsa")]}
    for name, b in R["bootstrap_hr"].items(): print(f"  {name}: {b}")

    step("10. The classifier I tested and set aside (appendix A4)")
    from src.classify import compare
    R["classifier"] = compare(g, early_text(complaints))
    c = R["classifier"]
    print(f"  train {c['train']:,} | test {c['test']:,} | base rate {c['base']}")
    for mdl in c["models"]: print(f"  {mdl['name']:28s} PR-AUC {mdl['pr_auc']}  ({mdl['lift']}x)  ROC {mdl['roc_auc']}")
    print(f"  top text terms: {', '.join(c['top_text_terms'])}")
    for name, t in c["triage"].items(): print(f"  triage ({name}): top 2,000 cases catch {t['2000']['caught']} of {t['2000']['of']} recalls")

step("12. Two checks for the final deck: late complaints, and a backtest of the monthly list")
from src.checks import late_complaints, monthly_list
R["checks"] = {"late_complaints": late_complaints(complaints, recalls),
               "monthly_list": monthly_list(complaints, recalls)}
for n, s in R["checks"]["late_complaints"].items():
    print(f"  complaint at least {n:>2} months before recall: NHTSA {s['ODI']['rate']}% | company {s['MFR']['rate']}%")
for x, s in R["checks"]["monthly_list"].items():
    print(f"  list at {x:>2} complaints in 6 months: {s['per_month']} a month, {s['hit_rate']}% recalled "
          f"({s['lift']}x), median lead {s['median_lead_months']} months")

step("11. Charts")
charts.complaints_per_year(R["data"]["per_year"], out / "slide2_complaints_per_year.png")
charts.case_study(R["kia"], out / "slide4_kia_soul.png")
charts.warning_rate(R["warning"], out / "slide5_warning_rate.png")
charts.years_waited(R["waiting"], out / "slide6_years_waited.png")
charts.speed_factors(R["plain"], out / "slide7_speed_factors.png")
charts.recall_timing(R["timing"]["cdf"], R["timing"], out / "slide8_recall_timing.png")
charts.old_route_chart_png(R["old_chart"], out / "appendixA10_old_chart.png")

(out / "results.json").write_text(json.dumps(R, indent=1))
step(f"Done. Wrote output/results.json and {len(list(out.glob('*.png')))} charts.")
