"""The two checks added for the final presentation (appendix A6 and A7).

Usage:
    python run_checks.py            (reads from DATA_DIR below)

Writes output/checks.json. run_all.py also runs these as step 12.
"""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from src.load import load_complaints, load_recalls
from src.checks import late_complaints, monthly_list

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA_DIR = r"/Users/oxy/Desktop/MSDS/prac 1/Data"
data = Path(args[0] if args else DATA_DIR); out = Path("output"); out.mkdir(exist_ok=True)
T0 = time.time()
complaints, _ = load_complaints([data / "COMPLAINTS_RECEIVED_2010-2014.txt",
                                 data / "COMPLAINTS_RECEIVED_2015-2019.txt"])
recalls, _ = load_recalls(data / "FLAT_RCL_POST_2010.txt")

R = {"late_complaints": late_complaints(complaints, recalls),
     "monthly_list": monthly_list(complaints, recalls)}
for n, s in R["late_complaints"].items():
    print(f"  buffer {n:>2} months: NHTSA {s['ODI']['rate']}% [{s['ODI']['lo']}, {s['ODI']['hi']}] | "
          f"company {s['MFR']['rate']}% [{s['MFR']['lo']}, {s['MFR']['hi']}] | gap {s['gap']['mean']} [{s['gap']['lo']}, {s['gap']['hi']}]")
for x, s in R["monthly_list"].items():
    print(f"  list at {x:>2} complaints in 6 months: {s}")
(out / "checks.json").write_text(json.dumps(R, indent=1))
print(f"done in {(time.time()-T0)/60:.1f} min")
