"""Load the three NHTSA ODI flat files.

The files are tab-delimited with no header row. Column positions were
checked against the data itself, not only the layout documents:
  - column 16 is the date NHTSA received the complaint; its values sit
    exactly inside each file's window (e.g. 2015-01-01 to 2019-12-31)
  - column 7 (failure date) is unusable: values run from year 1015 to 2160
"""
import csv
import pandas as pd

COMPLAINT_COLS = {0: "CMPLID", 3: "MAKE", 4: "MODEL", 5: "YEAR", 6: "CRASH",
                  8: "FIRE", 9: "INJURED", 10: "DEATHS", 11: "COMPDESC",
                  16: "LDATE", 19: "CDESCR", 45: "PROD_TYPE"}
RECALL_COLS = {1: "CAMPNO", 2: "MAKE", 3: "MODEL", 4: "YEAR", 6: "COMPNAME",
               10: "RCLTYPECD", 13: "INFLUENCED_BY", 15: "RCDATE", 18: "FMVSS"}


def _read(path, cols):
    df = pd.read_csv(path, sep="\t", header=None, encoding="latin-1",
                     quoting=csv.QUOTE_NONE, usecols=sorted(cols),
                     low_memory=False, on_bad_lines="skip", dtype=str)
    df.columns = [cols[i] for i in sorted(cols)]
    return df


def make_key(df, comp_col):
    """make | model | model year | component, upper-cased."""
    # fillna("nan") keeps pandas 2 and 3 identical: pandas 2 turned a missing value
    # into the text "nan"; pandas 3 keeps it missing, and the whole key drops out
    up = lambda s: s.fillna("nan").astype(str).str.strip().str.upper()
    yr = pd.to_numeric(df["YEAR"], errors="coerce")
    return (up(df["MAKE"]) + "|" + up(df["MODEL"]) + "|"
            + yr.astype("Int64").astype(str).fillna("<NA>") + "|" + up(df[comp_col]))


def load_complaints(paths):
    parts = []
    for p in paths:
        d = _read(p, COMPLAINT_COLS)
        print(f"  read {len(d):,} rows from {p.name}")
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    df = raw[raw["PROD_TYPE"] == "V"].copy()                  # vehicles only
    df["received"] = pd.to_datetime(df["LDATE"], format="%Y%m%d", errors="coerce")
    df["yr"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df[df["yr"].between(1980, 2021)].copy()               # drops 9999 and the like
    df["KEY"] = make_key(df, "COMPDESC")
    df["top_comp"] = df["COMPDESC"].fillna("nan").astype(str).str.upper().str.split(":").str[0].str.strip()
    df["make_up"] = df["MAKE"].fillna("nan").astype(str).str.strip().str.upper()
    df["narr_len"] = df["CDESCR"].fillna("nan").astype(str).str.len()
    counts = {"complaint_rows": int(len(raw)), "vehicle_complaints": int((raw["PROD_TYPE"] == "V").sum()),
              "vehicle_complaints_valid_year": int(len(df))}
    return df, counts


def load_recalls(path):
    raw = _read(path, RECALL_COLS)
    print(f"  read {len(raw):,} rows from {path.name}")
    df = raw[raw["RCLTYPECD"] == "V"].copy()
    df["recall_date"] = pd.to_datetime(df["RCDATE"], format="%Y%m%d", errors="coerce")
    df["KEY"] = make_key(df, "COMPNAME")
    df = df.dropna(subset=["recall_date"])
    return df, {"recall_rows": int(len(raw)), "vehicle_recall_rows": int(len(df))}
