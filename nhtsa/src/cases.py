"""Pull one vehicle's complaint history for the case-study slide."""
import pandas as pd


def case_timeline(complaints, recalls, key):
    r = recalls[recalls["KEY"] == key].sort_values("recall_date")
    first_recall = r.iloc[0]
    c = complaints[complaints["KEY"] == key]
    before = c[c["received"] < first_recall["recall_date"]]
    per_year = before.groupby(before["received"].dt.year).size()
    first = c["received"].min()
    return {"key": key, "first_complaint": str(first.date()), "recall_date": str(first_recall["recall_date"].date()),
            "campaign": first_recall["CAMPNO"], "route": first_recall["INFLUENCED_BY"],
            "complaints_before_recall": int(len(before)),
            "months": round((first_recall["recall_date"] - first).days / 30.44, 1),
            "years": [int(y) for y in per_year.index], "counts": [int(v) for v in per_year.values]}
