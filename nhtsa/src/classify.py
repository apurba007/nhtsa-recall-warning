"""The alternative I tested and set aside: a yes/no classifier.

Target: recalled within 36 months of the first complaint. Temporal split:
train on cases first complained about before 2017, test on 2017-2019.
Scored by average precision (PR-AUC) against the base rate, because
accuracy is meaningless when about 1% of cases are positive.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def structured(g):
    X = pd.DataFrame({"log_n": np.log1p(g["n_early"]), "crash": g["crash"], "fire": g["fire"],
                      "injury": g["injury"], "death": g["death"], "log_narr": np.log1p(g["narr_len"]),
                      "age": g["age"]}, index=g.index)
    for c in g["top_comp"].value_counts().head(10).index:
        X["C_" + c[:16]] = (g["top_comp"] == c).astype(int)
    for mk in g["make"].value_counts().head(12).index:
        X["M_" + mk[:12]] = (g["make"] == mk).astype(int)
    return X


def compare(g, text):
    X = structured(g); y = g["recalled_36mo"].values
    tr = (g["first"] < "2017-01-01").values; te = ~tr
    base = float(y[te].mean())
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    scores = {}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs[tr], y[tr])
    scores["Logistic regression"] = lr.predict_proba(Xs[te])[:, 1]
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight="balanced",
                                        random_state=0).fit(X[tr], y[tr])
    scores["Gradient boosting"] = gb.predict_proba(X[te])[:, 1]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=0).fit(X[tr], y[tr])
    scores["Random forest"] = rf.predict_proba(X[te])[:, 1]

    t = text.reindex(g.index).fillna("")
    tv = TfidfVectorizer(max_features=3000, min_df=20, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    Ttr = tv.fit_transform(t[tr]); Tte = tv.transform(t[te])
    lt = LogisticRegression(max_iter=1500, class_weight="balanced").fit(Ttr, y[tr])
    scores["Complaint text (TF-IDF)"] = lt.predict_proba(Tte)[:, 1]
    Atr = sp.hstack([sp.csr_matrix(Xs[tr]), Ttr]).tocsr(); Ate = sp.hstack([sp.csr_matrix(Xs[te]), Tte]).tocsr()
    la = LogisticRegression(max_iter=2500, class_weight="balanced").fit(Atr, y[tr])
    scores["Text + structured features"] = la.predict_proba(Ate)[:, 1]

    models = []
    for name, s in scores.items():
        ap = average_precision_score(y[te], s)
        models.append({"name": name, "pr_auc": round(float(ap), 4), "roc_auc": round(float(roc_auc_score(y[te], s)), 3),
                       "lift": round(float(ap / base), 1)})
    triage = {}                       # review the top k cases: how many eventual recalls are caught?
    for name in ("Logistic regression", "Gradient boosting"):
        hits = y[te][np.argsort(-scores[name])]
        triage[name] = {str(k): {"caught": int(hits[:k].sum()), "of": int(y[te].sum())} for k in (500, 1000, 2000)}
    terms = pd.Series(lt.coef_[0], index=tv.get_feature_names_out()).nlargest(12).index.tolist()
    return {"train": int(tr.sum()), "test": int(te.sum()), "base": round(base, 4),
            "models": models, "triage": triage, "top_text_terms": terms}
