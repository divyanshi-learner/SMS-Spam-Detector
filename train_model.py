"""
Train, compare and save the SMS spam model (v2).

Works in a Kaggle notebook (paste into a cell, or %run) and locally.
Outputs: spam_model.pkl, vectorizer.pkl, metrics.json  (+ PNG charts for your report)

Local use:  python train_model.py path/to/spam.csv
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = 42

# ---------- locate data ----------
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    DATA = sys.argv[1]
elif os.path.exists("spam.csv"):
    DATA = "spam.csv"
else:  # Kaggle: find spam.csv anywhere under /kaggle/input
    DATA = next((os.path.join(r, f) for r, _, fs in os.walk("/kaggle/input") for f in fs if f == "spam.csv"), None)
assert DATA, "spam.csv not found. Pass its path: python train_model.py path/to/spam.csv"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ---------- load & prepare ----------
df = pd.read_csv(DATA, encoding="latin-1")[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
df = df.dropna()
y = (df["label"] == "spam").astype(int).values
X = np.array(df["message"].astype(str).tolist(), dtype=object)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train {len(X_tr)} | Test {len(X_te)} | spam share {y.mean():.1%}")


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)


CANDIDATES = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", C=5),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
}
EXPLAINABLE = {"Naive Bayes", "Logistic Regression", "Logistic Regression (balanced)"}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def metrics_at(y_true, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return dict(threshold=round(float(thr), 2), accuracy=accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred), f1=f1_score(y_true, pred),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


results, fitted = {}, {}
for name, clf in CANDIDATES.items():
    pipe = make_pipeline(tfidf(), clf)
    # out-of-fold probabilities on TRAIN only -> used to choose the threshold (no test leakage)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thr = max(thresholds, key=lambda t: f1_score(y_tr, (oof >= t).astype(int)))
    cv_f1 = f1_score(y_tr, (oof >= best_thr).astype(int))
    pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    results[name] = dict(
        cv_f1_at_best_threshold=cv_f1,
        chosen_threshold=round(float(best_thr), 2),
        test_at_0_5=metrics_at(y_te, p_te, 0.5),
        test_at_chosen=metrics_at(y_te, p_te, best_thr),
        roc_auc=roc_auc_score(y_te, p_te),
        pr_auc=average_precision_score(y_te, p_te),
    )
    fitted[name] = (pipe, p_te)
    r5, rc = results[name]["test_at_0_5"], results[name]["test_at_chosen"]
    print(f"{name:32s} | @0.5: P={r5['precision']:.3f} R={r5['recall']:.3f} F1={r5['f1']:.3f} "
          f"| @{best_thr:.2f}: P={rc['precision']:.3f} R={rc['recall']:.3f} F1={rc['f1']:.3f}")

# ---------- pick the winner using CROSS-VALIDATION only (not the test set) ----------
pool = [n for n in results if n in EXPLAINABLE]
best = max(pool, key=lambda n: results[n]["cv_f1_at_best_threshold"])
print("\nSelected model (best CV F1, explainable):", best)

pipe, p_te = fitted[best]
vec, clf = pipe.named_steps["tfidfvectorizer"], pipe.steps[-1][1]

# threshold table for the report / app chart
thr_table = [metrics_at(y_te, p_te, t) for t in np.arange(0.1, 0.91, 0.05)]

# top spam / ham words
names = np.array(vec.get_feature_names_out())
w = (clf.feature_log_prob_[1] - clf.feature_log_prob_[0]) if hasattr(clf, "feature_log_prob_") else clf.coef_[0]
top_spam = names[np.argsort(w)[::-1][:15]].tolist()
top_ham = names[np.argsort(w)[:15]].tolist()

summary = dict(
    sklearn_version=sklearn.__version__, n_total=int(len(df)), n_train=int(len(X_tr)), n_test=int(len(X_te)),
    n_spam=int(y.sum()), n_ham=int((1 - y).sum()), selected_model=best,
    default_threshold=results[best]["chosen_threshold"], models=results,
    selected_threshold_table=thr_table, top_spam_words=top_spam, top_ham_words=top_ham,
)
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)
pickle.dump(clf, open(os.path.join(OUT, "spam_model.pkl"), "wb"))
pickle.dump(vec, open(os.path.join(OUT, "vectorizer.pkl"), "wb"))

# ---------- optional charts for the report ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, (p_te >= results[best]["chosen_threshold"]).astype(int),
                                            display_labels=["Ham", "Spam"], ax=ax[0], colorbar=False)
    ax[0].set_title(f"Confusion matrix ({best}, thr={results[best]['chosen_threshold']})")
    RocCurveDisplay.from_predictions(y_te, p_te, ax=ax[1]); ax[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_te, p_te, ax=ax[2]); ax[2].set_title("Precision-Recall curve")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "evaluation_charts.png"), dpi=150)
except Exception as e:
    print("Charts skipped:", e)

print("\nSaved: spam_model.pkl, vectorizer.pkl, metrics.json, evaluation_charts.png in", os.path.abspath(OUT))
