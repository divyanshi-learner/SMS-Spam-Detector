"""SMS Spam & Scam Detector v2 - Streamlit app with explainability, batch mode and model report."""
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="SMS Spam & Scam Detector", page_icon="🛡️", layout="wide")


# ======================= loading =======================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open(BASE / "spam_model.pkl", "rb"))
    vec = pickle.load(open(BASE / "vectorizer.pkl", "rb"))
    return model, vec


@st.cache_data
def load_metrics():
    p = BASE / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def spam_col(model):
    c = list(model.classes_)
    return c.index("spam") if "spam" in c else c.index(1)


# ======================= helpers =======================
INDICATORS = {
    "Contains a link": r"(https?://|www\.|bit\.ly|tinyurl|\b\w+\.(com|in|co|net|org|xyz|top|link|cc)\b)",
    "Urgency language": r"\b(urgent|immediately|now|today only|expires?|last chance|hurry|act fast|within \d+ (hours?|mins?|minutes?))\b",
    "Prize / reward claim": r"\b(won|winner|prize|reward|lottery|jackpot|gift|free|bonus|cashback|congratulations?)\b",
    "Money / account terms": r"(₹|rs\.?\s?\d|\$|£|\bcash\b|\bloan\b|\bupi\b|\bkyc\b|\bbank\b|\baccount\b|\bblocked\b|\bsuspended\b)",
    "Asks you to act (call/click/verify)": r"\b(call|text|reply|send|claim|click|verify|update|confirm)\b",
    "Phone number / short code": r"(\b\d{10,}\b|\b\d{5}\b)",
}


def find_indicators(text):
    return [n for n, p in INDICATORS.items() if re.search(p, text, flags=re.I)]


def explain(model, vec, text, top=8):
    """Words pushing the prediction towards spam / ham (works for Naive Bayes and Logistic Regression)."""
    X = vec.transform([text])
    if hasattr(model, "feature_log_prob_"):
        w = model.feature_log_prob_[spam_col(model)] - model.feature_log_prob_[1 - spam_col(model)]
    elif hasattr(model, "coef_"):
        w = model.coef_[0] if spam_col(model) == 1 else -model.coef_[0]
    else:
        return []
    names = vec.get_feature_names_out()
    idx = X.nonzero()[1]
    contrib = [(names[i], float(X[0, i] * w[i])) for i in idx]
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    return contrib[:top]


def predict_many(model, vec, texts):
    return model.predict_proba(vec.transform(texts))[:, spam_col(model)]


SAMPLES = {
    "🚨 Prize scam": "Congratulations! You have won a £1000 cash prize. Call 09061701461 now to claim. Claim code KL341. Valid 12 hours only.",
    "🚨 Free entry": "FREE entry in 2 a weekly competition to win FA Cup final tkts. Text FA to 87121 to receive entry question. T&C apply.",
    "⚠️ KYC scam": "URGENT: Your bank account will be blocked today. Update your KYC immediately: http://bit.ly/kyc-update-now",
    "✅ Casual chat": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "✅ College msg": "Don't forget to bring your notebook for the lab session at 10 tomorrow.",
}


def set_sample(t):
    st.session_state["sms_input"] = t


# ======================= startup =======================
try:
    model, vec = load_artifacts()
except FileNotFoundError:
    st.error("`spam_model.pkl` / `vectorizer.pkl` not found next to app.py. Run `train_model.py` first.")
    st.stop()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Usually a scikit-learn version mismatch. Pin the version used to train the model in requirements.txt.")
    st.stop()

M = load_metrics()
default_thr = float(M["default_threshold"]) if M else 0.5
st.session_state.setdefault("history", [])

# ======================= sidebar =======================
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Spam threshold", 0.05, 0.95, float(round(default_thr, 2)), 0.01,
                          help="Flag as spam when spam probability ≥ threshold. Lower = catches more spam, "
                               "but may flag more genuine messages.")
    st.caption(f"Default {default_thr:.2f} was chosen with cross-validation on the training data.")
    st.divider()
    if M:
        sel = M["selected_model"]
        t = M["models"][sel]["test_at_chosen"]
        st.markdown(f"**Model:** {sel}\n\n**Test recall:** {t['recall']:.1%}  \n**Test precision:** {t['precision']:.1%}  \n**Test F1:** {t['f1']:.3f}")
    st.divider()
    st.warning("Trained on an older UK/Singapore SMS dataset. It can miss modern or India-specific scams "
               "(UPI, KYC, courier fraud, Hinglish). Learning demo, not a security product.")

st.title("🛡️ SMS Spam & Scam Detector")
st.caption("Machine learning project · TF-IDF + Naive Bayes · Streamlit")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detector", "📂 Batch analysis", "📊 Model report", "ℹ️ About"])

# ======================= Tab 1: detector =======================
with tab1:
    st.write("**Try a sample**")
    cols = st.columns(len(SAMPLES))
    for c, (label, text) in zip(cols, SAMPLES.items()):
        c.button(label, on_click=set_sample, args=(text,), width="stretch")

    st.text_area("Message to analyse", key="sms_input", height=140, placeholder="Paste an SMS here...")

    if st.button("🔍 Analyze", type="primary", width="stretch"):
        msg = st.session_state.get("sms_input", "").strip()
        if not msg:
            st.warning("Please enter a message first.")
        else:
            p = float(predict_many(model, vec, [msg])[0])
            is_spam = p >= threshold
            st.session_state["history"].insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"), "Message": msg[:80] + ("…" if len(msg) > 80 else ""),
                "Spam probability": round(p, 3), "Verdict": "SPAM" if is_spam else "HAM"})

            left, right = st.columns([3, 2])
            with left:
                if is_spam:
                    st.error(f"🚨 **SPAM / SCAM likely**  ·  spam probability {p:.1%}")
                else:
                    st.success(f"✅ **Looks legitimate (HAM)**  ·  spam probability {p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))
                st.caption(f"Decision threshold: {threshold:.0%}. Spam probability above this is flagged.")
            with right:
                c1, c2 = st.columns(2)
                c1.metric("Spam", f"{p:.1%}")
                c2.metric("Ham", f"{1 - p:.1%}")

            st.subheader("Why did the model decide this?")
            contrib = explain(model, vec, msg)
            if contrib:
                df = pd.DataFrame(contrib, columns=["Word / phrase", "Push"])
                df["Direction"] = np.where(df["Push"] > 0, "→ spam", "→ ham")
                st.bar_chart(df.set_index("Word / phrase")["Push"], horizontal=True)
                st.caption("Bars to the right push towards SPAM, bars to the left towards HAM. "
                           "Only words the model knows from training appear here.")
            else:
                st.info("No known words from the training vocabulary were found in this message.")

            flags = find_indicators(msg)
            st.subheader("Warning signs (rule-based)")
            st.caption("Simple regex checks, shown separately. They are NOT part of the model's decision.")
            st.markdown("\n".join(f"- ⚠️ {f}" for f in flags) if flags else "- No common warning signs found.")
            if not is_spam and flags:
                st.info("The model says ham, but warning signs are present. Never share OTPs, PINs or passwords, "
                        "and don't click unknown links.")

    if st.session_state["history"]:
        with st.expander(f"🕘 Session history ({len(st.session_state['history'])})"):
            st.dataframe(pd.DataFrame(st.session_state["history"]), width="stretch", hide_index=True)
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()

# ======================= Tab 2: batch =======================
with tab2:
    st.write("Upload a CSV with one message per row, or paste messages (one per line).")
    up = st.file_uploader("CSV file", type=["csv"])
    pasted = st.text_area("…or paste messages, one per line", height=120, key="batch_text")
    texts, label = [], None
    if up is not None:
        try:
            raw = pd.read_csv(up, encoding="latin-1")
            col = st.selectbox("Which column has the messages?", raw.columns,
                               index=list(raw.columns).index("message") if "message" in raw.columns else 0)
            texts = raw[col].astype(str).tolist()
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
    elif pasted.strip():
        texts = [l.strip() for l in pasted.splitlines() if l.strip()]

    if texts and st.button("Run batch analysis", type="primary"):
        probs = predict_many(model, vec, texts)
        out = pd.DataFrame({"message": texts, "spam_probability": probs.round(4),
                            "verdict": np.where(probs >= threshold, "SPAM", "HAM")})
        a, b, c = st.columns(3)
        a.metric("Messages", len(out))
        b.metric("Flagged as spam", int((out.verdict == "SPAM").sum()))
        c.metric("Spam share", f"{(out.verdict == 'SPAM').mean():.1%}")
        st.dataframe(out.sort_values("spam_probability", ascending=False), width="stretch", hide_index=True)
        st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode(), "spam_results.csv", "text/csv")

# ======================= Tab 3: model report =======================
with tab3:
    if not M:
        st.info("`metrics.json` not found. Run `train_model.py` to generate the model report.")
    else:
        st.subheader("Dataset")
        a, b, c, d = st.columns(4)
        a.metric("Total messages", f"{M['n_total']:,}")
        b.metric("Ham", f"{M['n_ham']:,}")
        c.metric("Spam", f"{M['n_spam']:,}")
        d.metric("Train / Test", f"{M['n_train']:,} / {M['n_test']:,}")
        st.caption("Only ~13% of messages are spam (class imbalance), so accuracy alone is misleading; "
                   "always predicting 'ham' would already score ~87%.")

        st.subheader("Model comparison (held-out test set)")
        rows = []
        for name, r in M["models"].items():
            t5, tc = r["test_at_0_5"], r["test_at_chosen"]
            rows.append({"Model": name + (" ✅ selected" if name == M["selected_model"] else ""),
                         "CV F1": round(r["cv_f1_at_best_threshold"], 3),
                         "Precision @0.5": round(t5["precision"], 3), "Recall @0.5": round(t5["recall"], 3),
                         "F1 @0.5": round(t5["f1"], 3),
                         "Tuned thr.": r["chosen_threshold"], "Precision (tuned)": round(tc["precision"], 3),
                         "Recall (tuned)": round(tc["recall"], 3), "F1 (tuned)": round(tc["f1"], 3),
                         "ROC-AUC": round(r["roc_auc"], 3), "PR-AUC": round(r["pr_auc"], 3)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("The model was selected using cross-validation on the training data only. "
                   "Test-set numbers are reported once, for evaluation.")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix (selected model)")
            t = M["models"][M["selected_model"]]["test_at_chosen"]
            cm = pd.DataFrame([[t["tn"], t["fp"]], [t["fn"], t["tp"]]],
                              index=["Actual ham", "Actual spam"], columns=["Predicted ham", "Predicted spam"])
            st.dataframe(cm, width="stretch")
            st.caption(f"At threshold {t['threshold']}: {t['fp']} genuine messages wrongly flagged, "
                       f"{t['fn']} spam messages missed.")
        with right:
            st.subheader("Precision vs recall by threshold")
            tt = pd.DataFrame(M["selected_threshold_table"]).set_index("threshold")[["precision", "recall", "f1"]]
            st.line_chart(tt)
            st.caption("Lowering the threshold catches more spam (recall ↑) but flags more genuine messages (precision ↓).")

        l2, r2 = st.columns(2)
        l2.subheader("Top spam indicators")
        l2.write(", ".join(f"`{w}`" for w in M["top_spam_words"]))
        r2.subheader("Top ham indicators")
        r2.write(", ".join(f"`{w}`" for w in M["top_ham_words"]))

# ======================= Tab 4: about =======================
with tab4:
    st.markdown("""
### How it works
1. **Text → numbers:** each SMS is converted with **TF-IDF** (single words and word pairs, English stop-words removed).
2. **Classifier:** a **Multinomial Naive Bayes** model estimates the probability the message is spam.
3. **Decision:** the message is flagged as spam when that probability is above the chosen threshold.
4. **Explanation:** for each word we show how strongly it pushed the prediction towards spam or ham.

### Data
UCI SMS Spam Collection (5,572 labelled messages; 4,825 ham and 747 spam).

### Limitations (please read)
- The data is old and mostly UK/Singapore English. Modern and Indian scams (UPI, KYC, courier fraud, Hinglish) are under-represented, so the model can miss them.
- The threshold trades missed spam against false alarms. There is no setting that removes both.
- Regex warning signs are heuristics and not part of the model.
- This is an educational project, not a replacement for a real spam filter.

### Future work
Add Indian/Hinglish scam data, try transformer models (e.g. DistilBERT), add feedback-based retraining, deploy as an API or browser extension.
""")
