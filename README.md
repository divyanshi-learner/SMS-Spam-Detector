# 🛡️ SMS Spam & Scam Detector (v2)

A machine learning web app that classifies SMS messages as **spam** or **ham**, explains which words drove the decision, and reports how reliable the model really is.

**Live demo:** _add your Streamlit URL_  ·  **Team:** _add names_  ·  **Course/College:** _add_

## Problem statement
Spam and scam SMS messages (fake prizes, KYC/bank threats, phishing links) cause financial loss and loss of trust. This project builds and evaluates a text classifier that flags such messages, and is honest about where it fails.

## Features
| Tab | What it does |
|---|---|
| 🔍 Detector | Paste an SMS → verdict, spam probability, **word-level explanation**, rule-based warning signs, session history |
| 📂 Batch analysis | Upload a CSV (or paste many lines) → classify all, download results |
| 📊 Model report | Dataset stats, 4-model comparison, confusion matrix, precision/recall vs threshold, top spam/ham words |
| ℹ️ About | Method, data, limitations, future work |

## Method
1. **Data:** UCI SMS Spam Collection, 5,572 messages (4,825 ham / 747 spam, ~13% spam).
2. **Split:** 80/20 stratified (train 4,457 / test 1,115).
3. **Features:** TF-IDF, unigrams + bigrams, English stop words removed, sublinear TF.
4. **Models compared:** Multinomial Naive Bayes, Logistic Regression, Logistic Regression (balanced), Random Forest.
5. **Model selection:** 5-fold cross-validation on the training set only.
6. **Threshold tuning:** the decision threshold is chosen from out-of-fold predictions on the training set, so the test set stays untouched.
7. **Explainability:** per-word contribution = TF-IDF weight × (log P(word | spam) − log P(word | ham)).

## Results (held-out test set)
_Fill this table from the **Model report** tab after you run `train_model.py`._

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Naive Bayes (selected) | | | | |
| Logistic Regression | | | | |
| Logistic Regression (balanced) | | | | |
| Random Forest | | | | |

Version 1 of this project (plain TF-IDF + Naive Bayes at threshold 0.5) reached 96.23% accuracy with 72% spam recall. Version 2 improves recall by tuning smoothing, TF scaling and the decision threshold.

## Limitations
- Old, mostly UK/Singapore English data. Modern and Indian scams (UPI, KYC, courier fraud, Hinglish) are under-represented, so the model can miss them.
- Threshold trade-off: catching more spam means more genuine messages get flagged.
- Educational project, not a production spam filter.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py spam.csv   # creates spam_model.pkl, vectorizer.pkl, metrics.json
streamlit run app.py
```

## Structure
```
├── app.py            # Streamlit app
├── train_model.py    # training, comparison, evaluation
├── spam_model.pkl    # trained classifier
├── vectorizer.pkl    # TF-IDF vectorizer
├── metrics.json      # evaluation results used by the Model report tab
├── requirements.txt
└── README.md
```

## Future work
Add Indian/Hinglish scam data, try DistilBERT, collect user feedback for retraining, expose an API.
