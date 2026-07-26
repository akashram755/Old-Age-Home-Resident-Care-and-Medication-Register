"""
Task 4 — Predict which entries will need supervisor attention.

Target: needs_attention (Yes/No), set by the care staff who logged the entry.

Why this is a real prediction and not a copy of an if-statement:
  At the moment a prediction is actually useful — start of shift, before the
  dose is given and before anyone has written today's observation — we do NOT
  yet know today's `given` value or today's `observation` text. Both of those
  are decided/written AFTER (or at the same moment as) needs_attention, so
  using them as inputs would let the model see the answer before predicting
  it. This script deliberately excludes them.

  Instead it uses only things known in advance:
    - which resident, which medication, what time of day, what day of week
    - that resident's OWN track record up to (but not including) this entry:
      how often they've been given their meds on time, and how often past
      entries were flagged. This is exactly what a new staff member could
      look up in the file before the shift starts.

Run:
    python3 train_model.py
Produces:
    predictions.json   -- consumed by script.js in the browser
    (metrics are printed to the console)
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

RANDOM_SEED = 42
LOW_CONFIDENCE_THRESHOLD = 0.60  # below this, the UI shows "not confident" instead of forcing an answer

with open("data.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# Drop the one unlinked/junk row seeded on purpose (no resident_id to build history from)
df = df[df["resident_id"] != ""].copy()
df["resident_id"] = df["resident_id"].astype(int)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["resident_id", "date"]).reset_index(drop=True)

# ---- Feature engineering: everything here is knowable BEFORE this entry's
# ---- given/observation/needs_attention are decided. ----

def dose_bucket(t):
    if not t:
        return "unknown"
    hour = int(t.split(":")[0])
    if hour < 11:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"

df["dose_bucket"] = df["dose_time"].apply(dose_bucket)
df["weekday"] = df["date"].dt.day_name()

prior_given_rate, prior_flag_rate, prior_count = [], [], []
history = {}  # resident_id -> list of (given_yes: bool, flagged: bool)

for _, row in df.iterrows():
    past = history.get(row["resident_id"], [])
    if past:
        given_vals = [g for g, _ in past if g is not None]
        prior_given_rate.append(np.mean(given_vals) if given_vals else 0.5)
        prior_flag_rate.append(np.mean([fl for _, fl in past]))
        prior_count.append(len(past))
    else:
        # No history yet for this resident: use a neutral default, not the answer.
        prior_given_rate.append(0.5)
        prior_flag_rate.append(0.0)
        prior_count.append(0)

    g = row["given"]
    given_bool = True if g == "Yes" else (False if g == "No" else None)
    flagged_bool = row["needs_attention"] == "Yes"
    history.setdefault(row["resident_id"], []).append((given_bool, flagged_bool))

df["prior_given_rate"] = prior_given_rate
df["prior_flag_rate"] = prior_flag_rate
df["prior_count"] = prior_count

FEATURES_CATEGORICAL = ["resident_id", "medication", "dose_bucket", "weekday"]
FEATURES_NUMERIC = ["prior_given_rate", "prior_flag_rate", "prior_count"]
TARGET = "needs_attention"

X = df[FEATURES_CATEGORICAL + FEATURES_NUMERIC].copy()
X["resident_id"] = X["resident_id"].astype(str)  # treat as a category, not a number
y = (df[TARGET] == "Yes").astype(int)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=RANDOM_SEED, stratify=y
)

pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
    ], remainder="passthrough")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
])

pipeline.fit(X_train, y_train)
pred_test = pipeline.predict(X_test)

print("=== Test set metrics (25% held out, never seen during training) ===")
print(f"Accuracy:  {accuracy_score(y_test, pred_test):.2f}")
print(f"Precision: {precision_score(y_test, pred_test, zero_division=0):.2f}")
print(f"Recall:    {recall_score(y_test, pred_test, zero_division=0):.2f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:")
print(confusion_matrix(y_test, pred_test))

# ---- Predict on everything, for the UI, but keep the test-set numbers above
# ---- as the honest measure of how good the model actually is. ----
probs = pipeline.predict_proba(X)[:, 1]

predictions = {}
for i, record_id in enumerate(df["record_id"]):
    p = float(probs[i])
    confident = max(p, 1 - p) >= LOW_CONFIDENCE_THRESHOLD
    predictions[record_id] = {
        "predicted_risk": ("Yes" if p >= 0.5 else "No") if confident else None,
        "confidence": round(max(p, 1 - p), 2),
        "in_test_set": bool(i in set(idx_test)),
    }

with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

n_confident = sum(1 for v in predictions.values() if v["predicted_risk"] is not None)
print(f"\nWrote predictions.json — {n_confident}/{len(predictions)} entries above the "
      f"{LOW_CONFIDENCE_THRESHOLD:.0%} confidence threshold; the rest are left unpredicted on purpose.")
