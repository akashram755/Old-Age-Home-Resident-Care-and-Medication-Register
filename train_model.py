
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
LOW_CONFIDENCE_THRESHOLD = 0.60  

with open("data.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)


df = df[df["resident_id"] != ""].copy()
df["resident_id"] = df["resident_id"].astype(int)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["resident_id", "date"]).reset_index(drop=True)


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
history = {}  

for _, row in df.iterrows():
    past = history.get(row["resident_id"], [])
    if past:
        given_vals = [g for g, _ in past if g is not None]
        prior_given_rate.append(np.mean(given_vals) if given_vals else 0.5)
        prior_flag_rate.append(np.mean([fl for _, fl in past]))
        prior_count.append(len(past))
    else:
        
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
random_split_accuracy = accuracy_score(y_test, pred_test)


print("\n=== Change 1: new-category check ===")
new_med_rows = df[df["medication"] == "Clopidogrel 75mg"]
if new_med_rows.empty:
    print("No rows with a new medication found yet. Run "
          "add_new_medication_category.py first, then re-run this script.")
else:
    new_med_X = new_med_rows[FEATURES_CATEGORICAL + FEATURES_NUMERIC].copy()
    new_med_X["resident_id"] = new_med_X["resident_id"].astype(str)
    try:
        new_med_preds = pipeline.predict_proba(new_med_X)[:, 1]
        n_in_train = new_med_rows.index.isin(idx_train).sum()
        n_in_test = new_med_rows.index.isin(idx_test).sum()
        print(f"Found {len(new_med_rows)} rows with the new medication "
              f"'Clopidogrel 75mg' ({n_in_train} in train, {n_in_test} in test).")
        print(f"Pipeline scored all {len(new_med_rows)} of them without error — "
              f"e.g. predicted probability {new_med_preds[0]:.2f} for the first one.")
        print("The pipeline still runs correctly: the new category is one-hot "
              "encoded like any other when seen in training, and safely "
              "zero-encoded (via handle_unknown='ignore') for any occurrence "
              "the model didn't train on.")
    except Exception as e:
        print(f"Pipeline broke on the new category: {e}")


df_time_sorted = df.sort_values("date", kind="stable").reset_index(drop=True)
cutoff = int(len(df_time_sorted) * 0.8)

X_time = df_time_sorted[FEATURES_CATEGORICAL + FEATURES_NUMERIC].copy()
X_time["resident_id"] = X_time["resident_id"].astype(str)
y_time = (df_time_sorted[TARGET] == "Yes").astype(int)

X_train_t, X_test_t = X_time.iloc[:cutoff], X_time.iloc[cutoff:]
y_train_t, y_test_t = y_time.iloc[:cutoff], y_time.iloc[cutoff:]

pipeline_temporal = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
    ], remainder="passthrough")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
])
pipeline_temporal.fit(X_train_t, y_train_t)
pred_test_t = pipeline_temporal.predict(X_test_t)
temporal_accuracy = accuracy_score(y_test_t, pred_test_t)

print("\n=== Change 2: chronological holdout (last 20% of records, by date) ===")
print(f"Train rows: {len(X_train_t)}  |  Test rows: {len(X_test_t)}")
print(f"Test set date range: {df_time_sorted['date'].iloc[cutoff].date()} "
      f"to {df_time_sorted['date'].iloc[-1].date()}")
print(f"Accuracy:  {temporal_accuracy:.2f}")
print(f"Precision: {precision_score(y_test_t, pred_test_t, zero_division=0):.2f}")
print(f"Recall:    {recall_score(y_test_t, pred_test_t, zero_division=0):.2f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:")
print(confusion_matrix(y_test_t, pred_test_t))

print("\n--- Honest comparison ---")
print(f"Random 75/25 split accuracy:        {random_split_accuracy:.2f}")
print(f"Chronological last-20% accuracy:    {temporal_accuracy:.2f}")
diff = temporal_accuracy - random_split_accuracy
if diff < -0.01:
    print(f"The chronological split scores {abs(diff):.2f} LOWER. This is expected: "
          "the random split lets the model see examples from every date, including "
          "dates close to (or interleaved with) test rows, and every resident is "
          "usually represented in training. The chronological split instead tests "
          "on the most recent day using ONLY earlier days to learn from — some "
          "residents' most recent behaviour may differ from their earlier pattern, "
          "and any resident whose records fall entirely in the last 20% is unseen "
          "by the model. A drop here is a more honest estimate of how the model "
          "would perform in real deployment (predicting today from yesterday) than "
          "the random split is.")
elif diff > 0.01:
    print(f"The chronological split scores {diff:.2f} HIGHER, which can happen with "
          "a small dataset like this one if the held-out days happen to be easier "
          "to call (e.g. less varied risk levels) than the days in the random "
          "split's test set. It doesn't mean the model generalises better — with "
          "only a few dates of history, either split's number should be treated as "
          "a rough estimate, not a precise one.")
else:
    print("The two splits score about the same here, likely because the dataset "
          "is small enough that both test sets end up similarly balanced.")


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
