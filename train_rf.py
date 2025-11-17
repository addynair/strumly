import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# ------------------------------
# Load and clean dataset
# ------------------------------
df = pd.read_csv("hand_chords_dataset.csv")

# Drop any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how="all")
df = df.dropna()

print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# Some CSVs may have an extra empty numeric column at the end
# Ensure total = 63 features + 1 label
if df.shape[1] > 64:
    print(f"⚙️ Detected {df.shape[1]} columns, trimming to 64 (63 + 1 label)")
    df = df.iloc[:, :64]

# Split into features (X) and labels (y)
X_raw = df.iloc[:, :-1].values  # first 63 columns
y = df.iloc[:, -1].values       # last column = label

print(f"🧩 Feature count per sample: {X_raw.shape[1]}")

# ------------------------------
# Normalize landmarks
# ------------------------------
def normalize_landmarks(row):
    lm = row.reshape(-1, 3)
    center = lm[0]
    lm -= center
    hand_size = np.linalg.norm(lm[0] - lm[9])
    if hand_size > 0:
        lm /= hand_size
    return lm.flatten()

X = np.array([normalize_landmarks(r) for r in X_raw])

# ------------------------------
# Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Train RandomForest
# ------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

# ------------------------------
# Evaluate & Save
# ------------------------------
acc = rf.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

with open("hand_rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("💾 Saved as hand_rf_model.pkl")
