"""
Alzheimer's Disease Severity Classification — CA3 Predictive Analytics
Run this file from VS Code, or copy the cells into a Jupyter notebook.

Dataset expected: oasis_cross-sectional.csv (or the OASIS Excel file).
The script searches the current folder, this script's folder, and one level below them.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42

def locate_dataset():
    """Search common locations and supported OASIS file names."""
    names = [
        "oasis_cross-sectional.csv",
        "oasis-cross-sectional.csv",
        "oasis_cross-sectional-5708aa0a98d82080.xlsx",
        "oasis_cross-sectional.xlsx",
        "oasis_cross-sectional.xls",
        "oasis-cross-sectional.xlsx",
    ]
    roots = [Path.cwd(), Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()]
    candidates = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
            candidates.append(root / "Alzheimers_CA3_PartB" / name)
        # Search one directory level down, common when files are extracted from ZIP
        try:
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    for name in names:
                        candidates.append(child / name)
        except (PermissionError, OSError):
            pass
    seen = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    raise FileNotFoundError(
        "\nCould not find the OASIS dataset.\n"
        f"Python is currently looking from: {Path.cwd()}\n"
        "Put oasis_cross-sectional.csv in the same folder as this script/notebook, "
        "or edit DATA_FILE below with the full path to your CSV."
    )

# If auto-detection fails, replace None with a full path, e.g.
# DATA_FILE = r"C:\Users\YourName\Downloads\Alzheimers_CA3_PartB\oasis_cross-sectional.csv"
DATA_FILE = None

if DATA_FILE:
    data_path = Path(DATA_FILE)
    if not data_path.is_file():
        raise FileNotFoundError(f"DATA_FILE does not exist: {data_path}")
else:
    data_path = locate_dataset()

print(f"Loading dataset from: {data_path.resolve()}")
if data_path.suffix.lower() == ".csv":
    df = pd.read_csv(data_path)
else:
    df = pd.read_excel(data_path)

print("\n--- 1. DATA UNDERSTANDING ---")
print("Dataset shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Normalize column names and map common OASIS column variants.
def normalize_name(name):
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())

normalized = {normalize_name(c): c for c in df.columns}
aliases = {
    "gender": ["mf", "gender", "sex"],
    "hand": ["hand", "handedness"],
    "age": ["age"],
    "educ": ["educ", "education", "educationyears"],
    "ses": ["ses", "socioeconomicstatus"],
    "mmse": ["mmse"],
    "cdr": ["cdr", "clinicaldementiarating"],
    "etiv": ["etiv", "estimatedtotalintracranialvolume"],
    "nwbv": ["nwbv", "normalizedwholebrainvolume"],
    "asf": ["asf", "atlas scaling factor"],
    "delay": ["delay", "mrdelay"],
    "id": ["id", "subjectid"],
}
rename = {}
for canonical, options in aliases.items():
    for option in options:
        if normalize_name(option) in normalized:
            rename[normalized[normalize_name(option)]] = canonical
            break
df = df.rename(columns=rename)

if "cdr" not in df.columns:
    raise ValueError(
        "Could not identify the CDR target column. "
        f"Columns found: {df.columns.tolist()}"
    )

# Convert target and relevant numeric columns safely.
df["cdr"] = pd.to_numeric(df["cdr"], errors="coerce")
for col in ["age", "educ", "ses", "mmse", "etiv", "nwbv", "asf", "delay"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["cdr"]).copy()

# Project-defined CDR grouping: 0=Normal, 0.5=Very mild, >=1=Mild/moderate.
def group_cdr(value):
    if value == 0:
        return "Normal"
    elif value == 0.5:
        return "Very mild impairment"
    elif value >= 1:
        return "Mild/moderate impairment"
    return np.nan

df["severity"] = df["cdr"].apply(group_cdr)
df = df.dropna(subset=["severity"]).copy()

print("\n--- 2. TARGET DISTRIBUTION ---")
print(df["severity"].value_counts())

# Avoid target leakage and identifiers. Exclude CDR and ID from predictors.
exclude = {"cdr", "severity", "id", "hand"}
feature_cols = [c for c in df.columns if c not in exclude]
# Remove columns that are entirely empty.
feature_cols = [c for c in feature_cols if not df[c].isna().all()]
if not feature_cols:
    raise ValueError("No usable predictor columns were found.")

X = df[feature_cols].copy()
y = df["severity"].copy()

# Convert low-cardinality object/category columns to categorical and try numeric coercion
for col in X.columns:
    if X[col].dtype == "object":
        # Keep genuine categories as strings; numerical-looking fields become numeric.
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(0.8 * X[col].notna().sum())):
            X[col] = converted

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
], remainder="drop")

# Stratify only if each class has at least two examples.
class_counts = y.value_counts()
stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train)))),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
}

print("\n--- 3. TRAINING AND EVALUATION ---")
results = []
fitted_models = {}
for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    fitted_models[name] = pipeline
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0),
    })

results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

best_name = results_df.loc[0, "Model"]
best_model = fitted_models[best_name]
best_predictions = best_model.predict(X_test)

print(f"\nBest model by Macro F1: {best_name}")
print("\n--- 4. CLASSIFICATION REPORT ---")
print(classification_report(y_test, best_predictions, zero_division=0))

print("\n--- 5. CONFUSION MATRIX ---")
ConfusionMatrixDisplay.from_predictions(
    y_test, best_predictions, xticks_rotation=30, cmap="Blues", values_format="d"
)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.show()

print("\n--- 6. MODEL COMPARISON CHART ---")
results_df.set_index("Model")[["Accuracy", "Macro F1"]].plot(kind="bar", ylim=(0, 1), figsize=(10, 5))
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

print("\n--- 7. PERMUTATION FEATURE IMPORTANCE ---")
try:
    # Permutation importance on raw columns works with the full pipeline.
    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE,
        scoring="f1_macro"
    )
    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance.importances_mean
    }).sort_values("Importance", ascending=False)
    print(importance_df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    top = importance_df.head(10).sort_values("Importance")
    plt.figure(figsize=(8, 5))
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Mean decrease in macro F1 when shuffled")
    plt.title("Top Feature Importance (Permutation)")
    plt.tight_layout()
    plt.show()
except Exception as exc:
    print("Permutation importance could not be calculated:", exc)

print("\n--- 8. SAVE RESULTS ---")
output_path = Path.cwd() / "model_comparison_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Model comparison saved to: {output_path.resolve()}")

print("""
INTERPRETATION / LIMITATIONS
- This is a classification of recorded CDR-based severity groups, not prediction of future disease progression.
- CDR categories are simplified project-defined groups.
- MMSE and other cognitive variables may be closely associated with CDR, so discuss possible target-related feature dependence.
- Results are educational research results, not a clinical diagnostic tool.
- Report the metrics produced by this run; do not invent or copy example scores.
""")
