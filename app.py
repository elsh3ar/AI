import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ─────────────────────────────────────────────
# 1. Load Data  (relative path — works on any machine)
# ─────────────────────────────────────────────
print("Step 1: Loading data...")

CSV_PATH = os.path.join(os.path.dirname(__file__), "indian_liver_patient.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {CSV_PATH}\n"
        "Please place 'indian_liver_patient.csv' in the same folder as app.py."
    )

data = pd.read_csv(CSV_PATH)
print(f"  ✔ Loaded {len(data)} rows.")

# ─────────────────────────────────────────────
# 2. Clean Data  (fill missing values)
# ─────────────────────────────────────────────
missing_before = data.isnull().sum().sum()
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(
    data['Albumin_and_Globulin_Ratio'].mean()
)
print(f"  ✔ Filled {missing_before} missing value(s).")

# ─────────────────────────────────────────────
# 3. Encode  (text → numbers)
# ─────────────────────────────────────────────
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])   # Male=1, Female=0
data['Dataset'] = data['Dataset'].map({1: 1, 2: 0}) # 1=Liver patient, 0=Healthy

# ─────────────────────────────────────────────
# 4. Features & Target
# ─────────────────────────────────────────────
X = data.drop('Dataset', axis=1)
y = data['Dataset']

# ─────────────────────────────────────────────
# 5. Balance classes with SMOTE
# ─────────────────────────────────────────────
print("Step 2: Balancing data classes with SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"  ✔ Balanced dataset: {X_res.shape[0]} samples.")

# ─────────────────────────────────────────────
# 6. Scale features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# ─────────────────────────────────────────────
# 7. Train / Test split  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 8. Train XGBoost model
# ─────────────────────────────────────────────
print("Step 3: Training XGBoost model — please wait...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 9. Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"  ✔ Model Accuracy: {acc * 100:.2f}%")
print("-" * 40)
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 10. Save model & scaler
# ─────────────────────────────────────────────
joblib.dump(model,  'liver_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("  ✔ Saved: liver_model.pkl  &  scaler.pkl")
