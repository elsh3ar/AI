import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─────────────────────────────────────────────────────────────
#  train_and_save()
#  Called automatically by interface.py on first run (Streamlit Cloud)
#  or run this file directly: python app.py
# ─────────────────────────────────────────────────────────────
def train_and_save():

    # 1. Load Data
    print("Step 1: Loading data...")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indian_liver_patient.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Please place 'indian_liver_patient.csv' in the same folder as app.py"
        )
    data = pd.read_csv(csv_path)
    print(f"  Loaded {len(data)} rows, {data.shape[1]} columns.")

    # 2. Clean Data
    missing = data['Albumin_and_Globulin_Ratio'].isna().sum()
    data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(
        data['Albumin_and_Globulin_Ratio'].mean()
    )
    print(f"  Filled {missing} missing value(s) in Albumin_and_Globulin_Ratio.")

    # 3. Encode
    le = LabelEncoder()
    data['Gender']  = le.fit_transform(data['Gender'])
    data['Dataset'] = data['Dataset'].map({1: 1, 2: 0})

    # 4. Features & Target
    X = data.drop('Dataset', axis=1)
    y = data['Dataset']

    # 5. Balance with SMOTE
    print("Step 2: Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  Balanced dataset: {X_res.shape[0]} samples.")

    # 6. Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # 7. Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    # 8. Train XGBoost
    print("Step 3: Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 9. Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print("-" * 40)
    print(f"  Model Accuracy: {acc * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_test, y_pred))

    # 10. Save
    joblib.dump(model,  'liver_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("  Saved: liver_model.pkl  &  scaler.pkl")

    return acc


# ─────────────────────────────────────────────────────────────
#  Run directly: python app.py
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save()
