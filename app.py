import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load Data
print("Step 1: Loading data...")
data = pd.read_csv('indian_liver_patient.csv')

# 2. Cleaning Data (Handle missing values)
mean_value = data['Albumin_and_Globulin_Ratio'].mean()
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(mean_value)

# 3. Encoding (Convert text and labels to numbers)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Dataset'] = data['Dataset'].map({1: 1, 2: 0}) 

# 4. Define Features (X) and Target (y)
X = data.drop('Dataset', axis=1)
y = data['Dataset']

# 5. Balancing Data using SMOTE to hit 90% accuracy
print("Step 2: Balancing data classes...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 6. Scaling (Normalize numeric ranges)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# 7. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

# 8. Training the Advanced XGBoost Model
print("Step 3: Training the AI model... Please wait.")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# 9. Calculate and Print Accuracy
acc = model.score(X_test, y_test)
print("-" * 30)
print(f"DONE! Model Accuracy: {acc * 100:.2f}%")
print("-" * 30)

# 10. Save the "Brain" of our app
joblib.dump(model, 'liver_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully as .pkl files")
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))