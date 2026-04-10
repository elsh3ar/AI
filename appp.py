import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
try:
    data = pd.read_csv('indian_liver_patient.csv')
    print("Data loaded successfully!")
except:
    print("Error: File not found. Check the filename!")
    exit()

# 2. Preprocessing
# Fill missing values
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean())

# Convert Gender to numbers
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# 3. Split Data
X = data.drop('Dataset', axis=1)
y = data['Dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
print("Training the model... please wait.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# 6. Save Model
joblib.dump(model, 'liver_model.pkl')
print("Success! 'liver_model.pkl' has been created.")