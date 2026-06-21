import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("water_potability.csv")

# Features & Target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Handle missing values
X = X.fillna(X.mean())

# -------------------------------
# Balance dataset using SMOTE
# -------------------------------
print("🔄 Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("✅ Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# -------------------------------
# Train-test split (stratified)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# -------------------------------
# Train RandomForest
# -------------------------------
model = RandomForestClassifier(
    n_estimators=400,           # more trees for stability
    max_depth=15,               # avoid overfitting
    min_samples_split=5,        # split control
    min_samples_leaf=2,         # leaf control
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1                   # use all CPU cores
)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n✅ Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model with features
# -------------------------------
joblib.dump((model, X.columns.tolist()), "model.pkl")
print("\n💾 Model trained and saved as model.pkl")
