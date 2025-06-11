import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/The_Cancer_data_V2.csv")

# Features and Target
X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Numeric Preprocessing
num_features = X.select_dtypes(include=np.number).columns
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features)
])

# Define models to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Store performance
model_scores = {}
pipelines = {}

# Train and evaluate each model
for name, clf in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores[name] = acc
    pipelines[name] = pipeline
    print(f"{name} Accuracy: {acc:.4f}")

# Choose best model
best_model_name = max(model_scores, key=model_scores.get)
best_pipeline = pipelines[best_model_name]

# Save best model
joblib.dump(best_pipeline, "cancer_model.pkl")
print(f"\n✅ Best model: {best_model_name} saved as cancer_model.pkl")