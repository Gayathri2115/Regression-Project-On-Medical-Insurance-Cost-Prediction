# train.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("insurance.csv")

print("Dataset Loaded Successfully")
print(data.head())


# =========================
# 2. Define Features & Target
# =========================
X = data.drop("charges", axis=1)
y = data["charges"]


# =========================
# 3. Identify Categorical & Numerical Columns
# =========================
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]


# =========================
# 4. Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)


# =========================
# 5. Create Pipeline
# =========================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# 7. Train Model
# =========================
model.fit(X_train, y_train)

print("Model Training Completed")


# =========================
# 8. Evaluation
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =========================
# 9. Save Model
# =========================
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
