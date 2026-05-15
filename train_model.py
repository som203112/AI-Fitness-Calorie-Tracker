import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# =========================
# LOAD DATASETS
# =========================

calories = pd.read_csv("data/calories.csv")
exercise = pd.read_csv("data/exercise.csv")

# =========================
# MERGE DATASETS
# =========================

data = exercise.merge(calories, on="User_ID")

# =========================
# DATA PREPROCESSING
# =========================

# Convert Gender into numerical values
data.replace({"Gender": {"male": 0, "female": 1}}, inplace=True)

# Drop User_ID column
data.drop(columns=["User_ID"], inplace=True)

# =========================
# FEATURES AND TARGET
# =========================

X = data.drop(columns=["Calories"])
y = data["Calories"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = XGBRegressor()

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "app/model.pkl")

print("Model saved successfully!")