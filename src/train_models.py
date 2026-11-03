
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Configuration
DATA_PATH = "final_model_dataset.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.sort_values("year").reset_index(drop=True)

crops = {
    "bean": ("bean_yield", "bean_yield_lag1"),
    "maize": ("maize_yield", "maize_yield_lag1"),
    "potato": ("potato_yield", "potato_yield_lag1"),
}

yield_columns = [
    "bean_yield", "maize_yield", "potato_yield"
]

lag_columns = [
    "bean_yield_lag1",
    "maize_yield_lag1",
    "potato_yield_lag1"
]

# Avoid using other crops' current or lagged yields
base_features = [
    col for col in df.columns
    if col not in yield_columns + lag_columns
]

model_results = []

# Chronological 80:20 split
split_index = int(len(df) * 0.8)

for crop, (target, lag_feature) in crops.items():

    features = base_features + [lag_feature]

    X = df[features]
    y = df[target]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                random_state=42
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42
            ))
        ]),

        "ANN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=3000,
                early_stopping=False,
                alpha=0.01,
                random_state=42
            ))
        ])
    }

    print(f"\nTraining models for {crop.upper()}")

    for model_name, pipeline in models.items():

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        model_results.append({
            "Crop": crop,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        filename = (
            f"{MODEL_DIR}/{crop}_{model_name.lower().replace(' ', '_')}.joblib"
        )

        joblib.dump(pipeline, filename)

        print(
            f"{model_name}: "
            f"MAE={mae:.3f}, "
            f"RMSE={rmse:.3f}, "
            f"R2={r2:.3f}"
        )

# Save comparison results
results = pd.DataFrame(model_results)
results.to_csv(
    f"{RESULTS_DIR}/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

print("\nTraining completed.")
