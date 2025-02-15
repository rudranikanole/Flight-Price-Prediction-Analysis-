import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Load the processed dataset
df = pd.read_csv("data/Processed_Dataset.csv")

# ✅ Optional: Reduce dataset size to speed up training
df = df.sample(50000, random_state=42)  # Uncomment if dataset is too large

# Define Features (X) and Target (y)
X = df.drop(columns=["price"])  # Features
y = df["price"]  # Target Variable

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define hyperparameter grid (Optimized for Speed)
param_grid = {
    'n_estimators': [10, 50],         # Reduce tree count (faster training)
    'max_depth': [10, 20],            # Limit depth to avoid overfitting
    'min_child_samples': [5, 10]      # Similar to min_samples_split in RF
}

# ✅ Initialize LightGBM Model
lgbm = LGBMRegressor(random_state=42)

# ✅ Randomized Search to speed up training
grid_search = RandomizedSearchCV(
    estimator=lgbm, 
    param_distributions=param_grid, 
    n_iter=5,  # Only test 5 combinations instead of 36
    cv=2,  # Reduce cross-validation folds from 3 to 2
    scoring='neg_mean_absolute_error', 
    verbose=2, 
    n_jobs=4,  # Use only 4 cores to avoid memory issues
    random_state=42
)

grid_search.fit(X_train, y_train)

# ✅ Get the best model from Randomized Search
best_model = grid_search.best_estimator_

# ✅ Save the trained model
joblib.dump(best_model, "flight_price_model.joblib")
print("💾 ✅ Model saved as 'flight_price_model.joblib'!")

# ✅ Make predictions using the best model
y_pred = best_model.predict(X_test)

# ✅ Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Trained with Best Parameters!")
print("Best Parameters:", grid_search.best_params_)
print(f"📉 Mean Absolute Error (MAE): {mae}")
print(f"📊 Mean Squared Error (MSE): {mse}")
print(f"🎯 R² Score: {r2}")

# ✅ Example: Load the model for future use
loaded_model = joblib.load("flight_price_model.joblib")
print("🚀 Loaded the model successfully for future predictions!")
