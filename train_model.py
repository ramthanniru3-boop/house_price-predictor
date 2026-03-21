import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("train.csv")

# Target
y = np.log(df["SalePrice"])
X = df.drop("SalePrice", axis=1)

# ==========================
# PREPROCESSING
# ==========================
X = X.fillna(0)
X = pd.get_dummies(X)

# Save column names (VERY IMPORTANT)
pickle.dump(X.columns, open("columns.pkl", "wb"))

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# SCALING
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# MODELS
# ==========================
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100)
}

best_model = None
best_score = -999

# ==========================
# TRAIN & COMPARE
# ==========================
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{name} → R2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model

# ==========================
# SAVE FILES
# ==========================
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Best model saved!")