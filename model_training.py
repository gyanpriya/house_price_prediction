import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engineering import load_data, preprocess_data

def train_models(X, y):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred)** 0.5
        r2 = r2_score(y, y_pred)

        results[name] = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"\n{name} Performance:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2 Score: {r2:.2f}")

    return results

def save_model(model, filename="models/model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved to {filename}")

if __name__ == "__main__":
    df = load_data()
    X, y, encoder, scaler = preprocess_data(df)
    results = train_models(X, y)

    # Choose best model based on R²
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_model_name]["model"]
    print(f"\n✅ Best Model: {best_model_name}")

    # Save model
save_model(best_model)

# Save encoder and scaler
with open("models/encoder_scaler.pkl", "wb") as f:
    pickle.dump({"encoder": encoder, "scaler": scaler}, f)

