import pandas as pd
import joblib

df = pd.read_csv("../data/processed/features_with_clusters.csv", parse_dates=["date"])

df = df.sort_values(by=["store_id", "date"])

latest_df = (
    df.groupby("store_id")
      .tail(1)
      .reset_index(drop=True)
)

print("Latest rows per store:")
print(latest_df[["store_id", "date", "sales", "cluster"]].head())


feature_cols = [c for c in df.columns if c not in ["sales", "date", "cluster"]]

print("Using features:")
print(feature_cols)


clusters = sorted(df["cluster"].unique())
all_forecasts = []

for cl in clusters:
    model_path = f"../models/xgb_cluster_{cl}.pkl"
    print(f"\n🔹 Processing cluster {cl} with model {model_path}")
    
    try:
        model_c = joblib.load(model_path)
    except FileNotFoundError:
        print(f"  ⚠️ Model file not found for cluster {cl}, skipping.")
        continue
    

    subset = latest_df[latest_df["cluster"] == cl].copy()
    if subset.empty:
        print("  ⚠️ No stores in this cluster for prediction, skipping.")
        continue
    
    X_pred = subset[feature_cols]
    
   
    preds = model_c.predict(X_pred)
    
    subset["forecast_sales"] = preds
    subset["cluster"] = cl
    
    all_forecasts.append(subset)
if not all_forecasts:
    raise ValueError("No forecasts were generated. Check cluster assignments and models.")

forecast_df = pd.concat(all_forecasts, ignore_index=True)

last_date = df["date"].max()
forecast_month = last_date + pd.offsets.MonthEnd(1)

forecast_df["forecast_month"] = forecast_month

forecast_df = forecast_df[["store_id", "cluster", "date", "sales", "forecast_month", "forecast_sales"]]
forecast_df = forecast_df.rename(columns={"sales": "last_month_sales"})

print("\nSample of final forecast output:")
print(forecast_df.head())

output_path = "../data/processed/next_month_forecast_cluster.csv"
forecast_df.to_csv(output_path, index=False)
print(f"\n✅ Saved next-month cluster-wise forecast to {output_path}")
