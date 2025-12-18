import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Driven Sales Forecasting System")
st.markdown("""
**Objective:** Predict next-month sales for 50+ stores using Machine Learning.  
**Models:** Cluster-specific XGBoost/RandomForest models trained on historical data.
""")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Path to your processed data
    data_path = "data/processed/features_with_clusters.csv"
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        return None, None
    
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(by=["store_id", "date"])
    
    # We need the latest month's data to predict the NEXT month
    latest_df = df.groupby("store_id").tail(1).reset_index(drop=True)
    return latest_df, df.columns

# Load the data
with st.spinner("Loading historical data..."):
    latest_df, all_columns = load_data()

if latest_df is not None:
    # Show a preview of the data
    with st.expander("🔎 View Input Data (Latest Month)"):
        st.dataframe(latest_df.head())
        st.caption(f"Total Stores: {len(latest_df)} | Data Month: {latest_df['date'].max().date()}")

    # --- 2. Prediction Engine ---
    def run_prediction(input_df):
        # Filter out non-feature columns
        feature_cols = [c for c in all_columns if c not in ["sales", "date", "cluster"]]
        
        clusters = sorted(input_df["cluster"].unique())
        all_forecasts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, cl in enumerate(clusters):
            model_path = f"models/xgb_cluster_{cl}.pkl"
            status_text.text(f"Processing Store Cluster {cl}...")
            
            # Load Model
            try:
                model = joblib.load(model_path)
            except FileNotFoundError:
                st.warning(f" Model for Cluster {cl} not found. Skipping.")
                continue
            
            # Get data for this cluster
            subset = input_df[input_df["cluster"] == cl].copy()
            if subset.empty:
                continue
            
            # Predict
            X_pred = subset[feature_cols]
            preds = model.predict(X_pred)
            
            subset["forecast_sales"] = preds
            all_forecasts.append(subset)
            
            # Update progress
            progress_bar.progress((idx + 1) / len(clusters))
            
        progress_bar.empty()
        status_text.empty()
        
        if all_forecasts:
            return pd.concat(all_forecasts, ignore_index=True)
        return pd.DataFrame()

    # --- 3. User Controls ---
    st.divider()
    col_action, col_info = st.columns([1, 3])
    
    with col_action:
        run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    if run_btn:
        forecast_df = run_prediction(latest_df)
        
        if not forecast_df.empty:
            # --- Results Display ---
            st.success("Forecast generated successfully!")
            
            # Calculate Dates
            last_date = latest_df["date"].max()
            forecast_month = last_date + pd.offsets.MonthEnd(1)
            
            # Format Output
            final_df = forecast_df[["store_id", "cluster", "sales", "forecast_sales"]].copy()
            final_df.rename(columns={"sales": "Last Month Sales", "forecast_sales": "Forecasted Sales"}, inplace=True)
            
            # Metrics
            total_current = final_df["Last Month Sales"].sum()
            total_forecast = final_df["Forecasted Sales"].sum()
            delta = ((total_forecast - total_current) / total_current) * 100
            
            # KPI Cards
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Target Month", forecast_month.strftime("%B %Y"))
            kpi2.metric("Total Forecast Revenue", f"${total_forecast:,.0f}", delta=f"{delta:.2f}%")
            kpi3.metric("Stores Analyzed", len(final_df))
            
            # Charts
            st.subheader("Store Performance Overview")
            chart_data = final_df[["store_id", "Last Month Sales", "Forecasted Sales"]].set_index("store_id")
            st.bar_chart(chart_data.head(20)) # Show top 20 stores to keep chart clean
            
            # Data Table
            st.subheader("Detailed Forecast Data")
            st.dataframe(final_df.style.format({"Last Month Sales": "${:,.2f}", "Forecasted Sales": "${:,.2f}"}))
            
            # Download
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Forecast CSV",
                csv,
                "next_month_forecast.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.error("No forecasts were generated. Please check model files.")
