
📊 AI-Driven Sales Forecasting System

An end-to-end Machine Learning + Business Intelligence project that predicts next-month sales for 50+ stores, evaluates model performance against a business baseline, and visualizes results in an interactive dashboard using Power BI.

🚀 Project Objective
To help a retail company:
Predict next-month sales
Identify high-performing vs low-performing store groups
Improve planning for inventory, promotions, and staffing
Reduce manual reporting efforts

🧠 Key Outcomes
✅ Built a complete time-series forecasting pipeline
✅ Applied store clustering for segmented modeling
✅ Achieved 27.41% improvement in forecast accuracy over baseline
✅ Automated next-month predictions
✅ Designed a management-ready Power BI dashboard

🛠️ Tech Stack
Category	Tools
Programming	Python
Data Handling	Pandas, NumPy
Machine Learning	Scikit-learn (RandomForest, KMeans)
Visualization	Power BI
Data Source	Retail Sales CSV
Concepts	Time Series, Feature Engineering, Clustering, Model Evaluation
📁 Project Structure
sales-forecasting-project/
│
├── data/
│   ├── raw/
│   │   └── 50000 Sales Records.csv
│   └── processed/
│       ├── monthly_sales.csv
│       ├── features.csv
│       ├── features_with_clusters.csv
│       └── next_month_forecast_cluster.csv
│
├── models/
│   ├── xgb_cluster_0.pkl
│   ├── xgb_cluster_1.pkl
│   └── xgb_cluster_2.pkl
│
├── src/
│   ├── train_cluster_models.py
│   └── predict_next_month_cluster.py
│
├── dashboard/
│   └── AI_Driven_Sales_Forecasting.pbix
│
└── README.md

📊 Workflow Overview
1️⃣ Data Preprocessing

Loaded raw retail sales data
Converted daily transactions into monthly sales per store
Cleaned and structured the dataset

2️⃣ Feature Engineering

Created:
Time features → year, month, quarter, weekofyear
Lag features → lag_1, lag_2, lag_3, lag_6, lag_12, lag_15
Rolling averages → rolling_3_mean, rolling_6_mean

3️⃣ Store Clustering

Aggregated store-level statistics:
Average sales
Volatility
Max sales
Applied KMeans clustering
Stores grouped into:
Cluster 0 → Low sales
Cluster 1 → Medium sales
Cluster 2 → High sales

4️⃣ Model Training (Cluster-wise)

Trained separate RandomForest models for each cluster
Used a time-based train–validation split
Evaluation Metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)

5️⃣ Baseline vs ML Evaluation
Model	MAE
Baseline (Last Month = Next Month)	3,583,565
ML Forecast Model (Cluster-wise RF)	2,601,213

✅ Accuracy Improvement: 27.41%

6️⃣ Automated Next-Month Forecasting

Generated predictions for all stores using trained models
Output saved to:
data/processed/next_month_forecast_cluster.csv

7️⃣ Power BI Dashboard

Dashboard includes:
✅ KPI Card → Total Forecasted Sales
✅ Store-wise Forecast Bar Chart
✅ Cluster-wise Sales Contribution
✅ Last Month vs Forecast Line Chart
✅ Detailed Store Comparison Table

▶️ How to Run the Project
🔹 Step 1: Install Requirements
pip install pandas scikit-learn joblib

🔹 Step 2: Train Models
python src/train_cluster_models.py

🔹 Step 3: Generate Forecast
python src/predict_next_month_cluster.py

🔹 Step 4: Open Power BI Dashboard

Open:

dashboard/AI_Driven_Sales_Forecasting.pbix
🧪 Evaluation Metrics Used
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
% Improvement over Baseline
