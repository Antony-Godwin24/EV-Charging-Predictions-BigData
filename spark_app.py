from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col, hour, dayofweek, window, mean as _mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib
import os

# Initialize Spark (simplified for Codespaces)
spark = SparkSession.builder.appName("EV_Load_Predictions").getOrCreate()

# Load cleaned dataset
df = spark.read.option("header", True).option("inferSchema", True).csv("dataset/ev_charging_load_clean.csv")

# Convert Date_Time to timestamp
df = df.withColumn("timestamp", to_timestamp(col("Date_Time")))

# Aggregate average load per hour
agg_df = df.groupBy(window(col("timestamp"), "1 hour").alias("window")).agg(
    _mean("Charging_Load_kW").alias("avg_load")
)
agg_df = agg_df.select(col("window.start").alias("start_time"), col("avg_load"))

# Add time features
agg_df = agg_df.withColumn("hour", hour(col("start_time"))).withColumn("weekday", dayofweek(col("start_time")))

# Convert to Pandas
pdf = agg_df.toPandas().dropna().sort_values("start_time")

# Lag features
pdf["load_lag1"] = pdf["avg_load"].shift(1)
pdf["load_lag24"] = pdf["avg_load"].shift(24)
pdf = pdf.dropna()

# Split data
train = pdf[:-24*7]
test = pdf[-24*7:]

X_train = train[["load_lag1", "load_lag24", "hour", "weekday"]]
y_train = train["avg_load"]
X_test = test[["load_lag1", "load_lag24", "hour", "weekday"]]
y_test = test["avg_load"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Model Performance → MAE: {mae:.2f}, R²: {r2:.2f}")

# Save model + processed data
os.makedirs("model", exist_ok=True)
pdf.to_csv("dataset/ev_hourly_agg.csv", index=False)
joblib.dump(model, "model/ev_load_model.joblib")

spark.stop()
print("✅ Model trained and saved successfully!")
