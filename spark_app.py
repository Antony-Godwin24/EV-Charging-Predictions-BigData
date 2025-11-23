# spark_app.py

import os
from typing import Tuple, Dict

import joblib
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ----------------------------
# CONSTANTS
# ----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ev_load_model.joblib")

TIMESTAMP_COL = "Date_Time"
TARGET_COL = "Charging_Load_kW"

CATEGORICAL_COLS = [
    "Traffic_Data",
    "Road_Conditions",
    "Charging_Station_ID",
    "Fleet_Schedule",
    "Charging_Preferences",
    "Weather_Conditions",
    "Weekday",
]


def get_spark(app_name: str = "EV_Load_Predictions") -> SparkSession:
    """
    Create a Spark session (used only for big-data style preprocessing).
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Clean + parse the Date_Time column using multiple formats in Pandas
    BEFORE sending to Spark, so Spark doesn't choke on NaN/garbage.
    """
    series = series.astype(str)
    series = series.replace(["NaN", "nan", "None", ""], pd.NA)

    def parse_one(x: str):
        if pd.isna(x):
            return pd.NaT

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M",
            "%Y/%m/%d %H:%M",
        ]
        for fmt in formats:
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                continue
        return pd.NaT

    return series.apply(parse_one)


def preprocess_with_spark(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    1) Fix timestamp in Pandas using multiple formats
    2) Push to Spark for big-data style cleaning
    3) Drop categorical columns + null target rows
    """

    # 1) Handle timestamp in Pandas first
    if TIMESTAMP_COL in pdf.columns:
        pdf[TIMESTAMP_COL] = _parse_timestamp_series(pdf[TIMESTAMP_COL])
        pdf = pdf.dropna(subset=[TIMESTAMP_COL])

    # 2) Now send clean-ish frame to Spark
    spark = get_spark()
    sdf = spark.createDataFrame(pdf)

    # 3) Drop categorical columns
    drop_cols = [c for c in CATEGORICAL_COLS if c in sdf.columns]
    if drop_cols:
        sdf = sdf.drop(*drop_cols)

    # 4) Drop rows missing target
    if TARGET_COL in sdf.columns:
        sdf = sdf.dropna(subset=[TARGET_COL])

    cleaned = sdf.toPandas()
    spark.stop()
    return cleaned


def train_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train RandomForest on numeric columns, return model + metrics.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    feature_cols = [
        c for c in df.columns
        if c != TARGET_COL and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found to train on.")

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # NOTE: no 'squared' kwarg here (for older sklearn compat)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(mse ** 0.5)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": rmse,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, metrics


def predict_with_model(model: RandomForestRegressor, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c for c in df.columns
        if c != TARGET_COL and pd.api.types.is_numeric_dtype(df[c])
    ]

    preds = model.predict(df[feature_cols])
    df2 = df.copy()
    df2["Predicted_Load_kW"] = preds
    return df2


def run_full_pipeline_from_df(raw_df: pd.DataFrame):
    """
    Entry point used by Streamlit:
    - Pandas + Spark cleaning
    - Train RF model
    - Predict & return metrics
    """
    cleaned = preprocess_with_spark(raw_df)
    model, metrics = train_model(cleaned)
    predicted = predict_with_model(model, cleaned)
    return predicted, metrics


# Expose constants for Streamlit
__all__ = [
    "run_full_pipeline_from_df",
    "TIMESTAMP_COL",
    "TARGET_COL",
]
