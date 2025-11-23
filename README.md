---

# EV Charging Load Prediction — Big Data & Machine Learning Pipeline

### **End-to-End Real-Time Analytics using Apache Spark, PySpark, Random Forest & Streamlit Dashboard**

This project is a **full end-to-end EV Charging Load Prediction System** built using **Apache Spark** for large-scale data preprocessing and **RandomForest Regression** for machine learning.
A fully interactive **Streamlit dashboard** provides real-time visual insights, dataset processing, charts, predictions, and downloadable PDF reports.

This project is designed as a **complete 60-mark Big Data + ML practical**, with fully automated pipeline execution.

---

# Why This Project Stands Out

This isn’t a simple ML project. It is a **big-data powered forecasting engine**:

🔹 Handles *large CSV datasets* automatically using **Apache Spark**
🔹 Performs automatic **timestamp parsing**, **categorical removal**, **null handling**, and **feature extraction**
🔹 Uses **RandomForestRegressor** to predict hourly charging load
🔹 Produces **interactive visual analytics**
🔹 Generates **professional PDF reports** with plots and metrics
🔹 Supports **live dataset upload**
🔹 Works even if the model file is missing (auto-training included)

---

# Key Features

### **1. Big Data Preprocessing using Apache Spark**

* Multi-format timestamp parsing
* Automatic cleaning & null-row removal
* Spark DataFrame → Pandas conversion
* Categorical columns dropped for ML compatibility
* Ready for scale-up to HDFS, Spark Clusters, AWS EMR

### **2. Machine Learning using Random Forest Regression**

* Automatic train-test split
* RandomForest with 200 trees
* Evaluation metrics:

  * R² Score
  * RMSE
* Prediction column added as **Predicted_Load_kW**

### **3. Streamlit Dashboard — Real-Time Visualization**

* Upload custom datasets OR use the sample dataset
* Automatic Spark preprocessing
* Interactive charts:

  * Time Series (Actual vs Predicted)
  * Scatter Plot
  * Error Distribution Histogram
* Clean UI with theme support (Dark / Light)
* Project metadata display

### **4. Auto-Generated PDF Reports**

Includes:

* Model metrics
* Dataset summary
* All charts (embedded)
* Project heading & description

The report can directly be used in **viva submissions, 60-mark practical records, and documentation**.

### **5. Automated Model Training**

If model doesn’t exist:

* System automatically trains a new model
* Saves under `model/ev_load_model.joblib`

### **6. Modular & Scalable Architecture**

Ready for:

* Spark Streaming
* MLOps pipelines
* Deployments in cloud
* Integration with APIs

---

# Project Structure

```
EV-Charging-Predictions-BigData/
│
├── dataset/
│   └── ev_charging_load.csv         # Sample dataset
│
├── model/
│   └── ev_load_model.joblib         # Auto-generated ML model
│
├── spark_app.py                     # Apache Spark + ML pipeline
├── clean_ev_data.py                 # Optional data cleaning utilities
├── streamlit_app.py                 # Interactive dashboard UI
├── requirements.txt                 # All dependencies
└── README.md                        # Documentation
```

---

# End-to-End Pipeline Explanation (For Viva)

### **1. Dataset Input**

User uploads a CSV OR uses sample dataset.

### **2. Spark Processing**

* Spark session created
* Timestamps cleaned (supports 4+ formats)
* Categorical irrelevant columns removed
* Nulls eliminated
* Output returned as Pandas DataFrame

### **3. Model Training**

RandomForestRegressor is trained on numeric features.

### **4. Prediction Generation**

Adds:

```
Predicted_Load_kW
```

column based on Spark-cleaned features.

### **5. Streamlit Visualization**

Displays:

* Metrics (R², RMSE)
* Time-series plot
* Scatter plot
* Error histogram
* First 100 cleaned rows

### **6. Export Outputs**

Download:

* Predictions CSV
* Professional PDF report

---

# Sample Visuals (Displayed in Streamlit)

* Actual vs Predicted Load Over Time
* Scatter: Actual vs Predicted
* Error Distribution Histogram

Each chart is also embedded into the downloadable PDF.

---

# 📥 Setup & Run

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/Antony-Godwin24/EV-Charging-Predictions-BigData.git
cd EV-Charging-Predictions-BigData
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Data Cleaning (Optional)

```bash
python clean_ev_data.py
```

### 5️⃣ Train Spark Model (Auto-trains if missing)

```bash
python spark_app.py
```

Model saved at:

```
model/ev_load_model.joblib
```

### 6️⃣ Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

# Future Enhancements (Roadmap)

### Database Integration

* MySQL / MongoDB for fleet data storage
* Cloud data ingestion (AWS S3 / GCP / Azure Blob)

### Real-Time Forecasting

* Spark Structured Streaming
* Kafka integration

### Charging Station Optimization

* Predict wait times
* Suggest best charging slot
* Cluster EVs by usage pattern

### Geospatial Intelligence

* Heatmaps of EV charging demand
* Traffic + weather correlation

---

# Project Summary

This project demonstrates:

* **Big Data Engineering** with Spark
* **Machine Learning** using Random Forest
* **Data Cleaning**
* **Feature Engineering**
* **Full-stack visualization** using Streamlit
* **PDF report generation**
* **Practical deployment architecture**

It's a **complete, industry-grade** pipeline suitable for academic evaluation, resume showcase, and real-world EV infrastructure forecasting.

---

# 👨Author

**Antony Godwin**
🚀 MERN & Java Spring Boot Full-Stack Developer
⚡ Big Data & ML Engineer (in progress)
📍 Tamil Nadu, India

GitHub → [https://github.com/Antony-Godwin24](https://github.com/Antony-Godwin24)

---

# 🏷 License

MIT License © 2025 Antony Godwin

---
