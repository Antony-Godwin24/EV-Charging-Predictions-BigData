# ⚡ EV-Charging-Predictions-BigData

A Big Data–driven project for **forecasting EV charging load** using **Apache Spark, PySpark, and Scikit-Learn**.  
The model predicts **hourly charging demand** based on real-world parameters like SoC, distance, weather, and traffic.

---

## 🚀 Features
- ✅ Data preprocessing and aggregation with **PySpark**
- ⚙️ Hourly energy load prediction using **Random Forest Regression**
- 📊 Interactive **Streamlit dashboard** for visualization and insights
- 🔁 Auto-retrain support — rebuilds model automatically if missing
- 🧠 Modular structure for future integration with ML pipelines or APIs

---

## 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python 3.11 |
| Big Data Engine | Apache Spark (PySpark) |
| ML Framework | scikit-learn |
| Visualization | Streamlit, Matplotlib, Seaborn |
| Data Storage | CSV (can be scaled to Hadoop / HDFS) |

---

## 🗂️ Project Structure

```

EV-Charging-Predictions-BigData/
├── dataset/
│   └── ev_charging_load.csv          # Raw dataset
├── model/
│   └── ev_load_model.joblib          # Trained model (auto-generated)
├── spark_app.py                      # Spark + ML pipeline
├── clean_ev_data.py                  # Data cleaning script
├── streamlit_app.py                  # Interactive dashboard
├── requirements.txt                  # Dependencies
└── README.md                         # This file

````

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Antony-Godwin24/EV-Charging-Predictions-BigData.git
cd EV-Charging-Predictions-BigData
````

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run data cleaning

```bash
python clean_ev_data.py
```

### 5️⃣ Train the Spark model

```bash
python spark_app.py
```

> ⚠️ **Note:** The trained model file (`model/ev_load_model.joblib`) is not included in the repository (too large for GitHub).
> When you run `spark_app.py`, it will automatically **train and recreate the model** if it’s missing.

### 6️⃣ Launch the dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📈 Future Enhancements

* Integrate MongoDB / MySQL for dynamic EV fleet data.
* Add predictive analytics for **charging station optimization**.
* Deploy Streamlit app on cloud (Streamlit Cloud / AWS / Render).
* Include **real-time forecasting** using Spark Streaming.

---

## 👨‍💻 Author

**Antony Godwin**
🚀 MERN & Java Full Stack Developer | CSE @ BE | Data Engineering Enthusiast
📧 Reach me: [Antony-Godwin24](https://github.com/Antony-Godwin24)

---

## 🏷️ License

MIT License © 2025 Antony Godwin

````

