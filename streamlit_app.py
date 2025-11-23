# streamlit_app.py

import io
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import tempfile
from datetime import datetime
from fpdf import FPDF

from spark_app import run_full_pipeline_from_df, TARGET_COL, TIMESTAMP_COL


# ---------------------------
# THEME SETUP (LIGHT / DARK)
# ---------------------------
THEMES = {
    "light": {
        "bg": "#f5f7fb",
        "bg_elevated": "#ffffff",
        "bg_deep": "#e6edf7",
        "text": "#111827",
        "text_muted": "#6b7280",
        "accent": "#1f8eed",      # EV blue
        "accent_soft": "#dbeafe",
        "accent_alt": "#22c55e",  # green
        "danger": "#ef4444",
        "card_shadow": "0 18px 45px rgba(15, 23, 42, 0.07)",
    },
    "dark": {
        "bg": "#020617",
        "bg_elevated": "#020617",
        "bg_deep": "#020617",
        "text": "#e5e7eb",
        "text_muted": "#9ca3af",
        "accent": "#38bdf8",      # lighter blue
        "accent_soft": "#0f172a",
        "accent_alt": "#22c55e",
        "danger": "#f97373",
        "card_shadow": "0 22px 55px rgba(0,0,0,0.75)",
    },
}


def inject_css(theme_name: str):
    t = THEMES[theme_name]

    st.markdown(
        f"""
        <style>
        /* ------------------ GLOBAL ------------------ */
        .stApp {{
            background: radial-gradient(circle at top left, 
                #0f172a 0, #020617 18%, {t['bg']} 50%, {t['bg']} 100%);
            color: {t['text']};
        }}
        .block-container {{
            padding-top: 3rem;
            padding-bottom: 3rem;
            padding-left: 4rem;
            padding-right: 4rem;
            max-width: 100%;
        }}
        h1, h2, h3, h4 {{
            color: {t['text']} !important;
        }}
        p, label, span, div {{
            color: {t['text_muted']} !important;
        }}

        /* ------------------ NAVBAR ------------------ */
        .ev-navbar {{
            width: 100%;
            border-radius: 18px;
            padding: 1.1rem 1.8rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(90deg, #0f172a, {t['accent']});
            box-shadow: {t['card_shadow']};
        }}
        .ev-navbar-left {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}
        .ev-logo {{
            width: 38px;
            height: 38px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at 30% 0%, #facc15, #f97316 40%, #ef4444 75%);
            box-shadow: 0 0 18px rgba(248,250,252,0.45);
            font-size: 1.4rem;
        }}
        
        .ev-title-main {{
            font-weight: 600;
            font-size: 1.2rem;
            color: {t['text']} !important;
        }}
        .ev-title-sub {{
            font-size: 0.78rem;
            color: {t['text_muted']} !important;
        }}

        .ev-chip {{
            font-size: 0.78rem;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: {t['accent_soft']};
            color: {t['accent']} !important;
            font-weight: 600;
            border: 1px solid rgba(148,163,184,0.25);
        }}

        /* Theme toggle button */
        .ev-theme-btn {{
            padding: 10px 20px;
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.35);
            background: {t['accent']};
            color: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }}

        /* ------------------ CARDS ------------------ */
        .ev-card {{
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            background: {t['bg_elevated']};
            box-shadow: {t['card_shadow']};
            margin-top: 1.4rem;
        }}

        .ev-card-hero {{
            border-radius: 24px;
            padding: 1.7rem 2rem;
            background: linear-gradient(135deg, {t['bg_deep']}, {t['accent_soft']});
            box-shadow: {t['card_shadow']};
            margin-top: 1.4rem;
            margin-bottom: 1.6rem;
        }}

        .ev-section-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: {t['text']} !important;
        }}

        .ev-section-caption {{
            font-size: 0.88rem;
            color: {t['text_muted']} !important;
        }}

        /* ------------------ METRIC CARDS ------------------ */
        .ev-metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 1.1rem;
            margin-top: 1rem;
        }}
        .ev-metric-card {{
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: {t['bg_elevated']};
            box-shadow: {t['card_shadow']};
            border: 1px solid rgba(148,163,184,0.18);
        }}
        .ev-metric-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {t['text_muted']} !important;
            margin-bottom: 0.3rem;
        }}
        .ev-metric-value {{
            font-size: 1.32rem;
            font-weight: 600;
            color: {t['text']} !important;
        }}

        /* ------------------ BUTTONS ------------------ */
        .stButton>button, .stDownloadButton>button {{
            width: 100%;
            padding: 0.9rem 1.1rem;
            border-radius: 999px;
            border: none;
            font-weight: 500;
            font-size: 0.95rem;
            background: {t['accent']};
            color: white !important;
            box-shadow: 0 14px 30px rgba(15,23,42,0.35);
        }}

        .stButton>button:hover, .stDownloadButton>button:hover {{
            filter: brightness(1.05);
            transform: translateY(-1px);
        }}

        /* Secondary button styling */
        .ev-secondary-btn>button {{
            background: transparent !important;
            color: {t['accent']} !important;
            border: 1px solid rgba(148,163,184,0.6) !important;
            box-shadow: none !important;
        }}

        /* ------------------ TABLE / FILE UPLOADER ------------------ */
        section[data-testid="stFileUploader"] > div {{
            border-radius: 18px;
            background: {t['bg_elevated']};
        }}

        /* ------------------ CHART CARDS ------------------ */
        .ev-chart-card {{
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            background: {t['bg_elevated']};
            box-shadow: {t['card_shadow']};
        }}

        /* Small badge */
        .ev-badge-soft {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: {t['accent_soft']};
            color: {t['accent']};
            font-size: 0.75rem;
            font-weight: 500;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )



# ---------------------------
# PDF REPORT
# ---------------------------
def generate_pdf_report(df, metrics, charts):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EV Charging Load Prediction Report", ln=True, align="C")

    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        6,
        "This report summarizes Spark preprocessing and RandomForest predictions "
        "for EV charging load based on the uploaded dataset.",
    )

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Model Performance", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"R² Score: {metrics['r2']:.4f}", ln=True)
    pdf.cell(0, 6, f"RMSE: {metrics['rmse']:.4f}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Dataset Overview", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Rows: {len(df)}", ln=True)
    pdf.cell(0, 6, f"Columns: {len(df.columns)}", ln=True)

    # ----------------------------
    # FIXED: Save each fig safely
    # ----------------------------
    import matplotlib.pyplot as plt
    import tempfile

    for title, fig in charts.items():
        pdf.add_page()
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, title, ln=True)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches="tight")
            pdf.image(tmp.name, x=10, y=28, w=190)

        plt.close(fig)  # <-- CRITICAL FIX

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes



# ---------------------------
# CHARTS
# ---------------------------
def plot_time_series(df):
    if TIMESTAMP_COL not in df.columns or TARGET_COL not in df.columns:
        return None

    df2 = df.copy()
    df2[TIMESTAMP_COL] = pd.to_datetime(df2[TIMESTAMP_COL])
    df2 = df2.sort_values(TIMESTAMP_COL)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df2[TIMESTAMP_COL], df2[TARGET_COL], label="Actual Load", linewidth=1.2)
    if "Predicted_Load_kW" in df2.columns:
        ax.plot(
            df2[TIMESTAMP_COL],
            df2["Predicted_Load_kW"],
            linestyle="--",
            label="Predicted Load",
            linewidth=1.1,
        )
    ax.set_title("Actual vs Predicted Load Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("kW")
    ax.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


def plot_actual_vs_pred_scatter(df):
    if "Predicted_Load_kW" not in df.columns or TARGET_COL not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.scatter(df[TARGET_COL], df["Predicted_Load_kW"], alpha=0.4)
    ax.set_title("Actual vs Predicted (Scatter)")
    ax.set_xlabel("Actual Load (kW)")
    ax.set_ylabel("Predicted Load (kW)")
    plt.tight_layout()
    return fig


def plot_error_hist(df):
    if "Predicted_Load_kW" not in df.columns or TARGET_COL not in df.columns:
        return None
    errors = df["Predicted_Load_kW"] - df[TARGET_COL]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(errors, bins=40)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (Predicted - Actual)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


# ---------------------------
# MAIN UI
# ---------------------------
def main():
    # Session theme
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    theme = st.session_state.theme
    inject_css(theme)

    st.set_page_config(
        page_title="EV Charging Load Predictions",
        layout="wide",
    )

    # # --------------------------------------------------
    # # CLEAN FIXED NAVBAR (NO LEAKED CODE)
    # # --------------------------------------------------

    # col_nav = st.container()
    # with col_nav:

    #     navbar_html = f"""
    #     <style>
    #         .ev-navbar {{
    #             display: flex;
    #             justify-content: space-between;
    #             align-items: center;
    #             padding: 18px 30px;
    #             width: 100%;
    #             border-radius: 12px;
    #             background: {'#0d1117' if theme=='dark' else '#e9f5ff'};
    #             box-shadow: 0 3px 12px rgba(0,0,0,0.25);
    #         }}
    #         .ev-navbar-left {{
    #             display: flex;
    #             gap: 18px;
    #             align-items: center;
    #         }}
    #         .ev-logo {{
    #             font-size: 33px;
    #         }}
    #         .ev-title-main {{
    #             font-size: 21px;
    #             font-weight: 700;
    #             color: {'white' if theme=='dark' else '#000'};
    #         }}
    #         .ev-title-sub {{
    #             font-size: 13px;
    #             color: {'#9ca3af' if theme=='dark' else '#333'};
    #         }}
    #         .ev-navbar-right {{
    #             display: flex;
    #             align-items: center;
    #             gap: 20px;
    #         }}
    #         .ev-chip {{
    #             padding: 6px 14px;
    #             border-radius: 20px;
    #             font-size: 13px;
    #             background: {'#1f2937' if theme=='dark' else '#d0e7ff'};
    #             color: {'white' if theme=='dark' else '#003366'};
    #             font-weight: 600;
    #         }}
    #         .ev-theme-btn {{
    #             padding: 10px 20px;
    #             border-radius: 25px;
    #             border: none;
    #             background: #3ea8ff;
    #             color: black;
    #             font-size: 14px;
    #             font-weight: 600;
    #             cursor: pointer;
    #         }}
    #     </style>

    #     <div class="ev-navbar">

    #         <div class="ev-navbar-left">
    #             <div class="ev-logo">⚡</div>
    #             <div>
    #                 <div class="ev-title-main">EV Charging Load Predictions</div>
    #                 <div class="ev-title-sub">Apache Spark · RandomForest · Streamlit Dashboard</div>
    #             </div>
    #         </div>

    #         <div class="ev-navbar-right">
    #             <div class="ev-chip">Big Data Ready</div>
    #             <div class="ev-chip">PDF Reports</div>
    #         </div>

    #     </div>
    #     """

    #     st.markdown(navbar_html, unsafe_allow_html=True)

    #     # THEME TOGGLE BUTTON
    #     toggle_label = "☀ Light Mode" if theme == "dark" else "🌙 Dark Mode"

    #     if st.button(toggle_label, key="real_theme_toggle"):
    #         st.session_state.theme = "light" if theme == "dark" else "dark"
    #         st.rerun()



    # HERO SECTION
    st.markdown(
        """
        <div class="ev-card-hero">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
            <div style="max-width:60%;">
              <div class="ev-badge-soft">Live Big Data Practical Dashboard</div>
              <h2 style="margin-top:0.6rem; margin-bottom:0.3rem;">End-to-End EV Charging Analytics</h2>
              <p style="font-size:0.92rem;">
                Upload an EV charging CSV or use the built-in sample dataset. Spark will clean and preprocess the data, 
                a RandomForest model will predict charging load, and you get interactive charts plus a downloadable PDF report – 
                perfect for a 60-mark exam or viva.
              </p>
            </div>
            <div style="text-align:right; min-width:180px;">
              <div style="font-size:0.8rem; color:#9ca3af;">Current run</div>
              <div style="font-size:1.4rem; font-weight:600;">EV-Spark-RF</div>
              <div style="font-size:0.78rem; margin-top:0.3rem; color:#9ca3af;">v1.0 · Antony's project</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- DATA SOURCE CARD ----------
    st.markdown(
        """
        <div class="ev-card">
          <div class="ev-section-title">1. Choose Data Source</div>
          <div class="ev-section-caption">
            Either upload your own EV charging CSV or use the built-in dataset (<code>dataset/ev_charging_load.csv</code>).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Put actual widgets after that card (Streamlit always renders in order)
    mode_col = st.container()
    with mode_col:
        source_mode = st.radio(
            "Data Source",
            ["Upload CSV", "Use local sample dataset"],
            horizontal=True,
            label_visibility="collapsed",
        )

        df = None

        if source_mode == "Use local sample dataset":
            try:
                df = pd.read_csv("dataset/ev_charging_load.csv")
                st.success("Loaded local dataset: `dataset/ev_charging_load.csv`")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load local dataset: {e}")
                return
        else:
            uploaded = st.file_uploader(
                "Upload EV Charging Dataset (CSV)",
                type=["csv"],
                help="Supports large CSV files; processing is done with Apache Spark.",
            )
            if not uploaded:
                st.info("Upload a CSV file to continue.")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                tmp.flush()
                df = pd.read_csv(tmp.name)
            st.success("CSV uploaded successfully.")
            st.dataframe(df.head(), use_container_width=True)

    # ---------- RUN PIPELINE ----------
    st.markdown(
        """
        <div class="ev-card">
          <div class="ev-section-title">2. Run Spark + RandomForest Pipeline</div>
          <div class="ev-section-caption">
            Spark will clean timestamps and remove categorical columns, then a RandomForestRegressor will be trained 
            to predict <code>Charging_Load_kW</code>. You’ll get metrics, processed data and visualizations.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Running Spark preprocessing and training RandomForest model…"):
        try:
            predicted_df, metrics = run_full_pipeline_from_df(df)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

    st.success("Pipeline completed successfully ✅")

    # ---------- METRICS + DATA SUMMARY ----------
    rows, cols = predicted_df.shape

    st.markdown(
        f"""
        <div class="ev-metric-grid">
          <div class="ev-metric-card">
            <div class="ev-metric-label">Rows</div>
            <div class="ev-metric-value">{rows:,}</div>
          </div>
          <div class="ev-metric-card">
            <div class="ev-metric-label">Columns</div>
            <div class="ev-metric-value">{cols}</div>
          </div>
          <div class="ev-metric-card">
            <div class="ev-metric-label">R² Score</div>
            <div class="ev-metric-value">{metrics['r2']:.4f}</div>
          </div>
          <div class="ev-metric-card">
            <div class="ev-metric-label">RMSE</div>
            <div class="ev-metric-value">{metrics['rmse']:.4f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- PROCESSED DATA ----------
    st.markdown(
        """
        <div class="ev-card" style="margin-top:1.8rem;">
          <div class="ev-section-title">3. Processed Data with Predictions</div>
          <div class="ev-section-caption">
            This table shows the Spark-cleaned dataset with the <code>Predicted_Load_kW</code> column added by the model.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(predicted_df.head(100), use_container_width=True)

    # ---------- CHARTS ----------
    # ---------- CHARTS ----------
    st.markdown(
        """
        <div class="ev-card" style="margin-top:1.8rem; margin-bottom:1.2rem;">
        <div class="ev-section-title">4. Visual Analytics</div>
        <div class="ev-section-caption">
            Time series, scatter and error-distribution views help you explain model behaviour in your viva.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    charts = {}

    # TIME SERIES
    fig_ts = plot_time_series(predicted_df)
    if fig_ts:
        st.markdown('<div class="ev-chart-card">', unsafe_allow_html=True)
        st.pyplot(fig_ts)  # FIXED
        st.markdown("</div>", unsafe_allow_html=True)
        charts["Actual vs Predicted Load Over Time"] = fig_ts

    # SCATTER
    fig_scatter = plot_actual_vs_pred_scatter(predicted_df)
    if fig_scatter:
        st.markdown('<div class="ev-chart-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.pyplot(fig_scatter)  # FIXED
        st.markdown("</div>", unsafe_allow_html=True)
        charts["Actual vs Predicted Scatter"] = fig_scatter

    # ERROR HIST
    fig_err = plot_error_hist(predicted_df)
    if fig_err:
        st.markdown('<div class="ev-chart-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.pyplot(fig_err)  # FIXED
        st.markdown("</div>", unsafe_allow_html=True)
        charts["Prediction Error Distribution"] = fig_err


    # ---------- DOWNLOADS ----------
    st.markdown(
        """
        <div class="ev-card" style="margin-top:2rem;">
          <div class="ev-section-title">5. Export Results</div>
          <div class="ev-section-caption">
            Download the processed predictions as CSV and a PDF report to attach directly in your project record or viva submission.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cdl1, cdl2 = st.columns(2)

    with cdl1:
        csv_bytes = predicted_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Predictions CSV",
            data=csv_bytes,
            file_name=f"ev_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

    with cdl2:
        pdf_bytes = generate_pdf_report(predicted_df, metrics, charts)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"ev_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
