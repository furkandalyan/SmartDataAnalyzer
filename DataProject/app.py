import streamlit as st
import pandas as pd
import os
import uuid
from utils import *

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer")
st.markdown("Upload your dataset to begin automatic analysis, visualization, and reporting.")

uploaded_file = st.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_ext == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.success(f"File loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        
        charts = []

        for func in [
            analyze_missing_data,
            plot_missing_heatmap,
            analyze_trends,
            analyze_correlations,
            analyze_pca,
            analyze_scatter_matrix
        ]:
            result = func(df)
            if result:
                title, explanation, fig = result
                chart_path = f"charts/{uuid.uuid4().hex}.png"
                save_plot(fig, chart_path)
                charts.append((title, chart_path))
                with st.expander(title):
                    st.markdown(f"**{explanation}**")
                    st.pyplot(fig)

        for func_group in [
            analyze_categorical,
            analyze_value_counts,
            analyze_numeric_distributions,
            analyze_boxplots
        ]:
            results = func_group(df)
            if results:
                for title, explanation, fig in results:
                    chart_path = f"charts/{uuid.uuid4().hex}.png"
                    save_plot(fig, chart_path)
                    charts.append((title, chart_path))
                    with st.expander(title):
                        st.markdown(f"**{explanation}**")
                        st.pyplot(fig)

        
        insights = generate_smart_insights(df)
        if insights:
            st.markdown("### 🔍 Smart Insights")
            for item in insights:
                st.info(item)

        
        outlier_info = detect_outliers(df)
        if outlier_info:
            st.markdown("### 🚨 Outlier Detection")
            for line in outlier_info:
                st.warning(line)

        
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Creating PDF report..."):
                report_path = f"reports/report_{uuid.uuid4().hex}.pdf"
                generate_pdf_report(df, charts, report_path)
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="Download Report as PDF",
                        data=f,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf"
                    )
    except Exception as e:
        st.error(f"❌ Error: {e}")
