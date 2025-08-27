# app.py
# Enhanced Damper Failure Analysis Dashboard
# Modified for Streamlit Cloud deployment with secrets
import os
import time
import warnings
from io import BytesIO
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
warnings.filterwarnings("ignore")
import logging
import json

def setup_logging():
    """Setup proper logging for debugging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def safe_column_check(df, required_columns, operation_name="operation"):
    """Check if required columns exist before processing"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.warning(f"Missing columns for {operation_name}: {missing}")
        return False
    return True

def validate_environment():
    """Check if all required secrets exist in Streamlit Cloud"""
    issues = []
    
    # Check Google credentials in secrets
    if "gcp_service_account" not in st.secrets:
        issues.append("'gcp_service_account' secret not found in Streamlit Cloud secrets")
    
    # Check Google Sheet key in secrets
    if "google_sheet_key" not in st.secrets:
        issues.append("'google_sheet_key' secret not found in Streamlit Cloud secrets")
    
    return issues

def validate_data_quality(df):
    """Generate a data quality report"""
    if df.empty:
        return {"status": "empty", "issues": ["No data available"]}
    
    issues = []
    total_rows = len(df)
    
    # Check for required columns
    required = ["Test Result"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")
    
    # Check data completeness
    completeness = 0
    if "Test Result" in df.columns:
        valid_results = df["Test Result"].isin(["PASS", "FAIL"]).sum()
        completeness = valid_results / total_rows * 100
        if completeness < 50:
            issues.append(f"Low data quality: only {completeness:.1f}% valid test results")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    status = "good" if not issues else "warning" if len(issues) < 3 else "error"
    return {"status": status, "issues": issues, "completeness": completeness}

# Initialize logger
logger = setup_logging()

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Enhanced Damper Failure Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------
# CSS
# -----------------------------------
st.markdown(
    """
<style>
/* Cards & notices */
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 10px; color: white;
    text-align: center; margin: 0.5rem 0;
}
.insight-box {
    background: #f8f9fa; border-left: 4px solid #007bff;
    padding: 1rem; margin: 1rem 0; border-radius: 5px;
}
.warning-box {
    background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 1rem; margin: 1rem 0; border-radius: 5px;
}
.success-box {
    background: #d4edda; border-left: 4px solid #28a745;
    padding: 1rem; margin: 1rem 0; border-radius: 5px;
}
.error-box {
    background: #f8d7da; border-left: 4px solid #dc3545;
    padding: 1rem; margin: 1rem 0; border-radius: 5px; color: #721c24;
}
.drill-down-card {
    background: #e3f2fd; border: 1px solid #2196f3;
    padding: 1rem; margin: 0.5rem 0; border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Default Plotly layout
default_layout = dict(
    font=dict(family="Arial", size=12),
    margin=dict(l=50, r=50, t=100, b=50),
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=True,
)

# -----------------------------------
# Google Sheets Helpers (Modified for Streamlit Cloud)
# -----------------------------------
def _get_creds_and_client():
    """Get credentials and client from Streamlit Cloud secrets"""
    try:
        # Get credentials from Streamlit secrets
        service_account_info = st.secrets["gcp_service_account"]
        sheet_key = st.secrets["google_sheet_key"]
        
        # Convert to dictionary if it's not already
        if isinstance(service_account_info, str):
            service_account_info = json.loads(service_account_info)
        
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        
        creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
        client = gspread.authorize(creds)
        
        return creds, client, sheet_key
        
    except Exception as e:
        raise RuntimeError(f"Failed to get credentials from Streamlit secrets: {e}")

def test_google_sheets_connection():
    """Test Google Sheets connection and return status and sheet list."""
    try:
        _, client, sheet_key = _get_creds_and_client()
        spreadsheet = client.open_by_key(sheet_key)
        worksheets = [ws.title for ws in spreadsheet.worksheets()]
        return True, f"✅ Connected successfully. Available sheets: {', '.join(worksheets)}"
    except Exception as e:
        return False, f"Connection error: {e}"

@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def cached_load_sheet(sheet_name: str, row_limit: int = None):
    """Cached sheet load with optimization for large datasets"""
    try:
        _, client, sheet_key = _get_creds_and_client()
        spreadsheet = client.open_by_key(sheet_key)
        ws = spreadsheet.worksheet(sheet_name)
        
        if row_limit:
            # Only get first N rows for large datasets
            data = ws.get(f"A1:Z{row_limit}")
        else:
            data = ws.get_all_values()
            
        if not data or len(data) < 2:
            return pd.DataFrame()
            
        return pd.DataFrame(data[1:], columns=data[0])
        
    except gspread.exceptions.WorksheetNotFound:
        available = [ws.title for ws in spreadsheet.worksheets()]
        raise RuntimeError(f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(available)}")
    except Exception as e:
        logger.error(f"Sheet loading error: {e}")
        raise RuntimeError(f"Failed to load sheet: {e}")

def check_for_updates(sheet_name: str, current_row_count: int):
    """Quick check whether row count changed since initial load."""
    try:
        _, client, sheet_key = _get_creds_and_client()
        spreadsheet = client.open_by_key(sheet_key)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_values() or []
        new_row_count = len(data) - 1 if data else 0
        return new_row_count > current_row_count, new_row_count
    except Exception:
        return False, current_row_count

# -----------------------------------
# Enhanced Analytics Engine
# -----------------------------------
class EnhancedDamperAnalytics:
    def __init__(self):
        self.df = pd.DataFrame()
        self.insights = []
        self.connection_status = None
        self.last_update = None

    # ---------- LOAD & PREPROCESS ----------
    def load_sheet(self, sheet_name: str):
        try:
            is_connected, msg = test_google_sheets_connection()
            self.connection_status = msg
            if not is_connected:
                st.error(f"❌ Connection failed: {msg}")
                return pd.DataFrame()

            with st.spinner(f"Loading data from {sheet_name}..."):
                df = cached_load_sheet(sheet_name)

            if df.empty:
                st.warning(f"⚠️ '{sheet_name}' is empty or has only headers")
                return pd.DataFrame()

            self.last_update = datetime.now()
            st.sidebar.success(f"✅ {sheet_name}: {len(df)} rows loaded")
            st.sidebar.info(f"📋 Columns: {list(df.columns)}")
            return df

        except Exception as e:
            self.connection_status = f"Load error: {e}"
            st.error(f"❌ {self.connection_status}")
            return pd.DataFrame()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            st.warning("⚠️ No data to preprocess")
            return df

        df = df.copy()
        df.columns = df.columns.str.strip()
        st.sidebar.info(f"📊 Preprocessing started with {len(df)} rows")

        try:
            self._process_test_results(df)
            self._process_dates(df)
            self._process_age_data(df)
            self._process_categorical_data(df)
            st.sidebar.success(f"✅ Preprocessing completed: {len(df)} rows")
        except Exception as e:
            st.error(f"❌ Preprocessing error: {e}")
            st.write("Debug — shape:", df.shape)
            st.write("Debug — cols:", df.columns.tolist())

        return df

    def _process_test_results(self, df: pd.DataFrame):
        if "Test Result" not in df.columns:
            # Try alternative column names
            alt_names = ["TestResult", "Result", "test_result", "TEST_RESULT", "Status"]
            found_alt = None
            for alt in alt_names:
                if alt in df.columns:
                    df.rename(columns={alt: "Test Result"}, inplace=True)
                    found_alt = alt
                    st.sidebar.info(f"ℹ️ Renamed '{alt}' → 'Test Result'")
                    break
            
            if not found_alt:
                st.warning("⚠️ 'Test Result' column not found — creating default")
                df["Test Result"] = "UNKNOWN"

        df["Test Result"] = df["Test Result"].astype(str).str.strip().str.upper()
        
        mapping = {
            "PASS": "PASS", "PASSED": "PASS", "SUCCESS": "PASS", "OK": "PASS", "GOOD": "PASS",
            "FAIL": "FAIL", "FAILED": "FAIL", "FAILURE": "FAIL", "ERROR": "FAIL", "BAD": "FAIL", "REJECT": "FAIL",
            "NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN", " ": "UNKNOWN", "NULL": "UNKNOWN", "NA": "UNKNOWN", "N/A": "UNKNOWN",
        }
        
        unexpected = set(df["Test Result"].unique()) - set(mapping.keys())
        if unexpected:
            st.sidebar.warning(f"⚠️ Unexpected Test Result values (sample): {list(unexpected)[:5]}")

        df["Test Result"] = df["Test Result"].map(mapping).fillna("UNKNOWN")
        df["Is_Failure"] = (df["Test Result"] == "FAIL").astype(int)
        df["Is_Pass"] = (df["Test Result"] == "PASS").astype(int)
        
        st.sidebar.info(f"📊 Test Results: {df['Test Result'].value_counts().to_dict()}")

    def _process_dates(self, df: pd.DataFrame):
        if "Test date time" not in df.columns:
            st.sidebar.warning("⚠️ 'Test date time' column not found")
            return

        df["Original_Test_date_time"] = df["Test date time"].copy()
        df["Test date time"] = pd.to_datetime(df["Test date time"], errors="ignore", utc=False)
        df.loc[~pd.to_datetime(df["Test date time"], errors="coerce").notna(), "Test date time"] = np.nan
        df["Test date time"] = pd.to_datetime(df["Test date time"], errors="coerce", utc=False)

        fmts = [
            "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y",
            "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y",
        ]

        for fmt in fmts:
            mask = df["Test date time"].isna() & df["Original_Test_date_time"].notna()
            if not mask.any():
                break
            parsed = pd.to_datetime(df.loc[mask, "Original_Test_date_time"], format=fmt, errors="coerce")
            df.loc[mask, "Test date time"] = parsed

        failed = df["Test date time"].isna().sum()
        if failed:
            st.sidebar.warning(f"⚠️ Failed to parse {failed} dates")
        else:
            st.sidebar.success("✅ All dates parsed successfully")

        valid = df[df["Test date time"].notna()]
        if not valid.empty:
            df["Year"] = df["Test date time"].dt.year
            df["Month"] = df["Test date time"].dt.month
            df["Quarter"] = df["Test date time"].dt.quarter
            df["Weekday"] = df["Test date time"].dt.day_name()
            df["Hour"] = df["Test date time"].dt.hour
            st.sidebar.info(f"📅 Date range: {df['Test date time'].min().date()} → {df['Test date time'].max().date()}")

    def _process_age_data(self, df: pd.DataFrame):
        if "Age" not in df.columns:
            st.sidebar.warning("⚠️ 'Age' column not found")
            return

        df["Age"] = (
            df["Age"].astype(str)
            .str.replace(" days", "", regex=False)
            .str.replace(" day", "", regex=False)
            .str.replace("days", "", regex=False)
            .str.replace("day", "", regex=False)
            .str.strip()
        )

        df["Age_Numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        missing_ages = df["Age_Numeric"].isna().sum()
        if missing_ages > 0:
            st.sidebar.warning(f"⚠️ {missing_ages} missing age values → default 1460 days")
            df["Age_Numeric"] = df["Age_Numeric"].fillna(1460)

        df["Age_Years"] = df["Age_Numeric"] / 365.25
        
        bins = [-np.inf, 2, 3, 5, np.inf]
        labels = ["<2 years", "2–3 years", "3–5 years", ">5 years"]
        df["Age_Category"] = pd.cut(df["Age_Years"], bins=bins, labels=labels, right=True, include_lowest=True)

        def _risk(y):
            if y < 2: return 1
            if y < 3: return 2
            if y < 5: return 3
            return 4
        
        df["Age_Risk_Score"] = df["Age_Years"].apply(_risk)
        
        stats_age = df["Age_Years"].describe()
        if stats_age["count"] > 0:
            st.sidebar.info(f"📊 Age span: {stats_age['min']:.1f}–{stats_age['max']:.1f} years")

    def _process_categorical_data(self, df: pd.DataFrame):
        if "TYPE OF DAMPER" not in df.columns and "Shocker type" in df.columns:
            df.rename(columns={"Shocker type": "TYPE OF DAMPER"}, inplace=True)
            st.sidebar.info("ℹ️ Renamed 'Shocker type' → 'TYPE OF DAMPER'")

        for col in ["TYPE OF DAMPER", "Make"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
                st.sidebar.info(f"📊 {col}: {df[col].nunique()} unique")
            else:
                st.sidebar.warning(f"⚠️ Missing column: '{col}'")

    # ---------- ENHANCED CHARTS ----------
    def generate_sample_size_analysis(self, column: str):
        """Dual bar + line: failures/passes + failure %"""
        if self.df.empty or column not in self.df.columns:
            return None, pd.DataFrame()

        known_data = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])]
        if known_data.empty:
            return None, pd.DataFrame()

        analysis = (
            known_data.groupby(column)
            .agg({'Is_Failure': ['count', 'sum', 'mean'], 'Test Result': 'count'}).round(3)
        )
        analysis.columns = ['Total_Tests', 'Total_Failures', 'Failure_Rate', 'Sample_Size']
        analysis = analysis.reset_index()
        analysis['Failure_Rate_Pct'] = analysis['Failure_Rate'] * 100
        analysis['Pass_Count'] = analysis['Total_Tests'] - analysis['Total_Failures']
        analysis = analysis.sort_values('Failure_Rate_Pct', ascending=False)

        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                            subplot_titles=[f"Sample Size Analysis - {column}"])

        fig.add_trace(go.Bar(name="Failures", x=analysis[column].astype(str), y=analysis['Total_Failures'],
                             marker_color='red', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Bar(name="Passes", x=analysis[column].astype(str), y=analysis['Pass_Count'],
                             marker_color='green', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(name="Failure Rate", x=analysis[column].astype(str),
                                 y=analysis['Failure_Rate_Pct'], mode="lines+markers",
                                 line=dict(color='orange', width=3), marker=dict(size=8)), secondary_y=True)

        for _, row in analysis.iterrows():
            fig.add_annotation(x=row[column], y=row['Total_Tests'], text=f"n={row['Total_Tests']}",
                               showarrow=False, yshift=10, font=dict(size=10, color="black"))

        fig.update_xaxes(title_text=column, tickangle=-45)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Failure Rate (%)", secondary_y=True)
        fig.update_layout(title=f"Sample Size & Performance Analysis - {column}",
                          height=500, hovermode="x unified", barmode='stack', **default_layout)

        return fig, analysis

    def generate_pie_charts(self, column: str):
        """Pie charts for overall distribution and failure distribution"""
        if self.df.empty or column not in self.df.columns:
            return None, None, pd.DataFrame()

        known_data = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])]
        failure_data = self.df[self.df["Test Result"] == "FAIL"]

        if known_data.empty:
            return None, None, pd.DataFrame()

        overall_dist = known_data[column].value_counts()
        fig_overall = px.pie(values=overall_dist.values, names=overall_dist.index,
                             title=f"Overall Test Distribution by {column}", hole=0.3)
        fig_overall.update_traces(textposition='inside', textinfo='percent+label')
        fig_overall.update_layout(**default_layout)

        fig_failure = None
        if not failure_data.empty:
            failure_dist = failure_data[column].value_counts()
            fig_failure = px.pie(values=failure_dist.values, names=failure_dist.index,
                                 title=f"Failure Distribution by {column}", hole=0.3,
                                 color_discrete_sequence=px.colors.sequential.Reds_r)
            fig_failure.update_traces(textposition='inside', textinfo='percent+label')
            fig_failure.update_layout(**default_layout)

        summary_data = (
            known_data.groupby(column)['Is_Failure'].agg(['count', 'sum', 'mean']).round(3)
        )
        summary_data.columns = ['Total_Tests', 'Total_Failures', 'Failure_Rate']
        summary_data['Failure_Rate_Pct'] = summary_data['Failure_Rate'] * 100
        summary_data = summary_data.reset_index().sort_values('Failure_Rate_Pct', ascending=False)

        return fig_overall, fig_failure, summary_data

    def generate_performance_vs_age_make_analysis(self):
        """Performance matrix (Make × Age), plus bubble/grouped/adequacy charts"""
        if self.df.empty:
            return {}, pd.DataFrame()

        known = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])]
        if known.empty or "Make" not in self.df.columns or "Age_Category" not in self.df.columns:
            return {}, pd.DataFrame()

        perf_matrix = (
            known.groupby(['Make', 'Age_Category'])
            .agg({'Is_Failure': ['count', 'sum', 'mean'], 'Age_Years': 'mean'}).round(3)
        )
        perf_matrix.columns = ['Sample_Size', 'Failures', 'Failure_Rate', 'Avg_Age_Years']
        perf_matrix = perf_matrix.reset_index()
        perf_matrix['Failure_Rate_Pct'] = perf_matrix['Failure_Rate'] * 100

        charts = {}
        if not perf_matrix.empty:
            pivot_failure = perf_matrix.pivot(index='Make', columns='Age_Category', values='Failure_Rate_Pct')
            
            charts['heatmap'] = px.imshow(
                pivot_failure,
                title="Failure Rate Heatmap: Make vs Age Category (%)",
                labels=dict(x="Age Category", y="Make", color="Failure Rate (%)"),
                color_continuous_scale="Reds",
            )
            charts['heatmap'].update_layout(xaxis_tickangle=-45, height=700, **default_layout)

            charts['bubble'] = px.scatter(
                perf_matrix, x='Avg_Age_Years', y='Failure_Rate_Pct', size='Sample_Size', color='Make',
                hover_data=['Age_Category', 'Failures'],
                title="Performance Bubble Chart: Age vs Failure Rate by Make",
                labels={'Avg_Age_Years': 'Average Age (Years)', 'Failure_Rate_Pct': 'Failure Rate (%)'}
            )
            charts['bubble'].update_layout(**default_layout)

            charts['grouped_bar'] = px.bar(
                perf_matrix, x='Age_Category', y='Failure_Rate_Pct', color='Make', barmode='group',
                title="Failure Rate by Age Category and Make",
                labels={'Failure_Rate_Pct': 'Failure Rate (%)', 'Age_Category': 'Age Category'}
            )
            charts['grouped_bar'].update_layout(xaxis_tickangle=-45, **default_layout)

            perf_matrix['Sample_Adequacy'] = perf_matrix['Sample_Size'].apply(
                lambda x: 'High (>50)' if x > 50 else 'Medium (20-50)' if x >= 20 else 'Low (<20)')

            charts['sample_adequacy'] = px.scatter(
                perf_matrix, x='Make', y='Age_Category', size='Sample_Size', color='Sample_Adequacy',
                title="Sample Size Adequacy: Make vs Age Category",
                color_discrete_map={'High (>50)': 'green', 'Medium (20-50)': 'orange', 'Low (<20)': 'red'}
            )
            charts['sample_adequacy'].update_layout(xaxis_tickangle=-45, **default_layout)

        return charts, perf_matrix

    def generate_drill_down_insights(self, selected_make=None, selected_age_category=None, selected_type=None):
        """Plain‑language insights for Make / Age / Type slice"""
        insights = []
        if self.df.empty:
            return insights

        known = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])]
        if known.empty:
            return insights

        filtered = known.copy()
        desc_parts = []

        if selected_make:
            filtered = filtered[filtered["Make"] == selected_make]
            desc_parts.append(f"Make: {selected_make}")

        if selected_age_category:
            filtered = filtered[filtered["Age_Category"] == selected_age_category]
            desc_parts.append(f"Age: {selected_age_category}")

        if selected_type and "TYPE OF DAMPER" in filtered.columns:
            filtered = filtered[filtered["TYPE OF DAMPER"] == selected_type]
            desc_parts.append(f"Type: {selected_type}")

        desc = ", ".join(desc_parts) if desc_parts else "Overall"

        if filtered.empty:
            return [{"type": "warning", "content": f"No data available for {desc}"}]

        n = len(filtered)
        failures = int(filtered["Is_Failure"].sum())
        fr = failures / n if n else 0.0

        insights.append({"type": "info", "title": f"Performance Summary — {desc}",
                         "content": f"Sample Size: {n:,} | Failures: {failures:,} | Failure Rate: {fr:.1%}"})

        overall_fr = known["Is_Failure"].mean()

        if fr > overall_fr * 1.2:
            insights.append({"type": "critical", "title": "Performance Alert",
                             "content": f"Failure rate ({fr:.1%}) is {(fr/overall_fr-1)*100:.0f}% higher than overall average ({overall_fr:.1%})"})
        elif fr < overall_fr * 0.8:
            insights.append({"type": "success", "title": "Strong Performance",
                             "content": f"Failure rate ({fr:.1%}) is {(1-fr/overall_fr)*100:.0f}% better than overall average ({overall_fr:.1%})"})

        if "Age_Years" in filtered.columns and len(filtered) > 10:
            age_corr = filtered[["Age_Years", "Is_Failure"]].corr().iloc[0, 1]
            if abs(age_corr) > 0.3:
                label = ("Strong positive" if age_corr > 0.5 else
                         "Moderate positive" if age_corr > 0.3 else
                         "Moderate negative" if age_corr < -0.3 else "Strong negative")
                insights.append({"type": "warning" if age_corr > 0 else "info",
                                 "title": "Age Correlation",
                                 "content": f"{label} correlation ({age_corr:.3f}) between age and failure rate"})

        if n < 30:
            insights.append({"type": "warning", "title": "Limited Sample Size",
                             "content": f"Sample size ({n}) may be insufficient for reliable inference (recommend >30)"})

        return insights

    # ---------- OTHER VISUALS ----------
    def generate_pareto_chart(self, column: str, title_prefix: str = ""):
        if self.df.empty or column not in self.df.columns:
            return None, pd.DataFrame()

        failure_data = self.df[self.df["Test Result"] == "FAIL"]
        if failure_data.empty:
            return None, pd.DataFrame()

        vc = failure_data[column].value_counts(dropna=False)
        df_p = vc.reset_index()
        df_p.columns = [column, "Failure_Count"]
        df_p["Cumulative_Count"] = df_p["Failure_Count"].cumsum()
        total = df_p["Failure_Count"].sum()
        df_p["Percentage"] = df_p["Failure_Count"] / total * 100.0
        df_p["Cumulative_Percentage"] = df_p["Cumulative_Count"] / total * 100.0

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=df_p[column].astype(str), y=df_p["Failure_Count"], name="Failure Count"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=df_p[column].astype(str), y=df_p["Cumulative_Percentage"],
                                 name="Cumulative %", mode="lines+markers"), secondary_y=True)

        fig.add_shape(type="line", x0=-0.5, x1=len(df_p) - 0.5, y0=80, y1=80, xref="x", yref="y2",
                      line=dict(dash="dash"))

        fig.update_xaxes(title_text=column, tickangle=-45)
        fig.update_yaxes(title_text="Failure Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 100], secondary_y=True)
        fig.update_layout(title=f"{title_prefix}Pareto Analysis — Failures by {column}",
                          height=500, hovermode="x unified", **default_layout)

        return fig, df_p

    def generate_volume_pareto_chart(self, column: str, title_prefix: str = ""):
        """Generate Pareto chart for total test volume by category"""
        if self.df.empty or column not in self.df.columns:
            return None, pd.DataFrame()

        known_data = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])]
        if known_data.empty:
            return None, pd.DataFrame()

        vc = known_data[column].value_counts(dropna=False)
        df_p = vc.reset_index()
        df_p.columns = [column, "Test_Count"]
        df_p["Cumulative_Count"] = df_p["Test_Count"].cumsum()
        total = df_p["Test_Count"].sum()
        df_p["Percentage"] = df_p["Test_Count"] / total * 100.0
        df_p["Cumulative_Percentage"] = df_p["Cumulative_Count"] / total * 100.0

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=df_p[column].astype(str), y=df_p["Test_Count"], name="Test Count",
                            marker_color='blue'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_p[column].astype(str), y=df_p["Cumulative_Percentage"],
                                 name="Cumulative %", mode="lines+markers", line=dict(color='orange')),
                      secondary_y=True)

        fig.add_shape(type="line", x0=-0.5, x1=len(df_p) - 0.5, y0=80, y1=80, xref="x", yref="y2",
                      line=dict(dash="dash"))

        fig.update_xaxes(title_text=column, tickangle=-45)
        fig.update_yaxes(title_text="Test Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 100], secondary_y=True)
        fig.update_layout(title=f"{title_prefix}Pareto Analysis — Test Volume by {column}",
                          height=500, hovermode="x unified", **default_layout)

        return fig, df_p

    def generate_failure_percentage_pareto_chart(self, column: str, title_prefix: str = ""):
        """Generate Pareto chart for failure percentage by category"""
        if self.df.empty or column not in self.df.columns:
            return None, pd.DataFrame()

        failure_data = self.df[self.df["Test Result"] == "FAIL"]
        if failure_data.empty:
            return None, pd.DataFrame()

        vc = failure_data[column].value_counts(dropna=False)
        df_p = vc.reset_index()
        df_p.columns = [column, "Failure_Count"]
        total = df_p["Failure_Count"].sum()
        df_p["Percentage"] = df_p["Failure_Count"] / total * 100.0
        df_p = df_p.sort_values("Percentage", ascending=False)
        df_p["Cumulative_Percentage"] = df_p["Percentage"].cumsum()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=df_p[column].astype(str), y=df_p["Percentage"], name="Failure %",
                            marker_color='purple'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_p[column].astype(str), y=df_p["Cumulative_Percentage"],
                                 name="Cumulative %", mode="lines+markers", line=dict(color='orange')),
                      secondary_y=True)

        fig.add_shape(type="line", x0=-0.5, x1=len(df_p) - 0.5, y0=80, y1=80, xref="x", yref="y2",
                      line=dict(dash="dash"))

        fig.update_xaxes(title_text=column, tickangle=-45)
        fig.update_yaxes(title_text="Failure Percentage (%)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 100], secondary_y=True)
        fig.update_layout(title=f"{title_prefix}Pareto Analysis — Failure Percentage by {column}",
                          height=500, hovermode="x unified", **default_layout)

        return fig, df_p

    def generate_failure_rate_heatmap(self):
        """Make × Type failure-rate heatmap (independent of Age)"""
        if self.df.empty or "Test Result" not in self.df.columns:
            return None, pd.DataFrame()

        required = ["Make", "TYPE OF DAMPER"]
        if any(col not in self.df.columns for col in required):
            st.warning(f"Missing required columns for heatmap: {required}")
            return None, pd.DataFrame()

        known = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])].copy()
        if known.empty:
            return None, pd.DataFrame()

        grp = (
            known.groupby(["Make", "TYPE OF DAMPER"])["Is_Failure"]
            .agg(Sample_Size="count", Failure_Rate=lambda x: 100.0 * x.mean())
            .reset_index()
        )
        grp = grp[grp["Sample_Size"] > 0]

        if grp.empty:
            return None, pd.DataFrame()

        pivot = grp.pivot(index="Make", columns="TYPE OF DAMPER", values="Failure_Rate").fillna(0)
        fig = px.imshow(
            pivot,
            title="Failure Rate Heatmap (Make × Damper Type)",
            labels=dict(x="Damper Type", y="Make", color="Failure Rate (%)")
        )
        fig.update_layout(xaxis_tickangle=-45, height=700, **default_layout)

        return fig, grp

    def generate_statistical_insights(self):
        insights = []
        if self.df.empty or "Test Result" not in self.df.columns:
            return insights

        known = self.df[self.df["Test Result"].isin(["PASS", "FAIL"])]
        if known.empty:
            return insights

        if "Make" in self.df.columns:
            makes = []
            for mk, mk_df in known.groupby("Make"):
                if len(mk_df) >= 10:
                    p = mk_df["Is_Failure"].mean()
                    makes.append({"Make": mk, "Failure_Rate": p, "N": len(mk_df), "CI95": self._ci_prop(mk_df["Is_Failure"])})
            
            if makes:
                makes.sort(key=lambda z: z["Failure_Rate"], reverse=True)
                worst, best = makes[0], makes[-1]
                insights.append({
                    "type": "critical" if worst["Failure_Rate"] > 0.30 else "warning",
                    "title": "Make Performance",
                    "content": f"Highest failure: {worst['Make']} ({worst['Failure_Rate']:.1%}), Lowest: {best['Make']} ({best['Failure_Rate']:.1%})"
                })

        if "Age_Years" in self.df.columns:
            try:
                corr = self.df[["Age_Years", "Is_Failure"]].corr().iloc[0, 1]
                band = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                insights.append({"type": "info", "title": "Age–Failure Correlation",
                                 "content": f"Correlation: {corr:.3f} ({band})"})
            except Exception:
                pass

        if "Weekday" in self.df.columns and not self.df["Weekday"].isna().all():
            wd = known.groupby("Weekday")["Is_Failure"].agg(["mean", "count"])
            if not wd.empty:
                worst_day = wd["mean"].idxmax()
                insights.append({"type": "info", "title": "Temporal Pattern",
                                 "content": f"Highest failure rate on {worst_day}: {wd.loc[worst_day, 'mean']:.1%}"})

        return insights

    @staticmethod
    def _ci_prop(series, confidence=0.95):
        n = len(series)
        if n == 0:
            return (0.0, 0.0)
        p = series.mean()
        z = stats.norm.ppf((1 + confidence) / 2.0)
        m = z * np.sqrt(p * (1 - p) / n)
        return (max(0, p - m), min(1, p + m))

# -----------------------------------
# Initialize Enhanced Analytics
# -----------------------------------
analytics = EnhancedDamperAnalytics()

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("🎨 Theme Settings")
theme_choice = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown(
        """
    <style>
    .stApp { background-color: #2b2b2b; color: #ffffff; }
    .metric-card, .insight-box, .warning-box, .success-box, .error-box, .drill-down-card {
        background: #3c3c3c !important; color: #ffffff !important;
    }
    .stPlotlyChart { background: #3c3c3c !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

st.sidebar.header("🔧 Dashboard Configuration")

if analytics.connection_status:
    box_class = "success-box" if "Connected successfully" in analytics.connection_status else "error-box"
    st.sidebar.markdown(f"""<div class="{box_class}">{analytics.connection_status}</div>""", unsafe_allow_html=True)

colR1, colR2 = st.sidebar.columns(2)

with colR1:
    if st.button("🔄 Refresh Data", type="secondary"):
        try:
            st.cache_data.clear()
            if hasattr(st, "cache_resource"):
                st.cache_resource.clear()
            analytics = EnhancedDamperAnalytics()
            st.sidebar.success("✅ Cache cleared. Reloading…")
            time.sleep(0.7)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Refresh error: {e}")

with colR2:
    if st.button("🔍 Test Connection"):
        with st.spinner("Testing connection…"):
            ok, msg = test_google_sheets_connection()
            if ok:
                st.sidebar.success("✅ Connection OK")
            else:
                st.sidebar.error(f"❌ {msg}")

# Add environment validation
env_issues = validate_environment()
if env_issues:
    st.error("❌ Configuration Issues:")
    for issue in env_issues:
        st.write(f"• {issue}")
    st.stop()

st.sidebar.subheader("📊 Data Source")
available_sheets = ["Sheet1", "LHB", "VB", "OTHERS"]
sheet_choice = st.sidebar.selectbox("Select Data Source", available_sheets)

# -----------------------------------
# Load Data
# -----------------------------------
try:
    df_raw = analytics.load_sheet(sheet_choice)
    if df_raw.empty:
        st.sidebar.error(f"❌ No data loaded from {sheet_choice}")
        st.error(f"## ❌ No Data Available from {sheet_choice}")
        st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Loading error: {e}")
    st.error(f"## ❌ Failed to Load Data: {e}")
    st.stop()

# Update check
if not df_raw.empty:
    with st.spinner("Checking for data updates…"):
        has_update, new_count = check_for_updates(sheet_choice, len(df_raw))
        if has_update:
            st.markdown(
                f"""<div class="warning-box">
<strong>🔔 New Data Available!</strong><br>
New row count: {new_count}. Click <em>Refresh Data</em> to load updates.
</div>""",
                unsafe_allow_html=True,
            )

# Memory management for large datasets
if len(df_raw) > 10000:
    st.warning("⚠️ Large dataset detected. Some visualizations may use sampling for better performance.")
    if len(df_raw) > 50000:
        st.info(f"📊 Dataset size: {len(df_raw):,} rows. Using optimized processing.")

# Raw preview
with st.sidebar.expander("📋 Raw Data Preview"):
    st.dataframe(df_raw.head(5), use_container_width=True)

# -----------------------------------
# Preprocess
# -----------------------------------
df = analytics.preprocess(df_raw)
analytics.df = df

if df.empty:
    st.warning("⚠️ Data is empty after preprocessing")
    st.stop()

# Add data quality validation
quality_report = validate_data_quality(df)
if quality_report["status"] == "error":
    st.error("❌ Critical Data Quality Issues:")
    for issue in quality_report["issues"]:
        st.write(f"• {issue}")
elif quality_report["status"] == "warning":
    with st.expander("⚠️ Data Quality Warnings"):
        for issue in quality_report["issues"]:
            st.write(f"• {issue}")

# -----------------------------------
# Filters
# -----------------------------------
st.sidebar.subheader("🔍 Analysis Filters")

if "Test date time" in df.columns and df["Test date time"].notna().any():
    valid = df[df["Test date time"].notna()]
    min_date, max_date = valid["Test date time"].min().date(), valid["Test date time"].max().date()
    date_range = st.sidebar.date_input(
        "Test Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        help="Filter data by test date",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["Test date time"] >= pd.to_datetime(start_date)) & (
            df["Test date time"] <= pd.to_datetime(end_date) + timedelta(days=1)
        )
        df = df[mask | df["Test date time"].isna()].copy()
        analytics.df = df
        st.sidebar.info(f"📊 After date filter: {len(df)} rows")

for col, label in [("Make", "Manufacturer"), ("TYPE OF DAMPER", "Damper Type"), ("Age_Category", "Age Category")]:
    if col in df.columns:
        options = sorted({str(x) for x in df[col].dropna().astype(str).tolist()})
        sel = st.sidebar.multiselect(f"Filter by {label}", options=options, default=options)
        if sel:
            df = df[df[col].astype(str).isin(sel)].copy()
            analytics.df = df
            st.sidebar.info(f"📊 After {label} filter: {len(df)} rows")

# -----------------------------------
# Metrics
# -----------------------------------
st.title("🚀 Enhanced Damper Failure Analysis Dashboard")
st.markdown("*Professional analytics with advanced sample size analysis and drill-down capabilities*")

if analytics.last_update:
    st.info(f"📅 Last loaded: {analytics.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

total_tests = len(df)
known_results = df[df["Test Result"].isin(["PASS", "FAIL"])]
failures = int((df["Test Result"] == "FAIL").sum())
passes = int((df["Test Result"] == "PASS").sum())
failure_rate = (failures / len(known_results) * 100.0) if len(known_results) else 0.0
success_rate = 100.0 - failure_rate
mtbf_days = df["Age_Numeric"].mean() if "Age_Numeric" in df.columns and df["Age_Numeric"].notna().any() else np.nan

try:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Tests", f"{total_tests:,}")
    with c2:
        delta = failure_rate - 25.0
        st.metric("Failure Rate", f"{failure_rate:.1f}%", delta=f"{delta:+.1f} pts vs 25%")
    with c3:
        st.metric("Total Failures", f"{failures:,}")
    with c4:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with c5:
        st.metric("Avg Age (Days)", "N/A" if np.isnan(mtbf_days) else f"{mtbf_days:.0f}")
except Exception as e:
    st.error(f"Metric error: {e}")