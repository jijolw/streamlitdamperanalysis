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
    
    # Check Google Sheet key in secrets - FIXED: using consistent case
    if "GOOGLE_SHEET_KEY" not in st.secrets:
        issues.append("'GOOGLE_SHEET_KEY' secret not found in Streamlit Cloud secrets")
    
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
        # FIXED: using consistent case for sheet key
        sheet_key = st.secrets["GOOGLE_SHEET_KEY"]
        
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
        
    except KeyError as e:
        raise RuntimeError(f"Missing secret in Streamlit Cloud: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to get credentials from Streamlit secrets: {e}")

def debug_secrets():
    """Debug function to check what secrets are available"""
    st.write("Available secrets:")
    try:
        for key in st.secrets.keys():
            st.write(f"- {key}")
        
        # Check if the Google Sheet key exists with different cases
        possible_keys = ["GOOGLE_SHEET_KEY", "google_sheet_key", "Google_Sheet_Key"]
        for key in possible_keys:
            if key in st.secrets:
                st.write(f"✅ Found Google Sheet key as: {key}")
                break
        else:
            st.write("❌ Google Sheet key not found with any common naming")
            
    except Exception as e:
        st.write(f"Error checking secrets: {e}")
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
# -----------------------------------
# Tabs
# -----------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["📊 Overview", "📈 Sample Size Analysis", "🥧 Distribution Analysis", "🔍 Drill-Down", "🔥 Performance Matrix", "📉 Trends", "🧠 Advanced Analytics", "📊 Pareto Analysis"]
)
with tab1:
    st.header("📊 Executive Summary & Key Metrics")
    if known_results.empty:
        st.warning("⚠️ No valid PASS/FAIL results found")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Key Performance Indicators")
            kpi_data = {
                "Metric": [
                    "Total Tests Conducted",
                    "Data Completeness",
                    "Overall Failure Rate",
                    "High-Risk Categories",
                    "Sample Size Adequacy"
                ],
                "Value": [
                    f"{total_tests:,}",
                    f"{(len(known_results)/total_tests*100):.1f}%" if total_tests > 0 else "0%",
                    f"{failure_rate:.1f}%",
                    "TBD",
                    "Adequate" if len(known_results) > 100 else "Limited"
                ],
                "Status": [
                    "🟢" if total_tests > 100 else "🟡" if total_tests > 50 else "🔴",
                    "🟢" if (len(known_results)/total_tests) > 0.9 else "🟡" if (len(known_results)/total_tests) > 0.7 else "🔴",
                    "🟢" if failure_rate < 15 else "🟡" if failure_rate < 25 else "🔴",
                    "🔍",
                    "🟢" if len(known_results) > 100 else "🟡" if len(known_results) > 50 else "🔴"
                ]
            }
            st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)
        with col2:
            st.subheader("📈 Quick Performance Overview")
            if "Make" in df.columns:
                make_perf = known_results.groupby("Make")["Is_Failure"].agg(["count", "mean"]).round(3)
                make_perf.columns = ["Sample_Size", "Failure_Rate"]
                make_perf["Failure_Rate_Pct"] = make_perf["Failure_Rate"] * 100
                make_perf = make_perf.sort_values("Failure_Rate_Pct", ascending=False)
                fig_quick = px.bar(
                    make_perf.reset_index(),
                    x="Make", y="Failure_Rate_Pct",
                    title="Quick Make Performance Comparison",
                    color="Failure_Rate_Pct",
                    color_continuous_scale="RdYlBu_r"
                )
                fig_quick.update_layout(xaxis_tickangle=-45, height=400, **default_layout)
                st.plotly_chart(fig_quick, use_container_width=True)
        st.subheader("🔍 Interactive Data Explorer")
        st.data_editor(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Test date time": st.column_config.DatetimeColumn("Test Date Time"),
                "Age_Numeric": st.column_config.NumberColumn("Age (Days)", format="%.0f"),
                "Age_Years": st.column_config.NumberColumn("Age (Years)", format="%.1f"),
                "Is_Failure": st.column_config.CheckboxColumn("Failure"),
                "Is_Pass": st.column_config.CheckboxColumn("Pass"),
            },
            num_rows="dynamic", disabled=True, key=f"data_explorer_{sheet_choice}",
        )
with tab2:
    st.header("📈 Sample Size Analysis")
    st.markdown("*Comprehensive analysis showing sample sizes, failure counts, and rates*")
    if failures == 0:
        st.warning("⚠️ No failure data available for sample size analysis")
    else:
        analysis_columns = [("Make", "Manufacturer"), ("TYPE OF DAMPER", "Damper Type")]
        if "Age_Category" in df.columns:
            analysis_columns.append(("Age_Category", "Age Category"))
        for col, title in analysis_columns:
            if col in df.columns:
                st.subheader(f"📊 {title} Sample Size Analysis")
                try:
                    fig_sample, sample_data = analytics.generate_sample_size_analysis(col)
                    if fig_sample is not None and not sample_data.empty:
                        st.plotly_chart(fig_sample, use_container_width=True)
                        adequate = sample_data[sample_data['Total_Tests'] >= 30]
                        limited = sample_data[(sample_data['Total_Tests'] >= 10) & (sample_data['Total_Tests'] < 30)]
                        insufficient = sample_data[sample_data['Total_Tests'] < 10]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="success-box"><strong>🟢 Adequate Sample Size</strong><br>
                            {len(adequate)} categories (≥30 tests)<br>
                            <small>{', '.join(adequate[col].astype(str).tolist()[:3])}{' ...' if len(adequate) > 3 else ''}</small>
                            </div>""", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="warning-box"><strong>🟡 Limited Sample Size</strong><br>
                            {len(limited)} categories (10-29 tests)<br>
                            <small>{', '.join(limited[col].astype(str).tolist()[:3])}{' ...' if len(limited) > 3 else ''}</small>
                            </div>""", unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="error-box"><strong>🔴 Insufficient Sample Size</strong><br>
                            {len(insufficient)} categories (<10 tests)<br>
                            <small>{', '.join(insufficient[col].astype(str).tolist()[:3])}{' ...' if len(insufficient) > 3 else ''}</small>
                            </div>""", unsafe_allow_html=True)
                        with st.expander(f"📋 Detailed {title} Analysis Data"):
                            st.dataframe(
                                sample_data.style.format({'Failure_Rate': '{:.1%}', 'Failure_Rate_Pct': '{:.1f}%'}),
                                use_container_width=True
                            )
                    else:
                        st.info(f"No data available for {title} sample size analysis")
                except Exception as e:
                    st.error(f"Sample size analysis error for {title}: {e}")
with tab3:
    st.header("🥧 Distribution Analysis")
    st.markdown("*Visual distribution analysis using pie charts and summary statistics*")
    if known_results.empty:
        st.warning("⚠️ No valid data for distribution analysis")
    else:
        dist_columns = [("Make", "Manufacturer"), ("TYPE OF DAMPER", "Damper Type")]
        if "Age_Category" in df.columns:
            dist_columns.append(("Age_Category", "Age Category"))
        for col, title in dist_columns:
            if col in df.columns:
                st.subheader(f"📊 {title} Distribution")
                try:
                    fig_overall, fig_failure, summary_data = analytics.generate_pie_charts(col)
                    if fig_overall is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(fig_overall, use_container_width=True)
                        with c2:
                            if fig_failure is not None:
                                st.plotly_chart(fig_failure, use_container_width=True)
                            else:
                                st.info("No failure data available for failure distribution chart")
                        if not summary_data.empty:
                            st.markdown(f"**📈 {title} Performance Summary**")
                            best_performer = summary_data.loc[summary_data['Failure_Rate_Pct'].idxmin()]
                            worst_performer = summary_data.loc[summary_data['Failure_Rate_Pct'].idxmax()]
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"""
                                <div class="success-box">
                                <strong>🏆 Best Performer</strong><br>
                                <strong>{best_performer[col]}</strong><br>
                                Failure Rate: {best_performer['Failure_Rate_Pct']:.1f}%<br>
                                Sample Size: {best_performer['Total_Tests']}
                                </div>""", unsafe_allow_html=True)
                            with c2:
                                st.markdown(f"""
                                <div class="error-box">
                                <strong>⚠️ Needs Attention</strong><br>
                                <strong>{worst_performer[col]}</strong><br>
                                Failure Rate: {worst_performer['Failure_Rate_Pct']:.1f}%<br>
                                Sample Size: {worst_performer['Total_Tests']}
                                </div>""", unsafe_allow_html=True)
                            with st.expander(f"📋 Complete {title} Summary"):
                                st.dataframe(
                                    summary_data.style.format({'Failure_Rate': '{:.1%}', 'Failure_Rate_Pct': '{:.1f}%'}),
                                    use_container_width=True
                                )
                    else:
                        st.info(f"No data available for {title} distribution analysis")
                except Exception as e:
                    st.error(f"Distribution analysis error for {title}: {e}")
with tab4:
    st.header("🔍 Interactive Drill-Down Analysis")
    st.markdown("*Select specific categories to get detailed insights and comparisons*")
    if known_results.empty:
        st.warning("⚠️ No valid data for drill-down analysis")
    else:
        # Controls: Make, Age, Type
        col1, col2, col3 = st.columns(3)
        selected_make = None
        selected_age = None
        selected_type = None
        with col1:
            if "Make" in df.columns:
                make_options = ['All'] + sorted(df["Make"].dropna().unique().tolist())
                chosen = st.selectbox("🏭 Select Make", make_options)
                selected_make = None if chosen == 'All' else chosen
        with col2:
            if "Age_Category" in df.columns:
                age_options = ['All'] + sorted(df["Age_Category"].dropna().astype(str).unique().tolist())
                chosen = st.selectbox("📅 Select Age Category", age_options)
                selected_age = None if chosen == 'All' else chosen
        with col3:
            if "TYPE OF DAMPER" in df.columns:
                type_options = ['All'] + sorted(df["TYPE OF DAMPER"].dropna().astype(str).unique().tolist())
                chosen = st.selectbox("⚙️ Select Damper Type", type_options)
                selected_type = None if chosen == 'All' else chosen
        # Insights
        insights = analytics.generate_drill_down_insights(selected_make, selected_age, selected_type)
        if insights:
            for insight in insights:
                box = ("error-box" if insight["type"] == "critical" else
                       "warning-box" if insight["type"] == "warning" else
                       "success-box" if insight["type"] == "success" else "insight-box")
                title = insight.get("title", "Insight")
                st.markdown(f"""<div class="{box}"><strong>💡 {title}</strong><br>{insight['content']}</div>""",
                            unsafe_allow_html=True)
        # Filter the data for charts/tables + downloads
        filtered_data = known_results.copy()
        parts = []
        if selected_make:
            filtered_data = filtered_data[filtered_data["Make"] == selected_make]
            parts.append(f"Make: {selected_make}")
        if selected_age:
            filtered_data = filtered_data[filtered_data["Age_Category"] == selected_age]
            parts.append(f"Age: {selected_age}")
        if selected_type and "TYPE OF DAMPER" in filtered_data.columns:
            filtered_data = filtered_data[filtered_data["TYPE OF DAMPER"] == selected_type]
            parts.append(f"Type: {selected_type}")
        label = "Overall" if not parts else ", ".join(parts)
        if filtered_data.empty:
            st.warning("No data available for the selected filters")
        else:
            c1, c2 = st.columns(2)
            with c1:
                if "Test date time" in filtered_data.columns and filtered_data["Test date time"].notna().any():
                    monthly_data = filtered_data.copy()
                    monthly_data["YearMonth"] = monthly_data["Test date time"].dt.to_period("M")
                    monthly_summary = (
                        monthly_data.groupby("YearMonth")["Is_Failure"]
                        .agg(["count", "sum", "mean"]).reset_index()
                    )
                    monthly_summary.columns = ["YearMonth", "Total_Tests", "Failures", "Failure_Rate"]
                    monthly_summary["Date"] = monthly_summary["YearMonth"].dt.to_timestamp()
                    monthly_summary["Failure_Rate_Pct"] = monthly_summary["Failure_Rate"] * 100
                    if len(monthly_summary) > 1:
                        fig_trend = px.line(monthly_summary, x="Date", y="Failure_Rate_Pct",
                                            title=f"Trend — {label}", markers=True)
                        fig_trend.update_layout(**default_layout)
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Insufficient time data for trend analysis")
            with c2:
                if "Age_Years" in filtered_data.columns:
                    fig_age = px.histogram(filtered_data, x="Age_Years", color="Test Result",
                                           title=f"Age Distribution — {label}", nbins=20)
                    fig_age.update_layout(**default_layout)
                    st.plotly_chart(fig_age, use_container_width=True)
            with st.expander("📊 Detailed Statistics"):
                stats_data = {
                    "Metric": ["Sample Size", "Failures", "Passes", "Failure Rate",
                               "Average Age (Years)", "Age Range (Years)"],
                    "Value": [
                        len(filtered_data),
                        int(filtered_data["Is_Failure"].sum()),
                        int((filtered_data["Is_Failure"] == 0).sum()),
                        f"{filtered_data['Is_Failure'].mean():.1%}",
                        f"{filtered_data['Age_Years'].mean():.1f}" if "Age_Years" in filtered_data.columns else "N/A",
                        (f"{filtered_data['Age_Years'].min():.1f} - {filtered_data['Age_Years'].max():.1f}"
                         if "Age_Years" in filtered_data.columns else "N/A")
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            # ---- Drill-down downloads (CSV + Excel) ----
            st.markdown("#### 📥 Download This Drill‑Down Dataset")
            csv_bytes = filtered_data.to_csv(index=False).encode('utf-8-sig') # BOM fixes odd chars in Excel
            st.download_button(
                "Download CSV (UTF‑8)", data=csv_bytes,
                file_name=f"drilldown_{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True
            )
            # Excel
            xls_buf = BytesIO()
            with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
                filtered_data.to_excel(writer, sheet_name="DrillDown", index=False)
            st.download_button(
                "Download Excel (.xlsx)", data=xls_buf.getvalue(),
                file_name=f"drilldown_{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
with tab5:
    st.header("🔥 Performance Matrix Analysis")
    st.markdown("*Comprehensive performance analysis by Age vs Make with multiple visualization perspectives*")
    if known_results.empty or "Make" not in df.columns or "Age_Category" not in df.columns:
        st.warning("⚠️ Insufficient data or missing required columns (Make, Age_Category) for performance matrix analysis")
    else:
        try:
            charts, perf_matrix = analytics.generate_performance_vs_age_make_analysis()
            if charts and not perf_matrix.empty:
                if 'heatmap' in charts:
                    st.subheader("🔥 Performance Heatmap")
                    st.plotly_chart(charts['heatmap'], use_container_width=True)
                    st.markdown("""
                    <div class="insight-box"><strong>💡 Reading the Heatmap:</strong>
                    Darker red indicates higher failure rates. Compare patterns across age groups and manufacturers.</div>
                    """, unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    if 'bubble' in charts:
                        st.subheader("💫 Performance Bubble Chart")
                        st.plotly_chart(charts['bubble'], use_container_width=True)
                with c2:
                    if 'grouped_bar' in charts:
                        st.subheader("📊 Grouped Performance Comparison")
                        st.plotly_chart(charts['grouped_bar'], use_container_width=True)
                if 'sample_adequacy' in charts:
                    st.subheader("📏 Sample Size Adequacy Map")
                    st.plotly_chart(charts['sample_adequacy'], use_container_width=True)
                    st.markdown("""
                    <div class="warning-box"><strong>⚠️ Sample Size Guidelines:</strong>
                    Green (>50 tests) are reliable. Orange (20–50) and Red (<20) need more data.</div>
                    """, unsafe_allow_html=True)
                st.subheader("🎯 Key Performance Insights")
                best_combo = perf_matrix.loc[perf_matrix['Failure_Rate_Pct'].idxmin()]
                worst_combo = perf_matrix.loc[perf_matrix['Failure_Rate_Pct'].idxmax()]
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="success-box"><strong>🏆 Best Performance</strong><br>
                    <strong>{best_combo['Make']}</strong><br>Age: {best_combo['Age_Category']}<br>
                    Failure Rate: {best_combo['Failure_Rate_Pct']:.1f}%<br>
                    Sample: {best_combo['Sample_Size']} tests</div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="error-box"><strong>⚠️ Needs Attention</strong><br>
                    <strong>{worst_combo['Make']}</strong><br>Age: {worst_combo['Age_Category']}<br>
                    Failure Rate: {worst_combo['Failure_Rate_Pct']:.1f}%<br>
                    Sample: {worst_combo['Sample_Size']} tests</div>""", unsafe_allow_html=True)
                with c3:
                    adequate_samples = perf_matrix[perf_matrix['Sample_Size'] >= 30]
                    if not adequate_samples.empty:
                        variance = adequate_samples['Failure_Rate_Pct'].var()
                        st.markdown(f"""
                        <div class="insight-box"><strong>📊 Statistical Overview</strong><br>
                        Combos: {len(perf_matrix)} | Adequate samples: {len(adequate_samples)}<br>
                        Variance across adequate samples: {variance:.1f}</div>""", unsafe_allow_html=True)
                with st.expander("📋 Complete Performance Matrix"):
                    display_matrix = perf_matrix.sort_values('Failure_Rate_Pct', ascending=False)
                    st.dataframe(
                        display_matrix.style.format({
                            'Failure_Rate': '{:.1%}', 'Failure_Rate_Pct': '{:.1f}%',
                            'Avg_Age_Years': '{:.1f}'
                        }).background_gradient(subset=['Failure_Rate_Pct'], cmap='Reds'),
                        use_container_width=True
                    )
                st.subheader("📋 Strategic Recommendations")
                age_analysis = perf_matrix.groupby('Age_Category')['Failure_Rate_Pct'].agg(['mean', 'count']).reset_index()
                age_analysis.columns = ['Age_Category', 'Avg_Failure_Rate', 'Sample_Count']
                age_analysis = age_analysis.sort_values('Avg_Failure_Rate', ascending=False)
                if not age_analysis.empty:
                    highest_risk_age = age_analysis.iloc[0]['Age_Category']
                    st.markdown(f"""
                    <div class="drill-down-card"><strong>🎯 Priority Actions:</strong><br>
                    1. Focus on <strong>{highest_risk_age}</strong> category (highest avg failure)<br>
                    2. Investigate <strong>{worst_combo['Make']}</strong> quality issues<br>
                    3. Benchmark <strong>{best_combo['Make']}</strong> practices<br>
                    4. Increase sample sizes where <30 tests</div>""", unsafe_allow_html=True)
            else:
                st.info("Insufficient data for performance matrix analysis")
        except Exception as e:
            st.error(f"Performance matrix analysis error: {e}")
with tab6:
    st.header("📉 Trends & Statistical Process Control")
    if "Test date time" in df.columns and df["Test date time"].notna().any():
        try:
            monthly = known_results.copy()
            monthly["YearMonth"] = monthly["Test date time"].dt.to_period("M")
            ms = (
                monthly.groupby("YearMonth")
                .agg(Total_Tests=("Is_Failure", "count"),
                     Total_Failures=("Is_Failure", "sum"),
                     Failure_Rate=("Is_Failure", "mean"))
                .reset_index()
            )
            ms["Date"] = ms["YearMonth"].dt.to_timestamp()
            if len(ms) > 1:
                ms["Failure_Rate_Pct"] = ms["Failure_Rate"] * 100.0
                c1, c2 = st.columns(2)
                with c1:
                    fig_trend = px.line(ms, x="Date", y="Failure_Rate_Pct",
                                        title="Monthly Failure Rate Trend",
                                        markers=True, labels={"Failure_Rate_Pct": "Failure Rate (%)"})
                    if len(ms) > 3:
                        x_idx = np.arange(len(ms))
                        z = np.polyfit(x_idx, ms["Failure_Rate_Pct"].values, 1)
                        p = np.poly1d(z)
                        fig_trend.add_trace(
                            go.Scatter(x=ms["Date"], y=p(x_idx), mode="lines",
                                       name="Trend", line=dict(dash="dash"))
                        )
                        slope = z[0]
                        direction = "Improving" if slope < -0.1 else "Worsening" if slope > 0.1 else "Stable"
                        st.markdown(
                            f"""<div class="{'success-box' if direction=='Improving' else 'warning-box' if direction=='Worsening' else 'insight-box'}">
                            <strong>📈 Trend:</strong> {direction} ({slope:.2f} pts/month)</div>""",
                            unsafe_allow_html=True,
                        )
                    fig_trend.update_layout(**default_layout)
                    st.plotly_chart(fig_trend, use_container_width=True)
                with c2:
                    fig_sample = px.bar(ms, x="Date", y="Total_Tests",
                                        title="Monthly Test Volume",
                                        labels={"Total_Tests": "Number of Tests"})
                    fig_sample.update_layout(**default_layout)
                    st.plotly_chart(fig_sample, use_container_width=True)
                st.subheader("📉 Statistical Process Control Chart")
                if len(ms) > 3:
                    mean = ms["Failure_Rate_Pct"].mean()
                    std = ms["Failure_Rate_Pct"].std(ddof=1) if len(ms) > 1 else 0.0
                    ucl = mean + 3 * std
                    lcl = max(0.0, mean - 3 * std)
                    fig_ctrl = go.Figure()
                    fig_ctrl.add_trace(go.Scatter(x=ms["Date"], y=ms["Failure_Rate_Pct"],
                                                  mode="lines+markers", name="Failure Rate"))
                    fig_ctrl.add_hline(y=mean, line_dash="dash", line_color="green", annotation_text="CL")
                    fig_ctrl.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL")
                    fig_ctrl.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL")
                    ooc = ms[(ms["Failure_Rate_Pct"] > ucl) | (ms["Failure_Rate_Pct"] < lcl)]
                    if not ooc.empty:
                        fig_ctrl.add_trace(go.Scatter(x=ooc["Date"], y=ooc["Failure_Rate_Pct"],
                                                      mode="markers", name="Out of Control",
                                                      marker=dict(size=15, symbol="x", color="red")))
                    fig_ctrl.update_layout(title="Control Chart — Monthly Failure Rate",
                                           xaxis_title="Date", yaxis_title="Failure Rate (%)", **default_layout)
                    st.plotly_chart(fig_ctrl, use_container_width=True)
                    if not ooc.empty:
                        st.markdown(
                            f"""<div class="warning-box"><strong>🚨 Out-of-Control Points:</strong> {len(ooc)} month(s)<br>
                            Dates: {', '.join(ooc['YearMonth'].astype(str).tolist())}</div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("""<div class="success-box"><strong>✅ Process in Control:</strong> No points beyond control limits.</div>""",
                                    unsafe_allow_html=True)
                else:
                    st.info("Need >3 months for control chart")
            else:
                st.info("Need more than one month for trend analysis")
        except Exception as e:
            st.error(f"Trend/Control error: {e}")
    else:
        st.info("Date information not available for trend analysis")
with tab7:
    st.header("🧠 Advanced Analytics & AI Insights")
    st.markdown("*Simple explanations of complex patterns in your damper data*")
   
    if len(known_results) < 30:
        st.warning(f"⚠️ **Limited Data Alert**: You have {len(known_results)} valid test results. For reliable insights, we recommend at least 30 results. Current insights should be used cautiously.")
   
    # Section 1: Basic Statistics in Plain English
    st.subheader("📊 What Your Data is Telling Us")
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("#### 🔍 Data Overview")
        data_quality_pct = (len(known_results) / total_tests * 100.0) if total_tests else 0.0
       
        # Simple data quality explanation
        if data_quality_pct > 95:
            quality_msg = "**Excellent** - Almost all your test results are clear (Pass/Fail)"
            quality_color = "success-box"
        elif data_quality_pct > 85:
            quality_msg = "**Good** - Most of your test results are clear"
            quality_color = "success-box"
        elif data_quality_pct > 70:
            quality_msg = "**Fair** - Some test results are unclear or missing"
            quality_color = "warning-box"
        else:
            quality_msg = "**Poor** - Many test results are unclear or missing"
            quality_color = "error-box"
           
        st.markdown(f"""
        <div class="{quality_color}">
        <strong>📈 Data Quality Score: {data_quality_pct:.0f}%</strong><br>
        {quality_msg}<br><br>
        • Total tests conducted: <strong>{total_tests:,}</strong><br>
        • Clear results (Pass/Fail): <strong>{len(known_results):,}</strong><br>
        • Failure rate: <strong>{failure_rate:.1f}%</strong>
        </div>""", unsafe_allow_html=True)
   
    with col2:
        st.markdown("#### 🎯 Quick Assessment")
       
        # Simple failure rate assessment
        if failure_rate < 10:
            performance_msg = "**Excellent Performance** 🟢<br>Your dampers are performing very well"
            perf_color = "success-box"
        elif failure_rate < 20:
            performance_msg = "**Good Performance** 🟡<br>Most dampers are working fine, some room for improvement"
            perf_color = "success-box"
        elif failure_rate < 30:
            performance_msg = "**Fair Performance** 🟠<br>Notable failure rate - investigation recommended"
            perf_color = "warning-box"
        else:
            performance_msg = "**Poor Performance** 🔴<br>High failure rate - immediate attention needed"
            perf_color = "error-box"
           
        st.markdown(f"""
        <div class="{perf_color}">
        {performance_msg}<br><br>
        • Out of 100 dampers tested, about <strong>{failure_rate:.0f}</strong> would fail<br>
        • This means <strong>{100-failure_rate:.0f} out of 100</strong> work properly
        </div>""", unsafe_allow_html=True)
    # Section 2: Age Analysis in Plain English
    if "Age_Years" in df.columns and df["Age_Years"].notna().any():
        st.subheader("📅 How Age Affects Damper Performance")
       
        col1, col2 = st.columns(2)
       
        with col1:
            # Age distribution chart with clear explanation
            fig_age_simple = px.histogram(
                known_results,
                x="Age_Years",
                color="Test Result",
                title="Damper Age vs Performance",
                labels={"Age_Years": "Age (Years)", "count": "Number of Dampers"},
                nbins=15
            )
            fig_age_simple.update_layout(**default_layout)
            st.plotly_chart(fig_age_simple, use_container_width=True)
           
        with col2:
            # Age correlation explanation
            if len(known_results) > 10:
                age_fail_corr = known_results[["Age_Years", "Is_Failure"]].corr().iloc[0, 1]
               
                if age_fail_corr > 0.3:
                    corr_explanation = f"""
                    <div class="warning-box">
                    <strong>🔍 Key Finding: Older Dampers Fail More Often</strong><br><br>
                    • As dampers get older, they tend to fail more frequently<br>
                    • Correlation strength: <strong>{"Strong" if age_fail_corr > 0.5 else "Moderate"}</strong><br>
                    • Recommendation: Consider replacing dampers before they get too old<br>
                    • Set up preventive maintenance schedules
                    </div>"""
                elif age_fail_corr < -0.3:
                    corr_explanation = f"""
                    <div class="insight-box">
                    <strong>🔍 Interesting Finding: Newer Dampers Fail More</strong><br><br>
                    • Surprisingly, newer dampers seem to fail more often<br>
                    • This could indicate manufacturing issues or break-in problems<br>
                    • Recommendation: Review quality control for new dampers
                    </div>"""
                else:
                    corr_explanation = f"""
                    <div class="insight-box">
                    <strong>🔍 Finding: Age Doesn't Strongly Predict Failures</strong><br><br>
                    • Damper age alone doesn't strongly predict when they'll fail<br>
                    • Other factors (like manufacturer or type) may be more important<br>
                    • This suggests good overall durability across different ages
                    </div>"""
               
                st.markdown(corr_explanation, unsafe_allow_html=True)
               
                # Age statistics in simple terms
                age_stats = known_results["Age_Years"].describe()
                st.markdown(f"""
                **📊 Age Summary:**
                - Youngest damper tested: **{age_stats['min']:.1f} years**
                - Oldest damper tested: **{age_stats['max']:.1f} years**
                - Average age: **{age_stats['mean']:.1f} years**
                - Most dampers are around **{age_stats['50%']:.1f} years** old
                """)
    # Section 3: Manufacturer Performance in Plain English
    if "Make" in df.columns:
        st.subheader("🏭 Which Manufacturers Perform Best?")
       
        # Calculate manufacturer performance
        make_performance = (
            known_results.groupby("Make")["Is_Failure"]
            .agg(total_tests="count", failures="sum", failure_rate="mean")
            .reset_index()
        )
        make_performance["failure_rate_pct"] = make_performance["failure_rate"] * 100
        make_performance = make_performance[make_performance["total_tests"] >= 10] # Only include makes with sufficient data
       
        if not make_performance.empty:
            make_performance = make_performance.sort_values("failure_rate_pct")
           
            col1, col2 = st.columns(2)
           
            with col1:
                # Manufacturer performance chart
                fig_make = px.bar(
                    make_performance,
                    x="failure_rate_pct",
                    y="Make",
                    orientation="h",
                    title="Failure Rate by Manufacturer",
                    labels={"failure_rate_pct": "Failure Rate (%)", "Make": "Manufacturer"},
                    color="failure_rate_pct",
                    color_continuous_scale="RdYlGn_r"
                )
                fig_make.update_layout(**default_layout, height=400)
                st.plotly_chart(fig_make, use_container_width=True)
               
            with col2:
                # Best and worst performers explanation
                best_make = make_performance.iloc[0]
                worst_make = make_performance.iloc[-1]
               
                st.markdown(f"""
                <div class="success-box">
                <strong>🏆 Best Performer: {best_make['Make']}</strong><br>
                • Failure rate: <strong>{best_make['failure_rate_pct']:.1f}%</strong><br>
                • Based on <strong>{best_make['total_tests']:,}</strong> tests<br>
                • Meaning: Very reliable, few failures
                </div>
               
                <div class="error-box">
                <strong>⚠️ Needs Improvement: {worst_make['Make']}</strong><br>
                • Failure rate: <strong>{worst_make['failure_rate_pct']:.1f}%</strong><br>
                • Based on <strong>{worst_make['total_tests']:,}</strong> tests<br>
                • Meaning: More failures than others, needs attention
                </div>""", unsafe_allow_html=True)
               
                # Performance gap explanation
                gap = worst_make['failure_rate_pct'] - best_make['failure_rate_pct']
                if gap > 10:
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>💡 Key Insight:</strong><br>
                    There's a <strong>{gap:.1f} percentage point</strong> difference between
                    the best and worst manufacturers. This suggests manufacturer choice
                    significantly impacts damper reliability.
                    </div>""", unsafe_allow_html=True)
    # Section 4: Time Trends in Plain English
    if "Test date time" in df.columns and df["Test date time"].notna().any():
        st.subheader("📈 Are Things Getting Better or Worse Over Time?")
       
        # Monthly trend analysis
        monthly_data = known_results.copy()
        monthly_data["YearMonth"] = monthly_data["Test date time"].dt.to_period("M")
        monthly_summary = (
            monthly_data.groupby("YearMonth")["Is_Failure"]
            .agg(total_tests="count", failures="sum", failure_rate="mean")
            .reset_index()
        )
        monthly_summary["Date"] = monthly_summary["YearMonth"].dt.to_timestamp()
        monthly_summary["failure_rate_pct"] = monthly_summary["failure_rate"] * 100
       
        if len(monthly_summary) > 2:
            col1, col2 = st.columns(2)
           
            with col1:
                # Trend chart
                fig_trend = px.line(
                    monthly_summary,
                    x="Date",
                    y="failure_rate_pct",
                    title="Monthly Failure Rate Trend",
                    labels={"failure_rate_pct": "Failure Rate (%)", "Date": "Month"},
                    markers=True
                )
               
                # Add trend line if enough data
                if len(monthly_summary) >= 3:
                    z = np.polyfit(range(len(monthly_summary)), monthly_summary["failure_rate_pct"], 1)
                    trend_line = np.polyval(z, range(len(monthly_summary)))
                    fig_trend.add_trace(
                        go.Scatter(
                            x=monthly_summary["Date"],
                            y=trend_line,
                            mode="lines",
                            name="Trend",
                            line=dict(dash="dash", color="red")
                        )
                    )
                   
                fig_trend.update_layout(**default_layout)
                st.plotly_chart(fig_trend, use_container_width=True)
               
            with col2:
                # Trend explanation
                if len(monthly_summary) >= 3:
                    # Calculate trend
                    first_3_avg = monthly_summary.head(3)["failure_rate_pct"].mean()
                    last_3_avg = monthly_summary.tail(3)["failure_rate_pct"].mean()
                    trend_change = last_3_avg - first_3_avg
                   
                    if trend_change > 2:
                        trend_msg = f"""
                        <div class="error-box">
                        <strong>📈 Worsening Trend</strong><br>
                        • Failure rate is <strong>increasing</strong> over time<br>
                        • Recent months show <strong>{trend_change:.1f}%</strong> higher failure rate<br>
                        • <strong>Action needed:</strong> Investigate what changed recently
                        </div>"""
                    elif trend_change < -2:
                        trend_msg = f"""
                        <div class="success-box">
                        <strong>📉 Improving Trend</strong><br>
                        • Failure rate is <strong>decreasing</strong> over time<br>
                        • Recent months show <strong>{abs(trend_change):.1f}%</strong> lower failure rate<br>
                        • <strong>Good news:</strong> Whatever you're doing is working!
                        </div>"""
                    else:
                        trend_msg = f"""
                        <div class="insight-box">
                        <strong>📊 Stable Trend</strong><br>
                        • Failure rate is <strong>relatively stable</strong> over time<br>
                        • No major changes in recent months<br>
                        • <strong>Status:</strong> Consistent performance
                        </div>"""
                   
                    st.markdown(trend_msg, unsafe_allow_html=True)
                   
                    # Monthly summary
                    st.markdown(f"""
                    **📅 Monthly Summary:**
                    - Best month: **{monthly_summary.loc[monthly_summary['failure_rate_pct'].idxmin(), 'YearMonth']}** ({monthly_summary['failure_rate_pct'].min():.1f}% failures)
                    - Worst month: **{monthly_summary.loc[monthly_summary['failure_rate_pct'].idxmax(), 'YearMonth']}** ({monthly_summary['failure_rate_pct'].max():.1f}% failures)
                    - Average tests per month: **{monthly_summary['total_tests'].mean():.0f}**
                    """)
    # Section 5: Predictive Model Explanation (Simplified)
    if len(known_results) >= 50:
        st.subheader("🤖 Can We Predict Which Dampers Will Fail?")
       
        try:
            # Build simple prediction model
            features_for_prediction = []
            if "Age_Years" in df.columns:
                features_for_prediction.append("Age_Years")
            if "Make" in df.columns:
                features_for_prediction.extend(pd.get_dummies(df["Make"], prefix="Make").columns.tolist())
           
            if features_for_prediction and "Age_Years" in df.columns:
                # Simple model with just age
                X_simple = df[["Age_Years"]].fillna(df["Age_Years"].mean())
                y = (df["Test Result"] == "FAIL").astype(int)
               
                if 0 < y.sum() < len(y):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_simple, y, test_size=0.3, random_state=42, stratify=y
                    )
                   
                    model = LogisticRegression(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                   
                    # Calculate accuracy
                    from sklearn.metrics import accuracy_score, precision_score, recall_score
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                   
                    col1, col2 = st.columns(2)
                   
                    with col1:
                        # Model performance explanation
                        if accuracy > 0.8:
                            model_quality = "**Excellent** 🟢 - Very reliable predictions"
                            model_color = "success-box"
                        elif accuracy > 0.7:
                            model_quality = "**Good** 🟡 - Fairly reliable predictions"
                            model_color = "success-box"
                        elif accuracy > 0.6:
                            model_quality = "**Fair** 🟠 - Somewhat reliable predictions"
                            model_color = "warning-box"
                        else:
                            model_quality = "**Poor** 🔴 - Not very reliable predictions"
                            model_color = "error-box"
                       
                        st.markdown(f"""
                        <div class="{model_color}">
                        <strong>🎯 Prediction Model Quality</strong><br>
                        {model_quality}<br><br>
                        <strong>In simple terms:</strong><br>
                        • Out of 100 predictions, about <strong>{accuracy*100:.0f}</strong> are correct<br>
                        • When we predict a damper will fail, we're right <strong>{precision*100:.0f}%</strong> of the time<br>
                        • We catch <strong>{recall*100:.0f}%</strong> of actual failures
                        </div>""", unsafe_allow_html=True)
                   
                    with col2:
                        # Age-based risk explanation
                        age_ranges = [(0, 2, "New (0-2 years)"), (2, 4, "Medium (2-4 years)"),
                                     (4, 10, "Old (4+ years)")]
                       
                        risk_explanation = "<div class='insight-box'><strong>🔍 Age-Based Risk Levels:</strong><br><br>"
                       
                        for min_age, max_age, label in age_ranges:
                            mask = (df["Age_Years"] >= min_age) & (df["Age_Years"] < max_age) if max_age < 10 else (df["Age_Years"] >= min_age)
                            if mask.sum() > 0:
                                risk_rate = df[mask]["Is_Failure"].mean() * 100
                                if risk_rate > 25:
                                    risk_icon = "🔴 High Risk"
                                elif risk_rate > 15:
                                    risk_icon = "🟠 Medium Risk"
                                else:
                                    risk_icon = "🟢 Low Risk"
                               
                                risk_explanation += f"• {label}: {risk_icon} ({risk_rate:.0f}% fail)<br>"
                       
                        risk_explanation += "</div>"
                        st.markdown(risk_explanation, unsafe_allow_html=True)
       
        except Exception as e:
            st.info("Prediction model couldn't be built with current data structure.")
   
    else:
        st.info(f"**Prediction Model:** Need at least 50 test results to build a reliable prediction model. You currently have {len(known_results)} results.")
    # Section 6: Practical Recommendations
    st.subheader("🎯 What Should You Do? (Practical Recommendations)")
   
    recommendations = []
   
    # Data quality recommendations
    if data_quality_pct < 90:
        recommendations.append({
            "icon": "📋",
            "title": "Improve Data Quality",
            "description": f"You have {100-data_quality_pct:.0f}% unclear test results. Make sure all tests are recorded as either 'PASS' or 'FAIL' to get better insights.",
            "priority": "High"
        })
   
    # Failure rate recommendations
    if failure_rate > 20:
        recommendations.append({
            "icon": "🔧",
            "title": "Address High Failure Rate",
            "description": f"Your {failure_rate:.0f}% failure rate is concerning. Focus on the worst-performing manufacturers and oldest dampers first.",
            "priority": "High"
        })
   
    # Age-based recommendations
    if "Age_Years" in df.columns:
        old_dampers = df[df["Age_Years"] > 5]
        if len(old_dampers) > 0 and old_dampers["Is_Failure"].mean() > 0.3:
            recommendations.append({
                "icon": "⏰",
                "title": "Replace Old Dampers",
                "description": f"Dampers over 5 years old have a {old_dampers['Is_Failure'].mean()*100:.0f}% failure rate. Consider proactive replacement.",
                "priority": "Medium"
            })
   
    # Manufacturer recommendations
    if "Make" in df.columns and not make_performance.empty:
        if worst_make['failure_rate_pct'] > 25:
            recommendations.append({
                "icon": "🏭",
                "title": f"Review {worst_make['Make']} Dampers",
                "description": f"{worst_make['Make']} has a {worst_make['failure_rate_pct']:.0f}% failure rate. Consider switching suppliers or investigating quality issues.",
                "priority": "Medium"
            })
   
    # Sample size recommendations
    if len(known_results) < 100:
        recommendations.append({
            "icon": "📊",
            "title": "Collect More Data",
            "description": f"With {len(known_results)} test results, increase testing to get more reliable insights. Aim for at least 100 results.",
            "priority": "Low"
        })
   
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_color = {"High": "error-box", "Medium": "warning-box", "Low": "insight-box"}[rec["priority"]]
            st.markdown(f"""
            <div class="{priority_color}">
            <strong>{rec['icon']} {i}. {rec['title']} ({rec['priority']} Priority)</strong><br>
            {rec['description']}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
        <strong>🎉 Great Job!</strong><br>
        Your damper performance looks good overall. Keep monitoring and maintain current quality standards.
        </div>""", unsafe_allow_html=True)
   
    # Final summary
    st.markdown("---")
    st.subheader("📋 Executive Summary")
   
    summary_points = []
    summary_points.append(f"• **Data Quality:** {data_quality_pct:.0f}% of tests have clear results")
    summary_points.append(f"• **Overall Performance:** {failure_rate:.1f}% failure rate ({100-failure_rate:.0f}% success rate)")
   
    if "Make" in df.columns and not make_performance.empty:
        summary_points.append(f"• **Best Manufacturer:** {best_make['Make']} ({best_make['failure_rate_pct']:.1f}% failures)")
        summary_points.append(f"• **Needs Attention:** {worst_make['Make']} ({worst_make['failure_rate_pct']:.1f}% failures)")
   
    if "Age_Years" in df.columns:
        avg_age = df["Age_Years"].mean()
        summary_points.append(f"• **Average Damper Age:** {avg_age:.1f} years")
       
        if age_fail_corr > 0.3:
            summary_points.append(f"• **Key Finding:** Older dampers fail more often - consider preventive replacement")
        elif age_fail_corr < -0.3:
            summary_points.append(f"• **Key Finding:** Newer dampers fail more often - check manufacturing quality")
        else:
            summary_points.append(f"• **Key Finding:** Age doesn't strongly predict failures - other factors more important")
   
    st.markdown("\n".join(summary_points))
   
    # Confidence level
    if len(known_results) > 100:
        confidence_msg = "**High Confidence** 🟢 - These insights are reliable"
    elif len(known_results) > 50:
        confidence_msg = "**Medium Confidence** 🟡 - These insights are fairly reliable"
    else:
        confidence_msg = "**Low Confidence** 🔴 - More data needed for reliable insights"
   
    st.markdown(f"""
    <div class="insight-box">
    <strong>📊 Analysis Confidence Level:</strong> {confidence_msg}<br>
    Based on {len(known_results):,} valid test results from {total_tests:,} total records
    </div>""", unsafe_allow_html=True)
with tab8:
    st.header("📊 Pareto Analysis")
    st.markdown("*Pareto charts showing which categories contribute most to failures, failure percentages, and test volume*")
    if known_results.empty:
        st.warning("⚠️ No valid PASS/FAIL data available for Pareto analysis")
    elif failures == 0:
        st.warning("⚠️ No failure data available for failure Pareto charts")
    else:
        pareto_columns = [("Make", "Manufacturer"), ("TYPE OF DAMPER", "Damper Type")]
        if "Age_Category" in df.columns:
            pareto_columns.append(("Age_Category", "Age Category"))
       
        # Failure Count Pareto Charts
        st.subheader("🔴 Failure Count Contribution Analysis")
        st.markdown("These charts show which categories contribute most to the number of failures (80% line indicates where most issues originate).")
        for col, title in pareto_columns:
            if col in df.columns:
                try:
                    fig_pareto, pareto_data = analytics.generate_pareto_chart(col, title_prefix=f"{title} ")
                    if fig_pareto is not None and not pareto_data.empty:
                        st.markdown(f"**{title} Failure Count Pareto**")
                        st.plotly_chart(fig_pareto, use_container_width=True)
                        with st.expander(f"📋 {title} Failure Count Pareto Data"):
                            st.dataframe(
                                pareto_data.style.format({
                                    'Percentage': '{:.1f}%',
                                    'Cumulative_Percentage': '{:.1f}%'
                                }),
                                use_container_width=True
                            )
                    else:
                        st.info(f"No failure data available for {title} Failure Count Pareto analysis")
                except Exception as e:
                    st.error(f"Pareto analysis error for {title} (Failure Count): {e}")
        # Failure Percentage Pareto Charts
        st.subheader("🟣 Failure Percentage Contribution Analysis")
        st.markdown("These charts show the percentage contribution of each category to total failures (80% line indicates where most failure proportions originate).")
        for col, title in pareto_columns:
            if col in df.columns:
                try:
                    fig_pct, pct_data = analytics.generate_failure_percentage_pareto_chart(col, title_prefix=f"{title} ")
                    if fig_pct is not None and not pct_data.empty:
                        st.markdown(f"**{title} Failure Percentage Pareto**")
                        st.plotly_chart(fig_pct, use_container_width=True)
                        with st.expander(f"📋 {title} Failure Percentage Pareto Data"):
                            st.dataframe(
                                pct_data.style.format({
                                    'Percentage': '{:.1f}%',
                                    'Cumulative_Percentage': '{:.1f}%'
                                }),
                                use_container_width=True
                            )
                    else:
                        st.info(f"No failure data available for {title} Failure Percentage Pareto analysis")
                except Exception as e:
                    st.error(f"Pareto analysis error for {title} (Failure Percentage): {e}")
        # Volume Pareto Charts
        st.subheader("🔵 Test Volume Contribution Analysis")
        st.markdown("These charts show which categories contribute most to total tests conducted.")
        for col, title in pareto_columns:
            if col in df.columns:
                try:
                    fig_volume, volume_data = analytics.generate_volume_pareto_chart(col, title_prefix=f"{title} ")
                    if fig_volume is not None and not volume_data.empty:
                        st.markdown(f"**{title} Volume Pareto**")
                        st.plotly_chart(fig_volume, use_container_width=True)
                        with st.expander(f"📋 {title} Volume Pareto Data"):
                            st.dataframe(
                                volume_data.style.format({
                                    'Percentage': '{:.1f}%',
                                    'Cumulative_Percentage': '{:.1f}%'
                                }),
                                use_container_width=True
                            )
                    else:
                        st.info(f"No test volume data available for {title} Pareto analysis")
                except Exception as e:
                    st.error(f"Pareto analysis error for {title} (Volume): {e}")
# -----------------------------------
# Enhanced Export Options (Sidebar)
# -----------------------------------
# -----------------------------------
# Enhanced Export Options (Sidebar)
# -----------------------------------
with st.sidebar.expander("📥 Enhanced Export Options"):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_main = df.to_csv(index=False).encode('utf-8-sig') # Avoid odd characters in Excel
    st.download_button(
        label="📊 Download Filtered Dataset (CSV)",
        data=csv_main,
        file_name=f"enhanced_damper_analysis_{sheet_choice}_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    # Sample size analysis exports
    if not known_results.empty:
        for col in ["Make", "TYPE OF DAMPER", "Age_Category"]:
            if col in df.columns:
                try:
                    _, sample_data = analytics.generate_sample_size_analysis(col)
                    if not sample_data.empty:
                        st.download_button(
                            label=f"📈 Sample Size Analysis — {col} (CSV)",
                            data=sample_data.to_csv(index=False).encode('utf-8-sig'),
                            file_name=f"sample_analysis_{col}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception:
                    pass
        # Performance matrix export
        try:
            _, perf_matrix = analytics.generate_performance_vs_age_make_analysis()
            if not perf_matrix.empty:
                st.download_button(
                    label="🔥 Performance Matrix Data (CSV)",
                    data=perf_matrix.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f"performance_matrix_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception:
            pass
        # Pareto analysis exports
        for col in ["Make", "TYPE OF DAMPER", "Age_Category"]:
            if col in df.columns:
                try:
                    # Failure Count Pareto
                    _, pareto_data = analytics.generate_pareto_chart(col)
                    if not pareto_data.empty:
                        st.download_button(
                            label=f"📊 Pareto Failure Count Analysis — {col} (CSV)",
                            data=pareto_data.to_csv(index=False).encode('utf-8-sig'),
                            file_name=f"pareto_failure_count_{col}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    # Failure Percentage Pareto
                    _, pct_data = analytics.generate_failure_percentage_pareto_chart(col)
                    if not pct_data.empty:
                        st.download_button(
                            label=f"📊 Pareto Failure Percentage Analysis — {col} (CSV)",
                            data=pct_data.to_csv(index=False).encode('utf-8-sig'),
                            file_name=f"pareto_failure_pct_{col}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    # Volume Pareto
                    _, volume_data = analytics.generate_volume_pareto_chart(col)
                    if not volume_data.empty:
                        st.download_button(
                            label=f"📊 Pareto Volume Analysis — {col} (CSV)",
                            data=volume_data.to_csv(index=False).encode('utf-8-sig'),
                            file_name=f"pareto_volume_{col}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception:
                    pass
# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.subheader("📊 Enhanced Dashboard Status & Analytics Summary")
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    known_pct = (len(known_results) / total_tests * 100.0) if total_tests else 0.0
    badge = "🟢 Excellent" if known_pct > 95 else "🟡 Good" if known_pct > 85 else "🟠 Fair" if known_pct > 70 else "🔴 Poor"
    st.markdown(f"**📊 Data Quality**\n\n{badge}\n({known_pct:.1f}%)")
with f2:
    conf = "🟢 High" if len(known_results) > 100 else "🟡 Medium" if len(known_results) > 50 else "🟠 Low" if len(known_results) > 20 else "🔴 Very Low"
    st.markdown(f"**🎯 Confidence Level**\n\n{conf}\n(n={len(known_results)})")
with f3:
    sys = "🟢 Online" if analytics.connection_status and "Connected successfully" in analytics.connection_status else "🔴 Issues"
    st.markdown(f"**🔄 System Status**\n\n{sys}")
with f4:
    analytics_features = 8 # Updated for Pareto
    st.markdown(f"**🧠 Analytics Features**\n\n🟢 Active\n({analytics_features} modules)")
with f5:
    st.markdown("**⏱ Last Updated**\n\n" + (analytics.last_update.strftime("%H:%M:%S") if analytics.last_update else "Not updated"))
with st.expander("🔧 Enhanced Troubleshooting & Feature Guide"):
    st.markdown(
        """
## 🆕 Key Enhancements
- Bigger performance heatmap (height 700)
- Drill‑down across **Make, Age, Type** + **CSV/XLSX** downloads
- CSVs saved with **UTF‑8‑SIG** to avoid unknown characters in Excel
- Performance matrix, SPC, Pareto, predictive, anomalies
- New **Pareto Analysis Tab** for failure and volume contributions by Make, Type, and Age
## Tips
- Keep ≥30 samples per category for stable insights
- Use date and sidebar filters to narrow analysis
- If you update Google Sheet rows, use **Refresh Data**
"""
    )
st.markdown(
    f"""
<div style='text-align:center;color:gray;margin-top:2rem;padding:1rem;border-top:1px solid #eee;'>
🚀 <strong>Enhanced Damper Analytics Platform v4.1</strong><br>
<em>Sample Size • Drill‑Down • Performance Matrix • SPC • AI Insights • Pareto Analysis</em><br>
Built with Streamlit & Plotly<br>
<small>Processed {total_tests:,} records • {len(known_results):,} valid tests • {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>
</div>
""",
    unsafe_allow_html=True,
)