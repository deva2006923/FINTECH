"""
Smart Expense Tracker with AI-Powered Spending Insights
--------------------------------------------------------
Ledger / receipt-styled Streamlit dashboard.
ML: TF-IDF + Naive Bayes categorization, Isolation Forest anomaly detection,
    linear trend forecasting.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Import custom architecture modules
from ml_pipeline import train_categorizer, categorize, detect_anomalies, forecast_next_period
from assistant import parse_natural_language_query, execute_assistant_query, run_open_ended_analysis
from helpers import validate_csv, generate_heatmap_html

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Ledger — Smart Expense Tracker",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# THEME / CUSTOM CSS  — Load from style.css
# ----------------------------------------------------------------------
def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ----------------------------------------------------------------------
# SAMPLE DATA GENERATOR (used if no CSV uploaded)
# ----------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "starbucks", "dominos", "grocery"],
    "Travel": ["uber", "ola", "flight", "irctc", "petrol", "fuel", "metro"],
    "Bills": ["electricity", "recharge", "broadband", "water bill", "gas bill", "insurance premium"],
    "Shopping": ["amazon", "flipkart", "myntra", "mall", "shopping"],
    "Entertainment": ["netflix", "spotify", "movie", "pvr", "bookmyshow", "prime video"],
    "Health": ["pharmacy", "hospital", "doctor", "medical", "gym"],
}

def generate_sample_data(n=180, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime.today() - timedelta(days=90)
    merchants = [m for cat in CATEGORY_KEYWORDS.values() for m in cat]
    weights = rng.dirichlet(np.ones(len(merchants)) * 2)
    for _ in range(n):
        day_offset = int(rng.integers(0, 90))
        date = start + timedelta(days=day_offset)
        merchant = rng.choice(merchants, p=weights)
        base_amt = {
            "Food": (100, 900), "Travel": (50, 1500), "Bills": (300, 3000),
            "Shopping": (200, 5000), "Entertainment": (99, 800), "Health": (150, 4000),
        }
        cat = next(c for c, kws in CATEGORY_KEYWORDS.items() if merchant in kws)
        low, high = base_amt[cat]
        amount = round(float(rng.uniform(low, high)), 2)
        rows.append({"date": date.date(), "description": merchant.title(), "amount": amount, "true_category": cat})

    # inject a few anomalies
    for _ in range(4):
        day_offset = int(rng.integers(0, 90))
        date = start + timedelta(days=day_offset)
        rows.append({
            "date": date.date(),
            "description": rng.choice(["Unknown Merchant", "Cash Withdrawal", "Foreign Txn"]),
            "amount": round(float(rng.uniform(15000, 40000)), 2),
            "true_category": "Other",
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

# ----------------------------------------------------------------------
# RUNTIME ARCHITECTURE
# Helper modules: ml_pipeline, assistant, and helpers hold all underlying algorithms.
# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧾 Ledger")
    st.markdown("Upload transactions or use sample data.")
    uploaded = st.file_uploader("CSV (date, description, amount)", type=["csv"])
    use_sample = st.button("Use sample data")
    st.markdown("---")
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    st.markdown("---")
    api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    st.markdown("---")
    st.markdown('<div class="panel-title">Daily Expense Entry</div>', unsafe_allow_html=True)
    with st.form("daily_entry_form", clear_on_submit=True):
        entry_date = st.date_input("Date", value=datetime.today().date())
        entry_desc = st.text_input("Description", placeholder="e.g. Starbucks Coffee")
        entry_amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
        entry_category = st.selectbox("Category", ["Food", "Travel", "Bills", "Shopping", "Entertainment", "Health", "Other"])
        submit_entry = st.form_submit_button("Submit Entry")
        
    if submit_entry:
        if not entry_desc.strip():
            st.sidebar.error("Please enter a transaction description.")
        elif entry_amount <= 0:
            st.sidebar.error("Amount must be greater than ₹0.")
        elif st.session_state.data is None:
            st.sidebar.error("No ledger loaded. Please load sample data or upload a CSV first.")
        else:
            # Check for gap
            df_check = st.session_state.data
            last_date = pd.to_datetime(df_check["date"]).max().date()
            if entry_date > last_date + timedelta(days=1):
                # Trigger gap resolution state
                st.session_state.resolving_gap = True
                st.session_state.pending_entry = {
                    "date": entry_date,
                    "description": entry_desc,
                    "amount": entry_amount,
                    "category": entry_category
                }
                missing_dates = []
                curr = last_date + timedelta(days=1)
                while curr < entry_date:
                    missing_dates.append(curr)
                    curr += timedelta(days=1)
                st.session_state.missing_dates = missing_dates
                st.rerun()
            else:
                # No gap, just append the entry
                new_row = pd.DataFrame([{
                    "date": entry_date,
                    "description": entry_desc,
                    "amount": entry_amount,
                    "category": entry_category,
                    "anomaly": 1
                }])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.session_state.data = detect_anomalies(st.session_state.data)
                st.sidebar.success("Entry saved successfully!")
                st.rerun()

if "data" not in st.session_state:
    st.session_state.data = None

# Process uploaded file with change tracking
if uploaded is not None:
    file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("last_uploaded_key") != file_key:
        st.session_state.last_uploaded_key = file_key
        st.session_state.force_sample = False
        is_valid, validated_df, error_msg = validate_csv(uploaded)
        if is_valid:
            st.session_state.data = validated_df
            st.session_state.csv_error = None
        else:
            st.session_state.csv_error = error_msg
            st.session_state.data = None

if use_sample:
    st.session_state.data = generate_sample_data()
    st.session_state.csv_error = None
    st.session_state.force_sample = True

# Fallback check
if st.session_state.get("force_sample", False):
    if st.session_state.data is None:
        st.session_state.data = generate_sample_data()
else:
    if uploaded is None and st.session_state.data is None:
        st.session_state.data = generate_sample_data()

csv_error = st.session_state.get("csv_error", None)

# If validation failed and we have no data, set placeholder df so execution doesn't crash before header renders
if st.session_state.data is not None:
    df = st.session_state.data.copy()
else:
    # Minimal dummy dataframe to allow initial downstream code to read, but it will stop rendering at header anyway
    df = pd.DataFrame(columns=["date", "description", "amount", "category", "anomaly"])

# ----------------------------------------------------------------------
# ML PIPELINE
# ----------------------------------------------------------------------
vec, clf, metrics = train_categorizer(df)
df = categorize(df, vec, clf)
df = detect_anomalies(df)
daily, forecast_df = forecast_next_period(df, days_ahead=forecast_days)

total_spend = df["amount"].sum()
by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
anomalies = df[df["anomaly"] == -1]

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="header-block">
    <div class="app-title">Smart Expense Tracker</div>
    <div class="app-subtitle">AI-Powered Spending Insights &amp; Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# GAP RESOLUTION UI
# ----------------------------------------------------------------------
if st.session_state.get("resolving_gap", False):
    st.markdown(f"""
    <div class="panel" style="border: 1px solid var(--gold); background: rgba(212,175,55,0.05); margin-bottom: 16px;">
        <div class="panel-title" style="color: var(--gold);">Gap Resolution Required</div>
        <span class="mono" style="color: var(--paper-cream); font-size: 15px;">
            You have missed logging transactions between <b>{pd.to_datetime(df["date"]).max().date()}</b> and your entry date <b>{st.session_state.pending_entry["date"]}</b>.
            Please specify the spending details for each missing day to maintain complete history.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("gap_resolution_form"):
        for d in st.session_state.missing_dates:
            st.markdown(f"<span class='mono' style='font-size:15px; font-weight:600;'>Date: {d.strftime('%A, %b %d, %Y')}</span>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1.2, 1.5, 1.5])
            with col1:
                st.checkbox("No Spend (₹0)", key=f"zero_{d}", value=True)
            with col2:
                st.number_input("Amount (₹)", min_value=0.0, step=50.0, key=f"amt_{d}")
            with col3:
                st.selectbox("Category", options=["Food", "Travel", "Bills", "Shopping", "Entertainment", "Health", "Other"], index=6, key=f"cat_{d}")
            st.markdown("<hr style='border-top:1px dotted rgba(242,236,221,0.1); margin: 8px 0;'>", unsafe_allow_html=True)
            
        if st.form_submit_button("Save and Resolve Gaps"):
            new_rows = []
            for d in st.session_state.missing_dates:
                is_zero_val = st.session_state.get(f"zero_{d}", True)
                amt_val = st.session_state.get(f"amt_{d}", 0.0)
                cat_val = st.session_state.get(f"cat_{d}", "Other")
                
                if is_zero_val:
                    new_rows.append({
                        "date": d,
                        "description": "Zero Spend Baseline",
                        "amount": 0.0,
                        "category": "Other",
                        "anomaly": 1
                    })
                else:
                    new_rows.append({
                        "date": d,
                        "description": f"Gap entry for {d}",
                        "amount": float(amt_val),
                        "category": cat_val,
                        "anomaly": 1
                    })
            
            # Add pending entry
            pending = st.session_state.pending_entry
            new_rows.append({
                "date": pending["date"],
                "description": pending["description"],
                "amount": float(pending["amount"]),
                "category": pending["category"],
                "anomaly": 1
            })
            
            res_df = pd.DataFrame(new_rows)
            st.session_state.data = pd.concat([st.session_state.data, res_df], ignore_index=True)
            st.session_state.data = detect_anomalies(st.session_state.data)
            
            # Clear states
            st.session_state.resolving_gap = False
            st.session_state.pending_entry = None
            st.session_state.missing_dates = None
            st.success("All entries backfilled and saved!")
            st.rerun()
    st.stop()


if csv_error is not None:
    st.markdown(f"""
    <div class="panel" style="border: 1px solid var(--stamp-red); background: rgba(193,80,46,0.05); margin-bottom: 16px;">
        <div class="panel-title" style="color: var(--stamp-red);">Ledger Validation Error</div>
        <span class="mono" style="color: var(--paper-cream); font-size: 15px;">{csv_error}</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------- COLUMNS LAYOUT (40/60 Split) ----------------
col_left, col_right = st.columns([4, 6], gap="large")

# ---------------- LEFT Column: Receipt Card Only ----------------
with col_left:
    rows_html = ""
    for cat, amt in by_category.items():
        rows_html += f"""<div class="receipt-row"><span class="cat">{cat}</span><span class="amt">₹{amt:,.2f}</span></div>"""

    receipt_html = f"""
    <div class="receipt">
        <div class="receipt-header">
            <div class="label">Total Spend · Last 90 Days</div>
            <div class="amount">₹{total_spend:,.2f}</div>
        </div>
        {rows_html}
        <div class="receipt-footer">*** Thank you for tracking responsibly ***</div>
    </div>
    """
    st.markdown(receipt_html, unsafe_allow_html=True)

# ---------------- RIGHT Column: Card Stack ----------------
with col_right:
    # [1] Calendar Heatmap
    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📅 Calendar Heatmap (Last 90 Days)</div>', unsafe_allow_html=True)
        st.markdown(generate_heatmap_html(df), unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:8px; font-size:12px; font-family:'IBM Plex Mono', monospace; opacity:0.75; margin-top:8px;">
            <span>Less</span>
            <div style="width:12px; height:12px; background:#2c3b37; border-radius:2px;"></div>
            <div style="width:12px; height:12px; background:rgba(124, 152, 133, 0.25); border-radius:2px;"></div>
            <div style="width:12px; height:12px; background:rgba(124, 152, 133, 0.5); border-radius:2px;"></div>
            <div style="width:12px; height:12px; background:rgba(124, 152, 133, 0.75); border-radius:2px;"></div>
            <div style="width:12px; height:12px; background:rgb(124, 152, 133); border-radius:2px;"></div>
            <span>More</span>
        </div>
        """, unsafe_allow_html=True)

    # [2] Forecast Chart
    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📈 Forecast — Next Period</div>', unsafe_allow_html=True)
        chart_df = pd.concat([
            daily.rename(columns={"amount": "Actual"})[["date", "Actual"]].set_index("date"),
            forecast_df.rename(columns={"amount": "Forecast"})[["date", "Forecast"]].set_index("date"),
        ], axis=0)
        st.line_chart(chart_df, height=220)
        st.markdown(
            f'<span class="mono">Projected next {forecast_days} days: '
            f'₹{forecast_df["amount"].sum():,.2f}</span>',
            unsafe_allow_html=True,
        )

    # [3] AI Assistant Chat Panel
    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💬 AI Ledger Assistant</div>', unsafe_allow_html=True)
        
        query_input = st.text_input("Ask about your spending data:", placeholder="e.g., How much did I spend on Food this month?")
        
        if query_input:
            parsed = parse_natural_language_query(query_input, df)
            
            if parsed["type"] != "open_ended":
                response_html = execute_assistant_query(parsed, df)
            else:
                response_html = run_open_ended_analysis(query_input, df, api_key=api_key)
                
            st.markdown(f'<div class="stapled-note">{response_html}</div>', unsafe_allow_html=True)

    # [4] Anomaly Flags & Performance Metrics
    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🚨 Anomaly Flags & Performance Metrics</div>', unsafe_allow_html=True)
        
        if len(anomalies) == 0:
            st.markdown('<span class="mono">No unusual transactions detected.</span>', unsafe_allow_html=True)
        else:
            # Calculate category averages from normal transactions
            normal_df = df[df["anomaly"] == 1]
            category_averages = normal_df.groupby("category")["amount"].mean().to_dict() if not normal_df.empty else {}
            overall_category_averages = df.groupby("category")["amount"].mean().to_dict()
            global_average = df["amount"].mean()

            for _, row in anomalies.sort_values("amount", ascending=False).iterrows():
                cat = row["category"]
                amt = row["amount"]
                cat_avg = category_averages.get(cat, overall_category_averages.get(cat, global_average))

                if cat_avg > 0:
                    ratio = amt / cat_avg
                else:
                    ratio = 1.0

                if cat in ["Other", "Uncategorized"]:
                    if ratio >= 1.5:
                        explanation = f"{ratio:.1f}x higher than baseline standard (₹{cat_avg:,.2f})"
                    else:
                        explanation = "Unusual merchant descriptor pattern"
                else:
                    if ratio >= 1.5:
                        explanation = f"{ratio:.1f}x higher than {cat} average (₹{cat_avg:,.2f})"
                    elif ratio <= 0.25 and ratio > 0:
                        explanation = f"{1.0/ratio:.1f}x lower than {cat} average (₹{cat_avg:,.2f})"
                    else:
                        explanation = f"Unusual timing pattern for {cat}"

                st.markdown(
                    f'<div class="receipt-row" style="color:var(--paper-cream); border-bottom:1px dotted rgba(242,236,221,0.2); align-items: center; padding: 8px 0;">'
                    f'<div style="display: flex; flex-direction: column;">'
                    f'<span class="mono" style="font-weight: 500;">{row["date"]} · {row["description"]}</span>'
                    f'<span class="mono" style="font-size: 12px; color: var(--gold); opacity: 0.85; margin-top: 4px;">→ {explanation}</span>'
                    f'</div>'
                    f'<span class="stamp">₹{row["amount"]:,.0f}</span></div>',
                    unsafe_allow_html=True,
                )
                
        st.markdown('<hr style="border-top:1px dashed rgba(242,236,221,0.25); margin:16px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="color:var(--gold); margin-bottom:8px;">Model Performance</div>', unsafe_allow_html=True)
        if metrics is None:
            st.markdown('<span class="mono" style="opacity:0.65; font-size:15px;">No evaluation metrics available (ground-truth labels missing).</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="receipt-row" style="border-bottom:1px dotted rgba(242,236,221,0.15); padding:8px 0;">'
                f'<span class="mono" style="color:var(--gold);">Accuracy</span>'
                f'<span class="mono" style="font-weight:700;">{metrics["accuracy"]*100:.1f}%</span></div>'
                f'<div class="receipt-row" style="border-bottom:1px dotted rgba(242,236,221,0.15); padding:8px 0;">'
                f'<span class="mono" style="color:var(--gold);">Weighted F1-Score</span>'
                f'<span class="mono" style="font-weight:700;">{metrics["f1"]:.3f}</span></div>'
                f'<div class="receipt-row" style="border-bottom:1px dotted rgba(242,236,221,0.15); padding:8px 0;">'
                f'<span class="mono" style="color:var(--gold);">Weighted Precision</span>'
                f'<span class="mono" style="font-weight:700;">{metrics["precision"]:.3f}</span></div>',
                unsafe_allow_html=True
            )
            
            st.markdown('<div class="mono" style="font-size:12px; color:var(--gold); margin-top:16px; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;">Confusion Matrix</div>', unsafe_allow_html=True)
            
            labels = metrics["labels"]
            cm = metrics["confusion_matrix"]
            
            header_cols = "".join(f'<th style="text-align:center; padding:4px; font-weight:600; border-bottom:1px solid rgba(242,236,221,0.25); font-size:12px;">{lbl[:4]}</th>' for lbl in labels)
            thead = f'<thead><tr><th style="text-align:left; padding:4px; font-weight:600; border-bottom:1px solid rgba(242,236,221,0.25); font-size:12px;">True \\ Pred</th>{header_cols}</tr></thead>'
            
            rows_html = ""
            for i, true_label in enumerate(labels):
                cells = "".join(
                    f'<td style="text-align:center; padding:4px; border-bottom:1px dotted rgba(242,236,221,0.1); font-size:12px; '
                    f'background: {"rgba(124,152,133,0.15)" if i == j and cm[i][j] > 0 else "none"};'
                    f'color: {"var(--sage)" if i == j and cm[i][j] > 0 else "var(--paper-cream)"}; font-weight: {"700" if i == j else "400"};">'
                    f'{cm[i][j]}</td>'
                    for j in range(len(labels))
                )
                rows_html += f'<tr><td style="text-align:left; padding:4px; border-bottom:1px dotted rgba(242,236,221,0.1); font-weight:600; text-transform:uppercase; font-size:12px;">{true_label[:4]}</td>{cells}</tr>'
            
            cm_table = f"""
            <table class="mono" style="width:100%; border-collapse:collapse; color:var(--paper-cream); font-size:12px; margin-top:4px;">
                {thead}
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """
            st.markdown(cm_table, unsafe_allow_html=True)

# ---------------- LEDGER HISTORY VIEWS (Full Width) ----------------
st.markdown('<hr style="border-top:1px solid rgba(242,236,221,0.15); margin:32px 0;">', unsafe_allow_html=True)
st.markdown('<div class="panel-title" style="margin-top:16px;">Ledger History & Search Filters</div>', unsafe_allow_html=True)

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    unique_categories = sorted(df["category"].unique())
    selected_category = st.selectbox("Filter by Category", ["All Categories"] + list(unique_categories))
with col_filter2:
    min_date = df["date"].min()
    max_date = df["date"].max()
    if pd.isnull(min_date):
        min_date = datetime.today().date()
    if pd.isnull(max_date):
        max_date = datetime.today().date()
    
    date_range = st.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

# Extract date bounds safely
start_date = min_date
end_date = max_date
if isinstance(date_range, tuple):
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = date_range[0]
        end_date = date_range[0]
elif date_range:
    start_date = date_range
    end_date = date_range

# Apply filters
df_table = df.copy()
if selected_category != "All Categories":
    df_table = df_table[df_table["category"] == selected_category]
df_table = df_table[(df_table["date"] >= start_date) & (df_table["date"] <= end_date)]

st.dataframe(
    df_table.sort_values("date", ascending=False)[["date", "description", "amount", "category", "anomaly"]],
    use_container_width=True,
    height=300,
)
