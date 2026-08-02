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
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

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
# THEME / CUSTOM CSS  — "Ledger & Receipt" design system
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --ink-black: #1B2A26;
    --paper-cream: #F2ECDD;
    --stamp-red: #C1502E;
    --sage: #7C9885;
    --gold: #D4AF37;
    --paper-line: rgba(27,42,38,0.12);
}

/* Base canvas */
.stApp {
    background-color: var(--ink-black);
    color: var(--paper-cream);
    font-family: 'Space Grotesk', sans-serif;
}

/* Kill Streamlit default chrome look */
#MainMenu, header, footer {visibility: hidden;}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--paper-cream);
    letter-spacing: -0.01em;
}

.app-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}
.app-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--sage);
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ---- RECEIPT CARD (signature element) ---- */
.receipt {
    background: var(--paper-cream);
    color: var(--ink-black);
    padding: 2rem 1.8rem 1.5rem 1.8rem;
    font-family: 'IBM Plex Mono', monospace;
    position: relative;
    box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    clip-path: polygon(
        0% 0%, 100% 0%, 100% 96%,
        96% 100%, 92% 96%, 88% 100%, 84% 96%, 80% 100%,
        76% 96%, 72% 100%, 68% 96%, 64% 100%, 60% 96%,
        56% 100%, 52% 96%, 48% 100%, 44% 96%, 40% 100%,
        36% 96%, 32% 100%, 28% 96%, 24% 100%, 20% 96%,
        16% 100%, 12% 96%, 8% 100%, 4% 96%, 0% 100%
    );
}
.receipt-header {
    text-align: center;
    border-bottom: 1px dashed var(--ink-black);
    padding-bottom: 0.8rem;
    margin-bottom: 0.8rem;
}
.receipt-header .label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.65;
}
.receipt-header .amount {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--stamp-red);
}
.receipt-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    padding: 0.28rem 0;
    border-bottom: 1px dotted var(--paper-line);
}
.receipt-row .cat {
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.receipt-row .amt {
    font-weight: 600;
}
.receipt-footer {
    text-align: center;
    margin-top: 1rem;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    opacity: 0.55;
    text-transform: uppercase;
}

/* Stamp badge for anomalies */
.stamp {
    display: inline-block;
    border: 2px solid var(--stamp-red);
    color: var(--stamp-red);
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    transform: rotate(-3deg);
}

/* Panel (right column) */
.panel {
    background: rgba(242,236,221,0.05);
    border: 1px solid rgba(242,236,221,0.15);
    border-radius: 6px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.2rem;
}
.panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.9rem;
}

/* Numbers everywhere use mono */
.mono {
    font-family: 'IBM Plex Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #16211D;
    border-right: 1px solid rgba(242,236,221,0.1);
}
section[data-testid="stSidebar"] * {
    color: var(--paper-cream) !important;
    font-family: 'IBM Plex Mono', monospace;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--paper-cream);
    border-radius: 6px;
}

/* Buttons */
.stButton > button {
    background-color: var(--stamp-red);
    color: var(--paper-cream);
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.78rem;
    padding: 0.5rem 1.1rem;
}
.stButton > button:hover {
    background-color: #a63e21;
    color: var(--paper-cream);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
# ML: CATEGORIZATION (TF-IDF + Naive Bayes, trained on keyword rules as labels)
# ----------------------------------------------------------------------
@st.cache_resource
def train_categorizer(df):
    texts = df["description"].str.lower()
    labels = df["true_category"] if "true_category" in df.columns else None
    if labels is None:
        return None, None
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = MultinomialNB()
    clf.fit(X, labels)
    return vec, clf

def categorize(df, vec, clf):
    if vec is None:
        df["category"] = "Uncategorized"
        return df
    X = vec.transform(df["description"].str.lower())
    df["category"] = clf.predict(X)
    return df

# ----------------------------------------------------------------------
# ML: ANOMALY DETECTION (Isolation Forest on amount + day-of-month)
# ----------------------------------------------------------------------
def detect_anomalies(df):
    feats = pd.DataFrame({
        "amount": df["amount"],
        "day": pd.to_datetime(df["date"]).dt.day,
    })
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(feats)
    return df

# ----------------------------------------------------------------------
# ML: FORECAST (simple linear trend on daily totals)
# ----------------------------------------------------------------------
def forecast_next_period(df, days_ahead=30):
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily["t"] = (pd.to_datetime(daily["date"]) - pd.to_datetime(daily["date"]).min()).dt.days
    X = daily[["t"]].values
    y = daily["amount"].values
    model = LinearRegression().fit(X, y)
    future_t = np.arange(daily["t"].max() + 1, daily["t"].max() + 1 + days_ahead).reshape(-1, 1)
    preds = model.predict(future_t)
    last_date = pd.to_datetime(daily["date"]).max()
    future_dates = [last_date + timedelta(days=int(i) + 1) for i in range(days_ahead)]
    forecast_df = pd.DataFrame({"date": future_dates, "amount": np.clip(preds, 0, None)})
    return daily, forecast_df

def validate_csv(uploaded_file):
    try:
        # Read file without parsing dates yet to prevent initial crash
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return False, None, f"❌ PARSE ERROR: The uploaded file is not a valid CSV format or is corrupted. Details: {str(e)}"
    
    if df.empty:
        return False, None, "❌ LEDGER ERROR: The uploaded ledger file is empty. Please enter valid transaction lines."
    
    # Normalize column names to lowercase to be user friendly
    df.columns = [c.lower() for c in df.columns]
    
    required_cols = ["date", "description", "amount"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, None, f"❌ LEDGER ERROR: Missing required column(s): {', '.join(missing_cols)}. Please verify your column headers."
    
    # Validate date
    if df["date"].isna().any():
        return False, None, "❌ ENTRY ERROR: Missing date(s) found. All ledger lines must have dates."
    try:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        if parsed_dates.isna().any():
            return False, None, "❌ ENTRY ERROR: Invalid or unparseable date format(s) found in the ledger. Dates must be in YYYY-MM-DD or standard parseable format."
        # Assign parsed datetime
        df["date"] = parsed_dates.dt.date
    except Exception as e:
        return False, None, f"❌ ENTRY ERROR: Date validation failed. Details: {str(e)}"
    
    # Validate description
    if df["description"].isna().any() or (df["description"].astype(str).str.strip() == "").any():
        return False, None, "❌ ENTRY ERROR: Missing transaction description(s) found. All ledger lines must have descriptions."
    
    # Validate amount
    if df["amount"].isna().any():
        return False, None, "❌ ENTRY ERROR: Missing transaction amount(s) found. All ledger lines must have amounts."
    
    # Ensure numeric
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        try:
            df["amount"] = pd.to_numeric(df["amount"], errors="raise")
        except Exception:
            return False, None, "❌ VALUE ERROR: Non-numeric values found in the amount column."
    
    # Check for negative amounts
    negative_count = (df["amount"] < 0).sum()
    if negative_count > 0:
        return False, None, f"❌ AUDIT ERROR: Negative amount(s) detected ({negative_count} occurrences). Only positive expense records are accepted in this ledger."
    
    return True, df, None

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
vec, clf = train_categorizer(df)
df = categorize(df, vec, clf)
df = detect_anomalies(df)
daily, forecast_df = forecast_next_period(df, days_ahead=forecast_days)

total_spend = df["amount"].sum()
by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
anomalies = df[df["anomaly"] == -1]

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown('<div class="app-title">Smart Expense Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI-Powered Spending Insights &amp; Anomaly Detection</div>', unsafe_allow_html=True)

if csv_error is not None:
    st.markdown(f"""
    <div class="panel" style="border: 1px solid var(--stamp-red); background: rgba(193,80,46,0.05); margin-bottom: 2rem;">
        <div class="panel-title" style="color: var(--stamp-red);">Ledger Validation Error</div>
        <span class="mono" style="color: var(--paper-cream); font-size: 0.9rem;">{csv_error}</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

col_left, col_right = st.columns([1.1, 1.5], gap="large")

# ---------------- LEFT: RECEIPT CARD ----------------
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

# ---------------- RIGHT: PANELS ----------------
with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Forecast — Next Period</div>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Anomaly Flags</div>', unsafe_allow_html=True)
    if len(anomalies) == 0:
        st.markdown('<span class="mono">No unusual transactions detected.</span>', unsafe_allow_html=True)
    else:
        for _, row in anomalies.sort_values("amount", ascending=False).iterrows():
            st.markdown(
                f'<div class="receipt-row" style="color:var(--paper-cream); border-bottom:1px dotted rgba(242,236,221,0.2);">'
                f'<span class="mono">{row["date"]} · {row["description"]}</span>'
                f'<span class="stamp">₹{row["amount"]:,.0f}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FULL TABLE ----------------
st.markdown('<div class="panel-title" style="margin-top:1.5rem;">All Transactions</div>', unsafe_allow_html=True)

# Filters above the table
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
