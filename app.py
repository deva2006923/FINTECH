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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

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

# Supplemental style block to fix Streamlit's st.container(border=True) rendering and heatmap styling
PANEL_FIX_CSS = """
<style>
div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-marker) {
    background: rgba(242,236,221,0.05) !important;
    border: 1px solid rgba(242,236,221,0.15) !important;
    border-radius: 6px !important;
    padding: 1.4rem 1.5rem !important;
    margin-bottom: 1.2rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-marker) > div {
    padding: 0 !important;
}

/* Conversational AI Assistant note styled like a stapled receipt memo */
.stapled-note {
    background: #F2ECDD !important;
    color: #1B2A26 !important;
    border-left: 5px solid var(--gold) !important;
    padding: 1.2rem 1.4rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    position: relative !important;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.15) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
    border-radius: 2px !important;
}
.stapled-note::before {
    content: "" !important;
    position: absolute !important;
    top: -5px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 38px !important;
    height: 10px !important;
    background: rgba(193, 80, 46, 0.5) !important; /* Stamped ink dark red staple marker */
    border-left: 2px solid rgba(0,0,0,0.1) !important;
    border-right: 2px solid rgba(0,0,0,0.1) !important;
    border-radius: 1px !important;
}

/* Heatmap Container */
.heatmap-scroll-container {
    overflow-x: auto;
    width: 100%;
    padding: 0.5rem 0;
    margin: 1rem 0;
}
.heatmap-cell {
    position: relative;
    cursor: pointer;
}
/* Micro-interaction: styling for tooltip popups matching the physical receipt look */
.heatmap-cell .tooltip {
    visibility: hidden;
    width: 190px;
    background-color: var(--paper-cream) !important;
    color: #1B2A26 !important;
    text-align: left;
    border: 1px solid rgba(27, 42, 38, 0.2) !important;
    border-radius: 4px !important;
    padding: 10px !important;
    position: absolute;
    z-index: 999 !important;
    bottom: 130%;
    left: 50%;
    margin-left: -95px;
    opacity: 0;
    transition: opacity 0.2s ease-in-out;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}
.heatmap-cell:hover .tooltip {
    visibility: visible;
    opacity: 1;
}
/* Mini Receipt Aesthetics */
.mini-receipt-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    border-bottom: 1px dashed #1B2A26 !important;
    margin-bottom: 6px !important;
    padding-bottom: 4px !important;
    color: #1B2A26 !important;
    text-transform: uppercase;
}
.mini-receipt-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
    line-height: 1.1;
    color: #1B2A26 !important;
}
.mini-receipt-total {
    border-top: 1px dashed #1B2A26 !important;
    margin-top: 6px !important;
    padding-top: 4px !important;
    font-weight: 700 !important;
    display: flex;
    justify-content: space-between;
    color: #1B2A26 !important;
}
</style>
"""
st.markdown(PANEL_FIX_CSS, unsafe_allow_html=True)

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

# ======================================================================
# MACHINE LEARNING PIPELINE ARCHITECTURE
# ======================================================================
#
# +--------------------------------------------------------------------+
# |                        RAW TRANSACTION DATA                        |
# |                 (Date, Description, Amount columns)                |
# +--------------------------------------------------------------------+
#                                   |
#                                   v
# +--------------------------------------------------------------------+
# |                    STAGE 1: TEXT CATEGORIZATION                    |
# |  - Model: TF-IDF Vectorizer + Multinomial Naive Bayes              |
# |  - Training: Maps keyword-matching rules (Food, Travel, Bills,     |
# |    Shopping, Entertainment, Health) to standard category tags      |
# |  - Output: Predicts missing category names for custom CSV descriptions|
# +--------------------------------------------------------------------+
#                                   |
#                                   v
# +--------------------------------------------------------------------+
# |                    STAGE 2: ANOMALY DETECTION                      |
# |  - Model: Isolation Forest (contamination = 5%)                    |
# |  - Features: [Transaction Amount, Day of Month]                    |
# |  - Output: Outlier flags (-1 for anomalies, 1 for normal)          |
# |  - Reasoning: Computes deviation ratios vs. category averages to   |
# |    explain why outliers were flagged (e.g. 3x above average).      |
# +--------------------------------------------------------------------+
#                                   |
#                                   v
# +--------------------------------------------------------------------+
# |                    STAGE 3: SPEND FORECASTING                      |
# |  - Model: Ordinary Least Squares (OLS) Linear Regression           |
# |  - Aggregator: Groups transactions by date to compute daily totals |
# |  - Feature: Number of days since first transaction (t)             |
# |  - Output: Predicts daily spending trend for next 7-60 days        |
# +--------------------------------------------------------------------+
#
# ======================================================================

# ----------------------------------------------------------------------
# ML: CATEGORIZATION (TF-IDF + Naive Bayes, trained on keyword rules as labels)
# ----------------------------------------------------------------------
@st.cache_resource
def train_categorizer(df):
    labels = df["true_category"] if "true_category" in df.columns else None
    if labels is None:
        return None, None, None
        
    df_clean = df.dropna(subset=["description", "true_category"])
    if len(df_clean) < 10:
        return None, None, None
        
    # We do a train/test split to evaluate our categorizer
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_clean["description"].astype(str).str.lower(),
        df_clean["true_category"],
        test_size=0.2,
        random_state=42,
        stratify=df_clean["true_category"] if df_clean["true_category"].nunique() > 1 else None
    )
    
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(X_train_text)
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Evaluate
    X_test = vec.transform(X_test_text)
    y_pred = clf.predict(X_test)
    
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "labels": unique_labels,
        "confusion_matrix": cm.tolist()
    }
    
    # Retrain on full dataset to maximize classification power for the actual table categorization
    X_full = vec.fit_transform(df_clean["description"].astype(str).str.lower())
    clf_full = MultinomialNB()
    clf_full.fit(X_full, df_clean["true_category"])
    
    return vec, clf_full, metrics

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

def generate_heatmap_html(df):
    df_heatmap = df.copy()
    # Ensure dates are parsed to datetime.date objects
    df_heatmap["date_parsed"] = pd.to_datetime(df_heatmap["date"]).dt.date
    
    # Group spend by date
    daily_spend = df_heatmap.groupby("date_parsed")["amount"].sum().to_dict()
    
    # Group transactions by date
    daily_txns = {}
    for date_obj, group in df_heatmap.groupby("date_parsed"):
        daily_txns[date_obj] = group[["description", "amount"]].to_dict("records")
        
    if not daily_spend:
        return "<div class='mono' style='color:var(--paper-cream); opacity:0.6;'>No transaction data available for heatmap.</div>"
        
    max_date = max(daily_spend.keys())
    min_date = min(daily_spend.keys())
    
    # Range of 90 days
    end_date = max_date
    start_date = end_date - timedelta(days=90)
    
    # Align to start of the week (Monday)
    start_weekday = start_date.weekday() # Mon=0, Sun=6
    grid_start = start_date - timedelta(days=start_weekday)
    
    # Align end to Sunday
    end_weekday = end_date.weekday()
    grid_end = end_date + timedelta(days=(6 - end_weekday))
    
    weeks = []
    curr = grid_start
    while curr <= grid_end:
        week_dates = []
        for _ in range(7):
            week_dates.append(curr)
            curr += timedelta(days=1)
        weeks.append(week_dates)
        
    non_zero_spends = [v for v in daily_spend.values() if v > 0]
    if non_zero_spends:
        q25 = np.percentile(non_zero_spends, 25)
        q50 = np.percentile(non_zero_spends, 50)
        q75 = np.percentile(non_zero_spends, 75)
    else:
        q25, q50, q75 = 100, 500, 1500
        
    def get_level(amt):
        if amt == 0:
            return 0
        elif amt <= q25:
            return 1
        elif amt <= q50:
            return 2
        elif amt <= q75:
            return 3
        else:
            return 4
            
    colors = {
        0: "#2c3b37",
        1: "rgba(124, 152, 133, 0.25)",
        2: "rgba(124, 152, 133, 0.5)",
        3: "rgba(124, 152, 133, 0.75)",
        4: "rgb(124, 152, 133)"
    }
    
    day_names = ["Mon", "", "Wed", "", "Fri", "", "Sun"]
    rows_html = ""
    for d_idx in range(7):
        row_cells = f'<td style="font-family:\'Space Grotesk\', sans-serif; font-size:0.7rem; color:var(--paper-cream); opacity:0.6; padding-right:8px; text-align:right; vertical-align:middle; line-height:14px; min-width:24px;">{day_names[d_idx]}</td>'
        for week in weeks:
            date_obj = week[d_idx]
            amt = daily_spend.get(date_obj, 0.0)
            level = get_level(amt)
            bg_color = colors[level]
            
            txns = daily_txns.get(date_obj, [])
            txn_rows_html = ""
            for t in txns[:5]:
                txn_rows_html += f'<div class="mini-receipt-row"><span>{t["description"][:16]}</span><span>₹{t["amount"]:,.0f}</span></div>'
            if len(txns) > 5:
                txn_rows_html += f'<div class="mini-receipt-row" style="opacity:0.6;"><span>... +{len(txns)-5} more</span></div>'
                
            tooltip_html = f"""
            <span class="tooltip">
                <div class="mini-receipt-title">{date_obj.strftime('%b %d, %Y')}</div>
                {txn_rows_html if txns else '<div class="mini-receipt-row" style="opacity:0.6;">No activity</div>'}
                <div class="mini-receipt-total"><span>Total Spend</span><span>₹{amt:,.2f}</span></div>
            </span>
            """
            
            if date_obj < min_date or date_obj > max_date:
                row_cells += f'<td style="width:14px; height:14px; background:transparent; border-radius:2px;"></td>'
            else:
                row_cells += f'<td class="heatmap-cell" style="width:14px; height:14px; background:{bg_color}; border-radius:2px; position:relative;">{tooltip_html}</td>'
        rows_html += f'<tr>{row_cells}</tr>'
        
    heatmap_table = f"""
    <div class="heatmap-scroll-container">
        <table style="border-collapse:separate; border-spacing:3px; margin:0 auto;">
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return heatmap_table

def parse_natural_language_query(query, df):
    query = query.lower().strip()
    
    # Standardize keywords to detect timeframes
    today = datetime.today().date()
    df_dates = pd.to_datetime(df["date"]).dt.date
    min_date = df_dates.min() if not df_dates.empty else today
    max_date = df_dates.max() if not df_dates.empty else today
    
    start_date = None
    end_date = max_date
    
    # 1. Check date keywords
    if "last week" in query:
        start_date = max_date - timedelta(days=7)
    elif "this week" in query:
        start_date = max_date - timedelta(days=max_date.weekday())
    elif "last month" in query:
        first_day_this_month = max_date.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        start_date = last_day_last_month.replace(day=1)
        end_date = last_day_last_month
    elif "this month" in query:
        start_date = max_date.replace(day=1)
    elif "today" in query:
        start_date = today
        end_date = today
    elif "yesterday" in query:
        start_date = today - timedelta(days=1)
        end_date = start_date
        
    # Check specific month names in query
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    for m_name, m_val in months_map.items():
        if m_name in query:
            year = max_date.year
            start_date = datetime(year, m_val, 1).date()
            if m_val == 12:
                end_date = datetime(year, 12, 31).date()
            else:
                end_date = (datetime(year, m_val + 1, 1) - timedelta(days=1)).date()
            break
            
    # 2. Check category keywords (case insensitive)
    unique_categories = df["category"].unique()
    matched_category = None
    for cat in unique_categories:
        if cat.lower() in query:
            matched_category = cat
            break
            
    # 3. Check query type
    if "anomaly" in query or "flagged" in query or "unusual" in query:
        anom_df = df[df["anomaly"] == -1]
        return {
            "type": "anomaly_explain",
            "data": anom_df,
            "query": query
        }
        
    if "compare" in query or "vs" in query or "comparison" in query:
        return {
            "type": "compare",
            "category": matched_category,
            "query": query
        }
        
    if "top" in query or "highest" in query or "most" in query:
        import re
        nums = re.findall(r"\d+", query)
        limit = int(nums[0]) if nums else 3
        return {
            "type": "top_spend",
            "limit": limit,
            "category": matched_category,
            "start_date": start_date,
            "end_date": end_date,
            "query": query
        }
        
    if "how much" in query or "spend" in query or "total" in query or matched_category:
        return {
            "type": "spend_summary",
            "category": matched_category,
            "start_date": start_date,
            "end_date": end_date,
            "query": query
        }
        
    return {
        "type": "open_ended",
        "query": query
    }

def execute_assistant_query(parsed, df):
    q_type = parsed["type"]
    
    # Convert dates to datetime.date in df for comparisons
    df_eval = df.copy()
    df_eval["date_parsed"] = pd.to_datetime(df_eval["date"]).dt.date
    
    if q_type == "spend_summary":
        cat = parsed["category"]
        start = parsed["start_date"]
        end = parsed["end_date"]
        
        filtered = df_eval.copy()
        if cat:
            filtered = filtered[filtered["category"] == cat]
        if start:
            filtered = filtered[(filtered["date_parsed"] >= start) & (filtered["date_parsed"] <= end)]
            
        total = filtered["amount"].sum()
        
        scope = f"on **{cat}**" if cat else "in total"
        timeframe = f"between {start} and {end}" if start else "across all logged dates"
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"You spent a total of:<br>"
        resp += f"<span style='font-family:\"Space Grotesk\", sans-serif; font-size:1.6rem; font-weight:700; color:var(--gold);'>₹{total:,.2f}</span><br>"
        resp += f"{scope} {timeframe}.<br><br>"
        
        if not filtered.empty:
            resp += "<strong>Recent matching entries:</strong><br>"
            for _, r in filtered.sort_values("date", ascending=False).head(5).iterrows():
                resp += f"<div style='display:flex; justify-content:space-between; font-size:0.75rem; border-bottom:1px dotted rgba(27,42,38,0.15); padding:2px 0;'><span>{r['date']} · {r['description'][:14]}</span><span>₹{r['amount']:,.0f}</span></div>"
        else:
            resp += "*No matching transactions found in this range.*"
        resp += "</div>"
        return resp

    elif q_type == "top_spend":
        cat = parsed["category"]
        limit = parsed["limit"]
        start = parsed["start_date"]
        end = parsed["end_date"]
        
        filtered = df_eval.copy()
        if cat:
            filtered = filtered[filtered["category"] == cat]
        if start:
            filtered = filtered[(filtered["date_parsed"] >= start) & (filtered["date_parsed"] <= end)]
            
        top_items = filtered.sort_values("amount", ascending=False).head(limit)
        
        scope = f" for {cat}" if cat else ""
        timeframe = f" between {start} and {end}" if start else ""
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"<strong>Top {limit} spending days{scope}{timeframe}:</strong><br><br>"
        
        if not top_items.empty:
            for i, (_, r) in enumerate(top_items.iterrows()):
                resp += f"<div style='display:flex; justify-content:space-between; font-size:0.75rem; border-bottom:1px dotted rgba(27,42,38,0.15); padding:3px 0;'>" \
                        f"<span>#{i+1} {r['date']} · {r['description'][:16]} ({r['category']})</span>" \
                        f"<span style='font-weight:700;'>₹{r['amount']:,.0f}</span></div>"
        else:
            resp += "*No transaction records found.*"
        resp += "</div>"
        return resp

    elif q_type == "compare":
        cat = parsed["category"]
        today = datetime.today().date()
        
        first_this = today.replace(day=1)
        last_last = first_this - timedelta(days=1)
        first_last = last_last.replace(day=1)
        
        df_this = df_eval[(df_eval["date_parsed"] >= first_this) & (df_eval["date_parsed"] <= today)]
        df_last = df_eval[(df_eval["date_parsed"] >= first_last) & (df_eval["date_parsed"] <= last_last)]
        
        if cat:
            df_this = df_this[df_this["category"] == cat]
            df_last = df_last[df_last["category"] == cat]
            
        sum_this = df_this["amount"].sum()
        sum_last = df_last["amount"].sum()
        
        scope = f"on **{cat}**" if cat else "in total"
        diff = sum_this - sum_last
        diff_pct = (diff / sum_last * 100) if sum_last > 0 else 0.0
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"<strong>Month-over-Month Comparison ({scope}):</strong><br>"
        resp += f"<div style='display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed rgba(27,42,38,0.15);'><span>This Month:</span><span style='font-weight:700;'>₹{sum_this:,.2f}</span></div>"
        resp += f"<div style='display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed rgba(27,42,38,0.15);'><span>Last Month:</span><span style='font-weight:700;'>₹{sum_last:,.2f}</span></div>"
        
        if diff > 0:
            resp += f"<br><span style='color:var(--stamp-red); font-weight:700;'>▲ Spending is UP by ₹{diff:,.2f} (+{diff_pct:.1f}%)</span> compared to last month."
        elif diff < 0:
            resp += f"<br><span style='color:var(--sage); font-weight:700;'>▼ Spending is DOWN by ₹{abs(diff):,.2f} ({diff_pct:.1f}%)</span> compared to last month."
        else:
            resp += f"<br>Spending is unchanged compared to last month."
        resp += "</div>"
        return resp

    elif q_type == "anomaly_explain":
        anom_df = parsed["data"]
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"<strong>Anomaly Analysis:</strong><br><br>"
        
        if not anom_df.empty:
            for _, r in anom_df.sort_values("date", ascending=False).head(3).iterrows():
                normal_df = df[df["anomaly"] == 1]
                cat = r["category"]
                amt = r["amount"]
                
                cat_avg = normal_df[normal_df["category"] == cat]["amount"].mean() if not normal_df.empty else 0.0
                if pd.isnull(cat_avg) or cat_avg == 0:
                    cat_avg = df[df["category"] == cat]["amount"].mean()
                if pd.isnull(cat_avg) or cat_avg == 0:
                    cat_avg = df["amount"].mean()
                    
                ratio = amt / cat_avg if cat_avg > 0 else 1.0
                
                resp += f"<span style='color:var(--stamp-red); font-weight:700;'>Flagged:</span> {r['date']} · {r['description']} (₹{r['amount']:,.0f})<br>"
                if ratio >= 1.5:
                    resp += f"→ *Reason*: Amount is **{ratio:.1f}x higher** than the category average of ₹{cat_avg:,.2f}.<br><br>"
                else:
                    resp += f"→ *Reason*: Unusual timing descriptor or merchant pattern detected by Isolation Forest.<br><br>"
        else:
            resp += "*No anomalous transactions found in the database.*"
        resp += "</div>"
        return resp
        
def get_local_fallback_summary(query, df):
    total_spend = df["amount"].sum()
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    anoms = df[df["anomaly"] == -1]
    
    summary = f"**Ledger Financial Health Summary:**<br>"
    summary += f"Total logged expenses sum to ₹{total_spend:,.2f}.<br>"
    
    if not by_cat.empty:
        top_cat = by_cat.index[0]
        top_cat_spend = by_cat.iloc[0]
        pct = (top_cat_spend / total_spend * 100) if total_spend > 0 else 0
        summary += f"- Your highest spending category is **{top_cat}** at ₹{top_cat_spend:,.2f} ({pct:.1f}% of total).<br>"
        
    if len(anoms) > 0:
        summary += f"- Standard Isolation Forest model has flagged **{len(anoms)} unusual transaction(s)**. We recommend reviewing these flags in the 'Anomaly Flags' view.<br>"
    else:
        summary += f"- No critical spending anomalies have been flagged in your recent logs.<br>"
        
    # Budget tips based on top spending category
    summary += "<br>**Advice/Tips:**<br>"
    if not by_cat.empty:
        if top_cat == "Food":
            summary += "→ *Food Spend*: High restaurant/groceries spend. Consider meal-prepping or planning weekly dining budgets to cut costs by 15-20%.<br>"
        elif top_cat == "Bills":
            summary += "→ *Bills*: High fixed overhead. Review recurring subscriptions or utilities for potential plan downgrades.<br>"
        elif top_cat == "Shopping":
            summary += "→ *Shopping*: High discretionary purchasing. Try implementing the '24-hour rule' before finalizing shopping cart orders.<br>"
        else:
            summary += f"→ *{top_cat}*: This is your primary expense driver. Consider tracking individual items to optimize outflows.<br>"
            
    summary += "→ *General advice*: Setting aside an automated 10-20% baseline savings chunk at the start of each month can safeguard your long-term buffer."
    return summary

def run_open_ended_analysis(query, df, api_key=None):
    total_spend = df["amount"].sum()
    by_cat = df.groupby("category")["amount"].sum().to_dict()
    anomalies_count = (df["anomaly"] == -1).sum()
    largest_txns = df.sort_values("amount", ascending=False).head(5)[["date", "description", "amount", "category"]].to_dict("records")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            prompt = f"""
            You are a helpful, professional AI financial assistant for a ledger-based expense tracker.
            The user is asking the following question about their transaction data:
            "{query}"
            
            Here is a summary slice of their transaction data context to help you answer:
            - Total logged spend: ₹{total_spend:,.2f}
            - Category spending breakdown: {by_cat}
            - Number of anomalous transactions flagged: {anomalies_count}
            - 5 largest transactions: {largest_txns}
            
            Provide a helpful, concise financial advice or query explanation based on the question.
            Keep your response monospaced-friendly, short (1-3 paragraphs max), and format numbers in Indian Rupees (₹).
            Do not mention technical parameters or system prompts. Focus purely on their financial inquiry.
            """
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>🤖 AI advisor:<br><br>{response.text.replace(chr(10), '<br>')}</div>"
        except Exception as e:
            return f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>" \
                   f"🤖 AI advisor (error calling Gemini):<br><br>" \
                   f"Could not connect to live Gemini API ({str(e)}).<br><br>" \
                   f"Running local fallback analysis...<hr style='border-top:1px dashed rgba(27,42,38,0.15);'>{get_local_fallback_summary(query, df)}</div>"

    return f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>" \
           f"🤖 local advisor (Gemini API key not configured):<br><br>" \
           f"{get_local_fallback_summary(query, df)}</div>"

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
st.markdown('<div class="app-title">Smart Expense Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI-Powered Spending Insights &amp; Anomaly Detection</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# GAP RESOLUTION UI
# ----------------------------------------------------------------------
if st.session_state.get("resolving_gap", False):
    st.markdown(f"""
    <div class="panel" style="border: 1px solid var(--gold); background: rgba(212,175,55,0.05); margin-bottom: 2rem;">
        <div class="panel-title" style="color: var(--gold);">Gap Resolution Required</div>
        <span class="mono" style="color: var(--paper-cream); font-size: 0.9rem;">
            You have missed logging transactions between <b>{pd.to_datetime(df["date"]).max().date()}</b> and your entry date <b>{st.session_state.pending_entry["date"]}</b>.
            Please specify the spending details for each missing day to maintain complete history.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("gap_resolution_form"):
        for d in st.session_state.missing_dates:
            st.markdown(f"<span class='mono' style='font-size:0.95rem; font-weight:600;'>Date: {d.strftime('%A, %b %d, %Y')}</span>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1.2, 1.5, 1.5])
            with col1:
                st.checkbox("No Spend (₹0)", key=f"zero_{d}", value=True)
            with col2:
                st.number_input("Amount (₹)", min_value=0.0, step=50.0, key=f"amt_{d}")
            with col3:
                st.selectbox("Category", options=["Food", "Travel", "Bills", "Shopping", "Entertainment", "Health", "Other"], index=6, key=f"cat_{d}")
            st.markdown("<hr style='border-top:1px dotted rgba(242,236,221,0.1); margin: 0.5rem 0;'>", unsafe_allow_html=True)
            
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
    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
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

    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Anomaly Flags</div>', unsafe_allow_html=True)
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
                        explanation = "Unusual merchant or transaction descriptor pattern"
                else:
                    if ratio >= 1.5:
                        explanation = f"{ratio:.1f}x higher than {cat} average (₹{cat_avg:,.2f})"
                    elif ratio <= 0.25 and ratio > 0:
                        explanation = f"{1.0/ratio:.1f}x lower than {cat} average (₹{cat_avg:,.2f})"
                    else:
                        explanation = f"Unusual timing pattern (day of month) for {cat}"

                st.markdown(
                    f'<div class="receipt-row" style="color:var(--paper-cream); border-bottom:1px dotted rgba(242,236,221,0.2); align-items: center; padding: 0.4rem 0;">'
                    f'<div style="display: flex; flex-direction: column;">'
                    f'<span class="mono" style="font-weight: 500;">{row["date"]} · {row["description"]}</span>'
                    f'<span class="mono" style="font-size: 0.75rem; color: var(--gold); opacity: 0.85; margin-top: 0.15rem;">→ {explanation}</span>'
                    f'</div>'
                    f'<span class="stamp">₹{row["amount"]:,.0f}</span></div>',
                    unsafe_allow_html=True,
                )

    with st.container(border=True):
        st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Model Performance Metrics</div>', unsafe_allow_html=True)
        if metrics is None:
            st.markdown('<span class="mono" style="opacity:0.65;">No evaluation metrics available (ground-truth labels missing).</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="receipt-row" style="border-bottom:1px dotted rgba(242,236,221,0.15);">'
                f'<span class="mono" style="color:var(--gold);">Accuracy</span>'
                f'<span class="mono" style="font-weight:700;">{metrics["accuracy"]*100:.1f}%</span></div>'
                f'<div class="receipt-row" style="border-bottom:1px dotted rgba(242,236,221,0.15);">'
                f'<span class="mono" style="color:var(--gold);">Weighted F1-Score</span>'
                f'<span class="mono" style="font-weight:700;">{metrics["f1"]:.3f}</span></div>'
                f'<div class="receipt-row" style="border-bottom:1px dotted rgba(242,236,221,0.15);">'
                f'<span class="mono" style="color:var(--gold);">Weighted Precision</span>'
                f'<span class="mono" style="font-weight:700;">{metrics["precision"]:.3f}</span></div>',
                unsafe_allow_html=True
            )
            
            st.markdown('<div class="mono" style="font-size:0.75rem; color:var(--gold); margin-top:1.2rem; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.05em;">Confusion Matrix (True \\ Pred)</div>', unsafe_allow_html=True)
            
            labels = metrics["labels"]
            cm = metrics["confusion_matrix"]
            
            header_cols = "".join(f'<th style="text-align:center; padding:0.25rem; font-weight:600; border-bottom:1px solid rgba(242,236,221,0.25);">{lbl[:4]}</th>' for lbl in labels)
            thead = f'<thead><tr><th style="text-align:left; padding:0.25rem; font-weight:600; border-bottom:1px solid rgba(242,236,221,0.25);">True \\ Pred</th>{header_cols}</tr></thead>'
            
            rows_html = ""
            for i, true_label in enumerate(labels):
                cells = "".join(
                    f'<td style="text-align:center; padding:0.25rem; border-bottom:1px dotted rgba(242,236,221,0.1); '
                    f'background: {"rgba(124,152,133,0.15)" if i == j and cm[i][j] > 0 else "none"};'
                    f'color: {"var(--sage)" if i == j and cm[i][j] > 0 else "var(--paper-cream)"}; font-weight: {"700" if i == j else "400"};">'
                    f'{cm[i][j]}</td>'
                    for j in range(len(labels))
                )
                rows_html += f'<tr><td style="text-align:left; padding:0.25rem; border-bottom:1px dotted rgba(242,236,221,0.1); font-weight:600; text-transform:uppercase; font-size:0.75rem;">{true_label}</td>{cells}</tr>'
            
            cm_table = f"""
            <table class="mono" style="width:100%; border-collapse:collapse; color:var(--paper-cream); font-size:0.8rem; margin-top:0.3rem;">
                {thead}
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """
            st.markdown(cm_table, unsafe_allow_html=True)

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

# ---------------- LEDGER HISTORY VIEWS ----------------
st.markdown('<div class="panel-title" style="margin-top:1.5rem;">Ledger History & Analysis</div>', unsafe_allow_html=True)

view_mode = st.radio(
    "Select View Mode",
    options=["Table List", "Calendar Heatmap", "Spending Trend Chart"],
    horizontal=True,
    label_visibility="collapsed"
)

if view_mode == "Table List":
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

elif view_mode == "Calendar Heatmap":
    st.markdown(generate_heatmap_html(df), unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:center; gap:10px; font-size:0.75rem; font-family:'IBM Plex Mono', monospace; opacity:0.75; margin-top:8px;">
        <span>Less</span>
        <div style="width:12px; height:12px; background:#2c3b37; border-radius:2px;"></div>
        <div style="width:12px; height:12px; background:rgba(124, 152, 133, 0.25); border-radius:2px;"></div>
        <div style="width:12px; height:12px; background:rgba(124, 152, 133, 0.5); border-radius:2px;"></div>
        <div style="width:12px; height:12px; background:rgba(124, 152, 133, 0.75); border-radius:2px;"></div>
        <div style="width:12px; height:12px; background:rgb(124, 152, 133); border-radius:2px;"></div>
        <span>More</span>
    </div>
    """, unsafe_allow_html=True)

else:
    col_chart_toggle1, col_chart_toggle2 = st.columns([1.5, 3])
    with col_chart_toggle1:
        chart_type = st.radio("Chart Type", ["Line Chart", "Bar Chart"], horizontal=True)
        
    df_trend = df.copy()
    df_trend["date_parsed"] = pd.to_datetime(df_trend["date"])
    daily_spend_df = df_trend.groupby("date_parsed")["amount"].sum().reset_index()
    daily_spend_df = daily_spend_df.rename(columns={"amount": "Daily Spend"}).set_index("date_parsed")
    
    if chart_type == "Line Chart":
        st.line_chart(daily_spend_df, height=300)
    else:
        st.bar_chart(daily_spend_df, height=300)
