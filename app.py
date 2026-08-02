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
import json
import uuid
import hashlib
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

# ----------------------------------------------------------------------
# USER IDENTITY & GROUP ACCOUNT SYSTEM
# ----------------------------------------------------------------------
DATA_DIR = "ledger_data"
os.makedirs(DATA_DIR, exist_ok=True)

PROFILE_FILE = os.path.join(DATA_DIR, "user_profile.json")
GROUP_INDEX_FILE = os.path.join(DATA_DIR, "group_index.json")

def load_profile():
    """Load or create a unique user profile with an 8-char alphanumeric ID."""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    uid = uuid.uuid4().hex[:8].upper()
    profile = {"user_id": uid, "display_name": "Me", "group_code": None}
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f)
    return profile

def save_profile(profile):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f)

def load_group_index():
    """Load the group index mapping group_code -> list of user_ids."""
    if os.path.exists(GROUP_INDEX_FILE):
        with open(GROUP_INDEX_FILE, "r") as f:
            return json.load(f)
    return {}

def save_group_index(index):
    with open(GROUP_INDEX_FILE, "w") as f:
        json.dump(index, f)

def generate_group_code():
    """Generate a deterministic 6-char group code from a fresh UUID."""
    return uuid.uuid4().hex[:6].upper()

def get_user_ledger_path(uid):
    return os.path.join(DATA_DIR, f"ledger_{uid}.csv")

def save_ledger_data(df, uid):
    df.to_csv(get_user_ledger_path(uid), index=False)

def load_ledger_data(uid):
    path = get_user_ledger_path(uid)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            if "user_id" not in df.columns:
                df["user_id"] = uid
            return df
        except Exception:
            pass
    df = generate_sample_data()
    df["user_id"] = uid
    save_ledger_data(df, uid)
    return df

def load_group_ledger(group_code, my_uid):
    """Merge ledger data from all members of the group."""
    index = load_group_index()
    members = index.get(group_code, [my_uid])
    frames = []
    for uid in members:
        df_m = load_ledger_data(uid)
        df_m["user_id"] = uid
        frames.append(df_m)
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"]).dt.date
        return merged.sort_values("date").reset_index(drop=True)
    return load_ledger_data(my_uid)

# Migrate legacy flat ledger_data.csv if it exists
LEGACY_FILE = "ledger_data.csv"
if os.path.exists(LEGACY_FILE):
    _profile_tmp = load_profile()
    _legacy_path = get_user_ledger_path(_profile_tmp["user_id"])
    if not os.path.exists(_legacy_path):
        import shutil
        shutil.copy(LEGACY_FILE, _legacy_path)

# -- Load user profile and initialize session state --
if "profile" not in st.session_state:
    st.session_state.profile = load_profile()
profile = st.session_state.profile
my_uid = profile["user_id"]

if "data" not in st.session_state or st.session_state.data is None:
    st.session_state.data = load_ledger_data(my_uid)

if "view_group" not in st.session_state:
    st.session_state.view_group = False

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
with st.sidebar:
    # -- Identity Badge --
    st.markdown(
        f'<div class="sidebar-badge">'
        f'<div class="badge-label">YOUR LEDGER ID</div>'
        f'<div class="badge-uid">{my_uid}</div>'
        f'<div class="badge-hint">Share this with family members to link accounts</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Allow display name edit
    new_name = st.text_input("Display Name", value=profile.get("display_name", "Me"), key="display_name_input")
    if new_name != profile.get("display_name", "Me"):
        profile["display_name"] = new_name
        save_profile(profile)
        st.session_state.profile = profile

    st.markdown("---")

    # -- Family / Group Account Section --
    st.markdown('<div class="panel-title">👨‍👩‍👧 Family Account</div>', unsafe_allow_html=True)
    current_group = profile.get("group_code")

    if current_group:
        index = load_group_index()
        members = index.get(current_group, [my_uid])
        st.markdown(
            f'<div class="group-info">'
            f'<span class="group-label">GROUP CODE</span>'
            f'<span class="group-code">{current_group}</span>'
            f'<span class="group-members">{len(members)} member(s) linked</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        view_toggle = st.toggle("View Group Ledger", value=st.session_state.view_group)
        if view_toggle != st.session_state.view_group:
            st.session_state.view_group = view_toggle
            st.rerun()

        if st.button("Leave Group", key="leave_group_btn"):
            idx = load_group_index()
            if current_group in idx and my_uid in idx[current_group]:
                idx[current_group].remove(my_uid)
                if not idx[current_group]:
                    del idx[current_group]
                save_group_index(idx)
            profile["group_code"] = None
            save_profile(profile)
            st.session_state.profile = profile
            st.session_state.view_group = False
            st.success("Left group.")
            st.rerun()
    else:
        with st.expander("＋ Create a Group"):
            if st.button("Generate New Group Code", key="create_group_btn"):
                new_code = generate_group_code()
                idx = load_group_index()
                idx[new_code] = [my_uid]
                save_group_index(idx)
                profile["group_code"] = new_code
                save_profile(profile)
                st.session_state.profile = profile
                st.success(f"Group created! Code: **{new_code}**")
                st.rerun()

        with st.expander("→ Join Existing Group"):
            join_code = st.text_input("Enter Group Code", max_chars=6, placeholder="e.g. A1B2C3").strip().upper()
            if st.button("Join Group", key="join_group_btn"):
                if len(join_code) != 6:
                    st.error("Code must be exactly 6 characters.")
                else:
                    idx = load_group_index()
                    if join_code not in idx:
                        st.error("Group code not found. Ask your family member for the correct code.")
                    elif my_uid in idx[join_code]:
                        st.info("You are already a member of this group.")
                    else:
                        idx[join_code].append(my_uid)
                        save_group_index(idx)
                        profile["group_code"] = join_code
                        save_profile(profile)
                        st.session_state.profile = profile
                        st.success(f"Joined group {join_code}!")
                        st.rerun()

    st.markdown("---")
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    st.markdown("---")
    api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    st.markdown("---")

    # -- Daily Expense Entry Form --
    st.markdown('<div class="panel-title">📝 Daily Expense Entry</div>', unsafe_allow_html=True)
    with st.form("daily_entry_form", clear_on_submit=True):
        entry_date = st.date_input("Date", value=datetime.today().date())
        entry_desc = st.text_input("Description", placeholder="e.g. Starbucks Coffee")
        entry_amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
        entry_category = st.selectbox("Category", ["Food", "Travel", "Bills", "Shopping", "Entertainment", "Health", "Other"])
        submit_entry = st.form_submit_button("✚ Add Entry")

    if submit_entry:
        if not entry_desc.strip():
            st.sidebar.error("Please enter a transaction description.")
        elif entry_amount <= 0:
            st.sidebar.error("Amount must be greater than ₹0.")
        else:
            # Use only personal ledger for gap-check (not group)
            df_check = load_ledger_data(my_uid)
            last_date = pd.to_datetime(df_check["date"]).max().date() if not df_check.empty else datetime.today().date()
            if entry_date > last_date + timedelta(days=1):
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
                personal_df = load_ledger_data(my_uid)
                new_row = pd.DataFrame([{
                    "date": entry_date,
                    "description": entry_desc,
                    "amount": entry_amount,
                    "category": entry_category,
                    "anomaly": 1,
                    "user_id": my_uid,
                }])
                personal_df = pd.concat([personal_df, new_row], ignore_index=True)
                personal_df = detect_anomalies(personal_df)
                save_ledger_data(personal_df, my_uid)
                st.session_state.data = personal_df
                st.sidebar.success("Entry saved!")
                st.rerun()

# Determine which ledger to display
if st.session_state.view_group and profile.get("group_code"):
    df = load_group_ledger(profile["group_code"], my_uid)
else:
    df = load_ledger_data(my_uid)

if "user_id" not in df.columns:
    df["user_id"] = my_uid

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
# HEADER + ACCOUNT INFO BAR
# ----------------------------------------------------------------------
current_group = profile.get("group_code")
group_badge = ""
if st.session_state.view_group and current_group:
    group_badge = f'<span class="group-view-badge">👨\u200d👩\u200d👧 Group View · {current_group}</span>'

st.markdown(f"""
<div class="header-block">
    <div class="app-title">Smart Expense Tracker {group_badge}</div>
    <div class="app-subtitle">AI-Powered Spending Insights &amp; Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# -- Prominent Account Info Bar --
group_code_display = current_group if current_group else "—"
group_members_count = 0
if current_group:
    _gi = load_group_index()
    group_members_count = len(_gi.get(current_group, [my_uid]))

group_status_html = (
    f'<div class="acct-stat-value">{current_group}</div>'
    f'<div class="acct-stat-label">{group_members_count} member(s) linked · Toggle in sidebar</div>'
    if current_group else
    '<div class="acct-stat-value" style="opacity:0.4;">NOT IN A GROUP</div>'
    '<div class="acct-stat-label">Create or join one in the sidebar</div>'
)

st.markdown(f"""
<div class="account-bar">
    <div class="acct-stat">
        <div class="acct-stat-label">YOUR LEDGER ID</div>
        <div class="acct-stat-uid">{my_uid}</div>
        <div class="acct-stat-label">Share with family to link accounts</div>
    </div>
    <div class="acct-divider"></div>
    <div class="acct-stat">
        <div class="acct-stat-label">DISPLAY NAME</div>
        <div class="acct-stat-value">{profile.get("display_name", "Me")}</div>
        <div class="acct-stat-label">Edit in sidebar</div>
    </div>
    <div class="acct-divider"></div>
    <div class="acct-stat">
        <div class="acct-stat-label">FAMILY GROUP CODE</div>
        {group_status_html}
    </div>
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
            personal_df = load_ledger_data(my_uid)
            personal_df = pd.concat([personal_df, res_df], ignore_index=True)
            personal_df = detect_anomalies(personal_df)
            save_ledger_data(personal_df, my_uid)
            st.session_state.data = personal_df
            
            # Clear states
            st.session_state.resolving_gap = False
            st.session_state.pending_entry = None
            st.session_state.missing_dates = None
            st.success("All entries backfilled and saved!")
            st.rerun()
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

# Show user_id column in group view so members can be distinguished
table_cols = ["date", "description", "amount", "category", "anomaly"]
if st.session_state.view_group and profile.get("group_code") and "user_id" in df_table.columns:
    table_cols = ["user_id", "date", "description", "amount", "category", "anomaly"]

st.dataframe(
    df_table.sort_values("date", ascending=False)[table_cols],
    use_container_width=True,
    height=300,
)
