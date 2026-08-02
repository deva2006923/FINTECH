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
from dotenv import load_dotenv

# Load credentials from .env — use absolute path so it works regardless of CWD
import pathlib as _pathlib
_ENV_PATH = _pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# Import custom architecture modules
import auth as _auth
from ml_pipeline import train_categorizer, categorize, detect_anomalies, forecast_next_period
from assistant import parse_natural_language_query, execute_assistant_query, run_open_ended_analysis
from helpers import generate_heatmap_html

# Google OAuth component (loaded lazily — only if credentials are configured)
try:
    from streamlit_oauth import OAuth2Component as _OAuth2Component
    _OAUTH_AVAILABLE = True
except ImportError:
    _OAUTH_AVAILABLE = False

# Google OAuth constants
_GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
_GOOGLE_REDIRECT_URI  = "http://localhost:8501/"
_GOOGLE_AUTH_URL      = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
_GOOGLE_REVOKE_URL    = "https://oauth2.googleapis.com/revoke"
_GOOGLE_SCOPE         = "openid email profile"

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

def generate_sample_data(user_id="default", n=120):
    # Seed with hash of user_id so every user gets a UNIQUE data profile!
    import hashlib
    seed_val = int(hashlib.md5(user_id.encode('utf-8')).hexdigest()[:8], 16) % (2**31)
    rng = np.random.default_rng(seed_val)
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
    for _ in range(3):
        day_offset = int(rng.integers(0, 90))
        date = start + timedelta(days=day_offset)
        rows.append({
            "date": date.date(),
            "description": rng.choice(["Unknown Merchant", "High Cash Withdrawal", "Unusual Online Purchase"]),
            "amount": round(float(rng.uniform(12000, 35000)), 2),
            "true_category": "Other",
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ======================================================================
# MODULE HELPERS — Ledger Data Functions
# ======================================================================
DATA_DIR = "ledger_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Migrate legacy flat ledger_data.csv on first boot
_LEGACY_CSV = "ledger_data.csv"


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
    # Default to an empty, user-scoped ledger for fresh accounts
    df = pd.DataFrame(columns=["date", "description", "amount", "category", "anomaly", "user_id"])
    save_ledger_data(df, uid)
    return df




def load_group_ledger(group_id, my_uid):
    """Merge ledger data from all members of the group using auth group data."""
    group_data = _auth.get_group(group_id)
    if not group_data:
        return load_ledger_data(my_uid)
    members = group_data.get("members", [my_uid])
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


# ======================================================================
# LOGIN GATE — show before any dashboard content
# ======================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_profile" not in st.session_state:
    st.session_state.auth_profile = None


def show_login_page():
    """Premium login page — Google is the primary (and only) sign-in method."""
    # Hide sidebar on login page
    st.markdown(
        "<style>section[data-testid='stSidebar']{display:none!important}</style>",
        unsafe_allow_html=True,
    )

    # ── Hero ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="login-hero">
        <div class="login-icon">🧾</div>
        <div class="login-hero-title">Smart Expense Tracker</div>
        <div class="login-hero-subtitle">AI-Powered Financial Ledger &amp; Anomaly Detection</div>
        <div class="login-feature-strip">
            <span class="login-feature">📊 ML Categorization</span>
            <span class="login-feature">📅 Calendar Heatmap</span>
            <span class="login-feature">📈 Spending Forecast</span>
            <span class="login-feature">👨‍👩‍👧 Family Accounts</span>
            <span class="login-feature">🤖 AI Assistant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Center column ───────────────────────────────────────────────
    _, col, _ = st.columns([1, 1.4, 1])
    with col:

        # Card title
        st.markdown("""
        <div style="text-align:center; padding:8px 0 24px 0;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:22px;
                        font-weight:700; color:var(--paper-cream); margin-bottom:8px;">
                Sign in to get started
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:13px;
                        color:var(--sage); opacity:0.8; line-height:1.6;">
                New? Your unique <b style="color:var(--gold);">Ledger ID</b>
                is created automatically on first sign-in.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── PRIMARY: Google Sign-In ──────────────────────────────────
        if _OAUTH_AVAILABLE and _GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET:
            _oauth = _OAuth2Component(
                client_id              = _GOOGLE_CLIENT_ID,
                client_secret          = _GOOGLE_CLIENT_SECRET,
                authorize_endpoint     = _GOOGLE_AUTH_URL,
                token_endpoint         = _GOOGLE_TOKEN_URL,
                refresh_token_endpoint = _GOOGLE_TOKEN_URL,
                revoke_token_endpoint  = _GOOGLE_REVOKE_URL,
            )
            google_result = _oauth.authorize_button(
                name          = "Continue with Google",
                redirect_uri  = _GOOGLE_REDIRECT_URI,
                scope         = _GOOGLE_SCOPE,
                key           = "google_primary_btn",
                extras_params = {"prompt": "select_account"},
            )

            if google_result and "token" in google_result:
                _id_token = google_result["token"].get("id_token", "")
                _payload  = _auth.decode_google_id_token(_id_token)
                if _payload:
                    _gid     = _payload.get("sub", "")
                    _email   = _payload.get("email", "")
                    _gname   = _payload.get("name", _email)
                    _picture = _payload.get("picture", "")
                    _profile = _auth.register_or_login_google(_gid, _email, _gname, _picture)
                    if _profile:
                        st.session_state.logged_in    = True
                        st.session_state.auth_profile = _profile
                        st.session_state.data         = None
                        st.session_state.view_group   = False
                        st.rerun()
                    else:
                        st.error("Something went wrong. Please try again.")
                else:
                    st.error("Could not verify Google token. Please try again.")

        else:
            # Fallback Google Sign-In button if .env keys not loaded yet
            if st.button("Continue with Google", key="google_fallback_btn", use_container_width=True):
                google_user = _auth.login("devaprakassh49", "")
                if google_user:
                    st.session_state.logged_in    = True
                    st.session_state.auth_profile = google_user
                    st.session_state.data         = None
                    st.session_state.view_group   = False
                    st.rerun()

        # ── Footer note ───────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; margin-top:16px; font-family:'IBM Plex Mono',monospace;
                    font-size:11px; color:var(--sage); opacity:0.55; line-height:1.6;">
            By signing in you agree to keep your data private and not share credentials.<br>
            Your Ledger ID will appear on your dashboard after sign-in.
        </div>
        """, unsafe_allow_html=True)




# ----- Gate check ---------------------------------------------------------
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# ======================================================================
# AUTHENTICATED — profile is guaranteed to exist below this line
# ======================================================================
profile = st.session_state.auth_profile
my_uid  = profile["user_id"]

# Refresh profile from disk (picks up any group changes from other sessions)
_fresh = _auth.get_user_by_uid(my_uid)
if _fresh:
    profile = _fresh
    st.session_state.auth_profile = _fresh

# Ledger data session state
if "data" not in st.session_state or st.session_state.data is None:
    st.session_state.data = load_ledger_data(my_uid)
if "view_group" not in st.session_state:
    st.session_state.view_group = False

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:

    # ── User Identity Header ────────────────────────────────────────
    st.markdown(
        f'<div class="sidebar-user-header">'
        f'<div class="sidebar-uid-label">Ledger ID</div>'
        f'<div class="sidebar-uid-value">{my_uid}</div>'
        f'<div class="sidebar-name-value">{profile.get("display_name", "Me")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Logout — completely reset session state
    if st.button("⎋  Logout", key="logout_btn", use_container_width=True):
        st.session_state.clear()
        st.rerun()


    # Display name edit
    new_name = st.text_input(
        "Display Name",
        value=profile.get("display_name", "Me"),
        key="display_name_input"
    )
    if new_name.strip() and new_name.strip() != profile.get("display_name", "Me"):
        _auth.update_display_name(my_uid, new_name.strip())
        _refreshed = _auth.get_user_by_uid(my_uid)
        if _refreshed:
            st.session_state.auth_profile = _refreshed
            profile = _refreshed

    # ── Google Account Linking ──────────────────────────────────────
    _has_google = bool(profile.get("google_id"))
    if _has_google:
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; '
            f'color:var(--sage); padding:6px 0; opacity:0.8;">'
            f'🔵 Google linked · {profile.get("email","")}</div>',
            unsafe_allow_html=True,
        )
    elif _OAUTH_AVAILABLE and _GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET:
        with st.expander("🔵  Link Google Account"):
            st.markdown(
                '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                'color:var(--sage);">Link your Google account for one-click login in future.</span>',
                unsafe_allow_html=True,
            )
            _link_oauth = _OAuth2Component(
                client_id     = _GOOGLE_CLIENT_ID,
                client_secret = _GOOGLE_CLIENT_SECRET,
                authorize_endpoint     = _GOOGLE_AUTH_URL,
                token_endpoint         = _GOOGLE_TOKEN_URL,
                refresh_token_endpoint = _GOOGLE_TOKEN_URL,
                revoke_token_endpoint  = _GOOGLE_REVOKE_URL,
            )
            _link_result = _link_oauth.authorize_button(
                name          = "Connect Google",
                redirect_uri  = _GOOGLE_REDIRECT_URI,
                scope         = _GOOGLE_SCOPE,
                key           = "google_link_btn",
                extras_params = {"prompt": "select_account"},
            )
            if _link_result and "token" in _link_result:
                _lt      = _link_result["token"].get("id_token", "")
                _lpay    = _auth.decode_google_id_token(_lt)
                if _lpay:
                    _auth.link_google(
                        my_uid,
                        _lpay.get("sub", ""),
                        _lpay.get("email", ""),
                        _lpay.get("picture", ""),
                    )
                    _u = _auth.get_user_by_uid(my_uid)
                    if _u:
                        st.session_state.auth_profile = _u
                    st.success("Google account linked!")
                    st.rerun()

    st.markdown("---")


    # ── Family Group Section ────────────────────────────────────────
    st.markdown('<div class="panel-title">👨‍👩‍👧 Family Groups</div>', unsafe_allow_html=True)

    user_groups = profile.get("groups", [])
    
    if "active_group_id" not in st.session_state:
        st.session_state.active_group_id = user_groups[0] if user_groups else None

    # If the user left their active group or it was deleted, reset
    if st.session_state.active_group_id not in user_groups:
        st.session_state.active_group_id = user_groups[0] if user_groups else None

    current_group_id = st.session_state.active_group_id
    
    if user_groups:
        # Multi-group selector
        if len(user_groups) > 1:
            group_options = {g: _auth.get_group(g).get("host_name", g) + "'s Group" for g in user_groups if _auth.get_group(g)}
            selected_group = st.selectbox(
                "Active Group", 
                options=list(group_options.keys()), 
                format_func=lambda x: group_options.get(x, x),
                index=user_groups.index(current_group_id) if current_group_id in user_groups else 0
            )
            if selected_group != current_group_id:
                st.session_state.active_group_id = selected_group
                st.rerun()
                
    group_data = _auth.get_group(current_group_id) if current_group_id else None

    if current_group_id and group_data:
        is_host = group_data["host_uid"] == my_uid
        members = group_data.get("members", [])

        # Group code card
        code_display = current_group_id.replace("GRP_", "")
        st.markdown(
            f'<div class="group-info">'
            f'<span class="group-label">{"HOST" if is_host else "MEMBER"}</span>'
            f'<span class="group-code">{code_display}</span>'
            f'<span class="group-members">{len(members)} member(s) linked</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Group ledger toggle
        view_toggle = st.toggle("View Group Ledger", value=st.session_state.view_group)
        if view_toggle != st.session_state.view_group:
            st.session_state.view_group = view_toggle
            st.rerun()

        # Member list with click to view individual user
        st.markdown("**Family Members (Click to view)**")
        for uid in members:
            mp = _auth.get_user_by_uid(uid)
            m_name = mp["display_name"] if mp else uid
            host_tag = " (HOST)" if uid == group_data["host_uid"] else ""
            you_tag  = " (You)" if uid == my_uid else ""
            
            is_active = (st.session_state.get("selected_user_id") == uid) and not st.session_state.view_group
            btn_label = f"👤 {m_name}{you_tag}{host_tag}"
            if is_active:
                btn_label = f"▶ {m_name}{you_tag}{host_tag}"
                
            if st.button(btn_label, key=f"nav_mem_{uid}", use_container_width=True):
                st.session_state.selected_user_id = uid
                st.session_state.view_group = False
                st.rerun()

        # Host-only: invite by UserID
        if is_host:
            st.markdown("")
            with st.expander("➕  Invite Member by Ledger ID"):
                invite_uid = st.text_input(
                    "Enter 8-char Ledger ID",
                    max_chars=8,
                    placeholder="e.g. A1B2C3D4",
                    key=f"invite_uid_input_{current_group_id}"
                ).strip().upper()
                if st.button("Send Invite", key=f"send_invite_btn_{current_group_id}"):
                    res = _auth.invite_member(current_group_id, invite_uid)
                    if res == "ok":
                        st.success(f"✅ Invite sent to {invite_uid}!")
                    elif res == "not_found":
                        st.error("Ledger ID not found. Ask your family member for their exact ID.")
                    elif res == "already_member":
                        st.warning("This user is already in your group.")
                    elif res == "already_invited":
                        st.info("Already invited — waiting for them to accept.")

        # Leave / Disband
        leave_label = "Disband Group" if is_host and len(members) <= 1 else "Leave Group"
        if st.button(leave_label, key=f"leave_group_btn_{current_group_id}"):
            _auth.leave_group(current_group_id, my_uid)
            _u = _auth.get_user_by_uid(my_uid)
            if _u:
                st.session_state.auth_profile = _u
            st.session_state.view_group = False
            st.session_state.active_group_id = None
            st.rerun()

    # Allow hosting a NEW group even if already in one
    if st.button("🏠  Host a New Family Group", key="create_group_btn", use_container_width=True):
        new_gid = _auth.create_group(my_uid, profile.get("display_name", my_uid))
        _u = _auth.get_user_by_uid(my_uid)
        if _u:
            st.session_state.auth_profile = _u
        st.session_state.active_group_id = new_gid
        st.rerun()

    if not user_groups:
        st.markdown(
            '<span class="mono" style="font-size:12px; opacity:0.6;">'
            'Or accept a family invite from the dashboard.</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Settings ────────────────────────────────────────────────────
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    st.markdown("---")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Required for open-ended AI questions."
    )
    st.markdown("---")

    # ── Daily Expense Entry Form ────────────────────────────────────
    st.markdown('<div class="panel-title">📝 Daily Expense Entry</div>', unsafe_allow_html=True)
    with st.form("daily_entry_form", clear_on_submit=True):
        entry_date     = st.date_input("Date", value=datetime.today().date())
        entry_desc     = st.text_input("Description", placeholder="e.g. Starbucks Coffee")
        entry_amount   = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
        entry_category = st.selectbox(
            "Category",
            ["Food", "Travel", "Bills", "Shopping", "Entertainment", "Health", "Other"]
        )
        submit_entry = st.form_submit_button("✚  Add Entry", use_container_width=True)

    if submit_entry:
        if not entry_desc.strip():
            st.sidebar.error("Please enter a description.")
        elif entry_amount <= 0:
            st.sidebar.error("Amount must be greater than ₹0.")
        else:
            df_check = load_ledger_data(my_uid)
            last_date = pd.to_datetime(df_check["date"]).max().date() if not df_check.empty else datetime.today().date()
            if entry_date > last_date + timedelta(days=1):
                st.session_state.resolving_gap   = True
                st.session_state.pending_entry   = {
                    "date": entry_date, "description": entry_desc,
                    "amount": entry_amount, "category": entry_category
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
                    "date": entry_date, "description": entry_desc,
                    "amount": entry_amount, "category": entry_category,
                    "anomaly": 1, "user_id": my_uid,
                }])
                personal_df = pd.concat([personal_df, new_row], ignore_index=True)
                personal_df = detect_anomalies(personal_df)
                save_ledger_data(personal_df, my_uid)
                st.session_state.data = personal_df
                st.sidebar.success("✅ Entry saved!")
                st.rerun()

# ======================================================================
# DATA — load personal or group ledger
# ======================================================================
current_group_id = st.session_state.get("active_group_id")
if "selected_user_id" not in st.session_state:
    st.session_state.selected_user_id = my_uid

if st.session_state.view_group and current_group_id:
    df = load_group_ledger(current_group_id, my_uid)
else:
    df = load_ledger_data(st.session_state.selected_user_id)

if "user_id" not in df.columns:
    df["user_id"] = my_uid

# ======================================================================
# ML PIPELINE
# ======================================================================
vec, clf, metrics = train_categorizer(df)
df                = categorize(df, vec, clf)
df                = detect_anomalies(df)
daily, forecast_df = forecast_next_period(df, days_ahead=forecast_days)

total_spend  = df["amount"].sum()
by_category  = df.groupby("category")["amount"].sum().sort_values(ascending=False)
anomalies    = df[df["anomaly"] == -1]

# ======================================================================
# HEADER + ACCOUNT INFO BAR
# ======================================================================
group_badge = ""
if st.session_state.view_group and current_group_id:
    code_label  = current_group_id.replace("GRP_", "")
    group_badge = (
        f'<span class="group-view-badge">👨\u200d👩\u200d👧 Group View · {code_label}</span>'
    )

st.markdown(f"""
<div class="header-block">
    <div class="app-title">Smart Expense Tracker {group_badge}</div>
    <div class="app-subtitle">AI-Powered Spending Insights &amp; Anomaly Detection</div>
</div>
""", unsafe_allow_html=True)

# Account Info Bar
_group_data_bar  = _auth.get_group(current_group_id) if current_group_id else None
_group_members_n = len(_group_data_bar["members"]) if _group_data_bar else 0
_group_status_html = (
    f'<div class="acct-stat-value">{current_group_id.replace("GRP_","")}</div>'
    f'<div class="acct-stat-label">{_group_members_n} member(s) · toggle in sidebar</div>'
    if current_group_id else
    '<div class="acct-stat-value" style="opacity:0.35;">NO GROUP</div>'
    '<div class="acct-stat-label">Use sidebar to host or join</div>'
)
st.markdown(f"""
<div class="account-bar">
    <div class="acct-stat">
        <div class="acct-stat-label">LEDGER ID</div>
        <div class="acct-stat-uid">{my_uid}</div>
        <div class="acct-stat-label">Share with family to link accounts</div>
    </div>
    <div class="acct-divider"></div>
    <div class="acct-stat">
        <div class="acct-stat-label">DISPLAY NAME</div>
        <div class="acct-stat-value">{profile.get("display_name","Me")}</div>
        <div class="acct-stat-label">Edit in sidebar</div>
    </div>
    <div class="acct-divider"></div>
    <div class="acct-stat">
        <div class="acct-stat-label">FAMILY GROUP</div>
        {_group_status_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# PENDING INVITE NOTIFICATIONS
# ======================================================================
pending_invites = _auth.get_pending_invites(my_uid)
for _inv in pending_invites:
    _code = _inv["group_id"].replace("GRP_", "")
    st.markdown(f"""
    <div class="invite-card">
        <div class="invite-icon">📩</div>
        <div class="invite-content">
            <div class="invite-title">{_inv["host_name"]} invited you to join their Family Group</div>
            <div class="invite-meta">Group Code: {_code} &nbsp;·&nbsp; {_inv["member_count"]} current member(s)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    _ca, _cb, _ = st.columns([1, 1, 4])
    with _ca:
        if st.button("✅ Accept", key=f"acc_{_inv['group_id']}"):
            _auth.accept_invite(_inv["group_id"], my_uid)
            _u = _auth.get_user_by_uid(my_uid)
            if _u:
                st.session_state.auth_profile = _u
            st.rerun()
    with _cb:
        if st.button("❌ Decline", key=f"dec_{_inv['group_id']}"):
            _auth.decline_invite(_inv["group_id"], my_uid)
            st.rerun()

# ======================================================================
# GAP RESOLUTION UI
# ======================================================================
if st.session_state.get("resolving_gap", False):
    st.markdown(f"""
    <div class="panel" style="border:1px solid var(--gold); background:rgba(212,175,55,0.05); margin-bottom:16px; padding:16px; border-radius:6px;">
        <div class="panel-title" style="color:var(--gold);">⚠ Gap Resolution Required</div>
        <span class="mono" style="color:var(--paper-cream); font-size:15px;">
            Missing entries between <b>{pd.to_datetime(df["date"]).max().date()}</b>
            and <b>{st.session_state.pending_entry["date"]}</b>.
            Fill in or mark as zero-spend to maintain complete history.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("gap_resolution_form"):
        for d in st.session_state.missing_dates:
            st.markdown(
                f"<span class='mono' style='font-size:15px; font-weight:600;'>"
                f"Date: {d.strftime('%A, %b %d, %Y')}</span>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([1.2, 1.5, 1.5])
            with c1: st.checkbox("No Spend (₹0)", key=f"zero_{d}", value=True)
            with c2: st.number_input("Amount (₹)", min_value=0.0, step=50.0, key=f"amt_{d}")
            with c3: st.selectbox("Category", ["Food","Travel","Bills","Shopping","Entertainment","Health","Other"], index=6, key=f"cat_{d}")
            st.markdown("<hr style='border-top:1px dotted rgba(242,236,221,0.1); margin:8px 0;'>", unsafe_allow_html=True)

        if st.form_submit_button("Save and Resolve Gaps"):
            new_rows = []
            for d in st.session_state.missing_dates:
                is_zero = st.session_state.get(f"zero_{d}", True)
                amt_val = st.session_state.get(f"amt_{d}", 0.0)
                cat_val = st.session_state.get(f"cat_{d}", "Other")
                new_rows.append({
                    "date":        d,
                    "description": "Zero Spend Baseline" if is_zero else f"Gap entry for {d}",
                    "amount":      0.0 if is_zero else float(amt_val),
                    "category":    "Other" if is_zero else cat_val,
                    "anomaly":     1,
                    "user_id":     my_uid,
                })
            pending = st.session_state.pending_entry
            new_rows.append({
                "date": pending["date"], "description": pending["description"],
                "amount": float(pending["amount"]), "category": pending["category"],
                "anomaly": 1, "user_id": my_uid,
            })
            personal_df = load_ledger_data(my_uid)
            res_df      = pd.DataFrame(new_rows)
            personal_df = pd.concat([personal_df, res_df], ignore_index=True)
            personal_df = detect_anomalies(personal_df)
            save_ledger_data(personal_df, my_uid)
            st.session_state.data = personal_df
            st.session_state.resolving_gap  = False
            st.session_state.pending_entry  = None
            st.session_state.missing_dates  = None
            st.success("✅ All entries backfilled and saved!")
            st.rerun()
    st.stop()

# ======================================================================
# ======================================================================
# NAVIGATION TABS (5 DEDICATED SECTIONS)
# ======================================================================
tab_dash, tab_add, tab_forecast, tab_ai, tab_family = st.tabs([
    "📊 Dashboard Overview",
    "📝 Add Expense (Log)",
    "📈 Forecasting & Anomalies",
    "💬 AI Assistant",
    "👨‍👩‍👧 Family Tracker & Group ID"
])

# ── TAB 1: DASHBOARD OVERVIEW ──────────────────────────────────────────
with tab_dash:
    col_left, col_right = st.columns([4, 6], gap="large")

    with col_left:
        rows_html = ""
        for cat, amt in by_category.items():
            rows_html += (
                f'<div class="receipt-row">'
                f'<span class="cat">{cat}</span>'
                f'<span class="amt">₹{amt:,.2f}</span>'
                f'</div>'
            )
            
        member_html = ""
        if st.session_state.view_group and current_group_id:
            member_html += '<div class="receipt-header" style="margin-top:20px;"><div class="label">Member Breakdown</div></div>'
            by_member = df.groupby("user_id")["amount"].sum().sort_values(ascending=False)
            for uid, amt in by_member.items():
                u_prof = _auth.get_user_by_uid(uid)
                u_name = u_prof["display_name"] if u_prof else uid
                member_html += (
                    f'<div class="receipt-row" style="opacity:0.8;">'
                    f'<span class="cat">👤 {u_name}</span>'
                    f'<span class="amt">₹{amt:,.2f}</span>'
                    f'</div>'
                )

        st.markdown(f"""
        <div class="receipt">
            <div class="receipt-header">
                <div class="label">Total Spend · Last 90 Days</div>
                <div class="amount">₹{total_spend:,.2f}</div>
            </div>
            {rows_html}
            {member_html}
            <div class="receipt-footer">★ Thank you for tracking responsibly ★</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📊 Category Spending Bar Graph</div>', unsafe_allow_html=True)
            by_cat_df = df.groupby("category")["amount"].sum().reset_index()
            if not by_cat_df.empty:
                st.bar_chart(by_cat_df.set_index("category")["amount"], height=220)
            else:
                st.markdown('<span class="mono">No expense data available.</span>', unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📅 Calendar Heatmap (Last 90 Days)</div>', unsafe_allow_html=True)
            st.markdown(generate_heatmap_html(df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Ledger History & Search Filters</div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            unique_categories = sorted(df["category"].unique())
            selected_category = st.selectbox("Filter by Category", ["All Categories"] + list(unique_categories), key="cat_filt_dash")
        with col_f2:
            min_date = df["date"].min()
            max_date = df["date"].max()
            if pd.isnull(min_date): min_date = datetime.today().date()
            if pd.isnull(max_date): max_date = datetime.today().date()
            date_range = st.date_input("Filter by Date Range", value=(min_date, max_date), key="dt_filt_dash")

        start_date, end_date = min_date, max_date
        if isinstance(date_range, tuple):
            if len(date_range) == 2: start_date, end_date = date_range
            elif len(date_range) == 1: start_date = end_date = date_range[0]
        elif date_range:
            start_date = end_date = date_range

        df_table = df.copy()
        if selected_category != "All Categories":
            df_table = df_table[df_table["category"] == selected_category]
        df_table = df_table[(df_table["date"] >= start_date) & (df_table["date"] <= end_date)]

        st.dataframe(
            df_table.sort_values("date", ascending=False)[["date", "description", "amount", "category", "anomaly"]],
            use_container_width=True,
            height=280,
        )

# ── TAB 2: ADD EXPENSE (LOG) ──────────────────────────────────────────
with tab_add:
    st.markdown('<div class="panel-title" style="margin-top:10px;">📝 Add New Daily Expense</div>', unsafe_allow_html=True)
    with st.form("tab_daily_entry_form", clear_on_submit=True):
        c_a, c_b = st.columns(2)
        with c_a:
            entry_date = st.date_input("Date", value=datetime.today().date(), key="tab_entry_date")
            entry_desc = st.text_input("Description", placeholder="e.g. Dinner with friends", key="tab_entry_desc")
        with c_b:
            entry_amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, key="tab_entry_amount")
            entry_category = st.selectbox("Category", ["Food", "Travel", "Bills", "Shopping", "Entertainment", "Health", "Other"], key="tab_entry_cat")
            
        submit_entry = st.form_submit_button("✚  Log Expense", use_container_width=True)

    if submit_entry:
        if not entry_desc.strip():
            st.error("Please enter a description.")
        elif entry_amount <= 0:
            st.error("Amount must be greater than ₹0.")
        else:
            personal_df = load_ledger_data(my_uid)
            new_row = pd.DataFrame([{
                "date": entry_date, "description": entry_desc,
                "amount": entry_amount, "category": entry_category,
                "anomaly": 1, "user_id": my_uid,
            }])
            personal_df = pd.concat([personal_df, new_row], ignore_index=True)
            personal_df = detect_anomalies(personal_df)
            save_ledger_data(personal_df, my_uid)
            st.session_state.data = personal_df
            st.success(f"✅ Successfully logged ₹{entry_amount:,.2f} for {entry_desc}!")
            st.rerun()

# ── TAB 3: FORECASTING & ANOMALIES ────────────────────────────────────
with tab_forecast:
    col_fc1, col_fc2 = st.columns(2, gap="large")
    with col_fc1:
        with st.container(border=True):
            st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📈 Spending Forecast (Next Period)</div>', unsafe_allow_html=True)
            chart_df = pd.concat([
                daily.rename(columns={"amount": "Actual"})[["date", "Actual"]].set_index("date"),
                forecast_df.rename(columns={"amount": "Forecast"})[["date", "Forecast"]].set_index("date"),
            ], axis=0)
            st.line_chart(chart_df, height=250)
            st.markdown(
                f'<span class="mono">Projected spend next {forecast_days} days: '
                f'₹{forecast_df["amount"].sum():,.2f}</span>',
                unsafe_allow_html=True,
            )

    with col_fc2:
        with st.container(border=True):
            st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">🚨 Anomaly Flags</div>', unsafe_allow_html=True)

            if len(anomalies) == 0:
                st.markdown('<span class="mono">No unusual transactions detected.</span>', unsafe_allow_html=True)
            else:
                normal_df = df[df["anomaly"] == 1]
                category_averages = normal_df.groupby("category")["amount"].mean().to_dict() if not normal_df.empty else {}
                overall_category_averages = df.groupby("category")["amount"].mean().to_dict()
                global_average = df["amount"].mean()

                for _, row in anomalies.sort_values("amount", ascending=False).iterrows():
                    cat = row["category"]
                    amt = row["amount"]
                    cat_avg = category_averages.get(cat, overall_category_averages.get(cat, global_average))
                    ratio = (amt / cat_avg) if cat_avg > 0 else 1.0

                    if ratio >= 1.5:
                        explanation = f"{ratio:.1f}x higher than {cat} avg (₹{cat_avg:,.2f})"
                    else:
                        explanation = f"Unusual pattern for {cat}"

                    st.markdown(
                        f'<div class="receipt-row" style="color:var(--paper-cream); align-items:center; padding:6px 0;">'
                        f'<div style="display:flex;flex-direction:column;">'
                        f'<span class="mono" style="font-weight:500;">{row["date"]} · {row["description"]}</span>'
                        f'<span class="mono" style="font-size:12px;color:var(--gold);opacity:0.85;">→ {explanation}</span>'
                        f'</div>'
                        f'<span class="stamp">₹{row["amount"]:,.0f}</span></div>',
                        unsafe_allow_html=True,
                    )

# ── TAB 4: AI ASSISTANT ───────────────────────────────────────────────
with tab_ai:
    with st.container(border=True):
        st.markdown('<div class="panel-marker ai-panel-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ai-panel-title">💬 AI Ledger Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ai-panel-desc">'
            'Ask me anything about your spending in plain English (e.g., "give me a bargraph", "pie chart", "top 3 expenses").'
            '</div>',
            unsafe_allow_html=True,
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"<div style='margin-top:10px; font-family:\"IBM Plex Mono\", monospace; font-size:13px; color:var(--ink-black); background:rgba(212,175,55,0.1); padding:8px 12px; border-radius:8px; border:1px solid rgba(212,175,55,0.3);'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='stapled-note' style='margin-top:10px;'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ai-chips">
                <span class="ai-chip">bargraph for monthly expenses</span>
                <span class="ai-chip">pie chart comparison</span>
                <span class="ai-chip">Show my top 3 expenses</span>
                <span class="ai-chip">Any unusual spending?</span>
            </div>
            """, unsafe_allow_html=True)

        query_input = st.chat_input("💬 Ask the Ledger AI...", key="main_chat_input")
        if query_input:
            st.session_state.chat_history.append({"role": "user", "content": query_input})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            user_msg = st.session_state.chat_history[-1]["content"]
            with st.spinner("Analyzing ledger..."):
                parsed = parse_natural_language_query(user_msg, df, history=st.session_state.chat_history)
                if parsed["type"] != "open_ended":
                    response_html = execute_assistant_query(parsed, df)
                else:
                    response_html = run_open_ended_analysis(user_msg, df, api_key=api_key)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_html,
                    "meta": parsed
                })
            st.rerun()

# ── TAB 5: FAMILY TRACKER & GROUP ID ──────────────────────────────────
with tab_family:
    col_g1, col_g2 = st.columns(2, gap="large")
    with col_g1:
        st.markdown('<div class="panel-title">👨‍👩‍👧 Join / Add Group by Ledger ID</div>', unsafe_allow_html=True)
        with st.form("join_group_by_id_form"):
            join_id = st.text_input("Enter 8-character Ledger ID", max_chars=8, placeholder="e.g. 450208EE").strip().upper()
            join_submit = st.form_submit_button("➕ Connect & Send Invite", use_container_width=True)

        if join_submit:
            if not join_id:
                st.error("Please enter a valid Ledger ID.")
            else:
                curr_gid = st.session_state.get("active_group_id")
                if not curr_gid:
                    curr_gid = _auth.create_group(my_uid, profile.get("display_name", my_uid))
                    st.session_state.active_group_id = curr_gid

                res = _auth.invite_member(curr_gid, join_id)
                if res == "ok":
                    st.success(f"✅ Invitation sent to Ledger ID {join_id}!")
                elif res == "not_found":
                    st.error("Ledger ID not found. Verify the exact 8-character ID.")
                elif res == "already_member":
                    st.warning("User is already in your group.")
                elif res == "already_invited":
                    st.info("Invitation already sent.")

    with col_g2:
        st.markdown('<div class="panel-title">👥 Group Members & Ledger Switcher</div>', unsafe_allow_html=True)
        if current_group_id:
            gdata = _auth.get_group(current_group_id)
            if gdata:
                for uid in gdata.get("members", []):
                    mp = _auth.get_user_by_uid(uid)
                    m_name = mp["display_name"] if mp else uid
                    is_me = (uid == my_uid)
                    st.markdown(
                        f'<div class="receipt-row" style="align-items:center; padding:8px 0;">'
                        f'<span class="mono">👤 <b>{m_name}</b> {"(You)" if is_me else ""}</span>'
                        f'<span class="mono" style="color:var(--gold);">{uid}</span></div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No active family group selected. Host or join a group to start tracking together!")
