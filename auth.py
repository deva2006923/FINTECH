"""
auth.py — Authentication & Family Group Management
---------------------------------------------------
Handles user registration, login, per-user profiles,
family group creation, and UserID-based invitation system.

Data files (in ledger_data/):
  users.json   — { username: { user_id, password_hash, display_name, group_id } }
  groups.json  — { group_id: { host_uid, host_name, members, pending_invites } }

Security note: passwords are stored as SHA-256 hashes.
Suitable for a local personal finance tool — not production-grade auth.
"""

import json
import uuid
import hashlib
import os

DATA_DIR = "ledger_data"
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE  = os.path.join(DATA_DIR, "users.json")
GROUPS_FILE = os.path.join(DATA_DIR, "groups.json")


# ── Low-level file helpers ──────────────────────────────────────────────

def _load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _load_groups():
    if os.path.exists(GROUPS_FILE):
        with open(GROUPS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_groups(groups):
    with open(GROUPS_FILE, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)

def _hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ── User Account Management ─────────────────────────────────────────────

def register(username, password, display_name):
    """
    Create a new account.
    Returns the user profile dict on success, or None if username is taken.
    """
    users = _load_users()
    key = username.strip().lower()
    if not key:
        return None
    if key in users:
        return None

    uid = uuid.uuid4().hex[:8].upper()
    profile = {
        "user_id":       uid,
        "password_hash": _hash(password),
        "display_name":  display_name.strip() or username.strip(),
        "groups":        [],
    }
    users[key] = profile
    _save_users(users)
    return profile


def login(username, password):
    """
    Authenticate a user.
    Returns the profile dict on success, None on failure.
    """
    users = _load_users()
    key = username.strip().lower()
    user = users.get(key)
    if not user:
        return None
    if user["password_hash"] != _hash(password):
        return None
    return dict(user)


def get_user_by_uid(uid):
    """Find a user profile by their UserID. Returns None if not found."""
    users = _load_users()
    for profile in users.values():
        if profile.get("user_id") == uid:
            return dict(profile)
    return None


def update_display_name(uid, display_name):
    """Update a users display name in users.json."""
    users = _load_users()
    for key, profile in users.items():
        if profile.get("user_id") == uid:
            users[key]["display_name"] = display_name.strip()
            _save_users(users)
            return True
    return False


def add_to_group(uid, group_id):
    """Add a group_id to a user's groups list in users.json."""
    users = _load_users()
    for key, profile in users.items():
        if profile.get("user_id") == uid:
            groups_list = users[key].get("groups", [])
            if group_id not in groups_list:
                groups_list.append(group_id)
                users[key]["groups"] = groups_list
                _save_users(users)
            return True
    return False

def remove_from_group(uid, group_id):
    """Remove a group_id from a user's groups list in users.json."""
    users = _load_users()
    for key, profile in users.items():
        if profile.get("user_id") == uid:
            groups_list = users[key].get("groups", [])
            if group_id in groups_list:
                groups_list.remove(group_id)
                users[key]["groups"] = groups_list
                _save_users(users)
            return True
    return False


# ── Family Group Management ─────────────────────────────────────────────

def create_group(host_uid, host_name):
    """
    Create a new family group hosted by host_uid.
    Returns the new group_id string.
    """
    groups = _load_groups()
    group_id = "GRP_" + uuid.uuid4().hex[:6].upper()
    groups[group_id] = {
        "host_uid":        host_uid,
        "host_name":       host_name,
        "members":         [host_uid],
        "pending_invites": [],
    }
    _save_groups(groups)
    add_to_group(host_uid, group_id)
    return group_id


def get_group(group_id):
    """Return group data dict, or None if not found."""
    return _load_groups().get(group_id)


def invite_member(group_id, target_input):
    """
    Invite target_input (Ledger ID, email, or username) to group_id.
    Returns one of: ok | not_found | already_member | already_invited
    """
    groups = _load_groups()
    group = groups.get(group_id)
    if not group:
        return "not_found"
    
    users = _load_users()
    target_uid = None
    target_str = target_input.strip().lower()

    # 1. Search by exact User ID
    for ukey, uprof in users.items():
        if uprof.get("user_id", "").lower() == target_str:
            target_uid = uprof["user_id"]
            break
        if ukey.lower() == target_str:
            target_uid = uprof["user_id"]
            break
        if uprof.get("email", "").lower() == target_str:
            target_uid = uprof["user_id"]
            break

    if not target_uid:
        return "not_found"

    if target_uid in group["members"]:
        return "already_member"
    if target_uid in group.get("pending_invites", []):
        return "already_invited"
        
    group.setdefault("pending_invites", []).append(target_uid)
    groups[group_id] = group
    _save_groups(groups)
    return "ok"



def accept_invite(group_id, uid):
    """
    Move uid from pending_invites to members.
    Returns True on success.
    """
    groups = _load_groups()
    group = groups.get(group_id)
    if not group or uid not in group.get("pending_invites", []):
        return False
    group["pending_invites"].remove(uid)
    if uid not in group["members"]:
        group["members"].append(uid)
    groups[group_id] = group
    _save_groups(groups)
    add_to_group(uid, group_id)
    return True


def decline_invite(group_id, uid):
    """
    Remove uid from pending_invites without joining.
    """
    groups = _load_groups()
    group = groups.get(group_id)
    if not group or uid not in group.get("pending_invites", []):
        return False
    group["pending_invites"].remove(uid)
    groups[group_id] = group
    _save_groups(groups)
    return True


def leave_group(group_id, uid):
    """
    Remove uid from members. Transfers host if needed. Deletes empty groups.
    """
    groups = _load_groups()
    group = groups.get(group_id)
    if not group or uid not in group.get("members", []):
        return False

    group["members"].remove(uid)

    if group["host_uid"] == uid:
        if group["members"]:
            new_host_uid = group["members"][0]
            group["host_uid"] = new_host_uid
            new_host_profile = get_user_by_uid(new_host_uid)
            group["host_name"] = (
                new_host_profile["display_name"] if new_host_profile else new_host_uid
            )
        else:
            del groups[group_id]
            _save_groups(groups)
            remove_from_group(uid, group_id)
            return True

    groups[group_id] = group
    _save_groups(groups)
    remove_from_group(uid, group_id)
    return True


def get_pending_invites(uid):
    """
    Return list of dicts for all groups where uid has a pending invite.
    Each dict: { group_id, host_uid, host_name, member_count }
    """
    groups = _load_groups()
    result = []
    for gid, group in groups.items():
        if uid in group.get("pending_invites", []):
            result.append({
                "group_id":     gid,
                "host_uid":     group["host_uid"],
                "host_name":    group["host_name"],
                "member_count": len(group.get("members", [])),
            })
    return result


def get_user_groups(uid):
    """
    Return a list of all group dicts that the given user belongs to.
    """
    groups = _load_groups()
    res = []
    for gid, gdata in groups.items():
        if uid in gdata.get("members", []):
            res.append(dict(gdata))
    return res




def list_group_members(group_id):
    """
    Return list of profile dicts for all members in a group.
    """
    group = get_group(group_id)
    if not group:
        return []
    profiles = []
    for uid in group.get("members", []):
        p = get_user_by_uid(uid)
        if p:
            profiles.append(p)
    return profiles


# ── Google OAuth Support ────────────────────────────────────────────────

def decode_google_id_token(id_token):
    """
    Decode a Google ID token (JWT) without signature verification.
    Returns the payload dict: { sub, email, name, picture, ... }
    Google's token is already validated by the OAuth2 exchange — we just
    need the claims payload here.
    """
    import base64
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        # Add padding if needed
        padding = (4 - len(payload_b64) % 4) % 4
        payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload
    except Exception:
        return None


def register_or_login_google(google_id, email, display_name, picture=""):
    """
    Sign in (or sign up) via Google.

    - If a user with this google_id already exists → return their profile (login).
    - If the email is already registered (username/password account) → link Google
      to that account and return it.
    - Otherwise → create a brand new account automatically.

    Returns the user profile dict.
    """
    users = _load_users()

    # 1. Check for existing google_id match
    for key, profile in users.items():
        if profile.get("google_id") == google_id:
            return dict(profile)

    # 2. Check for email match (link Google to existing account)
    for key, profile in users.items():
        if profile.get("email", "").lower() == email.lower():
            users[key]["google_id"] = google_id
            users[key]["picture"]   = picture
            _save_users(users)
            return dict(users[key])

    # 3. New Google user — create account automatically
    uid = uuid.uuid4().hex[:8].upper()
    # Use email prefix as the "username" key (slugified)
    username_key = email.split("@")[0].lower().replace(".", "_").replace("+", "_")
    # Ensure uniqueness
    base_key, counter = username_key, 1
    while username_key in users:
        username_key = f"{base_key}_{counter}"
        counter += 1

    new_profile = {
        "user_id":       uid,
        "password_hash": None,          # no password — Google-only account
        "display_name":  display_name or email.split("@")[0],
        "email":         email,
        "google_id":     google_id,
        "picture":       picture,
        "groups":        [],
    }
    users[username_key] = new_profile
    _save_users(users)
    return new_profile


def link_google(uid, google_id, email, picture=""):
    """
    Link an existing username/password account to a Google account.
    Called when a logged-in user clicks 'Link Google Account'.
    Returns True on success.
    """
    users = _load_users()
    for key, profile in users.items():
        if profile.get("user_id") == uid:
            users[key]["google_id"] = google_id
            users[key]["email"]     = email
            users[key]["picture"]   = picture
            _save_users(users)
            return True
    return False
