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
        "group_id":      None,
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


def update_group_id(uid, group_id):
    """Link (or unlink) a user to a group_id in users.json."""
    users = _load_users()
    for key, profile in users.items():
        if profile.get("user_id") == uid:
            users[key]["group_id"] = group_id
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
    update_group_id(host_uid, group_id)
    return group_id


def get_group(group_id):
    """Return group data dict, or None if not found."""
    return _load_groups().get(group_id)


def invite_member(group_id, target_uid):
    """
    Invite target_uid to group_id.
    Returns one of: ok | not_found | already_member | already_invited
    """
    groups = _load_groups()
    group = groups.get(group_id)
    if not group:
        return "not_found"
    if get_user_by_uid(target_uid) is None:
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
    update_group_id(uid, group_id)
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
            update_group_id(uid, None)
            return True

    groups[group_id] = group
    _save_groups(groups)
    update_group_id(uid, None)
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
