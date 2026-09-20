# tools.py
"""
Local deterministic tools that operate on JSON and SQLite.
This file is a safe, non-crashing replacement for the original tools module.
It ensures the database schema exists at import time and returns structured
error objects instead of raising for common operational failures.

Tools:
 - knowledge_search(query: str) -> list[dict]
 - ticket_lookup(employee_id: Optional[str], ticket_id: Optional[str]) -> list[dict] | error dict
 - ticket_create(employee_id: str, title: str, description: str, serial_number: Optional[str]=None, priority: Optional[str]=None) -> dict
 - get_employee(employee_id: str) -> dict|None

Behavior:
 - DB initialization is performed via db_init.ensure_db() at import time.
 - All DB operations catch sqlite3.OperationalError and return {"error": "...", "message": "..."}.
 - JSON file loads are defensive; missing files fall back to sensible defaults.
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import re
import uuid
from typing import Optional, List, Dict, Any
import logging

# Module logger
logger = logging.getLogger("tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Paths and defaults
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
EMPLOYEES_FILE = DATA_DIR / "employees.json"
KB_FILE = DATA_DIR / "knowledge.json"

# Import DB initializer and ensure DB exists at import time.
try:
    from db_init import ensure_db, get_db_path  # type: ignore

    ensure_db()
    DB_FILE = get_db_path()
except Exception as exc:
    # Fallback path and ensure directory exists
    DB_FILE = DATA_DIR / "it_support.db"
    try:
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    logger.exception("db_init.ensure_db failed; using fallback DB_FILE=%s; error=%s", str(DB_FILE), exc)

# In-memory fallback KB (used if KB_FILE missing or unreadable)
# Replace the current _DEFAULT_KB definition with this block in tools.py

try:
    # Prefer the canonical sample KB from db_init if available
    from db_init import SAMPLE_KB as _SAMPLE_KB_FROM_DB_INIT  # type: ignore
    # Ensure it's a list and has at least a few entries
    if isinstance(_SAMPLE_KB_FROM_DB_INIT, list) and len(_SAMPLE_KB_FROM_DB_INIT) >= 3:
        _DEFAULT_KB = _SAMPLE_KB_FROM_DB_INIT
    else:
        raise ImportError("SAMPLE_KB missing or invalid in db_init")
except Exception:
    # Fallback minimal KB if db_init is not importable or SAMPLE_KB is invalid
    _DEFAULT_KB = [
        {
            "id": 1,
            "title": "Laptop Won't Boot",
            "tags": ["laptop", "boot", "hardware"],
            "content": "Try power cycle: hold power 10s, remove battery if removable, boot with AC only. If still failing, collect serial number and open a ticket.",
        },
        {
            "id": 2,
            "title": "Connect to Corporate WiFi",
            "tags": ["wifi", "network"],
            "content": "Select 'CorpNet' SSID, use your employee credentials, and accept the certificate prompt.",
        },
        {
            "id": 3,
            "title": "Reset VPN Password",
            "tags": ["vpn", "password", "reset"],
            "content": "To reset your VPN password: visit the VPN portal, enter your employee ID, and follow the emailed link to set a new password.",
        },
    ]



# -------------------------
# Helpers
# -------------------------


def _load_json_safe(path: Path) -> Any:
    """
    Load JSON from path. If file missing or invalid, return None.
    """
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load JSON from %s: %s", str(path), e)
        return None


def _connect_db() -> sqlite3.Connection:
    """
    Open a sqlite3 connection to DB_FILE. Caller must close connection.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# -------------------------
# Knowledge base
# -------------------------


def knowledge_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Deterministic KB search over the JSON KB file or in-memory fallback.
    Returns up to top_k results. Never raises.
    """
    try:
        kb = _load_json_safe(KB_FILE)
        if not isinstance(kb, list):
            kb = _DEFAULT_KB
        q = (query or "").lower()
        scored: List[tuple] = []
        for article in kb:
            score = 0
            text = (article.get("title", "") + " " + article.get("content", "") + " " + " ".join(article.get("tags", []))).lower()
            for token in re.findall(r"\w+", q):
                if token in text:
                    score += text.count(token)
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [a for s, a in scored[:top_k]]
        if not results:
            # return top_k defaults if no match
            return kb[:top_k]
        return results
    except Exception as e:
        logger.exception("Error in knowledge_search: %s", e)
        return _DEFAULT_KB[:top_k]


# -------------------------
# Employee helper
# -------------------------


def get_employee(employee_id: str) -> Optional[Dict[str, Any]]:
    """
    Return employee dict from employees.json or None if not found.
    Safe: returns None on file errors.
    """
    try:
        employees = _load_json_safe(EMPLOYEES_FILE)
        if not isinstance(employees, list):
            return None
        for e in employees:
            if e.get("employee_id") == employee_id:
                return e
        return None
    except Exception as e:
        logger.exception("Error in get_employee: %s", e)
        return None


# -------------------------
# Ticket functions (safe)
# -------------------------


def ticket_lookup(employee_id: Optional[str] = None, ticket_id: Optional[str] = None) -> Any:
    """
    Safe ticket lookup.

    Returns:
      - list[dict] on success (possibly empty)
      - {"error": "invalid_args", "message": "..."} if neither arg provided
      - {"error": "database_error", "message": "..."} on DB operational errors
      - {"error": "unknown_error", "message": "..."} on unexpected errors
    """
    if not ticket_id and not employee_id:
        return {"error": "invalid_args", "message": "Provide employee_id or ticket_id."}

    conn = None
    try:
        conn = _connect_db()
        cur = conn.cursor()
        if ticket_id:
            cur.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        else:
            cur.execute("SELECT * FROM tickets WHERE employee_id = ? ORDER BY created_at DESC", (employee_id,))
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            results.append({k: r[k] for k in r.keys()})
        return results
    except sqlite3.OperationalError as e:
        logger.exception("OperationalError in ticket_lookup: %s", e)
        return {"error": "database_error", "message": str(e)}
    except Exception as e:
        logger.exception("Unexpected error in ticket_lookup: %s", e)
        return {"error": "unknown_error", "message": str(e)}
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def _check_db_openable(timeout: float = 5.0) -> Optional[Dict[str, str]]:
    """
    Quick DB health check. Returns None on success, or an error dict on failure.
    Non-raising: callers should inspect the returned dict.
    """
    try:
        conn = sqlite3.connect(DB_FILE, timeout=timeout)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            _ = cur.fetchone()
        finally:
            conn.close()
        return None
    except sqlite3.OperationalError as e:
        logger.exception("DB open error: %s", e)
        return {"error": "database_unavailable", "message": str(e)}
    except Exception as e:
        logger.exception("Unexpected DB open error: %s", e)
        return {"error": "unknown_db_error", "message": str(e)}

def ticket_create(
    employee_id: str,
    title: str,
    description: str,
    serial_number: Optional[str] = None,
    priority: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new ticket with strict validation.

    Behavior:
    - Requires employee_id, title, description.
    - If data file employees.json exists, employee_id must match an entry.
    - If employees.json is missing, creation is blocked (explicit error).
    - Performs DB health check before writing.
    - Detects similar open tickets and returns a duplicate response.
    - Returns structured dicts on success or error; never raises.
    """
    # 1) Validate required args
    if not employee_id or not title or not description:
        return {"error": "invalid_args", "message": "employee_id, title, and description are required."}

    # 2) DB health check
    db_health = _check_db_openable()
    if db_health:
        return db_health

    # 3) Validate employee exists if employees.json is present
    employees = _load_json_safe(EMPLOYEES_FILE)
    if isinstance(employees, list):
        if not any(e.get("employee_id") == employee_id for e in employees):
            return {"error": "invalid_employee", "message": f"Employee ID {employee_id} not found."}
    else:
        # employees.json missing — do not silently accept unknown IDs
        return {"error": "missing_employee_data", "message": "Employee registry not available; cannot create ticket."}

    # 4) Duplicate check
    existing = ticket_lookup(employee_id=employee_id)
    if isinstance(existing, dict) and existing.get("error"):
        return existing
    for t in existing:
        try:
            status = (t.get("status") or "").lower()
            if status in ("open", "in progress") and title.lower() in (t.get("title") or "").lower():
                return {"duplicate": True, "message": "A similar open ticket already exists", "existing_ticket": t}
        except Exception:
            continue

    # 5) Insert ticket
    conn = None
    try:
        conn = _connect_db()
        cur = conn.cursor()
        now = datetime.utcnow().isoformat()
        ticket_id = "TCKT-" + uuid.uuid4().hex[:8].upper()
        cur.execute(
            """
            INSERT INTO tickets (ticket_id, employee_id, title, description, serial_number, priority, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, employee_id, title, description, serial_number, priority or "medium", "open", now, now),
        )
        conn.commit()
        cur.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cur.fetchone()
        ticket = dict(row) if row else {
            "ticket_id": ticket_id,
            "employee_id": employee_id,
            "title": title,
            "description": description,
            "serial_number": serial_number,
            "priority": priority or "medium",
            "status": "open",
            "created_at": now,
            "updated_at": now,
        }
        return {"duplicate": False, "ticket": ticket}
    except sqlite3.IntegrityError as e:
        logger.exception("IntegrityError in ticket_create: %s", e)
        return {"error": "integrity_error", "message": str(e)}
    except sqlite3.OperationalError as e:
        logger.exception("OperationalError in ticket_create: %s", e)
        return {"error": "database_error", "message": str(e)}
    except Exception as e:
        logger.exception("Unexpected error in ticket_create: %s", e)
        return {"error": "unknown_error", "message": str(e)}
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

# -------------------------
# Small deterministic helpers
# -------------------------


def reset_vpn_password_instructions() -> str:
    return "To reset your VPN password: visit the VPN portal, enter your employee ID, and follow the emailed link to set a new password."


def connect_to_corp_wifi_instructions() -> str:
    return "Select 'CorpNet' SSID, use your employee credentials, and accept the certificate prompt."


# -------------------------
# Module self-test (safe)
# -------------------------
if __name__ == "__main__":
    print("tools.py self-check")
    print("DB_FILE:", str(DB_FILE))
    print("Knowledge search sample:", knowledge_search("laptop boot"))
    print("Employee lookup (non-existent):", get_employee("EMP-UNKNOWN"))
    print("Ticket lookup without args (should return invalid_args):", ticket_lookup())
    print("Attempting to create a test ticket (may fail if employees.json exists and employee missing):")
    res = ticket_create("EMP-TEST", "Test ticket", "This is a test ticket created by tools.py self-check", serial_number="SN-TEST")
    print("Create result:", json.dumps(res, default=str))
    if isinstance(res, dict) and res.get("ticket"):
        tid = res["ticket"].get("ticket_id")
        print("Lookup by ticket_id:", ticket_lookup(ticket_id=tid))
        print("Lookup by employee_id:", ticket_lookup(employee_id="EMP-TEST"))
