# db_init.py
"""
Database and sample data initializer for the agentic IT support app.

- Ensures the data directory exists.
- Ensures sample JSON files (employees.json, knowledge.json) exist.
- Ensures the SQLite DB exists and the `tickets` table is created (idempotent).
- Exposes `ensure_db(db_path: Optional[Path])` and `get_db_path()` for other modules.
- Safe to import at module load time; logs errors but does not raise on failure.
"""

from __future__ import annotations
import sqlite3
import json
import logging
import pathlib
from typing import Optional

# Module logger
logger = logging.getLogger("db_init")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default paths
ROOT = pathlib.Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "it_support.db"
EMPLOYEES_FILE = DATA_DIR / "employees.json"
KB_FILE = DATA_DIR / "knowledge.json"

# 50 sample employees
SAMPLE_EMPLOYEES = [
    {"employee_id": "EMP1001", "name": "Alice Johnson", "email": "alice.johnson@example.com", "department": "Engineering"},
    {"employee_id": "EMP1002", "name": "Bob Singh", "email": "bob.singh@example.com", "department": "Sales"},
    {"employee_id": "EMP1003", "name": "Carlos Mendes", "email": "carlos.mendes@example.com", "department": "Marketing"},
    {"employee_id": "EMP1004", "name": "Diana Lee", "email": "diana.lee@example.com", "department": "Product"},
    {"employee_id": "EMP1005", "name": "Ethan Brown", "email": "ethan.brown@example.com", "department": "Engineering"},
    {"employee_id": "EMP1006", "name": "Fiona Patel", "email": "fiona.patel@example.com", "department": "HR"},
    {"employee_id": "EMP1007", "name": "George Kim", "email": "george.kim@example.com", "department": "Support"},
    {"employee_id": "EMP1008", "name": "Hannah Wilson", "email": "hannah.wilson@example.com", "department": "Finance"},
    {"employee_id": "EMP1009", "name": "Ian Thompson", "email": "ian.thompson@example.com", "department": "Engineering"},
    {"employee_id": "EMP1010", "name": "Jaya Rao", "email": "jaya.rao@example.com", "department": "Sales"},
    {"employee_id": "EMP1011", "name": "Kumar", "email": "kumar@example.com", "department": "HR"},
    {"employee_id": "EMP1012", "name": "Lina Gomez", "email": "lina.gomez@example.com", "department": "Marketing"},
    {"employee_id": "EMP1013", "name": "Marcus Allen", "email": "marcus.allen@example.com", "department": "Product"},
    {"employee_id": "EMP1014", "name": "Nadia Petrova", "email": "nadia.petrova@example.com", "department": "Engineering"},
    {"employee_id": "EMP1015", "name": "Omar Farouk", "email": "omar.farouk@example.com", "department": "Support"},
    {"employee_id": "EMP1016", "name": "Priya Sharma", "email": "priya.sharma@example.com", "department": "Finance"},
    {"employee_id": "EMP1017", "name": "Quentin Blake", "email": "quentin.blake@example.com", "department": "Engineering"},
    {"employee_id": "EMP1018", "name": "Rina Sato", "email": "rina.sato@example.com", "department": "Design"},
    {"employee_id": "EMP1019", "name": "Samir Khan", "email": "samir.khan@example.com", "department": "Operations"},
    {"employee_id": "EMP1020", "name": "Tara O'Neil", "email": "tara.oneil@example.com", "department": "Legal"},
    {"employee_id": "EMP1021", "name": "Umar Aziz", "email": "umar.aziz@example.com", "department": "Engineering"},
    {"employee_id": "EMP1022", "name": "Vera Novak", "email": "vera.novak@example.com", "department": "HR"},
    {"employee_id": "EMP1023", "name": "Will Turner", "email": "will.turner@example.com", "department": "Sales"},
    {"employee_id": "EMP1024", "name": "Kumar R", "email": "kumar.r@example.com", "department": "HR"},
    {"employee_id": "EMP1025", "name": "Xiao Li", "email": "xiao.li@example.com", "department": "Engineering"},
    {"employee_id": "EMP1026", "name": "Yara Haddad", "email": "yara.haddad@example.com", "department": "Marketing"},
    {"employee_id": "EMP1027", "name": "Zane Cooper", "email": "zane.cooper@example.com", "department": "Product"},
    {"employee_id": "EMP1028", "name": "Aisha Bello", "email": "aisha.bello@example.com", "department": "Support"},
    {"employee_id": "EMP1029", "name": "Ben Carter", "email": "ben.carter@example.com", "department": "Finance"},
    {"employee_id": "EMP1030", "name": "Chloe Martin", "email": "chloe.martin@example.com", "department": "Design"},
    {"employee_id": "EMP1031", "name": "Diego Alvarez", "email": "diego.alvarez@example.com", "department": "Engineering"},
    {"employee_id": "EMP1032", "name": "Elena Rossi", "email": "elena.rossi@example.com", "department": "Sales"},
    {"employee_id": "EMP1033", "name": "Farah Ahmed", "email": "farah.ahmed@example.com", "department": "HR"},
    {"employee_id": "EMP1034", "name": "Gavin Brooks", "email": "gavin.brooks@example.com", "department": "Operations"},
    {"employee_id": "EMP1035", "name": "Hiro Tanaka", "email": "hiro.tanaka@example.com", "department": "Engineering"},
    {"employee_id": "EMP1036", "name": "Isha Mehta", "email": "isha.mehta@example.com", "department": "Product"},
    {"employee_id": "EMP1037", "name": "Jamal White", "email": "jamal.white@example.com", "department": "Support"},
    {"employee_id": "EMP1038", "name": "Katerina Ivanova", "email": "katerina.ivanova@example.com", "department": "Marketing"},
    {"employee_id": "EMP1039", "name": "Liam O'Connor", "email": "liam.oconnor@example.com", "department": "Finance"},
    {"employee_id": "EMP1040", "name": "Maya Fernandez", "email": "maya.fernandez@example.com", "department": "Design"},
    {"employee_id": "EMP1041", "name": "Noah Green", "email": "noah.green@example.com", "department": "Engineering"},
    {"employee_id": "EMP1042", "name": "Olivia Park", "email": "olivia.park@example.com", "department": "HR"},
    {"employee_id": "EMP1043", "name": "Peter Novak", "email": "peter.novak@example.com", "department": "Sales"},
    {"employee_id": "EMP1044", "name": "Qi Wang", "email": "qi.wang@example.com", "department": "Engineering"},
    {"employee_id": "EMP1045", "name": "Rohit Desai", "email": "rohit.desai@example.com", "department": "Operations"},
    {"employee_id": "EMP1046", "name": "Sara Lind", "email": "sara.lind@example.com", "department": "Product"},
    {"employee_id": "EMP1047", "name": "Tomás Silva", "email": "tomas.silva@example.com", "department": "Support"},
    {"employee_id": "EMP1048", "name": "Uma Nair", "email": "uma.nair@example.com", "department": "Finance"},
    {"employee_id": "EMP1049", "name": "Victor Hugo", "email": "victor.hugo@example.com", "department": "Marketing"},
    {"employee_id": "EMP1050", "name": "Wendy Zhao", "email": "wendy.zhao@example.com", "department": "Design"},
]

# 50 sample knowledge base articles
SAMPLE_KB = [
    {"id": 1, "title": "Reset VPN Password", "tags": ["vpn","password","reset"], "content": "To reset your VPN password: visit the VPN portal, enter your employee ID, and follow the emailed link to set a new password."},
    {"id": 2, "title": "Connect to Corporate WiFi", "tags": ["wifi","network"], "content": "Select 'CorpNet' SSID, use your employee credentials, and accept the certificate prompt."},
    {"id": 3, "title": "Laptop Won't Boot", "tags": ["laptop","boot","hardware"], "content": "Try power cycle: hold power 10s, remove battery (if removable), boot with AC only. If still failing, collect serial number and open a ticket."},
    {"id": 4, "title": "Email Not Syncing on Mobile", "tags": ["email","mobile","sync"], "content": "Ensure device has internet, check account settings, remove and re-add the corporate email account if necessary."},
    {"id": 5, "title": "Forgot Windows Password", "tags": ["password","windows","login"], "content": "Use the corporate password reset portal or contact IT to initiate a password reset with identity verification."},
    {"id": 6, "title": "Printer Not Responding", "tags": ["printer","printing","hardware"], "content": "Check printer power, network connection, and driver status. Restart the print spooler service if needed."},
    {"id": 7, "title": "VPN Connection Drops", "tags": ["vpn","disconnect"], "content": "Check network stability, try switching networks, and ensure VPN client is up to date."},
    {"id": 8, "title": "Slow Laptop Performance", "tags": ["performance","laptop"], "content": "Close unused apps, check Task Manager for high CPU/memory processes, and run disk cleanup."},
    {"id": 9, "title": "Two-Factor Authentication Setup", "tags": ["2fa","security"], "content": "Install the approved authenticator app and follow the corporate enrollment steps in the security portal."},
    {"id": 10, "title": "Access Denied to Shared Drive", "tags": ["permissions","files","share"], "content": "Request access from the file owner or submit a ticket with the folder path and required permission level."},
    {"id": 11, "title": "Browser Certificate Error", "tags": ["browser","certificate","ssl"], "content": "Verify system date/time, accept the corporate certificate if prompted, or contact IT for certificate installation."},
    {"id": 12, "title": "Install Corporate Software", "tags": ["install","software"], "content": "Use the company software center to request or install approved applications; admin approval may be required."},
    {"id": 13, "title": "Bluetooth Not Working", "tags": ["bluetooth","hardware"], "content": "Ensure Bluetooth is enabled, remove and re-pair devices, and update Bluetooth drivers."},
    {"id": 14, "title": "External Monitor Not Detected", "tags": ["display","monitor"], "content": "Check cables, try different ports, and update display drivers; use display settings to detect monitors."},
    {"id": 15, "title": "Cannot Print to Network Printer", "tags": ["printer","network"], "content": "Confirm printer is online, check network path, and reinstall the network printer using the correct driver."},
    {"id": 16, "title": "Email Attachment Too Large", "tags": ["email","attachment"], "content": "Use the corporate file share or cloud storage link instead of attaching large files to email."},
    {"id": 17, "title": "Password Policy Overview", "tags": ["password","policy","security"], "content": "Passwords must meet complexity rules and be changed every 90 days; use the password manager for secure storage."},
    {"id": 18, "title": "How to Request New Hardware", "tags": ["hardware","procurement"], "content": "Submit a hardware request ticket with justification, preferred model, and cost center information."},
    {"id": 19, "title": "Set Up Out of Office Reply", "tags": ["email","oof"], "content": "In the email client, go to automatic replies and set your out-of-office message and date range."},
    {"id": 20, "title": "Recover Deleted Files", "tags": ["files","recovery"], "content": "Check the recycle bin and version history in the file share; submit a recovery ticket if needed."},
    {"id": 21, "title": "Corporate VPN Split Tunneling", "tags": ["vpn","network"], "content": "Split tunneling is disabled by default; contact IT if you require exceptions for specific services."},
    {"id": 22, "title": "Set Up Email on iPhone", "tags": ["email","mobile","ios"], "content": "Use the corporate email profile or manual IMAP/Exchange settings provided in the mobile setup guide."},
    {"id": 23, "title": "Windows Update Fails", "tags": ["windows","update"], "content": "Run Windows Update troubleshooter, ensure disk space, and retry; escalate if errors persist."},
    {"id": 24, "title": "MacOS VPN Setup", "tags": ["vpn","macos"], "content": "Install the corporate VPN client for Mac and follow the configuration steps in the Mac setup guide."},
    {"id": 25, "title": "How to Share a Calendar", "tags": ["calendar","share"], "content": "Open your calendar, choose share settings, and grant view or edit permissions to colleagues."},
    {"id": 26, "title": "Reset MFA Device", "tags": ["mfa","security"], "content": "If you lose access to your MFA device, contact IT to reset your MFA enrollment after identity verification."},
    {"id": 27, "title": "Corporate Single Sign-On (SSO)", "tags": ["sso","login"], "content": "Use your corporate credentials at the SSO portal to access integrated applications without separate logins."},
    {"id": 28, "title": "How to Request Admin Rights", "tags": ["permissions","admin"], "content": "Provide business justification and manager approval; admin rights are granted temporarily and audited."},
    {"id": 29, "title": "Troubleshoot Audio Issues", "tags": ["audio","sound"], "content": "Check volume, default device, and drivers; test with headphones to isolate hardware vs software issues."},
    {"id": 30, "title": "Set Up Remote Desktop", "tags": ["remote","rdp"], "content": "Enable remote access on the host, ensure firewall rules allow RDP, and use the corporate remote access client."},
    {"id": 31, "title": "How to Clear Browser Cache", "tags": ["browser","cache"], "content": "Open browser settings and clear cached images and files; restart the browser after clearing."},
    {"id": 32, "title": "Corporate VPN Client Update", "tags": ["vpn","update"], "content": "Install the latest VPN client from the software center to resolve compatibility and security issues."},
    {"id": 33, "title": "How to Request Software License", "tags": ["license","software"], "content": "Submit a license request with justification and number of seats required; procurement will process the request."},
    {"id": 34, "title": "Mobile Hotspot Not Working", "tags": ["mobile","hotspot"], "content": "Verify mobile data is enabled, hotspot settings are correct, and device limits are not exceeded."},
    {"id": 35, "title": "How to Use the Password Manager", "tags": ["password","manager"], "content": "Install the corporate password manager extension and follow the onboarding steps to store credentials securely."},
    {"id": 36, "title": "File Sync Conflicts", "tags": ["files","sync"], "content": "Resolve conflicts by choosing the correct version and saving; contact IT if many conflicts occur."},
    {"id": 37, "title": "How to Request a New Email Alias", "tags": ["email","alias"], "content": "Provide the desired alias and business reason; IT will create the alias and map it to your mailbox."},
    {"id": 38, "title": "How to Connect to VPN from Home", "tags": ["vpn","remote"], "content": "Install the VPN client, ensure internet connectivity, and authenticate with your corporate credentials."},
    {"id": 39, "title": "How to Report a Security Incident", "tags": ["security","incident"], "content": "Immediately notify the security team via the incident reporting portal and follow the incident response guidance."},
    {"id": 40, "title": "How to Change Display Resolution", "tags": ["display","settings"], "content": "Open display settings and choose the recommended resolution; update drivers if options are missing."},
    {"id": 41, "title": "How to Request a Guest Account", "tags": ["access","guest"], "content": "Provide guest details and duration; guest accounts are time-limited and require sponsor approval."},
    {"id": 42, "title": "How to Use Company VPN Split Tunneling (Exceptions)", "tags": ["vpn","network"], "content": "Submit a request with the service details; exceptions are reviewed by network security."},
    {"id": 43, "title": "How to Configure Email Forwarding", "tags": ["email","forward"], "content": "Set up forwarding in email settings; ensure compliance with data handling policies before forwarding externally."},
    {"id": 44, "title": "How to Request a Software Update", "tags": ["software","update"], "content": "Open a ticket with the application name and version; IT will schedule updates based on priority."},
    {"id": 45, "title": "How to Connect a Docking Station", "tags": ["hardware","dock"], "content": "Connect power and data cables, attach monitors, and install any required docking drivers."},
    {"id": 46, "title": "How to Use Virtual Meetings Securely", "tags": ["meetings","security"], "content": "Use meeting passwords, enable waiting rooms, and avoid sharing sensitive information in public meetings."},
    {"id": 47, "title": "How to Request Data Access", "tags": ["data","access"], "content": "Provide dataset name, business justification, and manager approval; data access is provisioned with least privilege."},
    {"id": 48, "title": "How to Troubleshoot USB Devices", "tags": ["usb","hardware"], "content": "Try different ports, check device manager for errors, and update USB drivers."},
    {"id": 49, "title": "How to Configure Email Signatures", "tags": ["email","signature"], "content": "Use the corporate signature template and set it in email client settings for new messages and replies."},
    {"id": 50, "title": "How to Request a System Reimage", "tags": ["reimage","support"], "content": "Open a ticket with the reason and backup confirmation; IT will schedule a reimage and restore data as needed."},
]


# SQL to create tickets table (idempotent)
_TICKETS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT UNIQUE,
    employee_id TEXT,
    title TEXT,
    description TEXT,
    serial_number TEXT,
    priority TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""

# Sample ticket to insert if DB is newly created
_SAMPLE_TICKET = {
    "ticket_id": "TCKT-1000",
    "employee_id": "EMP1001",
    "title": "VPN not connecting",
    "description": "VPN fails with authentication error",
    "status": "open",
}


def get_db_path() -> pathlib.Path:
    """Return the Path to the SQLite DB file used by the app."""
    return DB_PATH


def _ensure_data_dir() -> None:
    """Create data directory if missing."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create data directory %s: %s", str(DATA_DIR), exc)


def _write_json_if_missing(path: pathlib.Path, data) -> None:
    """Write JSON to path only if the file does not already exist."""
    try:
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Created sample file %s", str(path))
    except Exception as exc:
        logger.exception("Failed to write JSON file %s: %s", str(path), exc)


def _ensure_db_schema(conn: sqlite3.Connection) -> None:
    """Create required tables if they do not exist."""
    cur = conn.cursor()
    cur.execute(_TICKETS_TABLE_SQL)
    conn.commit()


def _insert_sample_ticket_if_missing(conn: sqlite3.Connection) -> None:
    """Insert a sample ticket if a ticket with the sample ticket_id does not exist."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tickets WHERE ticket_id = ? LIMIT 1", (_SAMPLE_TICKET["ticket_id"],))
        if cur.fetchone():
            return
        cur.execute(
            """
            INSERT INTO tickets (ticket_id, employee_id, title, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (_SAMPLE_TICKET["ticket_id"], _SAMPLE_TICKET["employee_id"], _SAMPLE_TICKET["title"], _SAMPLE_TICKET["description"], _SAMPLE_TICKET["status"]),
        )
        conn.commit()
        logger.info("Inserted sample ticket %s", _SAMPLE_TICKET["ticket_id"])
    except Exception as exc:
        logger.exception("Failed to insert sample ticket: %s", exc)


def ensure_db(db_path: Optional[pathlib.Path] = None) -> None:
    """
    Ensure the data directory, sample JSON files, and SQLite DB (with schema) exist.

    This function is idempotent and safe to call at import/startup. It logs errors but does not raise.
    """
    path = db_path or DB_PATH
    try:
        _ensure_data_dir()

        # Ensure sample JSON files exist
        _write_json_if_missing(EMPLOYEES_FILE, SAMPLE_EMPLOYEES)
        _write_json_if_missing(KB_FILE, SAMPLE_KB)

        # Ensure DB and schema
        conn = sqlite3.connect(path)
        try:
            _ensure_db_schema(conn)
            _insert_sample_ticket_if_missing(conn)
            logger.info("Ensured database and schema at %s", str(path))
        finally:
            conn.close()
    except Exception as exc:
        # Log but do not raise; callers should handle DB errors gracefully.
        logger.exception("Failed to ensure DB at %s: %s", str(path), exc)


# Backwards-compatible alias for older code that used initialize_data
def initialize_data(data_dir: Optional[pathlib.Path] = None) -> None:
    """
    Backwards-compatible wrapper that mirrors the previous initialize_data behavior.
    If data_dir is provided, it will create files under that directory and ensure DB there.
    """
    if data_dir:
        # Use provided directory
        global DATA_DIR, EMPLOYEES_FILE, KB_FILE, DB_PATH
        DATA_DIR = pathlib.Path(data_dir)
        EMPLOYEES_FILE = DATA_DIR / "employees.json"
        KB_FILE = DATA_DIR / "knowledge.json"
        DB_PATH = DATA_DIR / "it_support.db"
    ensure_db(DB_PATH)


# Ensure DB on import (idempotent)
try:
    ensure_db()
except Exception:
    # ensure_db already logs exceptions; swallow any unexpected error to avoid import-time crashes
    logger.exception("Unexpected error while ensuring DB on import")


# If run directly, perform initialization and print status
if __name__ == "__main__":
    print("Running db_init self-check")
    ensure_db()
    print("Data directory:", str(DATA_DIR))
    print("Employees file exists:", EMPLOYEES_FILE.exists())
    print("Knowledge file exists:", KB_FILE.exists())
    print("DB file exists:", DB_PATH.exists())
