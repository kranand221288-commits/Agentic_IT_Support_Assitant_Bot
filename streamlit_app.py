"""
Streamlit UI for the Agentic IT Support Assistant.
Replace your existing streamlit_app.py with this file.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

import streamlit as st

from agent import handle_user_message
from db_init import initialize_data

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
HISTORY_FILE = DATA_DIR / "conversation_history.jsonl"

# Ensure sample data exists
initialize_data(DATA_DIR)

st.set_page_config(page_title="Agentic IT Support Assistant [Anand_IT_Bot]", layout="wide")
st.caption("Anand_IT_Bot")                # small subtitle under the title


# --- Session state defaults ---
if "history" not in st.session_state:
    st.session_state.history = []
if "state" not in st.session_state:
    st.session_state.state = {}
if "last_tool_trace" not in st.session_state:
    st.session_state.last_tool_trace = []

# --- Persistence helper (optional) ---
def append_conversation_entry(entry: dict) -> None:
    """
    Append a timestamped conversation entry to data/conversation_history.jsonl.
    This is minimal and writes raw text; redact PII before calling if needed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry_with_ts = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")


def load_conversation_history(limit: int | None = None) -> list:
    """
    Load conversation history from conversation_history.jsonl into memory.
    Returns a list of entries (oldest first). If the file is missing, returns [].
    """
    entries = []
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        entries.append(obj)
                    except Exception:
                        # skip malformed lines
                        continue
                    if limit and len(entries) >= limit:
                        break
    except Exception:
        # Do not raise; return what we have
        pass
    return entries


def save_history_to_file(history: list) -> None:
    """
    Overwrite the conversation_history.jsonl file with the provided history list.
    Each item in history should be a dict; this writes one JSON object per line.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            for entry in history:
                # Ensure timestamp exists
                if "timestamp" not in entry:
                    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Do not block UI on save errors
        pass


# --- Load persisted history into session state on first run ---
if not st.session_state.history:
    persisted = load_conversation_history()
    # Convert persisted entries into the same shape used in session_state.history
    # persisted entries are expected to have keys: timestamp, user, assistant, state, tool_trace
    for entry in persisted:
        ts = entry.get("timestamp")
        user_text = entry.get("user") or entry.get("from_user") or ""
        assistant_text = entry.get("assistant") or entry.get("from_agent") or ""
        tool_trace = entry.get("tool_trace") or entry.get("trace") or []
        if user_text:
            st.session_state.history.append({"from": "user", "text": user_text, "timestamp": ts})
        if assistant_text:
            st.session_state.history.append({"from": "agent", "text": assistant_text, "tool_trace": tool_trace, "timestamp": ts})

# --- UI layout ---
st.title("Agentic IT Support Assistant")
st.markdown(
    "Local agent demonstrating tool calling, state, conditional workflows, and LLM-assisted proposals. "
    "LLM outputs are proposals only; all side-effects are validated and executed by local tools."
)

with st.sidebar:
    st.header("Controls")
    if st.button("Reset Conversation"):
        st.session_state.history = []
        st.session_state.state = {}
        st.session_state.last_tool_trace = []
        # Also clear persisted file
        try:
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
        except Exception:
            pass

    st.markdown("**Sample prompts**")
    st.markdown("- My VPN is not working. Please raise a ticket.")
    st.markdown("- How do I reset my VPN password?")
    st.markdown("- What is the status of my laptop issue? EMP1001")
    st.markdown("---")
    st.subheader("Data files")
    st.write(f"`{DATA_DIR}`")
    st.markdown("To reinitialize sample data, delete the `data` folder and restart the app.")
    st.subheader("Agent State")
    st.json(st.session_state.state)
    st.subheader("Last Tool Trace")
    st.json(st.session_state.last_tool_trace)

    st.markdown("---")
    st.subheader("Conversation Persistence")
    st.markdown("The app appends each turn to `conversation_history.jsonl`. You can export or overwrite the file below.")
    if st.button("Export history (download)"):
        # Prepare a downloadable JSONL blob
        try:
            blob = "\n".join(json.dumps(e, ensure_ascii=False) for e in load_conversation_history())
            st.download_button("Download conversation_history.jsonl", blob, file_name="conversation_history.jsonl", mime="application/json")
        except Exception:
            st.warning("Could not prepare export.")

    if st.button("Save current session to file (overwrite)"):
        # Convert session history into JSONL entries and save
        entries = []
        for turn in st.session_state.history:
            ts = turn.get("timestamp") or datetime.now(timezone.utc).isoformat()
            if turn.get("from") == "user":
                entries.append({"timestamp": ts, "user": turn.get("text")})
            else:
                entries.append({"timestamp": ts, "assistant": turn.get("text"), "tool_trace": turn.get("tool_trace", [])})
        save_history_to_file(entries)
        st.success("Saved current session to conversation_history.jsonl")

st.subheader("Chat")
col1, col2 = st.columns([3, 1])

# --- Callback for sending messages (avoids widget state errors) ---
def on_send():
    user_text = st.session_state.get("user_input", "")
    if not user_text or not user_text.strip():
        # Show a warning inside the callback so the user sees it
        st.warning("Please enter a message before sending.")
        return

    ts = datetime.now(timezone.utc).isoformat()

    # Append user turn
    st.session_state.history.append({"from": "user", "text": user_text.strip(), "timestamp": ts})

    try:
        response, new_state, tool_trace = handle_user_message(user_text.strip(), st.session_state.state)
    except Exception as e:
        response = f"Agent error: {str(e)}"
        new_state = st.session_state.state
        tool_trace = [{"tool": "agent_error", "error": str(e)}]

    # Update session state
    st.session_state.state = new_state
    st.session_state.last_tool_trace = tool_trace
    st.session_state.history.append({"from": "agent", "text": response, "tool_trace": tool_trace, "timestamp": ts})

    # Persist conversation incrementally (optional)
    try:
        append_conversation_entry({
            "user": user_text.strip(),
            "assistant": response,
            "state": new_state,
            "tool_trace": tool_trace
        })
    except Exception:
        # Do not block UI if persistence fails
        pass

    # Clear the text area (allowed inside callback)
    st.session_state.user_input = ""

    # No explicit st.experimental_rerun() call; Streamlit will rerun after the callback returns.

with col1:
    # Text area widget bound to session_state key "user_input"
    st.text_area("Your message", height=120, key="user_input")

with col2:
    # Wire Send button to callback
    st.button("Send", on_click=on_send)

st.markdown("---")

# Render conversation history (latest first)
# We render persisted timestamp if available and show tool trace for agent turns.
for turn in reversed(st.session_state.history):
    ts = turn.get("timestamp")
    if turn.get("from") == "agent":
        st.markdown("**Assistant:**")
        st.markdown(turn.get("text", ""))
        if ts:
            try:
                # show local time for readability
                parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                st.caption(f"At {parsed.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
            except Exception:
                st.caption(f"At {ts}")
        if "tool_trace" in turn and turn["tool_trace"]:
            with st.expander("Tool / Action Trace"):
                st.json(turn["tool_trace"])
    else:
        # user turn
        display_text = turn.get("text", "")
        if ts:
            try:
                parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                st.markdown(f"**You ({parsed.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}):** {display_text}")
            except Exception:
                st.markdown(f"**You:** {display_text}")
        else:
            st.markdown(f"**You:** {display_text}")

st.markdown("---")
st.markdown(
    "Notes: LLM proposals are visible in the tool trace. All ticket creation and lookups are performed by local tools and validated before execution."
)
