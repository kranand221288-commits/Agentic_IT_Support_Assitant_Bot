"""
llm_client.py

Unified LLM client with support for:
  - OpenAI (modern client openai>=1.0.0 or older openai package)
  - Google Gemini (google.genai preferred, fallback to google.generativeai)

Exposes:
  - invoke(messages, functions=None, function_call=None, model=None, ...)
  - propose_action_with_functions(user_message)
  - synthesize_kb_answer(query, kb_articles)
  - summarize_for_ticket(message)
  - generate_final_response(tool_result, context)
  - llm_to_points(text, max_steps=4)

Behavior:
  - Attempts real LLM calls when provider SDK and API key are available.
  - Falls back to deterministic local behavior when LLM is unavailable or errors occur.
  - Normalizes responses into a consistent shape for the agent to consume.
"""

from __future__ import annotations
import os
import json
import time
import re
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# Optional dotenv loader
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Basic config (env-first)
ROOT = pathlib.Path(__file__).parent.resolve()
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
# preferred .env names first, then fallbacks)
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_MY_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_KEY")
)

GOOGLE_API_KEY = (
    os.getenv("GEMINI_API_MY_KEY")
    or os.getenv("GOOGLE_API_MY_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_GENAI_KEY")
)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# --- Try to import OpenAI modern client or older package ---
_OPENAI_MODERN = False
_OPENAI_CLIENT = None
try:
    # modern client (openai>=1.0.0)
    from openai import OpenAI  # type: ignore

    _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
    _OPENAI_MODERN = True
except Exception:
    try:
        import openai as _old_openai  # type: ignore

        if OPENAI_API_KEY:
            _old_openai.api_key = OPENAI_API_KEY
        _OPENAI_CLIENT = _old_openai
        _OPENAI_MODERN = False
    except Exception:
        _OPENAI_CLIENT = None
        _OPENAI_MODERN = False

# --- Try to import Google GenAI (preferred) or google.generativeai (fallback) ---
_GEMINI_CLIENT = None
_GEMINI_MODERN = False
try:
    # google.genai is the newer package
    import google.genai as genai_mod  # type: ignore

    # configure if API key present (some versions use configure)
    try:
        if GOOGLE_API_KEY:
            genai_mod.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        pass
    _GEMINI_CLIENT = genai_mod
    _GEMINI_MODERN = True
except Exception:
    try:
        import google.generativeai as genai_legacy  # type: ignore

        try:
            if GOOGLE_API_KEY:
                genai_legacy.configure(api_key=GOOGLE_API_KEY)
        except Exception:
            pass
        _GEMINI_CLIENT = genai_legacy
        _GEMINI_MODERN = False
    except Exception:
        _GEMINI_CLIENT = None
        _GEMINI_MODERN = False


# --- Provider readiness helpers ---

def _ensure_provider_ready(provider: Optional[str] = None) -> None:
    """
    Ensure the requested provider client and API key are available.
    Call with provider='openai' or provider='gemini' or leave None to check the configured PROVIDER.
    Raises RuntimeError with a clear message if not ready.
    """
    prov = (provider or PROVIDER or "").lower()
    if prov in ("openai", ""):
        if _OPENAI_CLIENT is None:
            raise RuntimeError("OpenAI client not available. Install the 'openai' package and set OPENAI_API_KEY.")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
    if prov in ("gemini", "google", "google_gemini", ""):
        if _GEMINI_CLIENT is None:
            # only raise if provider explicitly requested
            if provider and provider.lower() in ("gemini", "google", "google_gemini"):
                raise RuntimeError("Gemini client not available. Install 'google.genai' or 'google-generative-ai' and set GOOGLE_API_KEY.")
        else:
            if not GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY not set in environment.")


def _is_provider_ready(provider: str) -> bool:
    if provider in ("openai", "openai_api"):
        return _OPENAI_CLIENT is not None and bool(OPENAI_API_KEY)
    if provider in ("gemini", "google", "google_gemini"):
        return _GEMINI_CLIENT is not None and bool(GOOGLE_API_KEY)
    return False


# --- Low-level unified chat wrappers ---


def _call_openai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Wrapper for OpenAI chat completions. Returns dict or {"error": "..."}.
    Works with modern OpenAI client and older openai package.
    """
    model = model or DEFAULT_MODEL
    temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)

    if not _is_provider_ready("openai"):
        return {"error": "OpenAI client not available or OPENAI_API_KEY not set."}

    try:
        if _OPENAI_MODERN:
            payload = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            if functions is not None:
                payload["functions"] = functions
                if function_call is not None:
                    payload["function_call"] = function_call
            resp = _OPENAI_CLIENT.chat.completions.create(**payload)
            try:
                return resp.to_dict()
            except Exception:
                return dict(resp)
        else:
            payload = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            if functions is not None:
                payload["functions"] = functions
                if function_call is not None:
                    payload["function_call"] = function_call
            resp = _OPENAI_CLIENT.ChatCompletion.create(**payload)
            try:
                return resp.to_dict()
            except Exception:
                return dict(resp)
    except Exception as e:
        return {"error": str(e)}


def _call_gemini_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Wrapper for Google Gemini chat. Returns a dict-like normalized response or {"error": "..."}.
    Supports both google.genai and google.generativeai shapes (best-effort).
    """
    model = model or DEFAULT_MODEL
    temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)

    if not _is_provider_ready("gemini"):
        return {"error": "Gemini client not available or GOOGLE_API_KEY not set."}

    try:
        if _GEMINI_MODERN:
            # Try modern genai.chat.completions.create
            try:
                resp = _GEMINI_CLIENT.chat.completions.create(model=model, messages=messages, temperature=temperature)
                try:
                    return dict(resp)
                except Exception:
                    return {"choices": [{"message": {"content": getattr(resp, "last", "")}}]}
            except Exception:
                # fallback to chat.create
                try:
                    resp = _GEMINI_CLIENT.chat.create(model=model, messages=messages, temperature=temperature, max_output_tokens=max_tokens)
                    try:
                        return dict(resp)
                    except Exception:
                        candidates = getattr(resp, "candidates", None)
                        if candidates:
                            return {"choices": [{"message": {"content": candidates[0].get("content", "")}}]}
                        return {"choices": [{"message": {"content": str(resp)}}]}
                except Exception as e:
                    return {"error": str(e)}
        else:
            # legacy google.generativeai
            try:
                resp = _GEMINI_CLIENT.chat.create(model=model, messages=messages, temperature=temperature, max_output_tokens=max_tokens)
                try:
                    return dict(resp)
                except Exception:
                    candidates = getattr(resp, "candidates", None)
                    if candidates:
                        return {"choices": [{"message": {"content": candidates[0].get("content", "")}}]}
                    return {"choices": [{"message": {"content": str(resp)}}]}
            except Exception as e:
                return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# --- Normalizers for provider responses ---


def _normalize_openai(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize OpenAI response to unified shape:
      {"raw": resp, "message": {"role","content"}, "function_call": {...}|None, "usage": {...}|None}
    """
    if not isinstance(resp, dict):
        return {"raw": resp, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None}
    if resp.get("error"):
        return {"raw": resp, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None, "error": resp.get("error")}
    choices = resp.get("choices", [])
    if not choices:
        return {"raw": resp, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": resp.get("usage")}
    choice0 = choices[0]
    msg = choice0.get("message") or {}
    content = msg.get("content") or choice0.get("text") or ""
    fc = msg.get("function_call") or choice0.get("function_call")
    function_call = None
    if fc:
        args = fc.get("arguments", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                try:
                    args = json.loads(args.replace("\n", " "))
                except Exception:
                    args = {"raw_arguments": args}
        function_call = {"name": fc.get("name"), "arguments": args}
    return {"raw": resp, "message": {"role": msg.get("role", "assistant"), "content": content}, "function_call": function_call, "usage": resp.get("usage")}


def _normalize_gemini(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Gemini response to unified shape.
    """
    if not isinstance(resp, dict):
        return {"raw": resp, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None}
    if resp.get("error"):
        return {"raw": resp, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None, "error": resp.get("error")}
    content = ""
    try:
        candidates = resp.get("candidates") or resp.get("choices") or []
        if candidates:
            first = candidates[0]
            content = first.get("content") or (first.get("message") or {}).get("content") or first.get("output") or ""
        else:
            content = resp.get("content") or resp.get("last") or ""
    except Exception:
        content = ""
    return {"raw": resp, "message": {"role": "assistant", "content": content}, "function_call": None, "usage": resp.get("usage")}


# --- Unified invoke function ---


def invoke(
    messages: List[Dict[str, str]],
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Optional[Any] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retries: int = 2,
    backoff: float = 0.5,
) -> Dict[str, Any]:
    """
    Unified entrypoint for LLM calls. Returns normalized dict:
      - raw
      - message: {role, content}
      - function_call: {name, arguments} or None
      - usage
    On failure returns an error-shaped dict rather than raising.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if PROVIDER in ("openai", "openai_api"):
                raw = _call_openai_chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, functions=functions, function_call=function_call)
                norm = _normalize_openai(raw)
                return norm
            elif PROVIDER in ("gemini", "google", "google_gemini"):
                raw = _call_gemini_chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
                norm = _normalize_gemini(raw)
                return norm
            else:
                # If provider not recognized, try OpenAI then Gemini as fallback
                if _OPENAI_CLIENT is not None:
                    raw = _call_openai_chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, functions=functions, function_call=function_call)
                    norm = _normalize_openai(raw)
                    return norm
                if _GEMINI_CLIENT is not None:
                    raw = _call_gemini_chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
                    norm = _normalize_gemini(raw)
                    return norm
                return {"raw": {}, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None, "error": "No LLM provider configured or available."}
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return {"raw": {}, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None, "error": str(last_exc)}
    return {"raw": {}, "message": {"role": "assistant", "content": ""}, "function_call": None, "usage": None, "error": "invoke failed"}


# --- Function-calling schema (OpenAI-style) ---
FUNCTIONS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "knowledge_search",
        "description": "Search the local knowledge base for relevant articles. Return top matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User question or search query"},
                "top_k": {"type": "integer", "description": "Number of top results to return", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ticket_lookup",
        "description": "Lookup tickets by employee_id or ticket_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "Employee ID like EMP1024"},
                "ticket_id": {"type": "string", "description": "Ticket ID like TCKT-1234"},
            },
            "required": [],
        },
    },
    {
        "name": "ticket_create",
        "description": "Create a new support ticket. Agent must validate before executing.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "Employee ID like EMP1024"},
                "title": {"type": "string", "description": "Short ticket title"},
                "description": {"type": "string", "description": "Detailed ticket description"},
                "serial_number": {"type": "string", "description": "Optional serial number"},
                "priority": {"type": "string", "description": "Optional priority: low/medium/high"},
            },
            "required": [],
        },
    },
]


# --- High-level helpers ---


def propose_action_with_functions(user_message: str) -> Dict[str, Any]:
    """
    Ask the LLM to propose an intent and optionally return a function_call.
    Returns a dict that may contain:
      - "function_call": {"name", "arguments"}
      - "intent_proposal": parsed JSON
      - "intent_proposal_text": raw text
      - "raw": normalized invoke result
      - "error": str
    """
    system = {
        "role": "system",
        "content": (
            "You are an assistant that decides which of the following tools to call: "
            "knowledge_search, ticket_lookup, ticket_create. "
            "If a tool should be called, return a function call using the provided function schema. "
            "If uncertain, prefer knowledge_search. Do not invent ticket IDs or authoritative facts."
        ),
    }
    user = {"role": "user", "content": user_message}
    try:
        resp = invoke(messages=[system, user], functions=FUNCTIONS_SCHEMA, function_call="auto", model=DEFAULT_MODEL)
        if resp.get("error"):
            return {"error": resp.get("error"), "raw": resp}
        result: Dict[str, Any] = {"raw": resp}
        fc = resp.get("function_call")
        if fc:
            result["function_call"] = fc
            return result
        content = resp.get("message", {}).get("content", "") or ""
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and ("intent" in parsed or "intent_proposal" in parsed):
                    result["intent_proposal"] = parsed.get("intent_proposal", parsed)
                else:
                    result["intent_proposal_text"] = content
            except Exception:
                result["intent_proposal_text"] = content
        return result
    except Exception as e:
        return {"error": str(e)}


def synthesize_kb_answer(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Synthesize a concise answer from KB articles. Prefer LLM if available, otherwise deterministic fallback.
    """
    if not results:
        return "I couldn't find any relevant knowledge base articles."
    system = {"role": "system", "content": "You are a helpful IT assistant. Use the provided KB articles and cite titles when referencing them. Do not invent facts."}
    kb_text = "\n\n".join([f"Title: {a.get('title')}\nContent: {a.get('content')}" for a in results[:5]])
    user = {"role": "user", "content": f"Question: {query}\n\nKB Articles:\n{kb_text}\n\nProvide a concise answer and next steps if unresolved."}
    try:
        resp = invoke(messages=[system, user], model=DEFAULT_MODEL)
        if resp.get("error"):
            return results[0].get("content", "")
        return resp.get("message", {}).get("content", "") or results[0].get("content", "")
    except Exception:
        return results[0].get("content", "")


def summarize_for_ticket(message: str) -> Dict[str, str]:
    """
    Produce a short title and description for ticket creation.
    Try LLM; fallback to deterministic extraction.
    """
    system = {"role": "system", "content": "You are a ticket summarizer. Return JSON: {\"title\":..., \"description\":...}."}
    user = {"role": "user", "content": message}
    try:
        resp = invoke(messages=[system, user], model=DEFAULT_MODEL)
        if resp.get("error"):
            raise RuntimeError(resp.get("error"))
        text = resp.get("message", {}).get("content", "") or ""
        try:
            parsed = json.loads(text)
            return {"title": parsed.get("title", message.strip().split(".")[0][:80]), "description": parsed.get("description", message)}
        except Exception:
            title = message.strip().split(".")[0][:80]
            return {"title": title or "IT issue reported", "description": message}
    except Exception:
        title = message.strip().split(".")[0][:80]
        return {"title": title or "IT issue reported", "description": message}


def generate_final_response(tool_result: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Use LLM to format a final user-facing message. Falls back to deterministic formatting.
    """
    system = {"role": "system", "content": "You are an assistant that composes final user messages. Use the provided tool_result as the single source of truth for facts."}
    user = {"role": "user", "content": f"Tool result (JSON): {json.dumps(tool_result)}\n\nContext: {json.dumps(context)}\n\nProduce a concise message to the user, labeling retrieved facts and any suggestions."}
    try:
        resp = invoke(messages=[system, user], model=DEFAULT_MODEL)
        if resp.get("error"):
            raise RuntimeError(resp.get("error"))
        return resp.get("message", {}).get("content", "") or ""
    except Exception:
        if "ticket" in tool_result:
            t = tool_result["ticket"]
            return f"Ticket created: {t.get('ticket_id', 'TICKET')} | {t.get('title', '')} | Status: {t.get('status', 'open')}"
        if "existing_ticket" in tool_result:
            e = tool_result["existing_ticket"]
            return f"Similar ticket exists: {e.get('ticket_id', 'TICKET')} | {e.get('title', '')} | Status: {e.get('status', '')}"
        if isinstance(tool_result, list):
            return "Found results:\n" + "\n".join([str(r) for r in tool_result])
        return str(tool_result)


def llm_to_points(text: str, max_steps: int = 4) -> str:
    """
    Reformat text into numbered steps. Try LLM; fallback to local deterministic formatter.
    """
    def _local_points(t: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', t.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return t
        parts = parts[:max_steps]
        return "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))

    if not (_OPENAI_CLIENT or _GEMINI_CLIENT):
        return _local_points(text)

    system = {"role": "system", "content": f"Reformat the following answer into {max_steps} concise numbered steps. Use plain sentences."}
    user = {"role": "user", "content": text}
    try:
        resp = invoke(messages=[system, user], model=DEFAULT_MODEL)
        if resp.get("error"):
            return _local_points(text)
        content = resp.get("message", {}).get("content", "")
        return content.strip() if content else _local_points(text)
    except Exception:
        return _local_points(text)


# --- Utility: safe JSON loader for arguments ---


def _safe_load_json(s: Any) -> Any:
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r'(\{.*\})', s, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return {}
    return {}


# --- Exports ---
__all__ = [
    "invoke",
    "propose_action_with_functions",
    "synthesize_kb_answer",
    "summarize_for_ticket",
    "generate_final_response",
    "llm_to_points",
    "_safe_load_json",
    "_ensure_provider_ready",
]

# --- Self-check when run directly ---
if __name__ == "__main__":
    print("llm_client self-check")
    print("Configured PROVIDER:", PROVIDER)
    print("OpenAI client present:", _OPENAI_CLIENT is not None, "modern:", _OPENAI_MODERN)
    print("Gemini client present:", _GEMINI_CLIENT is not None, "modern:", _GEMINI_MODERN)
    print("OPENAI_API_KEY set:", bool(OPENAI_API_KEY))
    print("GOOGLE_API_KEY set:", bool(GOOGLE_API_KEY))
