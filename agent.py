"""
Agent orchestration updated to use llm_client.propose_action_with_functions.

Public API:
  handle_user_message(message: str, state: dict) -> (response_str, new_state, tool_trace)
"""
from typing import Tuple, Dict, Any, List, Optional
import re
import traceback
import json

from tools import knowledge_search, ticket_lookup, ticket_create, get_employee
from llm_client import (
    propose_action_with_functions,
    synthesize_kb_answer,
    summarize_for_ticket,
    generate_final_response,
)

try:
    from langgraph_workflow import build_graph
    from langgraph_workflow import LANGGRAPH_AVAILABLE as LG_AVAILABLE
except Exception:
    LG_AVAILABLE = False

ALLOWED_INTENTS = {"knowledge_search", "ticket_lookup", "ticket_create"}


def _map_function_name_to_intent(fn_name: Optional[str]) -> str:
    if not fn_name:
        return "knowledge_search"
    name = fn_name.lower()
    if "knowledge" in name or "search" in name:
        return "knowledge_search"
    if "lookup" in name:
        return "ticket_lookup"
    if "create" in name or "ticket" in name:
        return "ticket_create"
    return "knowledge_search"


def _extract_employee_id(text: str) -> str:
    m = re.search(r"(EMP\d+)", (text or "").upper())
    return m.group(1) if m else ""


def _validate_employee(employee_id: str) -> bool:
    return bool(get_employee(employee_id))


def _safe_ticket_create(employee_id: str, title: str, description: str) -> Dict[str, Any]:
    """
    Wrapper around ticket_create that returns structured errors instead of raising.
    """
    if not employee_id:
        return {"error": "employee_not_provided"}
    if not _validate_employee(employee_id):
        return {"error": "employee_not_found", "employee_id": employee_id}
    if not title or len(title) < 3:
        return {"error": "title_too_short"}
    try:
        result = ticket_create(employee_id=employee_id, title=title, description=description)
        return result
    except Exception as e:
        return {"error": "tool_error", "message": str(e)}


def _interpret_llm_proposal(proposal: Dict[str, Any]) -> Tuple[str, Dict[str, Any], float]:
    """
    Normalize the propose_action_with_functions() output into (intent, entities, confidence).
    proposal may contain:
      - function_call: {"name":..., "arguments": {...}}
      - intent_proposal: parsed JSON object with keys intent/entities/confidence
      - intent_proposal_text: raw text
    """
    intent = "knowledge_search"
    entities: Dict[str, Any] = {}
    confidence = 0.0

    if not proposal or "error" in proposal:
        return intent, entities, confidence

    # Prefer explicit function_call
    fc = proposal.get("function_call")
    if fc:
        fn = fc.get("name")
        args = fc.get("arguments") or {}
        intent = _map_function_name_to_intent(fn)
        # Map common argument names to entities expected by the agent
        if intent == "knowledge_search":
            entities = {"query": args.get("query") or args.get("q") or ""}
            confidence = 0.9
        elif intent == "ticket_lookup":
            entities = {"employee_id": args.get("employee_id") or args.get("emp") or None,
                        "ticket_id": args.get("ticket_id") or args.get("ticket") or None}
            confidence = 0.9
        elif intent == "ticket_create":
            entities = {"employee_id": args.get("employee_id") or args.get("emp") or None,
                        "summary": args.get("title") or args.get("summary") or "",
                        "description": args.get("description") or args.get("body") or ""}
            confidence = 0.9
        return intent, entities, confidence

    # Next, check for structured intent_proposal
    ip = proposal.get("intent_proposal")
    if ip and isinstance(ip, dict):
        intent = ip.get("intent", intent)
        entities = ip.get("entities", entities) or {}
        try:
            confidence = float(ip.get("confidence", confidence))
        except Exception:
            confidence = confidence
        return intent, entities, confidence

    # Finally, try to parse free text proposal
    text = proposal.get("intent_proposal_text") or ""
    if text:
        try:
            parsed = json.loads(text)
            intent = parsed.get("intent", intent)
            entities = parsed.get("entities", entities) or {}
            try:
                confidence = float(parsed.get("confidence", confidence))
            except Exception:
                confidence = confidence
        except Exception:
            # fallback: leave defaults
            pass

    return intent, entities, confidence


def _fallback_orchestrator(message: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict]]:
    """
    Core fallback orchestrator with minimal, safe changes:
      - Treat LLM-proposed employee_id as a suggestion only.
      - Require employee id explicitly in the user's message or explicit confirmation before creating tickets.
      - Resume pending flows when the user provides or confirms an employee id.
    """
    tool_trace: List[Dict[str, Any]] = []
    new_state = dict(state or {})

    try:
        # --- Resume flows that are waiting for employee id or confirmation ---
        awaiting = new_state.get("awaiting_employee_id_for_ticket")
        confirming = new_state.get("confirming_employee_id")
        awaiting_confirmation_flag = new_state.get("awaiting_employee_confirmation", False)
        suggested_emp = new_state.get("suggested_employee_id")

        # If we are confirming an employee id (user was asked to confirm a different id)
        if confirming or (awaiting_confirmation_flag and suggested_emp):
            # Determine which id is being confirmed
            emp_to_confirm = confirming or suggested_emp
            txt = (message or "").strip().lower()
            if txt in ("yes", "y", "yeah", "yep", "confirm"):
                # proceed with the confirmed id
                emp = emp_to_confirm
                new_state.pop("confirming_employee_id", None)
                new_state.pop("awaiting_employee_confirmation", None)
                new_state.pop("suggested_employee_id", None)
                # retrieve pending payload if any
                pending = new_state.pop("awaiting_employee_id_for_ticket", None) or {}
                title = pending.get("title") or pending.get("summary") or "Support request"
                description = pending.get("description") or pending.get("body") or pending.get("message") or message
                # call create
                result = _safe_ticket_create(employee_id=emp, title=title, description=description)
                tool_trace.append({"tool": "ticket_create", "employee_id": emp, "title": title, "result": result})
                if result.get("error"):
                    if result["error"] == "employee_not_found":
                        return f"I couldn't find an employee with ID {emp}. Please provide a valid employee ID.", new_state, tool_trace
                    return f"An error occurred while creating the ticket: {result.get('message','unknown')}", new_state, tool_trace
                if result.get("duplicate"):
                    existing = result.get("existing_ticket")
                    new_state["suggested_existing_ticket"] = existing
                    final = generate_final_response({"existing_ticket": existing, "duplicate": True}, {"state": new_state, "message": message})
                    return final, new_state, tool_trace
                ticket = result.get("ticket")
                new_state["last_created_ticket"] = ticket
                new_state["employee_id"] = emp
                final = generate_final_response({"ticket": ticket}, {"state": new_state, "message": message})
                return final, new_state, tool_trace
            elif txt in ("no", "n", "nah"):
                new_state.pop("confirming_employee_id", None)
                new_state.pop("awaiting_employee_confirmation", None)
                new_state.pop("suggested_employee_id", None)
                return "Okay — please provide the correct employee ID (for example: EMP1055).", new_state, tool_trace
            else:
                # Not a clear yes/no; ask explicitly
                return f"You asked to use employee ID {emp_to_confirm}. Please reply 'yes' to confirm or provide the correct employee ID.", new_state, tool_trace

        # If we are awaiting an employee id for a pending ticket creation
        if awaiting:
            emp_from_msg = _extract_employee_id(message)
            if emp_from_msg:
                # resume creation with provided id
                emp = emp_from_msg
                pending = new_state.pop("awaiting_employee_id_for_ticket", None) or {}
                title = pending.get("title") or pending.get("summary") or "Support request"
                description = pending.get("description") or pending.get("body") or pending.get("message") or message
                # confirm if this differs from last known
                last_known = new_state.get("employee_id")
                if last_known and last_known != emp:
                    new_state["confirming_employee_id"] = emp
                    return f"I have {last_known} on file but you provided {emp}. Should I use {emp}? Reply 'yes' to proceed or 'no' to provide a different ID.", new_state, tool_trace
                # proceed to create
                result = _safe_ticket_create(employee_id=emp, title=title, description=description)
                tool_trace.append({"tool": "ticket_create", "employee_id": emp, "title": title, "result": result})
                if result.get("error"):
                    if result["error"] == "employee_not_found":
                        return f"I couldn't find an employee with ID {emp}. Please check and provide a valid employee ID.", new_state, tool_trace
                    return f"An error occurred while creating the ticket: {result.get('message','unknown')}", new_state, tool_trace
                if result.get("duplicate"):
                    existing = result.get("existing_ticket")
                    new_state["suggested_existing_ticket"] = existing
                    final = generate_final_response({"existing_ticket": existing, "duplicate": True}, {"state": new_state, "message": message})
                    return final, new_state, tool_trace
                ticket = result.get("ticket")
                new_state["last_created_ticket"] = ticket
                new_state["employee_id"] = emp
                final = generate_final_response({"ticket": ticket}, {"state": new_state, "message": message})
                return final, new_state, tool_trace
            else:
                # still waiting for an employee id
                return "I still need your employee ID to create the ticket. Please provide it (for example: EMP1055).", new_state, tool_trace

        # --- Normal LLM-driven proposal flow ---
        proposal = propose_action_with_functions(message)
        tool_trace.append({"step": "llm_proposal_raw", "proposal": proposal})

        intent, entities, confidence = _interpret_llm_proposal(proposal)
        # Normalize intent
        if intent not in ALLOWED_INTENTS:
            intent = "knowledge_search"

        new_state.setdefault("last_proposal", {})
        new_state["last_proposal"].update({"proposed_intent": intent, "entities": entities, "confidence": confidence})
        tool_trace.append({"step": "intent_mapping", "intent": intent, "confidence": confidence, "entities": entities})

        # Merge entities into state carefully:
        # - Do NOT automatically accept employee_id from LLM proposals into state.
        # - ticket_id can be merged safely.
        if entities.get("ticket_id"):
            new_state["ticket_id"] = entities.get("ticket_id")

        # Route to the appropriate tool
        if intent == "knowledge_search":
            # Use explicit query entity if provided, otherwise use the raw message
            query = entities.get("query") or message
            results = knowledge_search(query)
            tool_trace.append({"tool": "knowledge_search", "query": query, "results_count": len(results)})
            if results:
                synthesized = synthesize_kb_answer(query, results)
                new_state["last_kb_result"] = results[0]
                final = generate_final_response({"kb": results, "synthesized": synthesized}, {"state": new_state, "message": message})
            else:
                final = "I couldn't find a matching article in the knowledge base. Would you like me to create a ticket for this issue?"
            return final, new_state, tool_trace

        elif intent == "ticket_lookup":
            emp = new_state.get("employee_id") or entities.get("employee_id") or _extract_employee_id(message)
            ticket_id = new_state.get("ticket_id") or entities.get("ticket_id")
            if emp:
                new_state["employee_id"] = emp
            if not emp and not ticket_id:
                return "I need your employee ID or ticket ID to look up tickets. Please provide your employee ID (e.g., EMP1024).", new_state, tool_trace
            results = ticket_lookup(employee_id=emp, ticket_id=ticket_id)
            # ticket_lookup may return an error dict
            if isinstance(results, dict) and results.get("error"):
                tool_trace.append({"tool": "ticket_lookup", "employee_id": emp, "ticket_id": ticket_id, "error": results})
                return f"Unable to look up tickets: {results.get('message','unknown')}", new_state, tool_trace
            tool_trace.append({"tool": "ticket_lookup", "employee_id": emp, "ticket_id": ticket_id, "found": len(results)})
            if not results:
                final = "No tickets found for the provided information."
            else:
                final = generate_final_response({"tickets": results}, {"state": new_state, "message": message})
            return final, new_state, tool_trace

        elif intent == "ticket_create":
            # Determine employee id source:
            # - prefer an ID explicitly present in the user's current message (extracted)
            # - otherwise treat any employee_id coming from the LLM proposal as a suggestion
            #   and ask the user to confirm it before proceeding.
            emp_from_message = _extract_employee_id(message) or None
            emp_from_entities = entities.get("employee_id") or None

            # If user actually typed an ID in this message, accept it immediately
            if emp_from_message:
                emp = emp_from_message
            else:
                # If the LLM proposed an employee_id but the user did not type it,
                # do NOT accept it silently — ask the user to confirm.
                if emp_from_entities:
                    # Save the suggested id in state and ask for confirmation
                    new_state["suggested_employee_id"] = emp_from_entities
                    new_state["awaiting_employee_confirmation"] = True
                    return (
                        f"I detected employee ID {emp_from_entities} from the assistant's suggestion. "
                        "Do you want me to use that ID to create the ticket? Reply 'yes' to confirm or provide the correct employee ID.",
                        new_state,
                        tool_trace,
                    )
                # No id anywhere; fall back to last known in state
                emp = new_state.get("employee_id") or None

            # If we have an emp now, but it differs from last known, ask to confirm
            if emp:
                last_known = new_state.get("employee_id")
                if last_known and last_known != emp:
                    new_state["confirming_employee_id"] = emp
                    return f"I have {last_known} on file but you mentioned {emp}. Should I use {emp}? Reply 'yes' to proceed or provide the correct ID.", new_state, tool_trace
                new_state["employee_id"] = emp

            # If still no emp, ask and store pending payload
            if not emp:
                pending_payload = {}
                if entities.get("summary") or entities.get("description"):
                    pending_payload["title"] = entities.get("summary") or ""
                    pending_payload["description"] = entities.get("description") or ""
                else:
                    pending_payload = summarize_for_ticket(message)
                    pending_payload.setdefault("description", message)
                new_state["awaiting_employee_id_for_ticket"] = pending_payload
                return "To create a ticket I need your employee ID. Please provide it (e.g., EMP1024).", new_state, tool_trace

            # Prefer LLM-provided summary/description if present, otherwise summarize from message
            summary = {}
            if entities.get("summary") or entities.get("description"):
                summary["title"] = entities.get("summary") or (message.strip().split(".")[0][:80])
                summary["description"] = entities.get("description") or message
            else:
                summary = summarize_for_ticket(message)
            title = summary.get("title")
            description = summary.get("description")
            tool_trace.append({"step": "llm_ticket_summary", "summary": summary})

            # Call safe create wrapper
            result = _safe_ticket_create(employee_id=emp, title=title, description=description)
            tool_trace.append({"tool": "ticket_create", "employee_id": emp, "title": title, "result": result})

            if result.get("error"):
                if result["error"] == "employee_not_found":
                    # Ask user to provide a valid employee id
                    # Save pending payload so user can correct id and resume
                    new_state["awaiting_employee_id_for_ticket"] = {"title": title, "description": description}
                    return f"I couldn't find an employee with ID {emp}. Please check and provide a valid employee ID.", new_state, tool_trace
                if result["error"] == "employee_not_provided":
                    new_state["awaiting_employee_id_for_ticket"] = {"title": title, "description": description}
                    return "I need your employee ID to create the ticket. Please provide it (e.g., EMP1055).", new_state, tool_trace
                return f"An error occurred while creating the ticket: {result.get('message','unknown')}", new_state, tool_trace

            if result.get("duplicate"):
                existing = result.get("existing_ticket")
                new_state["suggested_existing_ticket"] = existing
                final = generate_final_response({"existing_ticket": existing, "duplicate": True}, {"state": new_state, "message": message})
                return final, new_state, tool_trace

            ticket = result.get("ticket")
            new_state["last_created_ticket"] = ticket
            new_state["employee_id"] = emp
            final = generate_final_response({"ticket": ticket}, {"state": new_state, "message": message})
            return final, new_state, tool_trace

        else:
            return "Sorry, I couldn't determine how to help with that. Could you rephrase?", new_state, tool_trace

    except Exception as e:
        tb = traceback.format_exc()
        tool_trace.append({"tool": "error", "error": str(e), "trace": tb})
        return f"An error occurred while processing your request: {str(e)}", new_state, tool_trace


def handle_user_message(message: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Public entrypoint used by the Streamlit UI.
    If LangGraph is available and build_graph works, it will be used; otherwise fallback orchestrator runs.
    """
    if LG_AVAILABLE:
        try:
            graph = build_graph()
            ctx = {"message": message, "state": state}
            result_ctx = graph.run(ctx)
            final = result_ctx.get("final_response", "Sorry, I couldn't produce a response.")
            new_state = result_ctx.get("state", state)
            tool_trace: List[Dict[str, Any]] = []
            if "proposed_intent" in result_ctx:
                tool_trace.append({"llm_proposal": {"intent": result_ctx.get("proposed_intent"), "entities": result_ctx.get("proposed_entities")}})
            if "kb_results" in result_ctx:
                tool_trace.append({"tool": "knowledge_search", "results_count": len(result_ctx.get("kb_results", []))})
            if "ticket_lookup_results" in result_ctx:
                tool_trace.append({"tool": "ticket_lookup", "found": len(result_ctx.get("ticket_lookup_results", []))})
            if "ticket_create_result" in result_ctx:
                tool_trace.append({"tool": "ticket_create", "result": result_ctx.get("ticket_create_result")})
            return final, new_state, tool_trace
        except Exception:
            return _fallback_orchestrator(message, state)
    else:
        return _fallback_orchestrator(message, state)
