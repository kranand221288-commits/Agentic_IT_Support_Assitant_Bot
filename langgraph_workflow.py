"""
LangGraph workflow builder (optional). If LangGraph is installed, this module builds
a graph that demonstrates nodes, edges, conditional routing, state, and tool execution.
If LangGraph is not available, the agent will use a fallback orchestrator.
"""
from typing import Dict, Any

try:
    import langgraph as lg  # type: ignore
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

from llm_client import propose_intent_and_entities, synthesize_kb_answer, summarize_for_ticket, generate_final_response
from tools import knowledge_search, ticket_lookup, ticket_create, get_employee

def build_graph():
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph not installed")

    graph = lg.Graph()

    def node_intent(ctx):
        message = ctx["message"]
        proposal = propose_intent_and_entities(message)
        ctx["proposed_intent"] = proposal.get("intent")
        ctx["proposed_entities"] = proposal.get("entities", {})
        ctx["proposed_confidence"] = proposal.get("confidence", 0.0)
        return ctx

    n_intent = lg.Node("intent_extraction", node_intent)

    def node_validate(ctx):
        intent = ctx.get("proposed_intent", "knowledge_search")
        if intent not in ("knowledge_search", "ticket_lookup", "ticket_create"):
            intent = "knowledge_search"
        ctx["intent"] = intent
        state = ctx.get("state", {})
        state.update(ctx.get("proposed_entities", {}))
        ctx["state"] = state
        return ctx

    n_validate = lg.Node("intent_validation", node_validate)

    def node_kb(ctx):
        q = ctx["message"]
        results = knowledge_search(q)
        ctx["kb_results"] = results
        ctx["synthesized_kb"] = synthesize_kb_answer(q, results) if results else None
        return ctx

    n_kb = lg.Node("knowledge_search", node_kb)

    def node_lookup(ctx):
        state = ctx.get("state", {})
        emp = state.get("employee_id")
        ticket_id = state.get("ticket_id")
        results = ticket_lookup(employee_id=emp, ticket_id=ticket_id)
        ctx["ticket_lookup_results"] = results
        return ctx

    n_lookup = lg.Node("ticket_lookup", node_lookup)

    def node_create(ctx):
        state = ctx.get("state", {})
        emp = state.get("employee_id")
        if not emp:
            ctx["tool_error"] = "missing_employee_id"
            return ctx
        summary = summarize_for_ticket(ctx["message"])
        title = summary.get("title")
        description = summary.get("description")
        if not get_employee(emp):
            ctx["tool_error"] = "employee_not_found"
            return ctx
        result = ticket_create(employee_id=emp, title=title, description=description)
        ctx["ticket_create_result"] = result
        return ctx

    n_create = lg.Node("ticket_create", node_create)

    def node_response(ctx):
        tool_result = {}
        if ctx.get("intent") == "knowledge_search":
            tool_result = {"kb": ctx.get("kb_results", []), "synthesized": ctx.get("synthesized_kb")}
        elif ctx.get("intent") == "ticket_lookup":
            tool_result = {"tickets": ctx.get("ticket_lookup_results", [])}
        elif ctx.get("intent") == "ticket_create":
            tool_result = ctx.get("ticket_create_result", {})
        final = generate_final_response(tool_result, {"state": ctx.get("state", {}), "message": ctx.get("message")})
        ctx["final_response"] = final
        return ctx

    n_response = lg.Node("response_generation", node_response)

    graph.add_nodes([n_intent, n_validate, n_kb, n_lookup, n_create, n_response])
    graph.add_edge(n_intent, n_validate)
    graph.add_edge(n_validate, n_kb, condition=lambda ctx: ctx.get("intent") == "knowledge_search")
    graph.add_edge(n_validate, n_lookup, condition=lambda ctx: ctx.get("intent") == "ticket_lookup")
    graph.add_edge(n_validate, n_create, condition=lambda ctx: ctx.get("intent") == "ticket_create")
    graph.add_edge(n_kb, n_response)
    graph.add_edge(n_lookup, n_response)
    graph.add_edge(n_create, n_response)

    return graph
