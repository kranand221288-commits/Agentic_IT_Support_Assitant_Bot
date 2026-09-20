# Agentic IT Support Assistant

Project title
Agentic IT Support Assistant

Problem statement
Employees need a lightweight, local IT support assistant that can understand natural language requests, decide which local tool to call, execute deterministic actions (search KB, lookup tickets, create tickets), maintain conversational state across turns, and return clear, auditable responses. The system must demonstrate agentic behavior (tool calling, routing, state, multi-step workflows) while preventing hallucinations and enforcing safety and validation rules.

Solution overview
This project implements a modular, local-first assistant that combines:

Deterministic local tools (JSON + SQLite) as the single source of truth for employees, tickets, and knowledge base.

An LLM as an advisor that proposes intents, extracts entities, and synthesizes user-facing text. The LLM only proposes actions; the agent validates and executes actions via local tools.

Workflow orchestration via LangGraph (optional). A fallback orchestrator demonstrates the same nodes/edges/state/conditional routing when LangGraph is not available.

Streamlit UI for a simple chat interface, conversation history, and tool/action trace visibility.

The assistant strictly enforces:

Only three tools may be used: Knowledge Search, Ticket Lookup, Ticket Creation.

Server-side validation before any side-effect.

Duplicate ticket prevention and clear provenance in responses.
--------------------------------------------------------------------------------------------------------------------------------------------
Architecture diagram
Code
User Query
    ↓
Intent Proposal (LLM)  ──> Intent Mapping & Validation (Agent)
    ↓
Conditional Routing
    ├── Knowledge Search Tool  ──> KB Results ──> LLM Synthesis (proposal) ──> Response Generation
    ├── Ticket Lookup Tool     ──> Ticket Results ──> Response Generation
    └── Ticket Creation Tool   ──> Validation ──> Create Ticket ──> Response Generation
             ↓
        Tool Result (authoritative)
             ↓
    Final Response (LLM formats; agent enforces provenance)
             ↓
        User sees response and tool trace

--------------------------------------------------------------------------------------------------------------------------------------------
Technology stack
Language: Python 3.9+

UI: Streamlit

Local storage: JSON files and SQLite database

LLM provider: OpenAI-compatible client (optional). LLM calls are centralized in llm_client.py.

Workflow orchestration: LangGraph (optional). Fallback orchestrator included.

Testing / packaging: standard Python tooling (pip, venv)
--------------------------------------------------------------------------------------------------------------------------------------------
Project structure
Code
agentic-it-support/
├─ run.py
├─ streamlit_app.py
├─ agent.py
├─ tools.py
├─ db_init.py
├─ llm_client.py
├─ langgraph_workflow.py
├─ prompts.py
├─ utils.py
├─ requirements.txt
├─ README.md
└─ data/
   ├─ employees.json
   ├─ knowledge.json
   └─ it_support.db

--------------------------------------------------------------------------------------------------------------------------------------------
Setup instructions
Clone or copy the repository into a local folder.

Create and activate a Python virtual environment:

bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows PowerShell
Install dependencies:

bash
pip install -r requirements.txt
(Optional) Configure LLM: set OPENAI_API_KEY if you want LLM features:

bash
export OPENAI_API_KEY="sk-..."
or on Windows PowerShell:

powershell
$env:OPENAI_API_KEY="sk-..."
Initialize data (the app will auto-initialize on first run). To force reinitialize, delete the data/ folder and restart the app.

Run the app:

bash
python run.py
or

bash
streamlit run streamlit_app.py
Environment variable requirements
OPENAI_API_KEY — optional; required only if you enable LLM features in llm_client.py.

LLM_MODEL — optional; override default model name (e.g., gpt-4o-mini) if desired.

No other environment variables are required for local-only operation.
--------------------------------------------------------------------------------------------------------------------------------------------
How to run the application
Start the app with python run.py or streamlit run streamlit_app.py.

Open the Streamlit UI in your browser (Streamlit prints the local URL).

Use the chat box to interact with the assistant. The sidebar shows controls and the agent state for auditing.
--------------------------------------------------------------------------------------------------------------------------------------------
Sample inputs
Knowledge question  
How do I reset my VPN password?

Ticket creation request  
My VPN is not working. Please raise a ticket. Employee ID EMP1024.

Ticket lookup  
What is the status of my laptop issue? EMP1001  
or
Check ticket TCKT-1000

Sample outputs
Knowledge search response

Code
KB Article: Reset VPN Password

To reset your VPN password: visit the VPN portal, enter your employee ID, and follow the emailed link to set a new password.

(From KB: Reset VPN Password)
Would you like me to open a ticket if this doesn't solve your problem?
Ticket lookup response

Code
Here are the matching tickets:
- TCKT-1000 | VPN not connecting | Status: open | Created: 2026-09-20T05:12:34
Ticket creation response

Code
Ticket created successfully: TCKT-1A2B3C4D
Title: VPN not working
Status: open
Created at: 2026-09-20T05:45:12

(This ticket ID is from the local ticket system.)
Key design decisions
Local authoritative data: Employees, KB, and tickets are stored locally (JSON + SQLite). The LLM is never the source of truth for these records.

LLM as advisor only: LLM proposals (intent, entities, summaries) are validated by the agent before any tool call. All side-effects are executed by deterministic tool wrappers.

Three-tool constraint: The agent exposes exactly three tools: Knowledge Search, Ticket Lookup, Ticket Creation. This keeps the scope focused and auditable.
--------------------------------------------------------------------------------------------------------------------------------------------

There are two kinds of “final output” to consider:

Authoritative, persistent outputs (tickets, DB records)

Saved incrementally to disk: ticket creation writes immediately to the local SQLite DB data/it_support.db via tools.ticket_create(...). That function inserts the new row and commits the transaction before returning the created ticket.

Authoritative source: the SQLite DB is the single source of truth for tickets; ticket IDs and statuses come from the DB, not the LLM.

Ephemeral UI outputs (chat history, tool traces, agent state)

Stored in memory: st.session_state.history, st.session_state.state, and st.session_state.last_tool_trace. These are not persisted to disk by default — they live only for the Streamlit session.

LLM-generated text (synthesized KB answers, final response formatting) is returned to the UI but not written to disk unless you add explicit persistence.

--------------------------------------------------------------------------------------------------------------------------------------------


Validation and safety:

Employee existence is validated before ticket creation.

Duplicate open-ticket detection prevents unnecessary duplicates.

The agent asks for missing required information (e.g., employee ID) rather than proceeding.

Tool failures are caught and surfaced in the UI with a tool_trace.

Workflow orchestration: LangGraph is used if available to demonstrate nodes, edges, conditional routing, and state. A fallback orchestrator mirrors the same behavior without external dependencies.

Provenance and transparency: Final responses clearly label retrieved facts (e.g., From KB, From Ticket System) and generated suggestions.

Modular architecture: Tools, LLM client, orchestrator, and UI are separated for testability and clarity.
--------------------------------------------------------------------------------------------------------------------------------------------
Limitations
LLM dependency: LLM features require network access and an API key. The app includes fallbacks but synthesis and advanced NLU will be limited without an LLM.

Simple KB retrieval: The knowledge search is keyword-based. For better RAG performance, integrate embeddings and a vector store.

LangGraph compatibility: LangGraph usage depends on the installed version and API; the code attempts to use it if present and falls back otherwise.

No authentication: This demo assumes a trusted local environment and does not implement user authentication or access controls.

Not production hardened: The project is intentionally realistic and runnable locally but lacks production features such as logging rotation, monitoring, rate limiting, and secure secret management.
--------------------------------------------------------------------------------------------------------------------------------------------