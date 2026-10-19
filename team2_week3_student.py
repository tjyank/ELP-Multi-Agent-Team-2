import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tavily import TavilyClient


# ---------------- Configuration ----------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ---------------- Team 2 Supervisor ----------------
# This carries forward Week 2 homework: each assignment includes output_fields.
TEAM_2_SYSTEM_PROMPT = """
You are the Supervisor for a multi-agent competitive-intelligence research system.

Divide the user's business research question across exactly three specialist roles:
Competitor, Market, and Tech/Regulatory.

Each specialist must receive a distinct, non-overlapping objective. Include a short
output_fields list describing what that specialist should return.

Return only valid JSON in exactly this shape:
{
  "tasks": [
    {
      "agent": "Competitor",
      "objective": "...",
      "output_fields": ["...", "..."]
    },
    {
      "agent": "Market",
      "objective": "...",
      "output_fields": ["...", "..."]
    },
    {
      "agent": "Tech/Regulatory",
      "objective": "...",
      "output_fields": ["...", "..."]
    }
  ]
}

Use exactly these three agent names.
Do not include markdown or text outside the JSON object.
""".strip()

SPECIALIST_PROMPTS = {
    "Competitor": """
You are the Competitor specialist. Research competitors, products, positioning,
partnerships, and differentiators relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Market or Tech/Regulatory agent's job.
""".strip(),
    "Market": """
You are the Market specialist. Research market direction, customers, adoption,
industry dynamics, and business context relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Competitor or Tech/Regulatory agent's job.
""".strip(),
    "Tech/Regulatory": """
You are the Tech/Regulatory specialist. Research technology developments,
technical capabilities, emerging technology, regulation, and official policy
context relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Competitor or Market agent's job.
""".strip(),
}

class ResearchState(TypedDict):
    question: str
    assignments: list[dict]
    specialist_results: list[dict]


def call_model_json(system_prompt: str, user_text: str) -> dict:
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def supervisor_node(state: ResearchState) -> dict:
    output = call_model_json(TEAM_2_SYSTEM_PROMPT, state["question"])
    tasks = output.get("tasks")

    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Supervisor must return exactly three assignments.")

    expected_agents = {"Competitor", "Market", "Tech/Regulatory"}
    seen_agents = set()
    cleaned = []

    for assignment in tasks:
        if not isinstance(assignment, dict):
            raise ValueError("Each assignment must be a JSON object.")

        agent = str(assignment.get("agent", "")).strip()
        objective = str(assignment.get("objective", "")).strip()
        output_fields = assignment.get("output_fields", [])

        if agent not in expected_agents:
            raise ValueError(f"Unexpected specialist role: {agent}")
        if not objective:
            raise ValueError(f"{agent} needs a non-empty objective.")
        if not isinstance(output_fields, list):
            raise ValueError(f"{agent} output_fields must be a list.")

        seen_agents.add(agent)
        cleaned.append(
            {
                "agent": agent,
                "objective": objective,
                "output_fields": [str(field) for field in output_fields],
            }
        )

    if seen_agents != expected_agents:
        raise ValueError(
            "Assignments must include Competitor, Market, and Tech/Regulatory exactly once."
        )

    return {"assignments": cleaned}


# ---------------- Week 3 specialist execution ----------------
def research_assignment(assignment: dict, original_question: str) -> dict:
    """Execute ONE specialist's assignment."""
    # TODO 1:
    # 1. Read:
    #       agent = assignment["agent"]
    #       objective = assignment["objective"]
    #       output_fields = assignment["output_fields"]
    #
    # 2. Search Tavily using the specialist's objective.
    #
    # 3. Format the returned Tavily results into readable source text containing
    #    title, URL, and content.
    #
    # 4. Call DeepSeek with SPECIALIST_PROMPTS[agent].
    #    Give the model:
    #       - the original question
    #       - this specialist's objective
    #       - requested output_fields
    #       - Tavily source text
    #
    # 5. Return ONE dictionary in exactly this shape:
    #
    #    {
    #        "agent": agent,
    #        "objective": objective,
    #        "output_fields": output_fields,
    #        "research": model_response_text,
    #        "sources": tavily_results,
    #    }
    #
    # This function should execute only ONE specialist. Do not loop over all
    # assignments here.

    agent = assignment["agent"]
    objective = assignment["objective"]
    output_fields = assignment["output_fields"]

    if not objective:
        raise ValueError("The first research assignment needs an 'objective'.")

    response = tavily.search(
        query=objective,
        search_depth="advanced",
        max_results=5,
    )

    results = output_fields.get("results", [])

    if not results:
        raise ValueError("Tavily did not return any search results.")

    source_text = "\n\n".join(
        f"Title: {result.get('title', 'Untitled')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Content: {result.get('content', '')}"
        for result in results
    )

    prompt = f"""You are a research assistant. This is one research assignment
    from a larger question. Answer only this assignment, using only the search
    results below. Keep the response concise and cite source URLs next to the
    claims they support.

    Overall question:
    {original_question}

    Research assignment to answer:
    {objective}

    Requested output fields:
    {output_fields}

    Search results:
    {source_text}
    """

    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": SPECIALIST_PROMPTS[agent]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3
    )

    response = response.choices[0].message.content

    return {
        "agent": agent,
        "objective": objective,
        "output_fields": output_fields,
        "research": response,
        "sources": results,
    }

def specialists_node(state: ResearchState) -> dict:
    """Run all three independent specialist assignments in parallel."""

    # TODO 2:
    # Use ThreadPoolExecutor(max_workers=3).
    #
    # Submit research_assignment(assignment, state["question"]) once for each
    # assignment in state["assignments"].
    #
    # Use as_completed(...) to collect the finished results and append each one
    # to specialist_results.
    #
    # Important:
    # - The specialists should execute independently.
    # - Do not call them one-by-one in a normal sequential for-loop.
    # - Week 4 will add the shared evidence store; do not build it here.
    
    specialist_results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_assignment = {
            executor.submit(research_assignment, assignment, state["question"]): assignment 
            for assignment in state["assignments"]
        }

        for future in as_completed(future_to_assignment):
            result = future.result()
            specialist_results.append(result)


    return {"specialist_results": specialist_results}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", END)

    return graph.compile()


research_graph = build_graph()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ELP Team 2 — Week 3", page_icon="🧩")
st.title("Team 2 — Agentic Research Team")
st.caption("Supervisor → Specialist Agents → Parallel Research")

question = st.text_input(
    "Enter a business research question",
    value="How is NVIDIA positioning itself against AMD in the AI chip market?",
)

if st.button("Research", type="primary") and question.strip():
    initial_state: ResearchState = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
    }

    try:
        with st.spinner("Supervising and running specialist research..."):
            result = research_graph.invoke(initial_state)
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
    else:
        st.subheader("Supervisor assignments")
        st.json(result["assignments"])

        st.subheader("Independent specialist results")
        for item in sorted(
            result["specialist_results"],
            key=lambda x: x.get("agent", ""),
        ):
            with st.expander(item.get("agent", "Specialist"), expanded=True):
                st.write(f"**Objective:** {item.get('objective', '')}")
                st.write(item.get("research", ""))
                st.caption(f"{len(item.get('sources', []))} Tavily sources returned")

# ---------------- Week 4: Shared Evidence Store ----------------

import sqlite3

EVIDENCE_DB = "team2_evidence.db"


def init_evidence_db():
    """Create the Week 4 shared evidence table if it does not exist."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                publication_date TEXT,
                source_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence TEXT,
                related_company TEXT,
                report_section TEXT,
                UNIQUE(claim, url)
            )
        """)
        conn.commit()


def save_evidence_records(records: list[dict]):
    """Save validated specialist evidence to SQLite."""
    
    with sqlite3.connect(EVIDENCE_DB) as conn:
        for record in records:
            conn.execute(
                "INSERT OR IGNORE INTO evidence (claim, source, url, publication_date," \
                "source_type, evidence, confidence, related_company, report_section) VALUES (" \
                "?, ?, ?, ?, ?, ?, ?, ?, ?)", (record.get("claim"), record.get("source"), record.get("url"),
                record.get("publication_date"), record.get("source_type"), record.get("evidence"),
                record.get("confidence"), record.get("related_company"), record.get("report_section"))
            )
        conn.commit()

def load_evidence_records() -> list[dict]:
    """Read the shared evidence store back from SQLite."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()

class Week4ResearchState(ResearchState):
    evidence_records: list[dict]

EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a business research workflow.

Convert useful specialist research sources into structured Evidence Records.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "web",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Rules:
- claim = a factual statement directly supported by the source.
- evidence = the passage that supports the claim.
- source and url must come from the supplied Tavily result.
- Do not guess a publication date.
- confidence must be high, medium, or low.
- Do not turn analysis or recommendations into facts.
""".strip()


def extract_specialist_evidence(specialist_results: list[dict]) -> list[dict]:
    """Convert specialist source results into the shared Evidence Record format."""
    records = []

    # TODO 2 — EVIDENCE EXTRACTION
    #
    # For each specialist in specialist_results:
    #   1. Read agent and objective.
    #   2. Loop through specialist["sources"].
    #   3. Build source_payload with:
    #        agent, objective, title, url, content, published_date
    #   4. Call call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt).
    #   5. Add every returned record to records.
    #
    # Every specialist must produce the SAME Evidence Record structure.

    for specialist in specialist_results:
        for source in specialist["sources"]:
            agent = specialist["agent"]
            objective = specialist["objective"]
            title = source["title"]
            url = source["url"]
            content = source["content"]
            published_date = source["published_date"]

            prompt = f"agent: {agent}, objective: {objective}, title: {title}, url: {url}, content: {content}, published_date: {published_date}"

            result = call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt)

            records.extend(result["records"])

    return records


def validate_shared_evidence(records: list[dict]) -> list[dict]:
    """Validate records and remove exact duplicate claim/source pairs."""
    required = ["claim", "source", "url", "publication_date", "source_type", "evidence"]

    validated = []

    # TODO 3 — VALIDATION + DUPLICATES
    #
    # Keep only dictionaries with non-empty:
    #   claim, source, url, source_type, evidence
    #
    # Remove exact duplicates using:
    #   (claim.lower(), url)
    #
    # Hint: use a set called seen.
    
    seen = set()

    for record in records:
        if all(record.get(field) for field in required):
            identifier = (record["claim"].lower(), record["url"])

            if identifier not in seen:
                validated.append(record)
                seen.add(identifier)

    return validated


def evidence_store_node(state: ResearchState) -> dict:
    """Week 4: build and persist the shared evidence store."""
    # TODO 4 — CONNECT SPECIALISTS TO SHARED STORE + DATABASE
    #
    # 1. Extract records from state["specialist_results"].
    # 2. Validate them.
    # 3. Save validated records to SQLite.
    # 4. Return {"evidence_records": ...}.


    raw_records = extract_specialist_evidence(state["specialist_results"])
    validated_records = validate_shared_evidence(raw_records)

    with sqlite3.connect(EVIDENCE_DB) as conn:
        for record in validated_records:
            conn.execute(
                "INSERT OR IGNORE INTO evidence (claim, source, url, publication_date," \
                "source_type, evidence, confidence, related_company, report_section) VALUES (" \
                "?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                
                (record.get("claim"), record.get("source"), record.get("url"),
                record.get("publication_date"), record.get("source_type"), record.get("evidence"),
                record.get("confidence"), record.get("related_company"), record.get("report_section"))
            )
        conn.commit()

    return {"evidence_records": validated_records}


def build_week4_graph():
    """
    Extend the existing Week 3 Team 2 flow:

    Question -> Supervisor -> Specialist Agents -> Evidence Records -> SQLite Store
    """
    graph = StateGraph(Week4ResearchState)

    # TODO 5 — GRAPH CONNECTION
    #
    # Reuse:
    #   supervisor_node
    #   specialists_node
    #
    # Add:
    #   evidence_store_node
    #
    # Connect:
    #   START -> supervise -> specialists -> evidence_store -> END
    
    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", END)

    return graph.compile()

week4_graph = build_week4_graph()


# ---------------- Week 4 UI ----------------

st.divider()
st.subheader("Week 4 — Shared Evidence Store")
st.caption("Specialist research → Evidence Records → SQLite Shared Store")

if st.button("Build Shared Evidence Store", type="primary") and question.strip():
    week4_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
    }

    try:
        with st.spinner("Running specialists and building the shared evidence store..."):
            evidence_result = week4_graph.invoke(week4_state)
    except Exception as error:
        st.error(f"The Week 4 evidence workflow could not finish: {error}")
    else:
        st.subheader("New Evidence Records")
        st.json(evidence_result["evidence_records"])

        stored_records = load_evidence_records()
        st.subheader("Shared Evidence Store (SQLite)")
        st.dataframe(stored_records, use_container_width=True)
        st.caption(f"{len(stored_records)} evidence records stored in {EVIDENCE_DB}")


"""
WEEK 5 TARGET: multi-agent Supervisor → Specialists → Shared Evidence → Synthesis → Report
Deliverable: the first complete DRAFT report; automated citation verification and
quality review are deliberately reserved for Week 6.
"""

# UTC timestamp is used when each report is stored in SQLite.
from datetime import datetime, timezone

# ---------------- Week 5: Save reports in SQLite ----------------

# WHAT THIS DOES — DATABASE SETUP: Creates reports table if missing; preserves all Week 4 evidence rows.
def init_report_db():
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                report_markdown TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


# WHAT THIS DOES — DATABASE WRITE: Saves question, generated markdown, evidence count, timestamp; returns report ID.
def save_report(question: str, report_markdown: str, evidence_count: int) -> int:
    """Save the synthesized report and return its database id."""
    # STUDENT: Replace the `pass` in this function with real code.
    # TODO 0 — REPORT PERSISTENCE
    # Insert question, report_markdown, evidence_count, and a current UTC timestamp
    # into reports using parameterized SQL. Return cursor.lastrowid.

    current_time = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(EVIDENCE_DB) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO reports (question, report_markdown, evidence_count, created_at)" \
            "VALUES (?, ?, ?, ?)",

            (question, report_markdown, evidence_count, current_time)
        )
        conn.commit()

    return cursor.lastrowid if cursor.lastrowid is not None else 0

# WHAT THIS DOES — DATABASE READ: Loads previously generated reports, most recent first, for the UI.
def load_reports() -> list[dict]:
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


init_report_db()  # Ensure reports table exists before any report is saved.


# SHARED OUTPUT CONTRACT: Both teams use the SAME nine model-written sections;
# Sources / References is appended in Python as section number ten.
REPORT_SECTION_KEYS = [
    ("executive_summary", "Executive Summary"),
    ("key_findings", "Key Findings"),
    ("competitor_comparison", "Competitor Comparison"),
    ("market_industry_trends", "Market / Industry Trends"),
    ("emerging_technologies", "Emerging Technologies"),
    ("regulatory_developments", "Regulatory Developments"),
    ("risks_opportunities", "Risks / Opportunities"),
    ("recommended_next_steps", "Recommended Next Steps"),
    ("limitations_information_gaps", "Limitations / Information Gaps"),
]
# MODEL INSTRUCTIONS: The model must work from evidence only and return structured JSON.
# Week 6, not this file, handles independent citation validation and report review.
SYNTHESIS_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT in a multi-agent business research system.

Use ONLY the supplied Shared Evidence Store to create ONE coherent report.
Return ONLY valid JSON with exactly these keys:
{
  "executive_summary": "...",
  "key_findings": "...",
  "competitor_comparison": "...",
  "market_industry_trends": "...",
  "emerging_technologies": "...",
  "regulatory_developments": "...",
  "risks_opportunities": "...",
  "recommended_next_steps": "...",
  "limitations_information_gaps": "..."
}

Rules:
- Cite factual statements using evidence labels such as [E1].
- Reorganize findings by business topic, not agent name.
- The report should read as if one coordinated team produced it.
- Do not invent facts, dates, metrics, source names, URLs, or citations.
- Clearly distinguish facts, analysis, recommendations, and uncertainty.
- If evidence is insufficient, say so.
- Do NOT add the Reviewer/Critic today. That is Week 6.
""".strip()


# WHAT THIS DOES — EVIDENCE PREP: Assigns [E1], [E2] labels to this run’s evidence, keeps sources and excerpts.
def build_evidence_packet(evidence_records: list[dict]) -> list[dict]:
    packet = []
    for index, record in enumerate(evidence_records, start=1):
        packet.append({
            "evidence_id": f"E{index}",
            "claim": record.get("claim"),
            "source": record.get("source"),
            "url": record.get("url"),
            "publication_date": record.get("publication_date"),
            "source_type": record.get("source_type"),
            "evidence": record.get("evidence"),
            "confidence": record.get("confidence"),
            "related_company": record.get("related_company"),
            "report_section": record.get("report_section"),
        })
    return packet


# WHAT THIS DOES — TEAM 2 SYNTHESIS AGENT: Combines all specialist evidence into ONE unified report.
def synthesize_report(question: str, evidence_records: list[dict]) -> dict:
    """Merge all specialist evidence into the common report structure."""
    # STUDENT: Replace the `pass` below with the report-generation implementation.
    # TODO 1 — SYNTHESIS AGENT
    # 1. Build the numbered evidence packet.
    # 2. Put the research question + shared evidence into the prompt.
    # 3. Call call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt).
    # 4. Make sure all REPORT_SECTION_KEYS exist.
    # 5. Return the report dict.
    
    evidence_packet = build_evidence_packet(evidence_records)
    evidence_packet = json.dumps(evidence_packet, indent=2)

    prompt = f"question: {question}, evidence_packet: {evidence_packet}"

    report = call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)

    for key, title in REPORT_SECTION_KEYS:
        if key not in report or not report[key]:
            report[key] = "Information not available in evidence store."

    return report

# WHAT THIS DOES — REFERENCES: Creates citation legend from original URLs, not invented model references.
def build_references(evidence_records: list[dict]) -> str:
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        lines.append(f"- [E{index}] {source} — {date} — {url}")
    return "\n".join(lines) if lines else "No source records were available."


# WHAT THIS DOES — REPORT FORMAT: Assembles nine model sections and the tenth Sources / References section.
def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)

class Week5ResearchState(Week4ResearchState):
    report: dict
    report_markdown: str
    report_id: int

# WHAT THIS DOES — TEAM 2 GRAPH NODE: Runs synthesis, formats it, saves to SQLite.
def synthesis_node(state: Week5ResearchState):
    """Synthesize only after all shared evidence has been produced."""
    # TODO 3 — CONNECT SHARED EVIDENCE TO SYNTHESIS
    # 1. Read state["evidence_records"].
    # 2. Stop with a helpful error if empty.
    # 3. Call synthesize_report(...).
    # 4. Render markdown.
    # 5. Save the report to SQLite.
    # 6. Return report, report_markdown, and report_id.
    
    evidence_records = state["evidence_records"]

    if not evidence_records:
        raise ValueError("Evidence Records are Empty.")

    report = synthesize_report(state["question"], evidence_records)
    report_markdown = render_report_markdown(report, evidence_records)
    report_id = save_report(state["question"], report_markdown, len(evidence_records))

    return {"report": report, "report_markdown": report_markdown, "report_id": report_id}


# WHAT THIS DOES — WORKFLOW WIRING: Reuses Week 3/4 nodes and adds Week 5 report/synthesis at the end.
def build_week5_graph():
    """Question -> Supervisor -> Specialists -> Shared Evidence -> Synthesis -> END"""
    # TODO 4 — EXTEND THE MULTI-AGENT GRAPH
    # Reuse supervisor_node, specialists_node, and evidence_store_node.
    # Add synthesis_node after the shared evidence store.
    # Use Week5ResearchState.

    graph = StateGraph(Week5ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


week5_graph = build_week5_graph()


# ---------------- Week 5 UI ----------------
# UI: Adds a separate Week 5 button and shows the report + evidence + saved history.
# A full run invokes Week 3 planning/research, Week 4 evidence, THEN Week 5 writing.

from typing import cast, Any

st.divider()
st.subheader("Week 5 — Multi-Agent Synthesis Pipeline")
st.caption("Question → Supervisor → Specialists → Shared Evidence → Synthesis Agent → Report")

# Input field for your research question
user_query = st.text_input("Enter your research topic:", "What are the latest breakthroughs in fusion energy?")

if st.button("Run Full Synthesis Pipeline", type="primary") and user_query.strip():
    # TODO 5 — RUN THE COMPLETE PIPELINE
    initial_state = {
        "question": user_query,
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
        "report": {},
        "report_markdown": "",
        "report_id": 0
    }

    with st.spinner("Running multi-agent research pipeline and synthesizing report..."):
        # Invoke the graph using cast to satisfy strict type checkers
        final_state = week5_graph.invoke(cast(Any, initial_state)) # 

    st.success("Research Complete!")
    
    # 1. Display the synthesized report markdown
    st.markdown("Synthesized Research Report")
    st.markdown(final_state.get("report_markdown", "No report markdown generated."))

    # 2. Display the shared evidence used
    st.markdown("Shared Evidence Store")
    evidence_records = final_state.get("evidence_records", [])
    st.info(f"Total verified evidence records collected: {len(evidence_records)}")
    
    with st.expander("View Collected Evidence Details"):
        for i, record in enumerate(evidence_records, 1):
            st.markdown(f"**Evidence #{i}** ({record.get('source', 'Unknown Source')}):")
            st.write(record.get('content', record))

# 3. Display saved reports history from SQLite
st.divider()
st.subheader("Saved Reports History (SQLite)")

try:
    with sqlite3.connect(EVIDENCE_DB) as conn:
        cursor = conn.execute("SELECT id, question, created_at FROM reports ORDER BY created_at DESC")
        saved_reports = cursor.fetchall()
        
        if saved_reports:
            for rep_id, q_text, created in saved_reports:
                with st.expander(f"Report ID #{rep_id} — {q_text} ({created})"):
                    # Fetch and display the full markdown for this history item
                    hist_cursor = conn.execute("SELECT report_markdown FROM reports WHERE id = ?", (rep_id,))
                    row = hist_cursor.fetchone()
                    if row:
                        st.markdown(row[0])
        else:
            st.info("No saved reports found in the database yet. Run a pipeline above to create one!")
except Exception as e:
    st.warning(f"Database history table not initialized yet or error reading: {e}")

REVIEW_SYSTEM_PROMPT = """
You are the REVIEWER / CRITIC AGENT in a multi-agent business research system.
Evaluate the draft against the research question and Shared Evidence Store.
Do not conduct new research and do not silently rewrite the report.

Return ONLY valid JSON:
{
  "overall_status": "PASS" or "NEEDS_REVISION",
  "scores": {
    "completeness": 0,
    "citation_support": 0,
    "source_quality": 0,
    "consistency": 0,
    "business_usefulness": 0,
    "clarity": 0
  },
  "issues": [
    {
      "category": "completeness | citation_support | source_quality | consistency | business_usefulness | clarity",
      "section": "report section or overall",
      "explanation": "specific problem",
      "suggested_fix": "specific bounded fix"
    }
  ],
  "research_gaps": ["gap that may require more research later"]
}

PASS only when all six scores are at least 7 and there are no material citation-support issues.
""".strip()


REVISION_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT performing ONE controlled revision after reviewer feedback.
Use ONLY the supplied shared evidence. Do not research new facts.
Keep the same nine-section JSON report contract used in Week 5.
If a true research gap cannot be fixed from existing evidence, acknowledge it in limitations.
""".strip()

# TODO 0 — REVIEWER AGENT
# Give it: question + draft report + build_evidence_packet(evidence_records).
# Call call_model_json(REVIEW_SYSTEM_PROMPT, prompt).
# Normalize six scores to 0-10 integers and return:
# overall_status, scores, issues, research_gaps.
# Treat invalid model JSON shapes as a failed review, not a PASS.
def review_report(question: str, report: dict, evidence_records: list[dict]) -> dict:
    pass


# TODO 1 — ONE CONTROLLED REVISION
# Give the synthesis model the current report, reviewer feedback, and existing evidence.
# It may revise wording/organization using existing evidence only.
# It may NOT search or invent facts.
# Ensure every REPORT_SECTION_KEYS key still exists.
def revise_report_once(question: str, report: dict, evidence_records: list[dict], review: dict) -> dict:
    pass


# TODO 2 — WEEK 6 STATE
# Extend Week5ResearchState with:
#   review: dict
#   initial_review: dict
#   final_report_markdown: str
#   revision_performed: bool


# TODO 3 — REVIEW GATE NODE
# 1. Review the Week 5 synthesized report.
# 2. If PASS, keep it.
# 3. If NEEDS_REVISION, revise ONCE using existing evidence.
# 4. Review the revised report exactly once more.
# 5. Render final markdown using render_report_markdown(...).
def reviewer_node(state):
    pass


# TODO 4 — EXTEND THE MULTI-AGENT GRAPH
# Reuse supervisor_node -> specialists_node -> evidence_store_node -> synthesis_node.
# Add reviewer_node after synthesis.
def build_week6_graph():

    pass


week6_graph = build_week6_graph()


# TODO 5 — WEEK 6 UI
# Add a "Run Reviewed Synthesis Pipeline" button.
# Show:
# - final report ONLY if overall_status == PASS
# - if review fails, label the report unapproved and put it in an expander
# - reviewer scorecard
# - issues/research gaps
# - initial review if a revision happened
# - shared evidence
