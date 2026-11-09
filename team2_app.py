"""ELP — Team 2 Multi-Agent Competitive-Intelligence Research Assistant (Weeks 3–8 combined)

Flow:
    Question -> Supervisor -> 3 Specialists in parallel -> Shared Evidence (SQLite)
             -> Gap check -> (at most ONE extra research round) -> Synthesis
             -> Reviewer (at most ONE revision) -> Quality gate -> Download

Run:
    python -m pip install -r requirements_team2.txt
    streamlit run team2_app.py

Tests:
    python -m pip install -r requirements-dev.txt
    python -m pytest

.env:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key
    # Optional settings:
    DEEPSEEK_MODEL=deepseek-flash
    TEAM2_EVIDENCE_DB=team2_evidence.db
"""

import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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

# Settings that can be changed in .env without editing code
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
EVIDENCE_DB = os.getenv("TEAM2_EVIDENCE_DB", "team2_evidence.db")
MAX_API_ATTEMPTS = int(os.getenv("TEAM2_MAX_API_ATTEMPTS", "3"))
RETRY_BASE_SECONDS = float(os.getenv("TEAM2_RETRY_BASE_SECONDS", "2"))
MAX_RECORDS_PER_SOURCE = 4  # keeps the evidence store focused (was ~9 claims per page)

SPECIALIST_ROLES = ["Competitor", "Market", "Tech/Regulatory"]

EVIDENCE_FIELDS = [
    "claim",
    "source",
    "url",
    "publication_date",
    "source_type",
    "evidence",
    "confidence",
    "related_company",
    "report_section",
]

st.set_page_config(page_title="Team 2 — Research Assistant", page_icon="🧩", layout="wide")

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()


@st.cache_resource
def get_clients():
    """Create the API clients once, not on every Streamlit rerun."""
    return (
        TavilyClient(api_key=TAVILY_API_KEY),
        OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"),
    )


tavily, deepseek = get_clients()


def with_retries(fn, *args, attempts=None, **kwargs):
    """Call fn, retrying with exponential backoff (2s, 4s, ...) on any exception."""
    attempts = attempts or MAX_API_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(RETRY_BASE_SECONDS * 2 ** (attempt - 1))


# ---------------- Week 3: Supervisor + specialists ----------------
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
    specialist_errors: list[str]


class Week4ResearchState(ResearchState):
    evidence_records: list[dict]


def call_model_json(system_prompt: str, user_text: str) -> dict:
    """Ask the model for JSON. API errors AND unparseable JSON are retried."""

    def _once():
        response = deepseek.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return with_retries(_once)


def _parse_assignments(output: dict) -> list[dict]:
    tasks = output.get("tasks") if isinstance(output, dict) else None

    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Supervisor must return exactly three assignments.")

    expected_agents = set(SPECIALIST_ROLES)
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

    cleaned.sort(key=lambda a: SPECIALIST_ROLES.index(a["agent"]))
    return cleaned


def supervisor_node(state: ResearchState) -> dict:
    """Split the question into three assignments; a malformed plan is re-requested once."""
    last_error = None
    for _attempt in range(2):
        try:
            output = call_model_json(TEAM_2_SYSTEM_PROMPT, state["question"])
            return {"assignments": _parse_assignments(output)}
        except ValueError as error:
            last_error = error
    raise last_error


def normalize_publication_date(value) -> str | None:
    """Turn Tavily/model dates into YYYY-MM-DD (or YYYY-MM / YYYY). Returns None if unusable."""
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none", "unknown", "n/a", "date unavailable"}:
        return None
    # ISO dates, e.g. 2026-10-09 or 2026-10-09T15:17:03Z
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    # RFC 2822 dates from Tavily news, e.g. "Fri, 09 Oct 2026 15:17:03 GMT"
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    # Partial dates the model may give
    for fmt, out in (("%Y-%m", "%Y-%m"), ("%Y", "%Y"), ("%B %d, %Y", "%Y-%m-%d"),
                     ("%b %d, %Y", "%Y-%m-%d"), ("%d %B %Y", "%Y-%m-%d"), ("%B %Y", "%Y-%m")):
        try:
            return datetime.strptime(text, fmt).strftime(out)
        except ValueError:
            continue
    return None


def tavily_search(query: str, **kwargs) -> list[dict]:
    return with_retries(tavily.search, query=query, **kwargs).get("results", [])


def research_assignment(assignment: dict, original_question: str, round_number: int = 1) -> dict:
    """Execute ONE specialist's assignment."""
    # 1. Read assignment metadata
    agent = assignment["agent"]
    objective = assignment["objective"]
    output_fields = assignment["output_fields"]

    # 2. Search Tavily using the specialist's objective
    # 3 general web results (official/company pages) + 2 news results.
    # Only Tavily's news topic returns published_date, so this mix gives most sources a real date.
    general_results = tavily_search(objective, max_results=3)
    try:
        news_results = tavily_search(objective, max_results=2, topic="news")
    except Exception:
        news_results = []  # news search is a bonus; never fail the specialist over it
    seen_urls = set()
    tavily_results = []
    for src in general_results + news_results:
        if src.get("url") and src["url"] not in seen_urls:
            seen_urls.add(src["url"])
            tavily_results.append(src)

    # 3. Format Tavily results into readable source text
    sources_text = ""
    for idx, src in enumerate(tavily_results, 1):
        title = src.get("title", "No Title")
        url = src.get("url", "No URL")
        content = src.get("content", "")
        published = normalize_publication_date(src.get("published_date")) or "date unknown"
        sources_text += f"Source [{idx}]: {title}\nURL: {url}\nPublished: {published}\nContent: {content}\n\n"

    # 4. Call DeepSeek with specialist prompt & context
    prompt_text = f"""
    Original User Question: {original_question}
    Your Assigned Objective: {objective}
    Requested Output Fields: {', '.join(output_fields)} Tavily Research Results:
    {sources_text} Please provide a clear and concise research brief based on the sources provided above.
    """

    system_prompt = SPECIALIST_PROMPTS.get(agent, "")
    model_response = with_retries(
        deepseek.chat.completions.create,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
    )
    research_text = model_response.choices[0].message.content

    # 5. Return formatted result
    return {
        "agent": agent,
        "objective": objective,
        "output_fields": output_fields,
        "research": research_text,
        "sources": tavily_results,
        "round": round_number,
    }


def run_specialists(assignments: list[dict], question: str, round_number: int = 1):
    """Run specialist assignments in parallel. Returns (results, error messages).

    Errors are returned as data instead of calling st.error here: this code can run
    in a worker thread where Streamlit calls are silently dropped.
    """
    results, errors = [], []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(research_assignment, assignment, question, round_number): assignment
            for assignment in assignments
        }

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                agent = futures[future].get("agent", "Specialist")
                errors.append(f"{agent} (round {round_number}) failed: {error}")

    order = {role: i for i, role in enumerate(SPECIALIST_ROLES)}
    results.sort(key=lambda r: order.get(r.get("agent"), len(order)))
    return results, errors


def specialists_node(state: ResearchState) -> dict:
    """Run all three independent specialist assignments in parallel."""
    results, errors = run_specialists(state["assignments"], state["question"], round_number=1)
    return {"specialist_results": results, "specialist_errors": errors}


# ---------------- Week 4: Shared Evidence Store ----------------
@contextmanager
def open_db():
    """Open the SQLite store, commit on success, and always close the connection."""
    conn = sqlite3.connect(EVIDENCE_DB)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_evidence_db():
    """Create the Week 4 shared evidence table if it does not exist."""
    with open_db() as conn:
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


def save_evidence_records(records: list[dict]):
    """Save validated specialist evidence to SQLite."""
    if not records:
        return

    columns = ", ".join(EVIDENCE_FIELDS)
    placeholders = ", ".join(f":{field}" for field in EVIDENCE_FIELDS)
    sql = f"INSERT OR IGNORE INTO evidence ({columns}) VALUES ({placeholders})"

    rows = [{field: record.get(field) for field in EVIDENCE_FIELDS} for record in records]

    with open_db() as conn:
        conn.executemany(sql, rows)


def load_evidence_records() -> list[dict]:
    """Read the shared evidence store back from SQLite."""
    with open_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()


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
- publication_date: if the input includes published_date, use it. Otherwise use a date
  only if the source content explicitly states when the article/page was published or
  last updated (format YYYY-MM-DD, or YYYY-MM / YYYY if that is all it gives).
  Never guess, and never use the date of an event described in the text. Use null if unstated.
- confidence must be high, medium, or low.
- Do not turn analysis or recommendations into facts.
""".strip() + (
    f"\n- Return at most {MAX_RECORDS_PER_SOURCE} records: the most decision-relevant "
    "claims for the objective, not every sentence."
)


def extract_from_source(agent: str, objective: str, source: dict) -> list[dict]:
    """Extract Evidence Records from ONE Tavily source."""
    source_payload = {
        "agent": agent,
        "objective": objective,
        "title": source.get("title"),
        "url": source.get("url"),
        "content": (source.get("content") or "")[:6000],
        "published_date": source.get("published_date"),
    }
    prompt = json.dumps(source_payload, ensure_ascii=False)

    try:
        result = call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt)
    except Exception:
        return []  # skip a bad source instead of failing the whole run

    raw = result.get("records") if isinstance(result, dict) else None
    records = [r for r in (raw if isinstance(raw, list) else []) if isinstance(r, dict)]
    records = records[:MAX_RECORDS_PER_SOURCE]

    # Source identity comes from Tavily, never from the model: a model-written URL
    # could be mistyped or invented and would then appear in Sources / References.
    tavily_date = normalize_publication_date(source.get("published_date"))
    for record in records:
        record["url"] = source.get("url")
        record["source"] = source.get("title") or source.get("url")
        record["source_type"] = str(record.get("source_type") or "web")
        # Tavily's own published_date is authoritative; otherwise keep only a date
        # the model read explicitly from the source text.
        record["publication_date"] = tavily_date or normalize_publication_date(record.get("publication_date"))
    return records


def extract_specialist_evidence(specialist_results: list[dict], skip_urls=()) -> list[dict]:
    """Convert specialist source results into the shared Evidence Record format.

    Each URL is extracted only once, even if several specialists (or an earlier
    round, via skip_urls) found the same page.
    """
    seen_urls = set(skip_urls)
    jobs = []
    for specialist in specialist_results:
        for source in specialist.get("sources", []):
            url = source.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            jobs.append((specialist.get("agent"), specialist.get("objective"), source))

    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_from_source, *job) for job in jobs]
        for future in as_completed(futures):
            records.extend(future.result())

    return records


def validate_shared_evidence(records: list[dict]) -> list[dict]:
    """Validate records and remove exact duplicate claim/source pairs."""
    required = ["claim", "source", "url", "source_type", "evidence"]

    validated = []
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue

        if not all(str(record.get(field) or "").strip() for field in required):
            continue

        key = (str(record["claim"]).lower(), record["url"])
        if key in seen:
            continue

        seen.add(key)
        validated.append(record)

    return validated


def evidence_store_node(state: Week4ResearchState) -> dict:
    """Week 4: build and persist the shared evidence store."""
    records = extract_specialist_evidence(state["specialist_results"])
    validated = validate_shared_evidence(records)
    save_evidence_records(validated)
    return {"evidence_records": validated}


# ---------------- Week 5: Synthesis + saved reports ----------------
# Week 6 review results are stored on the same row once the Reviewer finishes.
REPORT_REVIEW_COLUMNS = {
    "final_report_markdown": "TEXT",
    "review_status": "TEXT",
    "review_json": "TEXT",
    "revision_performed": "INTEGER",
    "reviewed_at": "TEXT",
}


def init_report_db():
    """Create the reports table if missing and add any newer columns to old databases."""
    with open_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                report_markdown TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(reports)")}
        for column, column_type in REPORT_REVIEW_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE reports ADD COLUMN {column} {column_type}")


def save_report(question: str, report_markdown: str, evidence_count: int) -> int:
    """Save the synthesized (pre-review) draft and return its database id."""
    created_at = datetime.now(timezone.utc).isoformat()
    with open_db() as conn:
        cursor = conn.execute(
            "INSERT INTO reports(question, report_markdown, evidence_count, created_at) "
            "VALUES (?, ?, ?, ?)",
            (question, report_markdown, evidence_count, created_at),
        )
        return cursor.lastrowid


def update_report_review(report_id: int, final_report_markdown: str, review: dict,
                         revision_performed: bool):
    """Store the final (possibly revised) report and the Reviewer's verdict."""
    if not report_id:
        return
    with open_db() as conn:
        conn.execute(
            "UPDATE reports SET final_report_markdown = ?, review_status = ?, review_json = ?, "
            "revision_performed = ?, reviewed_at = ? WHERE id = ?",
            (
                final_report_markdown,
                review.get("overall_status"),
                json.dumps(review, ensure_ascii=False),
                int(bool(revision_performed)),
                datetime.now(timezone.utc).isoformat(),
                report_id,
            ),
        )


def load_reports() -> list[dict]:
    """Load previously generated reports, most recent first."""
    with open_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


init_report_db()


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
    ("sources_references", "Sources / References"),
]

MODEL_SECTION_KEYS = [key for key, _title in REPORT_SECTION_KEYS if key != "sources_references"]
SOURCES_HEADING = "## Sources / References"


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
- Only cite evidence_id labels that appear in the supplied evidence.
- Reorganize findings by business topic, not agent name.
- The report should read as if one coordinated team produced it.
- Do not invent facts, dates, metrics, source names, URLs, or citations.
- Clearly distinguish facts, analysis, recommendations, and uncertainty.
- If evidence is insufficient, say so.
""".strip()


def build_evidence_packet(evidence_records: list[dict]) -> list[dict]:
    """Assign [E1], [E2] labels to this run's evidence, keeping sources and excerpts."""
    packet = []
    for index, record in enumerate(evidence_records, start=1):
        packet.append({
            "evidence_id": f"E{index}",
            **{field: record.get(field) for field in EVIDENCE_FIELDS},
        })
    return packet


def synthesize_report(question: str, evidence_records: list[dict]) -> dict:
    """Merge all specialist evidence into the common report structure."""
    evidence_packet = build_evidence_packet(evidence_records)

    prompt = (
        f"Research question:\n{question}\n\n"
        f"Shared Evidence Store (cite using evidence_id labels such as [E1]):\n"
        f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2)}"
    )

    report = call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)

    # Make sure every model-written section exists
    if not isinstance(report, dict):
        report = {}
    return {
        key: report.get(key) or "Insufficient evidence found for this section."
        for key in MODEL_SECTION_KEYS
    }


def build_references(evidence_records: list[dict]) -> str:
    """Citation legend built from the original URLs, not model-written references."""
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        lines.append(f"- [E{index}] {source} — {date} — {url}")
    return "\n".join(lines) if lines else "No source records were available."


def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        if key == "sources_references":
            continue
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append(f"{SOURCES_HEADING}\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)


class Week5ResearchState(Week4ResearchState):
    report: dict
    report_markdown: str
    report_id: int


def synthesis_node(state):
    """Synthesize only after all shared evidence has been produced."""
    evidence_records = state.get("evidence_records", [])

    if not evidence_records:
        raise ValueError(
            "No evidence was collected, so no report can be written. "
            "Check the specialist errors and your Tavily / DeepSeek keys."
        )

    question = state["question"]
    report = synthesize_report(question, evidence_records)
    report_markdown = render_report_markdown(report, evidence_records)
    report_id = save_report(question, report_markdown, len(evidence_records))

    return {
        "report": report,
        "report_markdown": report_markdown,
        "report_id": report_id,
    }


# ---------------- Week 6: Citation check + Reviewer ----------------
_BRACKET_GROUP = re.compile(r"\[([^\[\]]*)\]")
_CITATION_TOKEN = re.compile(r"E(\d+)(?:\s*[-–]\s*E?(\d+))?")


def extract_citation_numbers(text) -> list[int]:
    """Find evidence numbers cited as [E1], [E1][E2], [E1, E3] or [E2–E4]."""
    numbers = []
    for group in _BRACKET_GROUP.findall(str(text or "")):
        tokens = [t for t in re.split(r"\s*[,;]\s*", group.strip()) if t]
        matches = [_CITATION_TOKEN.fullmatch(t) for t in tokens]
        if not tokens or not all(matches):
            continue  # not a citation, e.g. a markdown link label
        for match in matches:
            numbers.append(int(match.group(1)))
            if match.group(2):
                numbers.append(int(match.group(2)))
    return numbers


def check_citations(text: str, evidence_count: int) -> dict:
    """Return the cited evidence numbers and any that point to no real record."""
    cited = sorted(set(extract_citation_numbers(text)))
    invalid = [n for n in cited if n < 1 or n > evidence_count]
    return {"cited": cited, "invalid": invalid}


def report_body(report_markdown: str) -> str:
    """The report without its Sources / References legend (which lists every [E#])."""
    return str(report_markdown or "").split(SOURCES_HEADING, 1)[0]


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
Only cite evidence_id labels that appear in the supplied evidence.
If a true research gap cannot be fixed from existing evidence, acknowledge it in limitations.
""".strip()


SCORE_KEYS = ["completeness", "citation_support", "source_quality", "consistency",
              "business_usefulness", "clarity"]


def review_report(question: str, report: dict, evidence_records: list[dict]) -> dict:
    # Safe fallback result: used whenever the reviewer breaks, so a failure is never a PASS
    failed = {
        "overall_status": "NEEDS_REVISION",
        "scores": {key: 0 for key in SCORE_KEYS},
        "issues": [],
        "research_gaps": [],
    }

    prompt = json.dumps(
        {
            "question": question,
            "draft_report": report,
            "evidence": build_evidence_packet(evidence_records),
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        result = call_model_json(REVIEW_SYSTEM_PROMPT, prompt)
    except Exception as error:
        return {**failed, "issues": [f"Reviewer call failed: {error}"]}

    if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
        return {**failed, "issues": ["Reviewer returned an invalid JSON shape."]}

    # Normalize each score to an integer from 0 to 10
    scores = {}
    for key in SCORE_KEYS:
        try:
            value = int(round(float(result["scores"].get(key, 0))))
        except (TypeError, ValueError):
            value = 0
        scores[key] = max(0, min(10, value))

    status = str(result.get("overall_status", "")).strip().upper()
    issues = result.get("issues")
    gaps = result.get("research_gaps")

    return {
        "overall_status": status if status in {"PASS", "NEEDS_REVISION"} else "NEEDS_REVISION",
        "scores": scores,
        "issues": issues if isinstance(issues, list) else [],
        "research_gaps": gaps if isinstance(gaps, list) else [],
    }


def apply_citation_check(review: dict, report: dict, evidence_count: int) -> dict:
    """Overrule a PASS when the report cites evidence that does not exist.

    This is checked in code, not left to the Reviewer model.
    """
    text = "\n".join(str(report.get(key) or "") for key in MODEL_SECTION_KEYS)
    citations = check_citations(text, evidence_count)
    review = {**review, "citation_check": citations}
    if citations["invalid"]:
        labels = ", ".join(f"E{n}" for n in citations["invalid"])
        review["overall_status"] = "NEEDS_REVISION"
        review["issues"] = list(review.get("issues") or []) + [{
            "category": "citation_support",
            "section": "overall",
            "explanation": f"Cites evidence that does not exist: {labels} "
                           f"(only E1–E{evidence_count} are available).",
            "suggested_fix": "Replace each invalid label with a real evidence_id or remove the claim.",
        }]
    return review


def revise_report_once(question: str, report: dict, evidence_records: list[dict], review: dict) -> dict:
    prompt = json.dumps(
        {
            "question": question,
            "current_report": report,
            "reviewer_feedback": {
                "issues": review.get("issues", []),
                "research_gaps": review.get("research_gaps", []),
            },
            "evidence": build_evidence_packet(evidence_records),
            "required_keys": MODEL_SECTION_KEYS,
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        revised = call_model_json(REVISION_SYSTEM_PROMPT, prompt)
    except Exception:
        return report

    if not isinstance(revised, dict):
        return report

    return {
        key: revised.get(key) or report.get(key) or "Insufficient evidence found for this section."
        for key in MODEL_SECTION_KEYS
    }


class Week6ResearchState(Week5ResearchState):
    review: dict
    initial_review: dict
    final_report_markdown: str
    revision_performed: bool


def reviewer_node(state):
    """Review the draft, allow at most ONE revision, render and save the final report."""
    question = state["question"]
    report = state["report"]
    evidence_records = state.get("evidence_records", [])
    evidence_count = len(evidence_records)

    initial_review = apply_citation_check(
        review_report(question, report, evidence_records), report, evidence_count
    )
    review = initial_review
    revision_performed = False

    if initial_review["overall_status"] != "PASS":
        report = revise_report_once(question, report, evidence_records, initial_review)
        revision_performed = True
        # Review the revised report exactly once more (no second revision, whatever the result)
        review = apply_citation_check(
            review_report(question, report, evidence_records), report, evidence_count
        )

    final_report_markdown = render_report_markdown(report, evidence_records)
    update_report_review(state.get("report_id", 0), final_report_markdown, review, revision_performed)

    return {
        "report": report,
        "review": review,
        "initial_review": initial_review,
        "final_report_markdown": final_report_markdown,
        "revision_performed": revision_performed,
    }


# ---------------- Week 7: Gap check + ONE bounded extra round ----------------
W7_T2_GAP_PROMPT = """
You are the existing supervisor's GAP AND CONFLICT CHECK, not a new agent.
Given the user's research question and evidence store, identify at most ONE
material research gap or meaningful contradiction that targeted research could fix.
Return JSON ONLY:
  {"needs_research": boolean, "reason": string,
   "targeted_question": string, "conflicts": [string], "gaps": [string]}
Missing evidence and genuinely conflicting facts matter; duplicate coverage
alone does not require searching. Do not invent missing facts or resolve a
conflict by guessing. Recommend NO further research when already adequate.
""".strip()


def week7_t2_find_gaps(state: dict) -> dict:
    no_research = {
        "needs_research": False,
        "reason": "",
        "targeted_question": "",
        "conflicts": [],
        "gaps": [],
    }

    prompt = json.dumps(
        {
            "question": state["question"],
            "evidence": build_evidence_packet(state.get("evidence_records", [])),
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        result = call_model_json(W7_T2_GAP_PROMPT, prompt)
    except Exception as error:
        return {
            "gap_check": {**no_research, "reason": f"Gap check failed: {error}"},
            "additional_research_performed": False,
        }

    if not isinstance(result, dict):
        return {
            "gap_check": {**no_research, "reason": "Gap check returned invalid JSON."},
            "additional_research_performed": False,
        }

    targeted_question = str(result.get("targeted_question") or "").strip()
    conflicts = result.get("conflicts")
    gaps = result.get("gaps")

    # Research only if the model said exactly true AND gave a question to search
    needs_research = result.get("needs_research") is True and bool(targeted_question)

    gap_check = {
        "needs_research": needs_research,
        "reason": str(result.get("reason") or "").strip(),
        "targeted_question": targeted_question,
        "conflicts": [str(c) for c in conflicts] if isinstance(conflicts, list) else [],
        "gaps": [str(g) for g in gaps] if isinstance(gaps, list) else [],
    }

    return {"gap_check": gap_check, "additional_research_performed": False}


def week7_t2_route(state: dict) -> str:
    gap_check = state.get("gap_check") or {}
    if gap_check.get("needs_research") is True and gap_check.get("targeted_question"):
        return "additional"
    return "synthesis"


def week7_t2_additional_research(state: dict) -> dict:
    old_evidence = state.get("evidence_records", [])
    old_results = state.get("specialist_results", [])
    old_errors = state.get("specialist_errors", []) or []
    targeted_question = (state.get("gap_check") or {}).get("targeted_question", "").strip()

    unchanged = {
        "evidence_records": old_evidence,
        "specialist_results": old_results,
        "specialist_errors": old_errors,
        "additional_research_performed": False,
        "new_evidence_count": 0,
    }

    if not targeted_question:
        return unchanged

    try:
        # Reuse the existing supervisor to split the targeted question into 3 assignments
        assignments = supervisor_node({"question": targeted_question})["assignments"]
        # Reuse the existing specialists, tagged as round 2
        new_results, new_errors = run_specialists(assignments, targeted_question, round_number=2)
        # Pages already extracted in round 1 are not extracted again
        known_urls = {src.get("url") for r in old_results for src in r.get("sources", [])}
        new_records = validate_shared_evidence(
            extract_specialist_evidence(new_results, skip_urls=known_urls)
        )
        save_evidence_records(new_records)
    except Exception as error:
        return {**unchanged, "specialist_errors": old_errors + [f"Extra research round failed: {error}"]}

    # Old first, so existing [E#] labels keep their numbers
    merged_evidence = validate_shared_evidence(old_evidence + new_records)

    return {
        "evidence_records": merged_evidence,
        "specialist_results": old_results + new_results,
        "specialist_errors": old_errors + new_errors,
        "additional_research_performed": True,
        "new_evidence_count": len(merged_evidence) - len(old_evidence),
    }


class Week7ResearchState(Week6ResearchState):
    gap_check: dict
    additional_research_performed: bool
    new_evidence_count: int


def build_week7_graph():
    """START -> supervise -> specialists -> evidence_store -> gap_check
    gap_check -> synthesis, or gap_check -> additional -> synthesis (at most one extra pass)
    synthesis -> reviewer -> END
    """
    graph = StateGraph(Week7ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("gap_check", week7_t2_find_gaps)
    graph.add_node("additional", week7_t2_additional_research)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "gap_check")
    graph.add_conditional_edges(
        "gap_check",
        week7_t2_route,
        {"additional": "additional", "synthesis": "synthesis"},
    )
    graph.add_edge("additional", "synthesis")
    graph.add_edge("synthesis", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


# Built on every rerun on purpose: it takes milliseconds, and caching it with
# st.cache_resource would keep running OLD node code after you edit this file.
week7_graph = build_week7_graph()


# =====================================================================
# WEEK 8 — FINAL RESEARCH WORKSPACE (the app's single user interface)
# =====================================================================

# All 10 report headings, as rendered by render_report_markdown (9 model sections + Sources)
_W8_REQUIRED_HEADINGS = [f"## {title}" for _key, title in REPORT_SECTION_KEYS]


def _w8_quality_checks(result) -> list[tuple[str, bool]]:
    """Conservative quality gate. Each check has a plain-language label so the UI
    can show WHY a report failed."""
    checks = []
    try:
        if not isinstance(result, dict):
            return [("The research pipeline returned a result", False)]

        review = result.get("review") if isinstance(result.get("review"), dict) else {}
        checks.append(("Reviewer agent gave the report a PASS verdict",
                       review.get("overall_status") == "PASS"))

        report_md = result.get("final_report_markdown")
        has_report = isinstance(report_md, str) and bool(report_md.strip())
        checks.append(("A final report was written", has_report))

        missing = [h[3:] for h in _W8_REQUIRED_HEADINGS if not has_report or h not in report_md]
        label = "All 10 report sections are present"
        if missing and has_report:
            label += f" (missing: {', '.join(missing)})"
        checks.append((label, not missing))

        evidence = result.get("evidence_records")
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        checks.append(("At least one piece of evidence was collected", evidence_count >= 1))

        citations = check_citations(report_body(report_md if has_report else ""), evidence_count)
        checks.append(("The report cites its evidence with [E#] labels", bool(citations["cited"])))
        label = "Every [E#] citation points to a real evidence record"
        if citations["invalid"]:
            label += f" (invalid: {', '.join(f'E{n}' for n in citations['invalid'])})"
        checks.append((label, not citations["invalid"]))

        specialists = result.get("specialist_results")
        specialists = specialists if isinstance(specialists, list) else []
        first_round = [s for s in specialists if isinstance(s, dict) and s.get("round", 1) == 1]
        roles = {s.get("agent") for s in first_round if str(s.get("research") or "").strip()}
        missing_roles = [r for r in SPECIALIST_ROLES if r not in roles]
        label = "All three specialists (Competitor, Market, Tech/Regulatory) returned research"
        if missing_roles:
            label += f" (missing: {', '.join(missing_roles)})"
        checks.append((label, len(first_round) >= 3 and not missing_roles))
    except Exception as error:
        checks.append((f"Quality check could not run: {error}", False))
    return checks


def _w8_t2_ready(result) -> bool:
    """FAILS CLOSED: approved only if every check exists and passes."""
    checks = _w8_quality_checks(result)
    return bool(checks) and all(ok for _label, ok in checks)


def _md(text) -> str:
    """Escape $ so dollar amounts don't render as LaTeX math."""
    return str(text or "").replace("$", "\\$")


W8_STEP_LABELS = {
    "supervise": "🧭 Supervisor split the question into three research tasks",
    "specialists": "🔍 Specialists finished searching the web",
    "evidence_store": "🗂️ Evidence extracted and saved",
    "gap_check": "🧩 Checked the evidence for gaps or conflicts",
    "additional": "➕ Extra research round finished",
    "synthesis": "✍️ Draft report written",
    "reviewer": "✅ Reviewer finished checking the report",
}

SPECIALIST_ICONS = {"Competitor": "🏁", "Market": "📈", "Tech/Regulatory": "⚖️"}


# ---------------- Page header ----------------
st.title("🧩 Competitive Intelligence Research Assistant")
st.write(
    "Ask a business question. A team of AI agents researches it on the web, "
    "writes a report, and checks its own work before handing it to you."
)

with st.expander("How does it work?"):
    st.markdown(
        """
1. **Supervisor** splits your question into three focused tasks.
2. **Three specialists** research in parallel: 🏁 Competitor, 📈 Market, ⚖️ Tech/Regulatory.
3. **Evidence** — key facts are saved with their source link (each web page is read once).
4. **Gap check** — if something important is missing, the team runs **one** extra research round (never more).
5. **Report** — the facts are combined into a 10-section report with citations like [E1].
6. **Reviewer** scores the report, and every citation is checked against the evidence. If it falls short, it gets **one** revision.
7. **Quality gate** — you can only download the report if it passed every check.

Typical run time: 1–3 minutes. Reports are a prototype and are not independently fact-checked.
"""
    )

# ---------------- Question input ----------------
with st.form("w8_t2_form"):
    question = st.text_area(
        "Your research question",
        value="How is NVIDIA positioning itself against AMD in the AI chip market?",
        height=80,
    )
    _w8_submitted = st.form_submit_button("Run research", type="primary")

# ---------------- Run the pipeline ----------------
if _w8_submitted:
    if not question.strip():
        st.warning("Please enter a research question first.")
    else:
        _w8_initial = {"question": question.strip(), "assignments": [],
                       "specialist_results": [], "specialist_errors": [], "evidence_records": [],
                       "report": {}, "report_markdown": "", "report_id": 0,
                       "review": {}, "initial_review": {}, "final_report_markdown": "",
                       "revision_performed": False, "gap_check": {},
                       "additional_research_performed": False, "new_evidence_count": 0}
        st.session_state.pop("w8_t2_result", None)
        _w8_start = time.perf_counter()
        with st.status("Researching... this usually takes 1–3 minutes.", expanded=True) as _w8_status:
            _w8_state = dict(_w8_initial)
            try:
                for _w8_update in week7_graph.stream(_w8_initial, stream_mode="updates"):
                    for _w8_node, _w8_output in _w8_update.items():
                        _w8_status.write(
                            f"{W8_STEP_LABELS.get(_w8_node, _w8_node)} "
                            f"({time.perf_counter() - _w8_start:.0f}s)"
                        )
                        if _w8_output:
                            _w8_state.update(_w8_output)
                _w8_seconds = round(time.perf_counter() - _w8_start, 1)
                st.session_state["w8_t2_result"] = _w8_state
                st.session_state["w8_t2_seconds"] = _w8_seconds
                st.session_state["w8_t2_question"] = question.strip()
                _w8_status.update(label=f"Done in {_w8_seconds:.0f} seconds",
                                  state="complete", expanded=False)
            except Exception as exc:
                _w8_status.update(label="Research failed", state="error")
                st.error(f"Research failed, so no report was produced. Details: {exc}")
                for _err in _w8_state.get("specialist_errors") or []:
                    st.warning(_err)

# ---------------- Results ----------------
if "w8_t2_result" in st.session_state:
    _w8_result = st.session_state["w8_t2_result"]
    _w8_checks = _w8_quality_checks(_w8_result)
    _w8_ok = _w8_t2_ready(_w8_result)
    _w8_evidence = _w8_result.get("evidence_records", []) or []
    _w8_specialists = _w8_result.get("specialist_results", []) or []
    _w8_errors = _w8_result.get("specialist_errors", []) or []
    _w8_review = _w8_result.get("review", {}) or {}
    _w8_gap = _w8_result.get("gap_check", {}) or {}

    st.divider()
    st.subheader(f"Results for: “{st.session_state.get('w8_t2_question', '')}”")

    if _w8_ok:
        st.success("✅ **Approved.** The report passed every quality check. "
                   "(Prototype: not independently fact-checked.)")
    else:
        _w8_failed = [label for label, ok in _w8_checks if not ok]
        st.error("❌ **Not approved.** The report did not pass the quality checks, "
                 "so it can't be downloaded. Failed: " + "; ".join(_w8_failed))

    if _w8_errors:
        st.warning("Some research steps failed: " + " | ".join(_w8_errors))

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Time taken", f"{st.session_state.get('w8_t2_seconds', 0):.0f} s")
    _m2.metric("Evidence collected", len(_w8_evidence))
    _m3.metric("Extra research round",
               "Yes" if _w8_result.get("additional_research_performed") else "No")
    _m4.metric("Report revised", "Yes" if _w8_result.get("revision_performed") else "No")

    _tab_report, _tab_quality, _tab_process, _tab_sources = st.tabs(
        ["📄 Report", "✅ Quality check", "🔎 How it was researched", "📚 Evidence & sources"]
    )

    # ---- Report ----
    with _tab_report:
        if _w8_result.get("report_id"):
            st.caption(f"Saved to the database as report #{_w8_result['report_id']}.")
        if _w8_ok:
            _w8_text = _w8_result.get("final_report_markdown", "")
            st.download_button("⬇️ Download report (.md)", _w8_text,
                               file_name="elp_team2_report.md", mime="text/markdown",
                               key="w8_t2_download")
            st.markdown(_md(_w8_text))
        else:
            st.warning("This draft was **not approved**. It is shown for debugging only "
                       "and should not be shared or trusted.")
            with st.expander("Show unapproved draft"):
                st.markdown(_md(_w8_result.get("final_report_markdown")
                                or _w8_result.get("report_markdown")))

    # ---- Quality check ----
    with _tab_quality:
        st.markdown("#### Quality gate")
        st.caption("The report can only be downloaded if every check below passes.")
        for _label, _passed in _w8_checks:
            st.markdown(f"{'✅' if _passed else '❌'} {_label}")

        st.markdown("#### Reviewer scores")
        st.caption("Each category is scored 0–10. A pass needs 7 or more in every category.")
        _w8_scores = _w8_review.get("scores", {}) or {}
        _score_cols = st.columns(3)
        for _i, (_name, _score) in enumerate(_w8_scores.items()):
            _score_cols[_i % 3].metric(_name.replace("_", " ").title(), f"{_score}/10",
                                       delta=_score - 7)

        st.markdown("#### Issues the reviewer found")
        _w8_issues = _w8_review.get("issues") or []
        if not _w8_issues:
            st.caption("No issues reported.")
        for _issue in _w8_issues:
            if isinstance(_issue, dict):
                st.markdown(
                    f"- **{_issue.get('section', 'Overall')}** — {_md(_issue.get('explanation', ''))}  \n"
                    f"  *Suggested fix:* {_md(_issue.get('suggested_fix', ''))}"
                )
            else:
                st.markdown(f"- {_md(_issue)}")

        if _w8_review.get("research_gaps"):
            st.markdown("#### Remaining research gaps")
            for _g in _w8_review["research_gaps"]:
                st.markdown(f"- {_md(_g)}")

        if _w8_result.get("revision_performed"):
            with st.expander("Reviewer's verdict before the revision"):
                st.json(_w8_result.get("initial_review", {}))

    # ---- How it was researched ----
    with _tab_process:
        st.markdown("#### 1. Supervisor's task assignments")
        _assign_cols = st.columns(3)
        for _col, _a in zip(_assign_cols, _w8_result.get("assignments", []) or []):
            with _col:
                st.markdown(f"**{SPECIALIST_ICONS.get(_a.get('agent'), '🔹')} {_a.get('agent', '')}**")
                st.write(_a.get("objective", ""))
                if _a.get("output_fields"):
                    st.caption("Looks for: " + ", ".join(_a["output_fields"]))

        st.markdown("#### 2. Specialist research briefs")
        st.caption("Each specialist's brief should cover a different angle of the question.")
        for _err in _w8_errors:
            st.error(_err)
        for _spec in _w8_specialists:
            _agent = _spec.get("agent", "Specialist")
            _round = "first round" if _spec.get("round", 1) == 1 else "extra round"
            _n_src = len(_spec.get("sources", []) or [])
            with st.expander(f"{SPECIALIST_ICONS.get(_agent, '🔹')} {_agent} — {_round} ({_n_src} web sources)"):
                st.write(f"**Task:** {_spec.get('objective', '')}")
                st.markdown(_md(_spec.get("research")))

        st.markdown("#### 3. Gap check")
        if _w8_result.get("additional_research_performed"):
            st.info(
                f"The supervisor found a gap and ran **one** extra research round on: "
                f"“{_w8_gap.get('targeted_question', '')}”. "
                f"It added **{_w8_result.get('new_evidence_count', 0)}** new evidence records."
            )
        elif _w8_gap.get("needs_research"):
            st.warning("A gap was found, but the extra research round failed, "
                       "so the report uses the original evidence only.")
        else:
            st.success("No important gaps were found, so no extra research was needed.")
        if _w8_gap.get("reason"):
            st.write(f"**Why:** {_w8_gap['reason']}")
        if _w8_gap.get("gaps"):
            st.markdown("**Gaps noted:** " + "; ".join(_w8_gap["gaps"]))
        if _w8_gap.get("conflicts"):
            st.markdown("**Conflicting facts noted:** " + "; ".join(_w8_gap["conflicts"]))

    # ---- Evidence & sources ----
    with _tab_sources:
        st.caption("Numbers match the [E#] citations in the report. "
                   "Click a link to check a source yourself.")
        st.dataframe(
            [
                {
                    "Ref": f"E{_i}",
                    "Claim": _rec.get("claim", ""),
                    "Source": _rec.get("source", ""),
                    "Link": _rec.get("url", ""),
                    "Confidence": _rec.get("confidence", ""),
                    "Date": _rec.get("publication_date") or "Unknown",
                }
                for _i, _rec in enumerate(_w8_evidence, start=1)
            ],
            column_config={"Link": st.column_config.LinkColumn("Link")},
            hide_index=True,
            width="stretch",
        )

# TODO 2 — Inspect each of the three specialist roles and verify
# the output is distinct. Check that the gap pass occurs at most once.
# TODO 3 — Manually open 3 cited source links and note problems.
