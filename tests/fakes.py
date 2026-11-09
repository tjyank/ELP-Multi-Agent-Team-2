"""Offline stand-ins for Tavily and DeepSeek, so tests cost nothing and need no network.

Tests change behaviour through the module-level MODE dict (reset before each test).
"""
import json
import types

DEFAULT_MODE = {
    "review": "PASS",        # what the Reviewer returns
    "gap": False,            # whether the gap check asks for an extra round
    "bad_cite": False,       # first synthesis cites [E999]
    "records_per_source": 1, # how many records evidence extraction returns
    "fake_url": False,       # extraction returns a made-up URL
    "fail_agent": None,      # specialist role whose model call always fails
}
MODE = dict(DEFAULT_MODE)
CALLS = []  # (kind, payload) for every fake model call

SECTIONS = ["executive_summary", "key_findings", "competitor_comparison",
            "market_industry_trends", "emerging_technologies", "regulatory_developments",
            "risks_opportunities", "recommended_next_steps", "limitations_information_gaps"]
SCORES = ["completeness", "citation_support", "source_quality", "consistency",
          "business_usefulness", "clarity"]


def reset():
    MODE.clear()
    MODE.update(DEFAULT_MODE)
    CALLS.clear()


class FakeTavily:
    def __init__(self, **_kwargs):
        pass

    def search(self, query, max_results=5, topic=None):
        key = abs(hash(query)) % 100000
        return {"results": [
            {"title": f"Page {i} for {query[:30]}",
             "url": f"https://example.com/{key}/{topic or 'web'}/{i}",
             "content": "Revenue grew to $5 billion.",
             "published_date": "Fri, 09 Oct 2026 15:17:03 GMT" if topic == "news" else None}
            for i in range(max_results)
        ]}


def _reply(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


class _Completions:
    def create(self, model, messages, response_format=None):
        system, user = messages[0]["content"], messages[1]["content"]

        if system.startswith("You are the Supervisor"):
            CALLS.append(("supervisor", user))
            out = {"tasks": [{"agent": a, "objective": f"{a}: {user[:40]}", "output_fields": ["x"]}
                             for a in ["Competitor", "Market", "Tech/Regulatory"]]}
        elif system.startswith("You are the evidence"):
            src = json.loads(user)
            CALLS.append(("extract", src["url"]))
            out = {"records": [
                {"claim": f"Claim {i} from {src['url']}", "source": "Invented Source Name",
                 "url": "https://hallucinated.example/x" if MODE["fake_url"] else src["url"],
                 "publication_date": None, "source_type": "web", "evidence": "passage",
                 "confidence": "high", "related_company": None, "report_section": None}
                for i in range(MODE["records_per_source"])
            ]}
        elif "GAP AND CONFLICT" in system:
            CALLS.append(("gap", None))
            out = {"needs_research": MODE["gap"], "reason": "Missing pricing data",
                   "targeted_question": "What are current prices?", "conflicts": [],
                   "gaps": ["pricing"]}
        elif system.startswith("You are the SYNTHESIS AGENT in"):
            CALLS.append(("synthesis", None))
            cite = "[E999]" if MODE["bad_cite"] else "[E1]"
            out = {k: f"{k} costs $5 {cite}" for k in SECTIONS}
        elif system.startswith("You are the REVIEWER"):
            CALLS.append(("review", None))
            passed = MODE["review"] == "PASS"
            out = {"overall_status": MODE["review"],
                   "scores": {k: 8 if passed else 5 for k in SCORES},
                   "issues": [] if passed else [{"category": "clarity", "section": "overall",
                                                 "explanation": "vague", "suggested_fix": "be specific"}],
                   "research_gaps": []}
        elif system.startswith("You are the SYNTHESIS AGENT performing"):
            CALLS.append(("revision", None))
            current = json.loads(user)["current_report"]
            out = {k: str(v).replace("[E999]", "[E1]") for k, v in current.items()}
        elif system.startswith("You are the "):  # one of the three specialists
            agent = system.split("You are the ", 1)[1].split(" specialist", 1)[0]
            CALLS.append(("specialist", agent))
            if agent == MODE["fail_agent"]:
                raise RuntimeError(f"simulated {agent} outage")
            return _reply(f"{agent} brief")
        else:
            raise AssertionError("Unexpected prompt: " + system[:60])

        return _reply(json.dumps(out))


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.chat = types.SimpleNamespace(completions=_Completions())
