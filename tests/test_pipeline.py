"""Unit tests for the pipeline's safety rules and helpers (no network, no real database)."""
import sqlite3

import pytest

import fakes


def _good_result(app, **overrides):
    evidence = [{"claim": "c", "source": "s", "url": "https://x", "source_type": "web", "evidence": "e"}]
    report = {k: "Fact [E1]" for k in app.MODEL_SECTION_KEYS}
    result = {
        "review": {"overall_status": "PASS"},
        "final_report_markdown": app.render_report_markdown(report, evidence),
        "evidence_records": evidence,
        "specialist_results": [{"agent": r, "research": "brief", "round": 1} for r in app.SPECIALIST_ROLES],
    }
    result.update(overrides)
    return result


# ---------------- Quality gate ----------------
def test_gate_approves_a_good_result(app):
    assert app._w8_t2_ready(_good_result(app))


@pytest.mark.parametrize("bad", [None, "text", [], {}])
def test_gate_fails_closed_on_garbage(app, bad):
    assert not app._w8_t2_ready(bad)


def test_gate_rejects_reviewer_failure(app):
    assert not app._w8_t2_ready(_good_result(app, review={"overall_status": "NEEDS_REVISION"}))


def test_gate_rejects_citation_to_missing_evidence(app):
    result = _good_result(app)
    result["final_report_markdown"] = result["final_report_markdown"].replace("[E1]", "[E7]", 1)
    checks = dict(app._w8_quality_checks(result))
    assert not app._w8_t2_ready(result)
    assert any("invalid: E7" in label for label in checks)


def test_gate_rejects_report_without_citations(app):
    result = _good_result(app)
    body, refs = result["final_report_markdown"].split(app.SOURCES_HEADING)
    result["final_report_markdown"] = body.replace("[E1]", "") + app.SOURCES_HEADING + refs
    assert not app._w8_t2_ready(result)


def test_gate_requires_all_three_first_round_specialists(app):
    specialists = [{"agent": "Competitor", "research": "b", "round": 1},
                   {"agent": "Market", "research": "b", "round": 1},
                   {"agent": "Tech/Regulatory", "research": "b", "round": 2}]
    assert not app._w8_t2_ready(_good_result(app, specialist_results=specialists))


# ---------------- Citations ----------------
@pytest.mark.parametrize("text, expected", [
    ("A [E1][E2].", [1, 2]),
    ("A [E1, E3; E5].", [1, 3, 5]),
    ("A [E2–E4] and [E6-7].", [2, 4, 6, 7]),
    ("See [the docs](https://x) and [note].", []),
])
def test_extract_citation_numbers(app, text, expected):
    assert app.extract_citation_numbers(text) == expected


def test_check_citations_flags_out_of_range(app):
    assert app.check_citations("[E0] [E2] [E3]", 2) == {"cited": [0, 2, 3], "invalid": [0, 3]}


def test_reviewer_overrules_pass_when_citations_are_invalid(app, fake_mode):
    fake_mode["review"] = "PASS"
    evidence = [{"claim": "c", "source": "s", "url": "u", "source_type": "web", "evidence": "e"}]
    state = {"question": "q", "report": {k: "x [E999]" for k in app.MODEL_SECTION_KEYS},
             "evidence_records": evidence, "report_id": 0}
    out = app.reviewer_node(state)
    assert out["initial_review"]["overall_status"] == "NEEDS_REVISION"
    assert out["revision_performed"] is True
    assert "[E999]" not in out["final_report_markdown"]  # fake revision repaired it
    assert out["review"]["overall_status"] == "PASS"


# ---------------- Evidence extraction ----------------
def test_extraction_uses_tavily_url_not_model_url(app, fake_mode):
    fake_mode["fake_url"] = True
    source = {"title": "Real Title", "url": "https://real.example/page", "content": "c"}
    records = app.extract_from_source("Market", "obj", source)
    assert records and all(r["url"] == "https://real.example/page" for r in records)
    assert all(r["source"] == "Real Title" for r in records)


def test_extraction_caps_records_per_source(app, fake_mode):
    fake_mode["records_per_source"] = 10
    records = app.extract_from_source("Market", "obj", {"title": "t", "url": "https://u", "content": "c"})
    assert len(records) == app.MAX_RECORDS_PER_SOURCE


def test_same_url_is_extracted_once_across_specialists(app):
    shared = {"title": "t", "url": "https://shared", "content": "c"}
    results = [{"agent": "Competitor", "objective": "o", "sources": [shared]},
               {"agent": "Market", "objective": "o", "sources": [shared, {"title": "t2", "url": "https://other"}]}]
    app.extract_specialist_evidence(results, skip_urls={"https://other"})
    assert [p for kind, p in fakes.CALLS if kind == "extract"] == ["https://shared"]


# ---------------- Specialists, retries, dates ----------------
def test_specialist_failure_is_reported_not_raised(app, fake_mode):
    fake_mode["fail_agent"] = "Market"
    assignments = [{"agent": r, "objective": "o", "output_fields": []} for r in app.SPECIALIST_ROLES]
    results, errors = app.run_specialists(assignments, "q", round_number=1)
    assert [r["agent"] for r in results] == ["Competitor", "Tech/Regulatory"]
    assert len(errors) == 1 and "Market" in errors[0]


def test_with_retries_recovers_then_gives_up(app):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("blip")
        return "ok"

    assert app.with_retries(flaky, attempts=3) == "ok"
    with pytest.raises(ConnectionError):
        app.with_retries(lambda: (_ for _ in ()).throw(ConnectionError("down")), attempts=2)


@pytest.mark.parametrize("raw, expected", [
    ("2026-10-09T15:17:03Z", "2026-10-09"),
    ("Fri, 09 Oct 2026 15:17:03 GMT", "2026-10-09"),
    ("October 2026", "2026-10"),
    ("unknown", None),
    (None, None),
])
def test_normalize_publication_date(app, raw, expected):
    assert app.normalize_publication_date(raw) == expected


# ---------------- Database ----------------
def test_old_reports_table_is_migrated(app, tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, "
                     "report_markdown TEXT NOT NULL, evidence_count INTEGER NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO reports(question, report_markdown, evidence_count, created_at) "
                     "VALUES ('q', 'md', 1, 't')")
    conn.close()
    monkeypatch.setattr(app, "EVIDENCE_DB", str(db))
    app.init_report_db()
    app.update_report_review(1, "final", {"overall_status": "PASS"}, True)
    row = app.load_reports()[0]
    assert row["final_report_markdown"] == "final"
    assert row["review_status"] == "PASS" and row["revision_performed"] == 1
