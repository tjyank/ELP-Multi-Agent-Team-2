"""End-to-end tests: run the real Streamlit app with fake APIs and click "Run research"."""
import json
import os
import sqlite3

from streamlit.testing.v1 import AppTest

from conftest import APP_PATH


def _run(mode_updates, fake_mode):
    fake_mode.update(mode_updates)
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _latest_report_row():
    conn = sqlite3.connect(os.environ["TEAM2_EVIDENCE_DB"])
    try:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT 1").fetchone())
    finally:
        conn.close()


def test_approved_run_with_extra_round(fake_mode):
    at = _run({"gap": True}, fake_mode)
    result = at.session_state["w8_t2_result"]

    assert any("Approved" in s.value for s in at.success)
    assert len(at.get("download_button")) == 1
    assert result["additional_research_performed"] is True
    assert [s["round"] for s in result["specialist_results"]] == [1, 1, 1, 2, 2, 2]
    assert result["new_evidence_count"] > 0

    row = _latest_report_row()
    assert row["review_status"] == "PASS"
    assert row["final_report_markdown"] == result["final_report_markdown"]
    assert json.loads(row["review_json"])["citation_check"]["invalid"] == []


def test_rejected_run_offers_no_download(fake_mode):
    at = _run({"review": "NEEDS_REVISION"}, fake_mode)
    result = at.session_state["w8_t2_result"]

    assert result["revision_performed"] is True
    assert any("Not approved" in e.value for e in at.error)
    assert len(at.get("download_button")) == 0
    assert _latest_report_row()["review_status"] == "NEEDS_REVISION"


def test_specialist_outage_is_shown_and_blocks_approval(fake_mode):
    at = _run({"fail_agent": "Market"}, fake_mode)

    assert any("Market (round 1) failed" in w.value for w in at.warning)
    assert any("missing: Market" in e.value for e in at.error)
    assert len(at.get("download_button")) == 0
