from pathlib import Path

from src.utils.database import JobRecord, JobTracker


def test_crud_cycle(tmp_path: Path):
    db = tmp_path / "test.db"
    tr = JobTracker(db_path=db)
    jid = tr.add_job(JobRecord(title="MLE", company="Acme", jd_text="JD", status="saved",
                               score=80.0, ats_score=70.0, keywords=["python"], missing_skills=["k8s"]))
    assert jid > 0
    rec = tr.get_job(jid)
    assert rec is not None
    assert rec.title == "MLE"
    assert rec.keywords == ["python"]

    rec.status = "applied"
    tr.update_job(rec)
    assert tr.get_job(jid).status == "applied"

    assert any(j.id == jid for j in tr.list_jobs())
    tr.delete_job(jid)
    assert tr.get_job(jid) is None


def test_conversations(tmp_path: Path):
    tr = JobTracker(db_path=tmp_path / "c.db")
    tr.log_message("user", "hi", "career_qa")
    tr.log_message("assistant", "hello", "career_qa")
    recent = tr.recent_messages("career_qa", limit=5)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
