import threading

from eq_report.pipeline.run_tracker import RunTracker
from webui.app import JOBS, _THREAD_JOB, app


def test_job_document_opens_allowlisted_artifact(tmp_path):
    artifact = tmp_path / "07_qa_repair.json"
    artifact.write_text('{"attempts": []}', encoding="utf-8")
    JOBS["testjob"] = {
        "artifacts": {
            "repair_json": {"label": "Repair log", "path": str(artifact)},
        },
    }
    try:
        response = app.test_client().get("/document/testjob/repair_json")
        assert response.status_code == 200
        assert response.mimetype == "application/json"
        assert response.get_json() == {"attempts": []}
    finally:
        JOBS.pop("testjob", None)


def test_job_document_rejects_unknown_kind():
    JOBS["testjob"] = {"artifacts": {}}
    try:
        response = app.test_client().get("/document/testjob/not-allowed")
        assert response.status_code == 404
    finally:
        JOBS.pop("testjob", None)


def test_webui_records_completed_duration_for_each_pipeline_stage():
    job_id = "timingjob"
    JOBS[job_id] = {
        "active_stages": [],
        "completed_stages": [],
        "stage_first_started_at": {},
        "stage_completed_at": {},
        "stage_durations_ms": {},
    }
    thread_id = threading.get_ident()
    _THREAD_JOB[thread_id] = job_id
    tracker = RunTracker("run_timing", {})
    try:
        with tracker.stage("planning"):
            assert "planning" in JOBS[job_id]["active_stages"]
        job = JOBS[job_id]
        assert "planning" in job["completed_stages"]
        assert "planning" not in job["active_stages"]
        assert job["stage_completed_at"]["planning"] >= job["stage_first_started_at"]["planning"]
        assert job["stage_durations_ms"]["planning"] >= 0
    finally:
        _THREAD_JOB.pop(thread_id, None)
        JOBS.pop(job_id, None)
