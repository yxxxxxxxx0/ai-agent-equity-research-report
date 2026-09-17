from webui.app import JOBS, app


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
