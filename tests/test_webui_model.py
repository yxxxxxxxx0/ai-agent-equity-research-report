"""The model chosen in the web UI is validated and reaches the pipeline's settings."""

import datetime as dt

import webui.app as webui


def _offline(monkeypatch):
    monkeypatch.setattr(webui, "_live_models", lambda: {
        "openai/gpt-5.6-luna": {"id": "openai/gpt-5.6-luna", "name": "Luna",
                                "pricing": {"prompt": "0.0000002", "completion": "0.0000012"}},
        "openai/gpt-5.6-sol": {"id": "openai/gpt-5.6-sol", "name": "Sol",
                               "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
    })


def test_models_endpoint_lists_only_live_models_with_prices(monkeypatch):
    _offline(monkeypatch)
    models = {m["id"]: m for m in webui.app.test_client().get("/api/models").get_json()["models"]}
    assert models["openai/gpt-5.6-sol"]["price"] == [2.0, 10.0]
    assert "anthropic/claude-opus-5.5" not in models  # curated but not live


def test_unknown_or_malformed_model_is_rejected(monkeypatch):
    _offline(monkeypatch)
    client = webui.app.test_client()
    for bad in ("not-a-model", "openai/does-not-exist", "x; rm -rf /"):
        response = client.post("/api/generate", json={"ticker": "AAPL", "model": bad})
        assert response.status_code == 400


def test_chosen_model_overrides_settings_for_the_run(monkeypatch):
    seen = {}

    def fake_generate(_request, settings, **_kw):
        seen["model"] = settings.model.model
        raise RuntimeError("stop before any real work")

    monkeypatch.setattr(webui, "generate_report_sync", fake_generate)
    webui.JOBS["j"] = {"status": "queued"}
    try:
        webui._run_job("j", "AAPL", dt.date(2026, 9, 30), None, "openai/gpt-5.6-sol")
        assert seen["model"] == "openai/gpt-5.6-sol"
        webui._run_job("j", "AAPL", dt.date(2026, 9, 30), None, None)
        assert seen["model"] == webui.Settings.from_env().model.model
    finally:
        webui.JOBS.pop("j", None)
