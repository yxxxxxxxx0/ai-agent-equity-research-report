"""Writes the structured artefacts that sit alongside the PDF.

The report JSON is the input the PDF was rendered from, and the run manifest is
the execution record. Both are written on every run, including failed ones, so
any stage can be inspected after the fact without re-running the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..domain.qa import QAResult
from ..domain.report import ReportDraft
from ..domain.run import ReportRun
from ..errors import RenderingError
from ..logging_setup import get_logger, log_event

logger = get_logger("rendering.json")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(payload, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RenderingError(f"could not write {path}: {exc}") from exc
    return path


def write_report_json(
    output_dir: Path | str, draft: ReportDraft, qa_result: QAResult | None = None
) -> Path:
    """Persist the structured report the PDF was generated from."""
    path = Path(output_dir) / f"report_{draft.report_run_id}.json"
    payload = draft.to_dict()
    payload["qa"] = qa_result.to_dict() if qa_result else None
    write_json(path, payload)
    log_event(logger, logging.INFO, "report JSON written", path=str(path))
    return path


def write_run_manifest(output_dir: Path | str, run: ReportRun) -> Path:
    """Persist the execution record for one pipeline run."""
    path = Path(output_dir) / f"run_{run.report_run_id}.json"
    write_json(path, run.to_dict())
    log_event(logger, logging.INFO, "run manifest written", path=str(path))
    return path
