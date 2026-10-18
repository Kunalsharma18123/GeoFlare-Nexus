from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock


_status_lock = Lock()
_status = {
    "status": "never_run",
    "service_started_at": datetime.now(timezone.utc).isoformat(),
    "last_attempt_at": None,
    "last_success_at": None,
    "last_result": None,
    "last_error": None,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_ingestion_started() -> None:
    with _status_lock:
        _status.update(
            status="running",
            last_attempt_at=_utc_now(),
            last_error=None,
        )


def mark_ingestion_succeeded(result: dict) -> None:
    status = (
        "partial"
        if result.get("intelligence_failed", 0) > 0
        else "success"
    )

    with _status_lock:
        _status.update(
            status=status,
            last_success_at=_utc_now(),
            last_result=dict(result),
            last_error=None,
        )


def mark_ingestion_failed() -> None:
    with _status_lock:
        _status.update(
            status="failed",
            last_error="Ingestion failed. Check backend logs for details.",
        )


def get_ingestion_status() -> dict:
    with _status_lock:
        return deepcopy(_status)
