from backend import ingestion
from backend.app.main import INGEST_INTERVAL_SECONDS, ingestion_status
from backend.ingestion_status import get_ingestion_status


def test_run_ingestion_records_empty_feed_as_success(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(ingestion, "fetch_firms", lambda: pd.DataFrame())

    result = ingestion.run_ingestion()

    assert result["fetched"] == 0
    assert result["inserted"] == 0
    assert get_ingestion_status()["status"] == "success"
    assert get_ingestion_status()["last_result"] == result


def test_run_ingestion_tracks_success_and_failure(monkeypatch):
    result = {
        "fetched": 4,
        "inserted": 3,
        "duplicates": 1,
        "events_created": 2,
        "events_updated": 1,
        "intelligence_processed": 3,
        "intelligence_failed": 0,
    }

    monkeypatch.setattr(ingestion, "fetch_firms", lambda: object())
    monkeypatch.setattr(ingestion, "ingest_dataframe", lambda _: result)

    assert ingestion.run_ingestion() == result
    successful_status = get_ingestion_status()
    assert successful_status["status"] == "success"
    assert successful_status["last_attempt_at"] is not None
    assert successful_status["last_success_at"] is not None
    assert successful_status["last_result"] == result
    assert successful_status["last_error"] is None

    def fail_fetch():
        raise RuntimeError("sensitive upstream details")

    monkeypatch.setattr(ingestion, "fetch_firms", fail_fetch)

    try:
        ingestion.run_ingestion()
    except RuntimeError as error:
        assert str(error) == "sensitive upstream details"
    else:
        raise AssertionError("Ingestion should propagate fetch failures")

    failed_status = get_ingestion_status()
    assert failed_status["status"] == "failed"
    assert failed_status["last_attempt_at"] is not None
    assert failed_status["last_success_at"] == successful_status["last_success_at"]
    assert failed_status["last_result"] == result
    assert "sensitive upstream details" not in failed_status["last_error"]


def test_ingestion_status_returns_a_copy():
    status = get_ingestion_status()
    status["last_result"] = {"unexpected": True}

    assert get_ingestion_status()["last_result"] != {"unexpected": True}


def test_status_endpoint_includes_poll_interval_and_latest_run():
    response = ingestion_status()

    assert response["interval_seconds"] == INGEST_INTERVAL_SECONDS
    assert response["status"] in {"never_run", "success", "partial", "failed"}
    assert "last_attempt_at" in response
    assert "last_success_at" in response
    assert "last_result" in response
