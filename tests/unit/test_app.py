import threading

from backup_manager import create_app


def test_healthz_returns_exact_minimal_healthy_response():
    app = create_app()

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.data == b'{"status":"ok"}\n'
    assert response.content_type == "application/json"


def test_creating_multiple_apps_does_not_start_background_threads(monkeypatch):
    started_threads = []

    def record_thread_start(thread):
        started_threads.append(thread)

    monkeypatch.setattr(threading.Thread, "start", record_thread_start)

    create_app()
    create_app()

    assert started_threads == []
