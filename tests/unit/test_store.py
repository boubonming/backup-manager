from datetime import datetime, timezone
import sqlite3

import pytest

from backup_manager.models import Job, Schedule
from backup_manager.store import (
    FileRecord,
    Store,
    StoreConflictError,
    StoreMigrationError,
    StoreNotFoundError,
    StoreTransitionError,
    StoreValidationError,
)

UTC = timezone.utc


def job(*, interval=False) -> Job:
    schedule = (
        Schedule.interval(timezone="Asia/Singapore", interval_seconds=3600)
        if interval
        else Schedule.daily(time="03:00", timezone="Asia/Singapore", weekdays=[1, 3, 7])
    )
    return Job(
        name="Nightly archives",
        enabled=True,
        source_root="source",
        destination_root="destination",
        source_path="archives/./world",
        destination_path="palworld",
        keep_count=10,
        schedule=schedule,
    )


def test_new_database_migrates_and_reopens_without_data_loss(tmp_path):
    path = tmp_path / "store.sqlite3"
    with Store(path) as store:
        assert store.schema_version == Store.SCHEMA_VERSION
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store.connection.execute("PRAGMA busy_timeout").fetchone()[0] == store.busy_timeout_ms
        created = store.create_job(job())
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"schema_version", "jobs", "job_revisions", "runs", "files"} <= tables
    with Store(path) as reopened:
        assert reopened.get_job(created.job_id).job == job()
        assert reopened.schema_version == Store.SCHEMA_VERSION


def test_failing_injected_migration_rolls_back_version_and_ddl(tmp_path):
    path = tmp_path / "migration.sqlite3"
    with Store(path):
        pass

    def fail(connection):
        connection.execute("CREATE TABLE should_rollback(value TEXT)")
        raise RuntimeError("deliberate migration failure")

    with pytest.raises(StoreMigrationError, match="migration 2 failed"):
        Store(path, migrations=(*Store.MIGRATIONS, fail))
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 1
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='should_rollback'"
        ).fetchone() is None


def test_newer_schema_is_rejected_without_modification(tmp_path):
    path = tmp_path / "newer.sqlite3"
    with Store(path):
        pass
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version=999 WHERE id=1")
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(StoreMigrationError, match="newer schema version 999"):
        Store(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1.5, "not-a-version"])
def test_malformed_schema_version_is_public_error_without_modification(tmp_path, version):
    path = tmp_path / "malformed-schema.sqlite3"
    with Store(path):
        pass
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version = ? WHERE id = 1", (version,))
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(StoreMigrationError, match="invalid schema version"):
        Store(path)
    assert path.read_bytes() == before


def test_job_revisions_round_trip_and_stale_update_is_atomic(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with Store(path) as store:
        first = store.create_job(job())
        assert first.revision == 1
        assert first.job.schedule.weekdays == (1, 3, 7)
        second = store.update_job(first.job_id, 1, job(interval=True))
        assert second.revision == 2
        assert store.get_job(first.job_id, 1).job == job()
        assert store.get_job(first.job_id).job == job(interval=True)
        with pytest.raises(StoreConflictError, match="job .* revision conflict"):
            store.update_job(first.job_id, 1, job())
        assert store.get_job(first.job_id).revision == 2
        assert store.count_job_revisions(first.job_id) == 2


def test_run_state_machine_and_utc_timestamps(tmp_path):
    now = iter(
        [
            datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        ]
    )
    path = tmp_path / "runs.sqlite3"
    with Store(path, clock=lambda: next(now)) as store:
        created = store.create_job(job())
        run = store.create_run(created.job_id, 1, trigger="manual")
        assert run.state == "queued"
        with pytest.raises(StoreTransitionError, match="run .* transition"):
            store.transition_run(run.run_id, "succeeded")
        running = store.transition_run(run.run_id, "running")
        assert running.started_at == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
        with pytest.raises(StoreValidationError, match="error_summary"):
            store.transition_run(run.run_id, "running", error_summary="bad")
        finished = store.transition_run(run.run_id, "failed", error_summary="copy failed")
        assert finished.ended_at == datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
        assert finished.error_summary == "copy failed"
        with pytest.raises(StoreTransitionError):
            store.transition_run(run.run_id, "succeeded")
        assert store.get_run(run.run_id) == finished


def test_run_transition_matrix_rejects_every_other_edge_without_mutation(tmp_path):
    states = {"queued", "running", "succeeded", "failed", "partial", "cancelled"}
    terminals = {"succeeded", "failed", "partial", "cancelled"}
    with Store(tmp_path / "run-matrix.sqlite3") as store:
        created = store.create_job(job())
        queued = store.create_run(created.job_id, 1, trigger="matrix")
        assert queued.started_at is None
        assert queued.ended_at is None
        for target in states - {"running"}:
            before = store.get_run(queued.run_id)
            with pytest.raises(StoreTransitionError):
                store.transition_run(queued.run_id, target)
            assert store.get_run(queued.run_id) == before

        running = store.transition_run(queued.run_id, "running")
        assert running.started_at is not None
        assert running.ended_at is None
        for target in {"queued", "running"}:
            before = store.get_run(queued.run_id)
            with pytest.raises(StoreTransitionError):
                store.transition_run(queued.run_id, target)
            assert store.get_run(queued.run_id) == before

        for terminal in terminals:
            candidate = store.create_run(created.job_id, 1, trigger="matrix")
            store.transition_run(candidate.run_id, "running")
            error = None if terminal == "succeeded" else f"{terminal} error"
            finished = store.transition_run(candidate.run_id, terminal, error_summary=error)
            assert finished.state == terminal
            assert finished.started_at is not None
            assert finished.ended_at is not None
            assert finished.error_summary == error
            for target in states:
                before = store.get_run(candidate.run_id)
                with pytest.raises(StoreTransitionError):
                    store.transition_run(candidate.run_id, target)
                assert store.get_run(candidate.run_id) == before


def test_file_identity_is_unique_and_distinguishes_path_and_content(tmp_path):
    path = tmp_path / "files.sqlite3"
    with Store(path) as store:
        created = store.create_job(job())
        common = dict(source_sha256="a" * 64, source_length=10, source_mtime_ns=20)
        one = store.create_file(created.job_id, "a.zip", **common)
        duplicate = store.create_file(created.job_id, "./a.zip", **common)
        assert duplicate.file_id == one.file_id
        assert store.create_file(created.job_id, "b.zip", **common).file_id != one.file_id
        assert store.create_file(
            created.job_id, "a.zip", source_sha256="b" * 64, source_length=10, source_mtime_ns=20
        ).file_id != one.file_id
        assert store.count_files(created.job_id) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_sha256": "A" * 64},
        {"source_sha256": "short"},
        {"source_sha256": "g" * 64},
        {"source_length": True},
        {"source_length": "10"},
        {"source_length": -1},
        {"source_mtime_ns": False},
        {"source_mtime_ns": 1.0},
        {"source_mtime_ns": -1},
    ],
)
def test_file_identity_validation_is_strict_before_write(tmp_path, kwargs):
    with Store(tmp_path / "validation.sqlite3") as store:
        created = store.create_job(job())
        values = dict(source_sha256="a" * 64, source_length=10, source_mtime_ns=20)
        values.update(kwargs)
        with pytest.raises(StoreValidationError):
            store.create_file(created.job_id, "a.zip", **values)
        assert store.count_files(created.job_id) == 0


def test_journal_transitions_and_tombstones_survive_reopen(tmp_path):
    path = tmp_path / "journal.sqlite3"
    with Store(path) as store:
        created = store.create_job(job())
        record = store.create_file(
            created.job_id,
            "a.zip",
            source_sha256="a" * 64,
            source_length=10,
            source_mtime_ns=20,
        )
        run = store.create_run(created.job_id, 1, trigger="manual")
        with pytest.raises(StoreTransitionError):
            store.transition_file(record.file_id, "published")
        record = store.transition_file(record.file_id, "verified")
        record = store.transition_file(
            record.file_id,
            "published",
            destination_relative_path="palworld/a.zip",
            destination_sha256="b" * 64,
            destination_length=10,
            publishing_run_id=run.run_id,
            retention_order=1,
        )
        record = store.transition_file(record.file_id, "pending_delete")
        record = store.transition_file(record.file_id, "pruned")
        assert record.state == "pruned"
    with Store(path) as reopened:
        found = reopened.find_file(created.job_id, "a.zip", "a" * 64, 10)
        assert found is not None
        assert found.state == "pruned"
        assert found.destination_relative_path == "palworld/a.zip"
        assert reopened.count_files(created.job_id) == 1
        with pytest.raises(StoreTransitionError):
            reopened.transition_file(found.file_id, "published")


def test_file_transition_matrix_rejects_every_other_edge_without_mutation(tmp_path):
    states = {"staged", "verified", "published", "pending_delete", "pruned"}
    next_state = {
        "staged": "verified",
        "verified": "published",
        "published": "pending_delete",
        "pending_delete": "pruned",
    }
    with Store(tmp_path / "file-matrix.sqlite3") as store:
        created = store.create_job(job())
        run = store.create_run(created.job_id, 1, trigger="manual")
        record = store.create_file(
            created.job_id,
            "matrix.zip",
            source_sha256="a" * 64,
            source_length=10,
            source_mtime_ns=20,
        )
        for source in ("staged", "verified", "published", "pending_delete", "pruned"):
            allowed = next_state.get(source)
            for target in states - ({allowed} if allowed else set()):
                before = store.get_file(record.file_id)
                with pytest.raises(StoreTransitionError):
                    store.transition_file(record.file_id, target)
                assert store.get_file(record.file_id) == before
            if allowed is None:
                break
            if allowed == "published":
                record = store.transition_file(
                    record.file_id,
                    allowed,
                    destination_relative_path="palworld/matrix.zip",
                    destination_sha256="b" * 64,
                    destination_length=10,
                    publishing_run_id=run.run_id,
                    retention_order=1,
                )
            else:
                record = store.transition_file(record.file_id, allowed)


def test_publication_requires_complete_same_job_metadata_and_is_immutable(tmp_path):
    with Store(tmp_path / "publication-integrity.sqlite3") as store:
        first = store.create_job(job())
        second = store.create_job(job(interval=True))
        run = store.create_run(first.job_id, 1, trigger="manual")
        foreign_run = store.create_run(second.job_id, 1, trigger="manual")
        record = store.create_file(
            first.job_id,
            "integrity.zip",
            source_sha256="a" * 64,
            source_length=10,
            source_mtime_ns=20,
        )
        store.transition_file(record.file_id, "verified")
        before = store.get_file(record.file_id)
        with pytest.raises(StoreValidationError, match="publication metadata"):
            store.transition_file(record.file_id, "published")
        assert store.get_file(record.file_id) == before
        with pytest.raises(StoreConflictError, match="different job"):
            store.transition_file(
                record.file_id,
                "published",
                destination_relative_path="palworld/integrity.zip",
                destination_sha256="b" * 64,
                destination_length=10,
                publishing_run_id=foreign_run.run_id,
                retention_order=1,
            )
        assert store.get_file(record.file_id) == before
        record = store.transition_file(
            record.file_id,
            "published",
            destination_relative_path="palworld/integrity.zip",
            destination_sha256="b" * 64,
            destination_length=10,
            publishing_run_id=run.run_id,
            retention_order=1,
        )
        metadata = (
            record.destination_relative_path,
            record.destination_sha256,
            record.destination_length,
            record.publishing_run_id,
            record.retention_order,
        )
        for state in ("pending_delete", "pruned"):
            target = "pending_delete" if state == "pending_delete" else "pruned"
            before = store.get_file(record.file_id)
            with pytest.raises(StoreTransitionError, match="immutable"):
                store.transition_file(
                    record.file_id,
                    target,
                    destination_relative_path="other.zip",
                    destination_sha256="c" * 64,
                    destination_length=20,
                    publishing_run_id=run.run_id,
                    retention_order=2,
                )
            assert store.get_file(record.file_id) == before
            record = store.transition_file(record.file_id, target)
            assert (
                record.destination_relative_path,
                record.destination_sha256,
                record.destination_length,
                record.publishing_run_id,
                record.retention_order,
            ) == metadata


def test_unknown_entities_have_stable_public_errors(tmp_path):
    with Store(tmp_path / "errors.sqlite3") as store:
        with pytest.raises(StoreNotFoundError, match="job 'missing' not found"):
            store.get_job("missing")
        with pytest.raises(StoreNotFoundError, match="run 'missing' not found"):
            store.transition_run("missing", "running")
        with pytest.raises(StoreNotFoundError, match="file 999 not found"):
            store.get_file(999)
