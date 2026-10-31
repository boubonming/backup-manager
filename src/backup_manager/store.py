"""Transactional SQLite persistence for jobs, runs, and file journals.

This module stores metadata only.  It never opens, copies, hashes, publishes, or
removes backup files.  Job revisions and journal rows are append-only from the
caller's perspective; state changes are guarded by the state machines documented
on :class:`Store`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
import posixpath
import re
import sqlite3
from typing import Callable, Iterator, Sequence
from uuid import uuid4

from .models import Job, Schedule

UTC = timezone.utc
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class StoreError(Exception):
    """Base class for all public persistence errors."""


class StoreValidationError(StoreError):
    """An input violates the storage contract."""


class StoreConflictError(StoreError):
    """An optimistic-concurrency or uniqueness conflict occurred."""


class StoreNotFoundError(StoreError):
    """The requested entity does not exist."""


class StoreTransitionError(StoreError):
    """A state-machine transition is not legal."""


class StoreMigrationError(StoreError):
    """The database schema could not be migrated safely."""


@dataclass(frozen=True, slots=True)
class JobRevision:
    job_id: str
    revision: int
    job: Job


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    job_id: str
    config_revision: int
    trigger: str
    state: str
    queued_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    error_summary: str | None


@dataclass(frozen=True, slots=True)
class FileRecord:
    file_id: int
    job_id: str
    source_relative_path: str
    source_sha256: str
    source_length: int
    source_mtime_ns: int
    destination_relative_path: str | None
    destination_sha256: str | None
    destination_length: int | None
    state: str
    publishing_run_id: str | None
    retention_order: int | None
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None
    pending_delete_at: datetime | None
    pruned_at: datetime | None


def _migration_1(connection: sqlite3.Connection) -> None:
    """Create the v1 schema; called inside the migration transaction."""
    connection.execute(
        """CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL CHECK (version >= 0)
        )"""
    )
    connection.execute("INSERT INTO schema_version(id, version) VALUES (1, 0)")
    connection.execute(
        """CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY NOT NULL,
            current_revision INTEGER NOT NULL CHECK (current_revision >= 1),
            created_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE TABLE job_revisions (
            job_id TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            name TEXT,
            enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
            source_root TEXT NOT NULL,
            destination_root TEXT NOT NULL,
            source_path TEXT NOT NULL,
            destination_path TEXT NOT NULL,
            keep_count INTEGER NOT NULL CHECK (keep_count >= 1),
            schedule_kind TEXT NOT NULL CHECK (schedule_kind IN ('daily', 'interval')),
            schedule_timezone TEXT NOT NULL,
            schedule_time TEXT,
            schedule_weekdays TEXT,
            interval_seconds INTEGER,
            created_at TEXT NOT NULL,
            PRIMARY KEY (job_id, revision),
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        )"""
    )
    connection.execute(
        """CREATE TABLE runs (
            run_id TEXT PRIMARY KEY NOT NULL,
            job_id TEXT NOT NULL,
            config_revision INTEGER NOT NULL,
            trigger TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'partial', 'cancelled')),
            queued_at TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            error_summary TEXT,
            FOREIGN KEY (job_id, config_revision) REFERENCES job_revisions(job_id, revision)
        )"""
    )
    connection.execute(
        """CREATE TABLE files (
            file_id INTEGER PRIMARY KEY,
            job_id TEXT NOT NULL,
            source_relative_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_length INTEGER NOT NULL CHECK (source_length >= 0),
            source_mtime_ns INTEGER NOT NULL CHECK (source_mtime_ns >= 0),
            destination_relative_path TEXT,
            destination_sha256 TEXT,
            destination_length INTEGER CHECK (destination_length IS NULL OR destination_length >= 0),
            state TEXT NOT NULL CHECK (state IN ('staged', 'verified', 'published', 'pending_delete', 'pruned')),
            publishing_run_id TEXT,
            retention_order INTEGER CHECK (retention_order IS NULL OR retention_order >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            published_at TEXT,
            pending_delete_at TEXT,
            pruned_at TEXT,
            UNIQUE (job_id, source_relative_path, source_sha256, source_length),
            FOREIGN KEY (job_id) REFERENCES jobs(job_id),
            FOREIGN KEY (publishing_run_id) REFERENCES runs(run_id)
        )"""
    )
    connection.execute(
        "CREATE INDEX files_job_state_order ON files(job_id, state, retention_order)"
    )
    connection.execute("CREATE INDEX runs_job_queued ON runs(job_id, queued_at)")


MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_migration_1,)


def _strict_nonnegative_int(field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StoreValidationError(f"{field}: must be a non-negative integer")
    return value


def _strict_positive_int(field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StoreValidationError(f"{field}: must be a positive integer")
    return value


def _sha256(field: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StoreValidationError(f"{field}: must be lowercase 64-hex SHA-256")
    return value


def _relative_path(field: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise StoreValidationError(f"{field}: must be a relative POSIX path")
    if "\\" in value or _WINDOWS_DRIVE_RE.match(value):
        raise StoreValidationError(f"{field}: must use POSIX path separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part == ".." for part in value.split("/")):
        raise StoreValidationError(f"{field}: must not be absolute or contain '..'")
    normalized = posixpath.normpath(path.as_posix())
    if normalized == ".." or normalized.startswith("../"):
        raise StoreValidationError(f"{field}: must remain relative")
    return normalized


def _timestamp(value: object, *, field: str = "clock") -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise StoreValidationError(f"{field}: must be an aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _weekdays_text(weekdays: tuple[int, ...] | None) -> str | None:
    return None if weekdays is None else ",".join(str(day) for day in weekdays)


def _weekdays_value(value: str | None) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(day) for day in value.split(","))


class Store:
    """A small transactional SQLite store.

    Runs may only transition ``queued -> running -> terminal``.  File journals
    may only transition ``staged -> verified -> published -> pending_delete ->
    pruned``.  ``pruned`` rows remain as tombstones.  ``clock`` must return an
    aware :class:`datetime`; it is injected to make timestamps deterministic.
    """

    SCHEMA_VERSION = len(MIGRATIONS)
    MIGRATIONS = MIGRATIONS
    TERMINAL_RUN_STATES = frozenset({"succeeded", "failed", "partial", "cancelled"})
    FILE_STATES = frozenset({"staged", "verified", "published", "pending_delete", "pruned"})

    def __init__(
        self,
        path: str | bytes,
        *,
        clock: Callable[[], datetime] | None = None,
        busy_timeout_ms: int = 5_000,
        migrations: Sequence[Callable[[sqlite3.Connection], None]] | None = None,
    ) -> None:
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or not 1 <= busy_timeout_ms <= 60_000:
            raise StoreValidationError("busy_timeout_ms: must be an integer from 1 through 60000")
        self.busy_timeout_ms = busy_timeout_ms
        self._clock = clock or (lambda: datetime.now(UTC))
        self._migrations = tuple(migrations) if migrations is not None else self.MIGRATIONS
        self._connection = sqlite3.connect(path, timeout=busy_timeout_ms / 1000, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            if self._connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise StoreMigrationError("foreign keys could not be enabled")
            if self._connection.execute("PRAGMA busy_timeout").fetchone()[0] != busy_timeout_ms:
                raise StoreMigrationError("busy timeout could not be configured")
            self._migrate()
        except Exception:
            self._connection.close()
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT version FROM schema_version WHERE id = 1"
        ).fetchone()
        if row is None:
            raise StoreMigrationError("schema version row is missing")
        return int(row[0])

    def _migrate(self) -> None:
        table = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        recorded = 0 if table is None else self.schema_version
        supported = len(self._migrations)
        if recorded > supported:
            raise StoreMigrationError(
                f"newer schema version {recorded} is not supported (maximum {supported})"
            )
        if recorded == supported:
            return
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            for number in range(recorded + 1, supported + 1):
                try:
                    self._migrations[number - 1](self._connection)
                except Exception as exc:
                    raise StoreMigrationError(f"migration {number} failed: {exc}") from None
                self._connection.execute(
                    "UPDATE schema_version SET version = ? WHERE id = 1", (number,)
                )
            self._connection.commit()
        except StoreMigrationError:
            self._connection.rollback()
            raise
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise StoreMigrationError(f"migration failed: {exc}") from None

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except StoreError:
            self._connection.rollback()
            raise
        except sqlite3.IntegrityError:
            self._connection.rollback()
            raise StoreConflictError("database integrity conflict") from None
        except sqlite3.Error:
            self._connection.rollback()
            raise StoreError("database operation failed") from None
        except OverflowError:
            self._connection.rollback()
            raise StoreValidationError("integer value exceeds SQLite range") from None
        except Exception:
            # Preserve the original injected/application exception, but never
            # leave a failed transaction holding the connection lock.
            self._connection.rollback()
            raise

    def _now(self) -> tuple[datetime, str]:
        value = self._clock()
        text = _timestamp(value)
        return value.astimezone(UTC), text

    @staticmethod
    def _job_values(job: Job) -> tuple[object, ...]:
        schedule = job.schedule
        return (
            job.name,
            int(job.enabled),
            job.source_root,
            job.destination_root,
            job.source_path,
            job.destination_path,
            job.keep_count,
            schedule.kind,
            schedule.timezone,
            schedule.time,
            _weekdays_text(schedule.weekdays),
            schedule.interval_seconds,
        )

    @staticmethod
    def _row_job(row: sqlite3.Row) -> Job:
        if row["schedule_kind"] == "daily":
            schedule = Schedule.daily(
                time=row["schedule_time"],
                timezone=row["schedule_timezone"],
                weekdays=_weekdays_value(row["schedule_weekdays"]),
            )
        else:
            schedule = Schedule.interval(
                timezone=row["schedule_timezone"],
                interval_seconds=row["interval_seconds"],
            )
        return Job(
            name=row["name"],
            enabled=bool(row["enabled"]),
            source_root=row["source_root"],
            destination_root=row["destination_root"],
            source_path=row["source_path"],
            destination_path=row["destination_path"],
            keep_count=row["keep_count"],
            schedule=schedule,
        )

    def create_job(self, job: Job) -> JobRevision:
        if not isinstance(job, Job):
            raise StoreValidationError("job: must be a validated Job")
        job_id = uuid4().hex
        _, created_at = self._now()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO jobs(job_id, current_revision, created_at) VALUES (?, 1, ?)",
                (job_id, created_at),
            )
            connection.execute(
                """INSERT INTO job_revisions(
                    job_id, revision, name, enabled, source_root, destination_root,
                    source_path, destination_path, keep_count, schedule_kind,
                    schedule_timezone, schedule_time, schedule_weekdays,
                    interval_seconds, created_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, *self._job_values(job), created_at),
            )
        return JobRevision(job_id, 1, job)

    def update_job(self, job_id: str, expected_revision: int, job: Job) -> JobRevision:
        if not isinstance(job_id, str) or not job_id:
            raise StoreValidationError("job_id: must be a non-empty string")
        _strict_positive_int("expected_revision", expected_revision)
        if not isinstance(job, Job):
            raise StoreValidationError("job: must be a validated Job")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT current_revision FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise StoreNotFoundError(f"job '{job_id}' not found")
            current = int(row[0])
            if current != expected_revision:
                raise StoreConflictError(
                    f"job '{job_id}' revision conflict: expected {expected_revision}, current {current}"
                )
            _, created_at = self._now()
            next_revision = current + 1
            connection.execute(
                """INSERT INTO job_revisions(
                    job_id, revision, name, enabled, source_root, destination_root,
                    source_path, destination_path, keep_count, schedule_kind,
                    schedule_timezone, schedule_time, schedule_weekdays,
                    interval_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, next_revision, *self._job_values(job), created_at),
            )
            connection.execute(
                "UPDATE jobs SET current_revision = ? WHERE job_id = ?",
                (next_revision, job_id),
            )
        return JobRevision(job_id, next_revision, job)

    def get_job(self, job_id: str, revision: int | None = None) -> JobRevision:
        if not isinstance(job_id, str) or not job_id:
            raise StoreValidationError("job_id: must be a non-empty string")
        if revision is not None:
            _strict_positive_int("revision", revision)
            row = self._connection.execute(
                "SELECT * FROM job_revisions WHERE job_id = ? AND revision = ?",
                (job_id, revision),
            ).fetchone()
        else:
            row = self._connection.execute(
                """SELECT r.* FROM job_revisions AS r
                   JOIN jobs AS j ON j.job_id = r.job_id AND j.current_revision = r.revision
                   WHERE r.job_id = ?""",
                (job_id,),
            ).fetchone()
        if row is None:
            if revision is None:
                raise StoreNotFoundError(f"job '{job_id}' not found")
            raise StoreNotFoundError(f"job '{job_id}' revision {revision} not found")
        return JobRevision(job_id, int(row["revision"]), self._row_job(row))

    get_job_revision = get_job

    def count_job_revisions(self, job_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM job_revisions WHERE job_id = ?", (job_id,)
        ).fetchone()
        return int(row[0])

    def create_run(
        self,
        job_id: str,
        config_revision: int,
        *,
        trigger: str,
        state: str = "queued",
    ) -> RunRecord:
        if not isinstance(job_id, str) or not job_id:
            raise StoreValidationError("job_id: must be a non-empty string")
        _strict_positive_int("config_revision", config_revision)
        if not isinstance(trigger, str) or not trigger:
            raise StoreValidationError("trigger: must be a non-empty string")
        if state != "queued":
            raise StoreValidationError("state: runs must be created queued")
        queued_dt, queued_at = self._now()
        run_id = uuid4().hex
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone() is None:
                raise StoreNotFoundError(f"job '{job_id}' not found")
            if connection.execute(
                "SELECT 1 FROM job_revisions WHERE job_id = ? AND revision = ?",
                (job_id, config_revision),
            ).fetchone() is None:
                raise StoreNotFoundError(
                    f"job '{job_id}' revision {config_revision} not found"
                )
            connection.execute(
                """INSERT INTO runs(run_id, job_id, config_revision, trigger, state, queued_at)
                   VALUES (?, ?, ?, ?, 'queued', ?)""",
                (run_id, job_id, config_revision, trigger, queued_at),
            )
        return RunRecord(run_id, job_id, config_revision, trigger, "queued", queued_dt, None, None, None)

    def _row_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            row["run_id"], row["job_id"], int(row["config_revision"]), row["trigger"],
            row["state"], _parse_timestamp(row["queued_at"]), _parse_timestamp(row["started_at"]),
            _parse_timestamp(row["ended_at"]), row["error_summary"],
        )  # type: ignore[arg-type]

    def get_run(self, run_id: str) -> RunRecord:
        if not isinstance(run_id, str) or not run_id:
            raise StoreValidationError("run_id: must be a non-empty string")
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreNotFoundError(f"run '{run_id}' not found")
        return self._row_run(row)

    def transition_run(self, run_id: str, state: str, *, error_summary: str | None = None) -> RunRecord:
        if not isinstance(run_id, str) or not run_id:
            raise StoreValidationError("run_id: must be a non-empty string")
        if state not in {"running", *self.TERMINAL_RUN_STATES}:
            raise StoreTransitionError(f"run '{run_id}' transition to '{state}' is not allowed")
        if error_summary is not None and not isinstance(error_summary, str):
            raise StoreValidationError("error_summary: must be a string or None")
        if state in {"running", "succeeded"} and error_summary is not None:
            raise StoreValidationError(f"error_summary: forbidden for {state} runs")
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise StoreNotFoundError(f"run '{run_id}' not found")
            current = row["state"]
            allowed = current == "queued" and state == "running" or current == "running" and state in self.TERMINAL_RUN_STATES
            if not allowed:
                raise StoreTransitionError(
                    f"run '{run_id}' transition from '{current}' to '{state}' is not allowed"
                )
            now_dt, now_text = self._now()
            if state == "running":
                connection.execute(
                    "UPDATE runs SET state = 'running', started_at = ? WHERE run_id = ?",
                    (now_text, run_id),
                )
            else:
                connection.execute(
                    """UPDATE runs SET state = ?, ended_at = ?, error_summary = ?
                       WHERE run_id = ?""",
                    (state, now_text, error_summary, run_id),
                )
            updated = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_run(updated)  # type: ignore[arg-type]

    @staticmethod
    def _row_file(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            int(row["file_id"]), row["job_id"], row["source_relative_path"], row["source_sha256"],
            int(row["source_length"]), int(row["source_mtime_ns"]), row["destination_relative_path"],
            row["destination_sha256"], row["destination_length"], row["state"], row["publishing_run_id"],
            row["retention_order"], _parse_timestamp(row["created_at"]), _parse_timestamp(row["updated_at"]),
            _parse_timestamp(row["published_at"]), _parse_timestamp(row["pending_delete_at"]),
            _parse_timestamp(row["pruned_at"]),
        )  # type: ignore[arg-type]

    def create_file(
        self,
        job_id: str,
        source_relative_path: str,
        *,
        source_sha256: str,
        source_length: int,
        source_mtime_ns: int,
        destination_relative_path: str | None = None,
        destination_sha256: str | None = None,
        destination_length: int | None = None,
        publishing_run_id: str | None = None,
        retention_order: int | None = None,
    ) -> FileRecord:
        if not isinstance(job_id, str) or not job_id:
            raise StoreValidationError("job_id: must be a non-empty string")
        source_relative_path = _relative_path("source_relative_path", source_relative_path)
        source_sha256 = _sha256("source_sha256", source_sha256)
        source_length = _strict_nonnegative_int("source_length", source_length)
        source_mtime_ns = _strict_nonnegative_int("source_mtime_ns", source_mtime_ns)
        if destination_relative_path is not None:
            destination_relative_path = _relative_path("destination_relative_path", destination_relative_path)
        if destination_sha256 is not None:
            destination_sha256 = _sha256("destination_sha256", destination_sha256)
        if destination_length is not None:
            destination_length = _strict_nonnegative_int("destination_length", destination_length)
        if retention_order is not None:
            retention_order = _strict_nonnegative_int("retention_order", retention_order)
        created_dt, created_at = self._now()
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
                raise StoreNotFoundError(f"job '{job_id}' not found")
            if publishing_run_id is not None:
                self._validate_publishing_run(connection, job_id, publishing_run_id)
            existing = connection.execute(
                """SELECT * FROM files WHERE job_id = ? AND source_relative_path = ?
                   AND source_sha256 = ? AND source_length = ?""",
                (job_id, source_relative_path, source_sha256, source_length),
            ).fetchone()
            if existing is not None:
                return self._row_file(existing)
            connection.execute(
                """INSERT INTO files(
                    job_id, source_relative_path, source_sha256, source_length,
                    source_mtime_ns, destination_relative_path, destination_sha256,
                    destination_length, state, publishing_run_id, retention_order,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'staged', ?, ?, ?, ?)""",
                (job_id, source_relative_path, source_sha256, source_length, source_mtime_ns,
                 destination_relative_path, destination_sha256, destination_length,
                 publishing_run_id, retention_order, created_at, created_at),
            )
            row = connection.execute(
                "SELECT * FROM files WHERE file_id = last_insert_rowid()"
            ).fetchone()
        return self._row_file(row)  # type: ignore[arg-type]

    @staticmethod
    def _validate_publishing_run(connection: sqlite3.Connection, job_id: str, run_id: str) -> None:
        row = connection.execute("SELECT job_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreNotFoundError(f"run '{run_id}' not found")
        if row[0] != job_id:
            raise StoreConflictError(f"run '{run_id}' belongs to a different job")

    def get_file(self, file_id: int) -> FileRecord:
        _strict_positive_int("file_id", file_id)
        row = self._connection.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        if row is None:
            raise StoreNotFoundError(f"file {file_id} not found")
        return self._row_file(row)

    def find_file(
        self, job_id: str, source_relative_path: str, source_sha256: str, source_length: int
    ) -> FileRecord | None:
        if not isinstance(job_id, str) or not job_id:
            raise StoreValidationError("job_id: must be a non-empty string")
        source_relative_path = _relative_path("source_relative_path", source_relative_path)
        source_sha256 = _sha256("source_sha256", source_sha256)
        source_length = _strict_nonnegative_int("source_length", source_length)
        row = self._connection.execute(
            """SELECT * FROM files WHERE job_id = ? AND source_relative_path = ?
               AND source_sha256 = ? AND source_length = ?""",
            (job_id, source_relative_path, source_sha256, source_length),
        ).fetchone()
        return None if row is None else self._row_file(row)

    get_file_by_identity = find_file

    def count_files(self, job_id: str) -> int:
        if not isinstance(job_id, str) or not job_id:
            raise StoreValidationError("job_id: must be a non-empty string")
        row = self._connection.execute("SELECT COUNT(*) FROM files WHERE job_id = ?", (job_id,)).fetchone()
        return int(row[0])

    def transition_file(
        self,
        file_id: int,
        state: str,
        *,
        destination_relative_path: str | None = None,
        destination_sha256: str | None = None,
        destination_length: int | None = None,
        publishing_run_id: str | None = None,
        retention_order: int | None = None,
    ) -> FileRecord:
        _strict_positive_int("file_id", file_id)
        if state not in self.FILE_STATES:
            raise StoreTransitionError(f"file {file_id} transition to '{state}' is not allowed")
        if destination_relative_path is not None:
            destination_relative_path = _relative_path("destination_relative_path", destination_relative_path)
        if destination_sha256 is not None:
            destination_sha256 = _sha256("destination_sha256", destination_sha256)
        if destination_length is not None:
            destination_length = _strict_nonnegative_int("destination_length", destination_length)
        if retention_order is not None:
            retention_order = _strict_nonnegative_int("retention_order", retention_order)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
            if row is None:
                raise StoreNotFoundError(f"file {file_id} not found")
            current = row["state"]
            allowed = {
                "staged": "verified", "verified": "published",
                "published": "pending_delete", "pending_delete": "pruned",
            }.get(current)
            if allowed != state:
                raise StoreTransitionError(
                    f"file {file_id} transition from '{current}' to '{state}' is not allowed"
                )
            if publishing_run_id is not None:
                self._validate_publishing_run(connection, row["job_id"], publishing_run_id)
            _, now_text = self._now()
            updates = ["state = ?", "updated_at = ?"]
            values: list[object] = [state, now_text]
            if destination_relative_path is not None:
                updates.append("destination_relative_path = ?")
                values.append(destination_relative_path)
            if destination_sha256 is not None:
                updates.append("destination_sha256 = ?")
                values.append(destination_sha256)
            if destination_length is not None:
                updates.append("destination_length = ?")
                values.append(destination_length)
            if publishing_run_id is not None:
                updates.append("publishing_run_id = ?")
                values.append(publishing_run_id)
            if retention_order is not None:
                updates.append("retention_order = ?")
                values.append(retention_order)
            timestamp_column = {
                "published": "published_at", "pending_delete": "pending_delete_at", "pruned": "pruned_at"
            }.get(state)
            if timestamp_column is not None:
                updates.append(f"{timestamp_column} = ?")
                values.append(now_text)
            values.append(file_id)
            connection.execute(f"UPDATE files SET {', '.join(updates)} WHERE file_id = ?", values)
            updated = connection.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return self._row_file(updated)  # type: ignore[arg-type]


__all__ = [
    "FileRecord", "JobRevision", "MIGRATIONS", "RunRecord", "Store", "StoreConflictError",
    "StoreError", "StoreMigrationError", "StoreNotFoundError", "StoreTransitionError",
    "StoreValidationError",
]
