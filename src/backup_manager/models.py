"""Immutable, validated configuration models for backup jobs.

The models in this module validate configuration shape only.  They do not touch
configured roots, resolve paths, or calculate schedule occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import posixpath
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ValidationError(ValueError):
    """Raised when a job or schedule contains invalid configuration.

    Messages begin with the invalid field name (for example,
    ``"keep_count: must be an integer"``) so callers can display errors next
to the corresponding configuration control.
    """


MIN_INTERVAL_SECONDS = 60
_MISSING = object()
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_TIME_RE = re.compile(r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _validate_alias(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field}: must be a non-empty alias")
    if not value or value != value.strip() or not _ALIAS_RE.fullmatch(value):
        raise ValidationError(
            f"{field}: must contain only letters, digits, '_' or '-' and start with a letter or digit"
        )
    return value


def _validate_relative_path(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field}: must be a relative POSIX path")
    if not value or not value.strip():
        raise ValidationError(f"{field}: must be a non-empty relative POSIX path")
    if "\x00" in value:
        raise ValidationError(f"{field}: must not contain NUL")
    if "\\" in value or _WINDOWS_DRIVE_RE.match(value):
        raise ValidationError(f"{field}: must use POSIX path separators")

    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValidationError(f"{field}: must be relative")
    if any(component == ".." for component in value.split("/")):
        raise ValidationError(f"{field}: must not contain '..' traversal components")

    normalized = posixpath.normpath(value)
    if normalized == ".." or normalized.startswith("../"):
        raise ValidationError(f"{field}: normalized path escapes its configured root")
    return path.as_posix()


def _validate_timezone(value: object) -> str:
    field = "timezone"
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValidationError(f"{field}: must be an explicit IANA timezone")
    # ZoneInfo contains a few short aliases.  Require a region-style IANA key,
    # while retaining UTC as the standard global-zone special case.
    if "/" not in value and value != "UTC":
        raise ValidationError(f"{field}: must be an IANA region timezone")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValidationError(f"{field}: unknown IANA timezone") from None
    return value


def _validate_weekdays(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValidationError("weekdays: must be a non-empty list or set of ISO weekdays")
    if not value:
        raise ValidationError("weekdays: must not be empty when provided")
    result: list[int] = []
    for weekday in value:
        if isinstance(weekday, bool) or not isinstance(weekday, int):
            raise ValidationError("weekdays: each value must be an integer from 1 through 7")
        if not 1 <= weekday <= 7:
            raise ValidationError("weekdays: each value must be an integer from 1 through 7")
        result.append(weekday)
    if len(result) != len(set(result)):
        raise ValidationError("weekdays: values must be unique")
    return tuple(sorted(result))


def _validate_schedule_field_values(
    *,
    kind: object,
    timezone: object,
    time: object,
    weekdays: object,
    interval_seconds: object,
) -> tuple[str, str, str | None, tuple[int, ...] | None, int | None]:
    if not isinstance(kind, str) or not kind:
        raise ValidationError("kind: must be 'daily' or 'interval'")
    if kind not in {"daily", "interval"}:
        raise ValidationError("kind: must be 'daily' or 'interval'")
    checked_timezone = _validate_timezone(timezone)

    if kind == "daily":
        if interval_seconds is not None:
            raise ValidationError("interval_seconds: is only valid for interval schedules")
        if not isinstance(time, str) or not _TIME_RE.fullmatch(time):
            raise ValidationError("time: must use strict zero-padded HH:MM 24-hour format")
        checked_weekdays = _validate_weekdays(weekdays)
        return kind, checked_timezone, time, checked_weekdays, None

    if time is not None:
        raise ValidationError("time: is only valid for daily schedules")
    if weekdays is not None:
        raise ValidationError("weekdays: is only valid for daily schedules")
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int)
        or interval_seconds < MIN_INTERVAL_SECONDS
    ):
        raise ValidationError(
            f"interval_seconds: must be an integer of at least {MIN_INTERVAL_SECONDS}"
        )
    return kind, checked_timezone, None, None, interval_seconds


@dataclass(frozen=True, slots=True, init=False)
class Schedule:
    """An immutable v1 daily or interval schedule.

    Daily schedules use ISO weekdays (Monday=1 through Sunday=7), stored as a
    sorted tuple.  Interval schedules use ``interval_seconds`` with a minimum
    of :data:`MIN_INTERVAL_SECONDS`.  Every schedule requires an IANA timezone.
    """

    kind: str
    timezone: str
    time: str | None
    weekdays: tuple[int, ...] | None
    interval_seconds: int | None

    def __init__(
        self,
        kind: object = _MISSING,
        timezone: object = _MISSING,
        time: object = None,
        weekdays: object = None,
        interval_seconds: object = None,
        **extra: object,
    ) -> None:
        if extra:
            field = next(iter(extra))
            raise ValidationError(f"schedule: unknown field '{field}'")
        if kind is _MISSING:
            raise ValidationError("kind: is required")
        if timezone is _MISSING:
            raise ValidationError("timezone: is required")
        values = _validate_schedule_field_values(
            kind=kind,
            timezone=timezone,
            time=time,
            weekdays=weekdays,
            interval_seconds=interval_seconds,
        )
        object.__setattr__(self, "kind", values[0])
        object.__setattr__(self, "timezone", values[1])
        object.__setattr__(self, "time", values[2])
        object.__setattr__(self, "weekdays", values[3])
        object.__setattr__(self, "interval_seconds", values[4])

    @classmethod
    def daily(
        cls,
        *,
        time: object = _MISSING,
        timezone: object = _MISSING,
        weekdays: object = None,
    ) -> "Schedule":
        if time is _MISSING:
            raise ValidationError("time: is required for daily schedules")
        if timezone is _MISSING:
            raise ValidationError("timezone: is required")
        return cls(kind="daily", timezone=timezone, time=time, weekdays=weekdays)

    @classmethod
    def interval(
        cls,
        *,
        timezone: object = _MISSING,
        interval_seconds: object = _MISSING,
    ) -> "Schedule":
        if timezone is _MISSING:
            raise ValidationError("timezone: is required")
        if interval_seconds is _MISSING:
            raise ValidationError("interval_seconds: is required for interval schedules")
        return cls(
            kind="interval",
            timezone=timezone,
            interval_seconds=interval_seconds,
        )


@dataclass(frozen=True, slots=True, init=False)
class Job:
    """An immutable backup job configuration with lexical path validation."""

    source_root: str
    destination_root: str
    source_path: str
    destination_path: str
    keep_count: int
    schedule: Schedule
    enabled: bool
    name: str | None

    def __init__(
        self,
        source_root: object = _MISSING,
        destination_root: object = _MISSING,
        source_path: object = _MISSING,
        destination_path: object = _MISSING,
        keep_count: object = _MISSING,
        schedule: object = _MISSING,
        enabled: object = False,
        name: object = None,
        **extra: object,
    ) -> None:
        if extra:
            field = next(iter(extra))
            raise ValidationError(f"job: unknown field '{field}'")
        required = {
            "source_root": source_root,
            "destination_root": destination_root,
            "source_path": source_path,
            "destination_path": destination_path,
            "keep_count": keep_count,
            "schedule": schedule,
        }
        for field, value in required.items():
            if value is _MISSING or value is None:
                raise ValidationError(f"{field}: is required")

        checked_source_root = _validate_alias("source_root", source_root)
        checked_destination_root = _validate_alias("destination_root", destination_root)
        if checked_source_root == checked_destination_root:
            raise ValidationError("destination_root: must differ from source_root")
        checked_source_path = _validate_relative_path("source_path", source_path)
        checked_destination_path = _validate_relative_path(
            "destination_path", destination_path
        )
        if isinstance(keep_count, bool) or not isinstance(keep_count, int):
            raise ValidationError("keep_count: must be an integer")
        if keep_count < 1:
            raise ValidationError("keep_count: must be at least 1")
        if not isinstance(schedule, Schedule):
            raise ValidationError("schedule: must be a Schedule")
        if not isinstance(enabled, bool):
            raise ValidationError("enabled: must be a boolean")
        if name is not None and (
            not isinstance(name, str) or not name or not name.strip()
        ):
            raise ValidationError("name: must be a non-empty string when provided")

        object.__setattr__(self, "source_root", checked_source_root)
        object.__setattr__(self, "destination_root", checked_destination_root)
        object.__setattr__(self, "source_path", checked_source_path)
        object.__setattr__(self, "destination_path", checked_destination_path)
        object.__setattr__(self, "keep_count", keep_count)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "name", name)


# Explicit aliases keep the public API discoverable for callers that prefer a
# configuration-oriented name while retaining the short model names.
JobConfig = Job
ScheduleConfig = Schedule

__all__ = [
    "Job",
    "JobConfig",
    "MIN_INTERVAL_SECONDS",
    "Schedule",
    "ScheduleConfig",
    "ValidationError",
]
