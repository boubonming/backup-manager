"""Public mapping-to-model constructors for backup manager configuration."""

from __future__ import annotations

from collections.abc import Mapping

from .models import Job, Schedule, ValidationError

_SCHEDULE_FIELDS = frozenset({"kind", "timezone", "time", "weekdays", "interval_seconds"})
_JOB_FIELDS = frozenset(
    {
        "name",
        "enabled",
        "source_root",
        "destination_root",
        "source_path",
        "destination_path",
        "keep_count",
        "schedule",
    }
)


def _mapping_values(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field}: must be a mapping")
    result = dict(value)
    return result


def schedule_from_mapping(value: object) -> Schedule:
    """Validate and construct a :class:`~backup_manager.models.Schedule` mapping."""

    values = _mapping_values(value, field="schedule")
    unknown = [key for key in values if key not in _SCHEDULE_FIELDS]
    if unknown:
        raise ValidationError(f"schedule: unknown field '{unknown[0]}'")
    has_daily = values.get("time") is not None or values.get("weekdays") is not None
    has_interval = values.get("interval_seconds") is not None
    if has_daily and has_interval:
        raise ValidationError("schedule: daily and interval fields cannot be combined")
    return Schedule(**values)


def job_from_mapping(value: object) -> Job:
    """Validate and construct an immutable :class:`~backup_manager.models.Job`.

    A nested schedule mapping is accepted for configuration-file callers;
    omitted ``enabled`` remains the model's explicit disabled default.
    """

    values = _mapping_values(value, field="job")
    unknown = [key for key in values if key not in _JOB_FIELDS]
    if unknown:
        raise ValidationError(f"job: unknown field '{unknown[0]}'")
    if "schedule" in values and isinstance(values["schedule"], Mapping):
        values["schedule"] = schedule_from_mapping(values["schedule"])
    return Job(**values)


__all__ = ["job_from_mapping", "schedule_from_mapping"]
