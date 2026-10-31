from dataclasses import FrozenInstanceError

import pytest

from backup_manager.config import job_from_mapping, schedule_from_mapping
from backup_manager.models import (
    MIN_INTERVAL_SECONDS,
    Job,
    Schedule,
    ValidationError,
)


def valid_job_kwargs() -> dict[str, object]:
    return {
        "source_root": "palworld_backups",
        "destination_root": "raid",
        "source_path": ".",
        "destination_path": "palworld",
        "keep_count": 100,
        "schedule": Schedule.daily(time="03:00", timezone="Asia/Singapore"),
    }


def test_job_is_disabled_by_default_and_requires_configuration_fields():
    job = Job(**valid_job_kwargs())

    assert job.enabled is False
    with pytest.raises(ValidationError, match=r"^source_root: is required"):
        Job(**{**valid_job_kwargs(), "source_root": None})
    with pytest.raises(ValidationError, match=r"^keep_count: is required"):
        Job(**{**valid_job_kwargs(), "keep_count": None})


@pytest.mark.parametrize(
    "field, value",
    [
        ("source_root", ""),
        ("source_root", "   "),
        ("source_root", "source/child"),
        ("source_root", "../source"),
        ("source_root", "source\x00root"),
        ("source_root", 42),
        ("destination_root", ""),
        ("destination_root", "destination root"),
        ("destination_root", "/destination"),
        ("destination_root", "destination\\root"),
    ],
)
def test_root_aliases_reject_invalid_values(field: str, value: object):
    with pytest.raises(ValidationError, match=rf"^{field}:"):
        Job(**{**valid_job_kwargs(), field: value})


def test_root_aliases_must_be_distinct():
    with pytest.raises(ValidationError, match=r"^destination_root: must differ"):
        Job(**{**valid_job_kwargs(), "destination_root": "palworld_backups"})


@pytest.mark.parametrize("field", ["source_path", "destination_path"])
@pytest.mark.parametrize("value", [None, "", "   ", "/absolute", "../escape", "nested/../../escape", "has\x00nul", "C:\\absolute"])
def test_paths_reject_missing_absolute_blank_nul_and_escape_values(field: str, value: object):
    with pytest.raises(ValidationError, match=rf"^{field}:"):
        Job(**{**valid_job_kwargs(), field: value})


def test_paths_are_lexical_and_accept_relative_posix_examples():
    job = Job(
        **{
            **valid_job_kwargs(),
            "source_path": ".",
            "destination_path": "archives/2026/./daily",
        }
    )

    assert job.source_path == "."
    assert job.destination_path == "archives/2026/daily"


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", object()])
def test_keep_count_requires_an_integer_at_least_one(value: object):
    with pytest.raises(ValidationError, match=r"^keep_count:"):
        Job(**{**valid_job_kwargs(), "keep_count": value})


def test_keep_count_accepts_positive_integers():
    assert Job(**{**valid_job_kwargs(), "keep_count": 1}).keep_count == 1
    assert Job(**{**valid_job_kwargs(), "keep_count": 100}).keep_count == 100


def test_daily_schedule_accepts_explicit_iana_timezone_and_canonical_weekdays():
    schedule = Schedule.daily(
        time="03:00", timezone="Asia/Singapore", weekdays=[7, 1, 3]
    )

    assert schedule.kind == "daily"
    assert schedule.time == "03:00"
    assert schedule.timezone == "Asia/Singapore"
    assert schedule.weekdays == (1, 3, 7)
    assert schedule.interval_seconds is None


@pytest.mark.parametrize("timezone", [None, "", "   ", "SGT", "Not/AZone"])
def test_daily_schedule_rejects_missing_or_unknown_timezone(timezone: object):
    with pytest.raises(ValidationError, match=r"^timezone:"):
        Schedule.daily(time="03:00", timezone=timezone)


@pytest.mark.parametrize("time", [None, "", "3:00", "03:0", "24:00", "12:60", "noon"])
def test_daily_schedule_requires_strict_24_hour_time(time: object):
    with pytest.raises(ValidationError, match=r"^time:"):
        Schedule.daily(time=time, timezone="Asia/Singapore")


@pytest.mark.parametrize("weekdays", [[], [1, 1], [0], [8], ["1"], "123", object()])
def test_daily_schedule_rejects_empty_duplicate_or_invalid_weekdays(weekdays: object):
    with pytest.raises(ValidationError, match=r"^weekdays:"):
        Schedule.daily(time="03:00", timezone="Asia/Singapore", weekdays=weekdays)


def test_daily_schedule_rejects_interval_fields_and_unknown_kind():
    with pytest.raises(ValidationError, match=r"^interval_seconds:"):
        Schedule(kind="daily", timezone="Asia/Singapore", time="03:00", interval_seconds=60)
    with pytest.raises(ValidationError, match=r"^kind:"):
        Schedule(kind="weekly", timezone="Asia/Singapore", time="03:00")


@pytest.mark.parametrize("duration", [MIN_INTERVAL_SECONDS, 3600])
def test_interval_schedule_accepts_documented_positive_durations(duration: int):
    schedule = Schedule.interval(timezone="Asia/Singapore", interval_seconds=duration)

    assert schedule.kind == "interval"
    assert schedule.interval_seconds == duration
    assert schedule.time is None
    assert schedule.weekdays is None


@pytest.mark.parametrize("duration", [None, True, False, 0, -1, 1.5, "60", MIN_INTERVAL_SECONDS - 1])
def test_interval_schedule_rejects_wrong_nonpositive_or_busy_durations(duration: object):
    with pytest.raises(ValidationError, match=r"^interval_seconds:"):
        Schedule.interval(timezone="Asia/Singapore", interval_seconds=duration)


def test_interval_schedule_rejects_daily_fields_and_mixed_mapping():
    with pytest.raises(ValidationError, match=r"^time:"):
        Schedule(kind="interval", timezone="Asia/Singapore", interval_seconds=60, time="03:00")
    with pytest.raises(ValidationError, match=r"^weekdays:"):
        Schedule(kind="interval", timezone="Asia/Singapore", interval_seconds=60, weekdays=[1])
    with pytest.raises(ValidationError, match=r"^schedule:"):
        schedule_from_mapping(
            {
                "kind": "daily",
                "timezone": "Asia/Singapore",
                "time": "03:00",
                "interval_seconds": 60,
            }
        )


def test_mapping_construction_rejects_unknown_fields_with_stable_error():
    with pytest.raises(ValidationError, match=r"^schedule: unknown field 'unexpected'"):
        schedule_from_mapping(
            {
                "kind": "daily",
                "timezone": "Asia/Singapore",
                "time": "03:00",
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError, match=r"^job: unknown field 'unexpected'"):
        job_from_mapping({**valid_job_kwargs(), "unexpected": True})


def test_configuration_values_are_immutable():
    job = Job(**valid_job_kwargs())

    with pytest.raises(FrozenInstanceError):
        job.enabled = True
    with pytest.raises(FrozenInstanceError):
        job.schedule.time = "04:00"


def test_validation_error_is_one_public_value_error_type():
    assert issubclass(ValidationError, ValueError)
    with pytest.raises(ValidationError, match=r"^source_root:"):
        Job(**{**valid_job_kwargs(), "source_root": "bad/root"})
