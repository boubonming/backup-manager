"""Pure, deterministic retention planning for immutable backup objects.

This module consumes captured metadata only.  It never touches a filesystem,
store, clock, network, or credential and never authorizes a later mutation.
Execution must revalidate every proposed transfer or deletion independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import posixpath
import re
from pathlib import PurePosixPath
from typing import Final, TypeVar, cast


class PlannerError(ValueError):
    """Base class for expected planner failures."""


class PlannerValidationError(PlannerError):
    """A public planner input is malformed, unsafe, or ambiguous."""


_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:")
_ISSUE_CODES: Final = frozenset(
    {
        "source_missing",
        "source_unhealthy",
        "source_incomplete",
        "source_issue",
        "duplicate_source",
        "duplicate_published",
        "duplicate_destination",
        "conflicting_identity",
        "destination_collision",
        "unmanaged_collision",
        "tombstone_conflict",
    }
)
_RecordT = TypeVar("_RecordT")


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or not value.strip() or "\x00" in value:
        raise PlannerValidationError(f"{field_name}: must be non-empty text")
    return value


def _relative_path(value: object, field_name: str) -> str:
    value = _text(value, field_name)
    if "\\" in value or _WINDOWS_DRIVE_RE.match(value) or value.startswith("/"):
        raise PlannerValidationError(f"{field_name}: must be a normalized relative POSIX path")
    if value in {".", ".."} or value.endswith("/"):
        raise PlannerValidationError(f"{field_name}: must be a normalized relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PlannerValidationError(f"{field_name}: must be a normalized relative POSIX path")
    normalized = posixpath.normpath(PurePosixPath(value).as_posix())
    if normalized != value or normalized.startswith("../"):
        raise PlannerValidationError(f"{field_name}: must be a normalized relative POSIX path")
    return value


def _sha(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PlannerValidationError(f"{field_name}: must be lowercase 64-hex SHA-256")
    return value


def _integer(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" at least {minimum}" if minimum is not None else ""
        raise PlannerValidationError(f"{field_name}: must be an integer{suffix}")
    return value


def _tuple_input(value: object, field_name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise PlannerValidationError(f"{field_name}: must be an immutable tuple")
    return value


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Captured identity and age metadata for one eligible source file."""

    source_relative_path: str
    source_sha256: str
    source_length: int
    source_mtime_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_relative_path", _relative_path(self.source_relative_path, "source_relative_path"))
        object.__setattr__(self, "source_sha256", _sha(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "source_length", _integer(self.source_length, "source_length", minimum=0))
        object.__setattr__(self, "source_mtime_ns", _integer(self.source_mtime_ns, "source_mtime_ns"))

    @property
    def identity_key(self) -> tuple[str, str, int]:
        """The durable source identity, excluding the age observation."""
        return (self.source_relative_path, self.source_sha256, self.source_length)

    @property
    def fingerprint(self) -> tuple[str, str, int, int]:
        return self.identity_key + (self.source_mtime_ns,)

    @property
    def path(self) -> str:
        return self.source_relative_path

    @property
    def relative_path(self) -> str:
        return self.source_relative_path

    @property
    def sha256(self) -> str:
        return self.source_sha256

    @property
    def byte_length(self) -> int:
        return self.source_length

    @property
    def length(self) -> int:
        return self.source_length

    @property
    def mtime_ns(self) -> int:
        return self.source_mtime_ns


@dataclass(frozen=True, slots=True, init=False)
class Tombstone:
    """A previously pruned source identity that must not be copied again."""

    source_relative_path: str
    source_sha256: str
    source_length: int

    def __init__(self, source_relative_path: object, source_sha256: object = None, source_length: object = None) -> None:
        if isinstance(source_relative_path, SourceIdentity):
            if source_sha256 is not None or source_length is not None:
                raise PlannerValidationError("tombstone: ambiguous identity arguments")
            identity = source_relative_path
            source_relative_path, source_sha256, source_length = identity.identity_key
        object.__setattr__(self, "source_relative_path", _relative_path(source_relative_path, "source_relative_path"))
        object.__setattr__(self, "source_sha256", _sha(source_sha256, "source_sha256"))
        object.__setattr__(self, "source_length", _integer(source_length, "source_length", minimum=0))

    @property
    def identity_key(self) -> tuple[str, str, int]:
        return (self.source_relative_path, self.source_sha256, self.source_length)


@dataclass(frozen=True, slots=True)
class PublishedDestination:
    """Trusted publication metadata, or an explicitly untrusted observation."""

    source_relative_path: str
    source_sha256: str
    source_length: int
    source_mtime_ns: int
    destination_relative_path: str
    destination_sha256: str
    destination_length: int
    trusted: bool = True
    job_owned: bool = True

    def __post_init__(self) -> None:
        source = SourceIdentity(
            self.source_relative_path,
            self.source_sha256,
            self.source_length,
            self.source_mtime_ns,
        )
        object.__setattr__(self, "source_relative_path", source.source_relative_path)
        object.__setattr__(self, "source_sha256", source.source_sha256)
        object.__setattr__(self, "source_length", source.source_length)
        object.__setattr__(self, "source_mtime_ns", source.source_mtime_ns)
        object.__setattr__(self, "destination_relative_path", _relative_path(self.destination_relative_path, "destination_relative_path"))
        object.__setattr__(self, "destination_sha256", _sha(self.destination_sha256, "destination_sha256"))
        object.__setattr__(self, "destination_length", _integer(self.destination_length, "destination_length", minimum=0))
        if type(self.trusted) is not bool or type(self.job_owned) is not bool:
            raise PlannerValidationError("publication ownership flags: must be booleans")

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            self.source_relative_path,
            self.source_sha256,
            self.source_length,
            self.source_mtime_ns,
        )

    @property
    def identity_key(self) -> tuple[str, str, int]:
        return self.identity.identity_key

    @property
    def owned(self) -> bool:
        return self.trusted and self.job_owned

    @property
    def destination_path(self) -> str:
        return self.destination_relative_path


@dataclass(frozen=True, slots=True)
class DestinationEntry:
    """A captured destination object used only for collision classification."""

    destination_relative_path: str
    destination_sha256: str
    destination_length: int
    job_owned: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination_relative_path", _relative_path(self.destination_relative_path, "destination_relative_path"))
        object.__setattr__(self, "destination_sha256", _sha(self.destination_sha256, "destination_sha256"))
        object.__setattr__(self, "destination_length", _integer(self.destination_length, "destination_length", minimum=0))
        if type(self.job_owned) is not bool:
            raise PlannerValidationError("job_owned: must be a boolean")

    @property
    def owned(self) -> bool:
        return self.job_owned

    @property
    def path(self) -> str:
        return self.destination_relative_path

    @property
    def sha256(self) -> str:
        return self.destination_sha256

    @property
    def byte_length(self) -> int:
        return self.destination_length


@dataclass(frozen=True, slots=True)
class PlannerIssue:
    """Stable, non-absolute-path explanation of a blocked observation."""

    code: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not str or self.code not in _ISSUE_CODES:
            raise PlannerValidationError("issue code: unknown category")
        if self.relative_path is not None:
            object.__setattr__(self, "relative_path", _relative_path(self.relative_path, "relative_path"))

    @property
    def reason(self) -> str:
        return self.code

    @property
    def message(self) -> str:
        return self.code.replace("_", " ")


@dataclass(frozen=True, slots=True)
class SkippedItem:
    identity: SourceIdentity
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceIdentity):
            raise PlannerValidationError("skipped identity: must be a SourceIdentity")
        if type(self.reason) is not str or not self.reason or "\x00" in self.reason:
            raise PlannerValidationError("skip reason: must be stable non-empty text")


@dataclass(frozen=True, slots=True)
class TransferAction:
    identity: SourceIdentity
    destination_relative_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceIdentity):
            raise PlannerValidationError("transfer identity: must be a SourceIdentity")
        object.__setattr__(self, "destination_relative_path", _relative_path(self.destination_relative_path, "destination_relative_path"))

    @property
    def path(self) -> str:
        return self.destination_relative_path


@dataclass(frozen=True, slots=True)
class DeleteAction:
    identity: SourceIdentity
    destination_relative_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceIdentity):
            raise PlannerValidationError("delete identity: must be a SourceIdentity")
        object.__setattr__(self, "destination_relative_path", _relative_path(self.destination_relative_path, "destination_relative_path"))

    @property
    def path(self) -> str:
        return self.destination_relative_path


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """Immutable preview; its actions are proposals and never authorization."""

    transfers: tuple[TransferAction, ...]
    deletions: tuple[DeleteAction, ...]
    retained: tuple[SourceIdentity, ...]
    skipped: tuple[SkippedItem, ...]
    issues: tuple[PlannerIssue, ...]
    unmanaged: tuple[DestinationEntry, ...]
    managed_count: int
    managed_bytes: int
    keep_count: int
    blocked: bool
    deletion_authorized: bool = field(default=False, init=False)
    non_authorizing: bool = field(default=True, init=False)
    lower_retention_confirmation_required: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("transfers", self.transfers),
            ("deletions", self.deletions),
            ("retained", self.retained),
            ("skipped", self.skipped),
            ("issues", self.issues),
            ("unmanaged", self.unmanaged),
        ):
            _tuple_input(value, field_name)
        if any(not isinstance(item, TransferAction) for item in self.transfers):
            raise PlannerValidationError("transfers: invalid action")
        if any(not isinstance(item, DeleteAction) for item in self.deletions):
            raise PlannerValidationError("deletions: invalid action")
        if any(not isinstance(item, SourceIdentity) for item in self.retained):
            raise PlannerValidationError("retained: invalid identity")
        if any(not isinstance(item, SkippedItem) for item in self.skipped):
            raise PlannerValidationError("skipped: invalid item")
        if any(not isinstance(item, PlannerIssue) for item in self.issues):
            raise PlannerValidationError("issues: invalid issue")
        if any(not isinstance(item, DestinationEntry) for item in self.unmanaged):
            raise PlannerValidationError("unmanaged: invalid destination entry")
        _integer(self.managed_count, "managed_count", minimum=0)
        _integer(self.managed_bytes, "managed_bytes", minimum=0)
        _integer(self.keep_count, "keep_count", minimum=1)
        if type(self.blocked) is not bool:
            raise PlannerValidationError("blocked: must be a boolean")
        if self.blocked != bool(self.issues):
            raise PlannerValidationError("blocked: must match whether issues are present")
        if self.blocked and (self.transfers or self.deletions):
            raise PlannerValidationError("blocked: cannot contain proposed actions")
        transfer_paths = tuple(action.destination_relative_path for action in self.transfers)
        deletion_paths = tuple(action.destination_relative_path for action in self.deletions)
        if len(transfer_paths) != len(set(transfer_paths)):
            raise PlannerValidationError("transfers: destination paths must be unique")
        if len(deletion_paths) != len(set(deletion_paths)):
            raise PlannerValidationError("deletions: destination paths must be unique")
        if set(transfer_paths) & set(deletion_paths):
            raise PlannerValidationError("actions: transfer and deletion destinations overlap")
        if self.managed_count == 0 and self.managed_bytes != 0:
            raise PlannerValidationError("managed_bytes: inconsistent with managed_count")
        if self.managed_count < len(self.deletions):
            raise PlannerValidationError("managed_count: fewer records than deletions")
        if self.managed_bytes < sum(action.identity.source_length for action in self.deletions):
            raise PlannerValidationError("managed_bytes: fewer bytes than deletions")
        object.__setattr__(
            self,
            "lower_retention_confirmation_required",
            self.managed_count > self.keep_count,
        )

    @property
    def authorizing(self) -> bool:
        return False

    @property
    def authorizes_deletion(self) -> bool:
        return False

    @property
    def retained_set(self) -> tuple[SourceIdentity, ...]:
        return self.retained

    @property
    def deletion_proposals(self) -> tuple[DeleteAction, ...]:
        return self.deletions

    @property
    def transfer_proposals(self) -> tuple[TransferAction, ...]:
        return self.transfers

    @property
    def blockers(self) -> tuple[PlannerIssue, ...]:
        return self.issues

    @property
    def transfer_actions(self) -> tuple[TransferAction, ...]:
        return self.transfers

    @property
    def delete_actions(self) -> tuple[DeleteAction, ...]:
        return self.deletions

    @property
    def retained_identities(self) -> tuple[SourceIdentity, ...]:
        return self.retained

    @property
    def managed_file_count(self) -> int:
        return self.managed_count

    @property
    def managed_byte_count(self) -> int:
        return self.managed_bytes

    @property
    def execution_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SourceObservation:
    """Immutable source snapshot accepted by the planner façade."""

    candidates: tuple[SourceIdentity, ...]
    healthy: bool = True
    present: bool = True
    complete: bool = True
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _tuple_input(self.candidates, "candidates")
        if any(not isinstance(item, SourceIdentity) for item in self.candidates):
            raise PlannerValidationError("candidates: contains an invalid source identity")
        for field_name, value in (
            ("healthy", self.healthy),
            ("present", self.present),
            ("complete", self.complete),
        ):
            if type(value) is not bool:
                raise PlannerValidationError(f"{field_name}: must be a boolean")
        _tuple_input(self.issues, "issues")
        for value in self.issues:
            if type(value) is not str or not value or "\x00" in value or "/" in value:
                raise PlannerValidationError("issues: must contain stable issue codes")
        if not self.present and self.candidates:
            raise PlannerValidationError("candidates: cannot be populated when source is missing")

    @property
    def records(self) -> tuple[SourceIdentity, ...]:
        return self.candidates


# Convenient vocabulary aliases keep the pure API compatible with callers that
# use "record", "file", or "identity" terminology.
SourceRecord = SourceIdentity
EligibleSource = SourceIdentity
SourceFile = SourceIdentity
PublishedRecord = PublishedDestination
PublishedFile = PublishedDestination
PublishedFileRecord = PublishedDestination
PrunedIdentity = Tombstone
RetentionTombstone = Tombstone
DestinationObject = DestinationEntry
DestinationRecord = DestinationEntry
RetentionResult = RetentionPlan
PlannerResult = RetentionPlan
DiscoveryObservation = SourceObservation


def _validate_records(value: object, field_name: str, expected: type[_RecordT]) -> tuple[_RecordT, ...]:
    values = _tuple_input(value, field_name)
    if any(not isinstance(item, expected) for item in values):
        raise PlannerValidationError(f"{field_name}: contains an invalid record")
    return tuple(cast(_RecordT, item) for item in values)


def _issue_key(issue: PlannerIssue) -> tuple[str, str]:
    return (issue.code, issue.relative_path or "")


def _source_sort(identity: SourceIdentity) -> tuple[object, ...]:
    return (-identity.source_mtime_ns, identity.source_relative_path, identity.source_sha256, identity.source_length)


def _identity_sort(identity: SourceIdentity) -> tuple[object, ...]:
    return (identity.source_relative_path, identity.source_sha256, identity.source_length, identity.source_mtime_ns)


def plan_retention(
    keep_count: object,
    source_records: object,
    published_records: object = (),
    tombstones: object = (),
    *,
    destination_entries: object = (),
    source_present: object = True,
    source_healthy: object = True,
    source_complete: object = True,
    source_issues: object = (),
    source_observation: object = None,
) -> RetentionPlan:
    """Build a deterministic, read-only retention preview from captured records."""

    if source_observation is not None:
        if not isinstance(source_observation, SourceObservation):
            raise PlannerValidationError("source_observation: must be a SourceObservation")
        if source_records != () or source_present is not True or source_healthy is not True or source_complete is not True or source_issues != ():
            raise PlannerValidationError("source_observation: ambiguous duplicate source inputs")
        source_records = source_observation.candidates
        source_present = source_observation.present
        source_healthy = source_observation.healthy
        source_complete = source_observation.complete
        source_issues = source_observation.issues
    elif isinstance(source_records, SourceObservation):
        observation = source_records
        if source_present is not True or source_healthy is not True or source_complete is not True or source_issues != ():
            raise PlannerValidationError("source_observation: ambiguous duplicate source inputs")
        source_records = observation.candidates
        source_present = observation.present
        source_healthy = observation.healthy
        source_complete = observation.complete
        source_issues = observation.issues

    keep = _integer(keep_count, "keep_count", minimum=1)
    sources = _validate_records(source_records, "source_records", SourceIdentity)
    published = _validate_records(published_records, "published_records", PublishedDestination)
    tombstone_values = _validate_records(tombstones, "tombstones", Tombstone)
    destinations = _validate_records(destination_entries, "destination_entries", DestinationEntry)
    source_issue_values = _tuple_input(source_issues, "source_issues")
    for value in source_issue_values:
        if type(value) is not str or not value or "\x00" in value or "/" in value:
            raise PlannerValidationError("source_issues: must contain stable issue codes")
    for field_name, value in (
        ("source_present", source_present),
        ("source_healthy", source_healthy),
        ("source_complete", source_complete),
    ):
        if type(value) is not bool:
            raise PlannerValidationError(f"{field_name}: must be a boolean")
    if not source_present and sources:
        raise PlannerValidationError("source_records: cannot be populated when source is missing")

    issues: set[PlannerIssue] = set()
    if not source_present:
        issues.add(PlannerIssue("source_missing"))
    if not source_healthy:
        issues.add(PlannerIssue("source_unhealthy"))
    if not source_complete:
        issues.add(PlannerIssue("source_incomplete"))
    for _ in source_issue_values:
        issues.add(PlannerIssue("source_issue"))

    # Sorting before reduction makes the chosen representative deterministic
    # even when contradictory input records are permuted.
    sources = tuple(sorted(sources, key=_source_sort))
    published = tuple(sorted(published, key=lambda item: (
        item.destination_relative_path,
        item.source_relative_path,
        item.source_sha256,
        item.source_length,
        item.source_mtime_ns,
        item.destination_sha256,
        item.destination_length,
        item.trusted,
        item.job_owned,
    )))
    destinations = tuple(sorted(destinations, key=lambda item: (
        item.destination_relative_path,
        item.destination_sha256,
        item.destination_length,
        item.job_owned,
    )))
    tombstone_keys = {item.identity_key for item in tombstone_values}
    source_paths: dict[str, set[tuple[str, str, int]]] = {}
    for item in sources:
        source_paths.setdefault(item.source_relative_path, set()).add(item.identity_key)
    for path, identities in source_paths.items():
        if len(identities) > 1:
            issues.add(PlannerIssue("destination_collision", path))

    source_by_key: dict[tuple[str, str, int], SourceIdentity] = {}
    seen_source_records: dict[tuple[str, str, int], SourceIdentity] = {}
    tombstoned_sources: list[SourceIdentity] = []
    for item in sources:
        prior = seen_source_records.get(item.identity_key)
        if prior is None:
            seen_source_records[item.identity_key] = item
            if item.identity_key in tombstone_keys:
                tombstoned_sources.append(item)
            else:
                source_by_key[item.identity_key] = item
        elif prior == item:
            # Repeated byte-for-byte observations are one union member.
            continue
        else:
            issues.add(PlannerIssue("conflicting_identity", item.source_relative_path))

    published_by_key: dict[tuple[str, str, int], PublishedDestination] = {}
    published_by_destination: dict[str, PublishedDestination] = {}
    for item in published:
        if item.identity_key in tombstone_keys:
            issues.add(PlannerIssue("tombstone_conflict", item.destination_relative_path))
            continue
        if not item.owned:
            continue
        if (item.destination_sha256, item.destination_length) != (item.source_sha256, item.source_length):
            issues.add(PlannerIssue("conflicting_identity", item.destination_relative_path))
        prior = published_by_key.get(item.identity_key)
        if prior is not None:
            if prior == item:
                # Repeated trusted publication records are one union member.
                continue
            else:
                issues.add(PlannerIssue("conflicting_identity", item.source_relative_path))
            continue
        prior_destination = published_by_destination.get(item.destination_relative_path)
        if prior_destination is not None:
            issues.add(PlannerIssue("destination_collision", item.destination_relative_path))
            continue
        published_by_key[item.identity_key] = item
        published_by_destination[item.destination_relative_path] = item

    unmanaged: dict[str, DestinationEntry] = {}
    destination_by_path: dict[str, DestinationEntry] = {}
    for item in destinations:
        prior = destination_by_path.get(item.destination_relative_path)
        if prior is not None:
            issues.add(PlannerIssue("duplicate_destination", item.destination_relative_path))
            continue
        destination_by_path[item.destination_relative_path] = item
        if not item.job_owned:
            unmanaged[item.destination_relative_path] = item

    for item in published:
        if not item.owned:
            observed = DestinationEntry(item.destination_relative_path, item.destination_sha256, item.destination_length, False)
            prior = destination_by_path.get(observed.destination_relative_path)
            if prior is not None:
                issues.add(PlannerIssue("unmanaged_collision", observed.destination_relative_path))
            else:
                destination_by_path[observed.destination_relative_path] = observed
            unmanaged[observed.destination_relative_path] = observed
        else:
            observed = DestinationEntry(item.destination_relative_path, item.destination_sha256, item.destination_length, True)
            prior = destination_by_path.get(observed.destination_relative_path)
            if prior is not None and not prior.job_owned:
                issues.add(PlannerIssue("unmanaged_collision", observed.destination_relative_path))
            elif prior is not None and (prior.destination_sha256, prior.destination_length) != (observed.destination_sha256, observed.destination_length):
                issues.add(PlannerIssue("destination_collision", observed.destination_relative_path))
            elif prior is None:
                destination_by_path[observed.destination_relative_path] = observed

    # A source path is also the implicit transfer destination.  Compare every
    # source observation with every captured destination before retention
    # selection so an older conflicting source cannot be hidden by keep_count.
    for item in sources:
        existing = destination_by_path.get(item.source_relative_path)
        if existing is not None and (
            existing.destination_sha256,
            existing.destination_length,
        ) != (item.source_sha256, item.source_length):
            issues.add(
                PlannerIssue(
                    "unmanaged_collision" if not existing.job_owned else "destination_collision",
                    item.source_relative_path,
                )
            )
        publication = published_by_destination.get(item.source_relative_path)
        if publication is not None and publication.identity_key != item.identity_key:
            issues.add(PlannerIssue("destination_collision", item.source_relative_path))

    # A source and a trusted publication with one durable identity are one union
    # member.  A changed age observation is ambiguity, not an invitation to guess.
    union: dict[tuple[str, str, int], SourceIdentity] = dict(source_by_key)
    for key, record in published_by_key.items():
        source = source_by_key.get(key)
        if source is not None and source.source_mtime_ns != record.source_mtime_ns:
            issues.add(PlannerIssue("conflicting_identity", source.source_relative_path))
        union[key] = source or record.identity

    ordered = tuple(sorted(union.values(), key=_source_sort))
    retained = ordered[:keep]
    retained_keys = {item.identity_key for item in retained}
    skipped = [
        SkippedItem(item, "outside_retained_set")
        for item in sorted(source_by_key.values(), key=_identity_sort)
        if item.identity_key not in retained_keys
    ]
    skipped.extend(SkippedItem(item, "tombstoned") for item in sorted(tombstoned_sources, key=_identity_sort))

    transfers: list[TransferAction] = []
    collision_codes = {"duplicate_source", "duplicate_published", "duplicate_destination", "conflicting_identity", "destination_collision", "unmanaged_collision", "tombstone_conflict"}
    has_collision = any(issue.code in collision_codes for issue in issues)
    for item in retained:
        if item.identity_key in published_by_key:
            publication = published_by_key[item.identity_key]
            if publication.destination_relative_path != item.source_relative_path:
                # A published destination name is authoritative for an existing
                # item; the source identity itself does not imply a rename.
                continue
            continue
        destination_path = item.source_relative_path
        existing = destination_by_path.get(destination_path)
        if existing is not None:
            if existing.destination_sha256 != item.source_sha256 or existing.destination_length != item.source_length:
                issues.add(PlannerIssue("unmanaged_collision" if not existing.job_owned else "destination_collision", destination_path))
            else:
                issues.add(PlannerIssue("destination_collision", destination_path))
            continue
        if not has_collision:
            transfers.append(TransferAction(item, destination_path))

    managed_records = tuple(sorted(published_by_key.values(), key=lambda item: (item.destination_relative_path, item.identity_key)))
    managed_count = len(managed_records)
    managed_bytes = sum(item.destination_length for item in managed_records)
    deletion_gate = bool(source_present and source_healthy and source_complete and source_by_key and not source_issue_values)
    all_issues = tuple(sorted(issues, key=_issue_key))
    if any(issue.code in collision_codes for issue in all_issues):
        transfers = []
    deletions: list[DeleteAction] = []
    if deletion_gate and not all_issues:
        for item in managed_records:
            if item.identity_key not in retained_keys and item.identity_key not in tombstone_keys:
                deletions.append(DeleteAction(item.identity, item.destination_relative_path))
    skipped.sort(key=lambda item: (item.reason, _identity_sort(item.identity)))
    unmanaged_values = tuple(sorted(unmanaged.values(), key=lambda item: (item.destination_relative_path, item.destination_sha256, item.destination_length)))
    return RetentionPlan(
        tuple(sorted(transfers, key=lambda action: _identity_sort(action.identity))),
        tuple(sorted(deletions, key=lambda action: (action.destination_relative_path, _identity_sort(action.identity)))),
        retained,
        tuple(skipped),
        all_issues,
        unmanaged_values,
        managed_count,
        managed_bytes,
        keep,
        bool(all_issues),
    )


# Alternate spelling used by preview-oriented callers.
build_retention_plan = plan_retention
preview_retention = plan_retention


class RetentionPlanner:
    """Stateless façade around :func:`plan_retention`."""

    @staticmethod
    def plan(*args: object, **kwargs: object) -> RetentionPlan:
        return plan_retention(*args, **kwargs)


__all__ = [
    "DeleteAction",
    "DiscoveryObservation",
    "DestinationEntry",
    "DestinationObject",
    "DestinationRecord",
    "EligibleSource",
    "PlannerError",
    "PlannerIssue",
    "PlannerValidationError",
    "PublishedDestination",
    "PublishedFile",
    "PublishedFileRecord",
    "PublishedRecord",
    "PrunedIdentity",
    "RetentionTombstone",
    "PlannerResult",
    "RetentionPlan",
    "RetentionPlanner",
    "RetentionResult",
    "SkippedItem",
    "SourceFile",
    "SourceIdentity",
    "SourceObservation",
    "SourceRecord",
    "Tombstone",
    "TransferAction",
    "build_retention_plan",
    "plan_retention",
    "preview_retention",
]
