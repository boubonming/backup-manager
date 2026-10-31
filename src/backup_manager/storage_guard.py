"""Fail-closed identity validation for an approved Linux storage root.

The marker is deliberately not provisioned here.  A deployment operator must
create the fixed ``.backup-manager-storage.json`` file on the destination
filesystem.  Validation is a point-in-time, non-authorizing observation; callers
must call :func:`revalidate_storage` immediately before any future mutation.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

from .paths import FinalObjectPolicy, PathSafetyError, open_trusted_root

MARKER_FILENAME: Final = ".backup-manager-storage.json"
STORAGE_MARKER_FILENAME: Final = MARKER_FILENAME
MARKER_VERSION: Final = 1
MARKER_MAX_BYTES: Final = 256


class StorageGuardError(ValueError):
    """Base class for every expected storage-identity guard failure."""


class StorageConfigError(StorageGuardError):
    """The expected identity configuration is invalid."""


class StorageMarkerError(StorageGuardError):
    """The dedicated destination marker is missing or invalid."""


class StorageIdentityError(StorageGuardError):
    """Kernel-observed filesystem, device, or mount identity is unusable."""


class StorageRevalidationError(StorageIdentityError):
    """A later point-in-time observation differs from its prior snapshot."""


class StorageIOError(StorageGuardError):
    """An I/O or descriptor operation prevented a safe decision."""


@dataclass(frozen=True, slots=True)
class MountIdentity:
    """A Linux kernel device major/minor pair."""

    major: int
    minor: int

    def __post_init__(self) -> None:
        _validate_int(self.major, "device major")
        _validate_int(self.minor, "device minor")


@dataclass(frozen=True, slots=True)
class StorageIdentityConfig:
    """Strict, immutable expectations for exactly one destination root."""

    root: str
    marker_id: str
    filesystem_type: str
    device_major: int
    device_minor: int
    mount_id: int | None = None

    def __post_init__(self) -> None:
        _validate_root(self.root)
        _validate_text(self.marker_id, "marker_id")
        _validate_text(self.filesystem_type, "filesystem_type")
        _validate_int(self.device_major, "device_major")
        _validate_int(self.device_minor, "device_minor")
        if self.mount_id is not None:
            _validate_int(self.mount_id, "mount_id", positive=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StorageIdentityConfig:
        """Build a configuration without inferring, ignoring, or correcting keys."""

        if not isinstance(value, Mapping):
            raise StorageConfigError("config: expected a mapping")
        required = {"root", "marker_id", "filesystem_type", "device_major", "device_minor"}
        optional = {"mount_id"}
        unknown = set(value) - required - optional
        missing = required - set(value)
        if unknown:
            raise StorageConfigError("config: unknown field")
        if missing:
            raise StorageConfigError("config: missing field")
        try:
            return cls(
                root=value["root"],  # type: ignore[arg-type]
                marker_id=value["marker_id"],  # type: ignore[arg-type]
                filesystem_type=value["filesystem_type"],  # type: ignore[arg-type]
                device_major=value["device_major"],  # type: ignore[arg-type]
                device_minor=value["device_minor"],  # type: ignore[arg-type]
                mount_id=value.get("mount_id"),  # type: ignore[arg-type]
            )
        except StorageGuardError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageConfigError("config: invalid field") from None


@dataclass(frozen=True, slots=True)
class MountInfoRecord:
    """One strictly parsed row from Linux ``mountinfo``."""

    mount_id: int
    parent_id: int
    device: MountIdentity
    root: str
    mount_point: str
    filesystem_type: str
    source: str
    super_options: str


@dataclass(frozen=True, slots=True)
class ObservedStorageIdentity:
    """Identity observed from a held root descriptor and the kernel."""

    root_device: int
    root_inode: int
    filesystem_type: str
    device: MountIdentity
    mount_id: int
    mount_point: str
    marker_device: int | None = None
    marker_inode: int | None = None
    marker_size: int | None = None


class StorageIdentityAdapter(Protocol):
    """Small injectable seam used by unit tests and the Linux implementation."""

    def observe(self, root_fd: int, root_path: str) -> ObservedStorageIdentity:
        ...


@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    """Point-in-time and explicitly non-authorizing storage observation.

    This record is evidence about one observation only.  It is never durable
    permission to publish or delete files, and must be passed to
    :func:`revalidate_storage` before a later mutation.
    """

    root: str
    marker_id: str
    marker_version: int
    marker_device: int
    marker_inode: int
    marker_size: int
    root_device: int
    root_inode: int
    filesystem_type: str
    device: MountIdentity
    mount_id: int
    mount_point: str
    point_in_time: bool = field(default=True, init=False)
    non_authorizing: bool = field(default=True, init=False)
    authorizing: bool = field(default=False, init=False)

    @property
    def identity_key(self) -> tuple[object, ...]:
        return (
            self.root,
            self.marker_id,
            self.marker_version,
            self.marker_device,
            self.marker_inode,
            self.marker_size,
            self.root_device,
            self.root_inode,
            self.filesystem_type,
            self.device,
            self.mount_id,
            self.mount_point,
        )


class LinuxStorageIdentityAdapter:
    """Observe a root's device and selected mountinfo record using stdlib only."""

    def __init__(self, mountinfo_path: str = "/proc/self/mountinfo") -> None:
        if not isinstance(mountinfo_path, str) or not mountinfo_path or "\x00" in mountinfo_path:
            raise StorageConfigError("mountinfo: invalid path")
        self.mountinfo_path = mountinfo_path

    def observe(self, root_fd: int, root_path: str) -> ObservedStorageIdentity:
        try:
            root_stat = os.fstat(root_fd)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise StorageIdentityError("identity: root descriptor is not a directory")
            with open(self.mountinfo_path, "rb") as stream:
                mountinfo = stream.read()
        except StorageGuardError:
            raise
        except (OSError, UnicodeError) as exc:
            raise StorageIOError("storage I/O failure while reading mount identity") from None
        records = parse_mountinfo(mountinfo)
        record = _select_mount(records, root_path)
        actual_device = MountIdentity(os.major(root_stat.st_dev), os.minor(root_stat.st_dev))
        if actual_device != record.device:
            raise StorageIdentityError("identity: root device disagrees with mount record")
        return ObservedStorageIdentity(
            root_device=root_stat.st_dev,
            root_inode=root_stat.st_ino,
            filesystem_type=record.filesystem_type,
            device=record.device,
            mount_id=record.mount_id,
            mount_point=record.mount_point,
        )


class StorageGuard:
    """Validate and revalidate one configured storage identity."""

    def __init__(self, config: StorageIdentityConfig, adapter: StorageIdentityAdapter | None = None) -> None:
        if not isinstance(config, StorageIdentityConfig):
            raise StorageConfigError("config: expected StorageIdentityConfig")
        self.config = config
        self.adapter = adapter if adapter is not None else LinuxStorageIdentityAdapter()

    def validate(self) -> StorageSnapshot:
        """Return a point-in-time observation, never an authorization token."""

        try:
            with open_trusted_root(self.config.root) as trusted:
                root_stat = os.fstat(trusted.fd)
                marker_id, marker_version, marker_stat = _read_marker(trusted, self.config.marker_id)
                if marker_stat.st_dev != root_stat.st_dev:
                    raise StorageIdentityError("identity: marker device differs from root device")
                if os.major(root_stat.st_dev) != self.config.device_major or os.minor(root_stat.st_dev) != self.config.device_minor:
                    raise StorageIdentityError("identity: root device mismatch")
                observed = self.adapter.observe(trusted.fd, self.config.root)
                _check_observation(self.config, root_stat, observed)
                return StorageSnapshot(
                    root=self.config.root,
                    marker_id=marker_id,
                    marker_version=marker_version,
                    marker_device=marker_stat.st_dev,
                    marker_inode=marker_stat.st_ino,
                    marker_size=marker_stat.st_size,
                    root_device=root_stat.st_dev,
                    root_inode=root_stat.st_ino,
                    filesystem_type=observed.filesystem_type,
                    device=observed.device,
                    mount_id=observed.mount_id,
                    mount_point=observed.mount_point,
                )
        except StorageGuardError:
            raise
        except PathSafetyError as exc:
            raise StorageIOError("root: safe descriptor access failed") from None
        except Exception as exc:
            raise StorageIOError("storage identity observation failed") from None

    def revalidate(self, prior: StorageSnapshot) -> StorageSnapshot:
        """Repeat validation and reject every identity-relevant change."""

        if not isinstance(prior, StorageSnapshot) or prior.root != self.config.root:
            raise StorageRevalidationError("revalidation: invalid prior snapshot")
        try:
            current = self.validate()
        except StorageMarkerError:
            raise
        except StorageGuardError:
            raise StorageRevalidationError("revalidation: storage identity changed") from None
        if current.identity_key != prior.identity_key:
            raise StorageRevalidationError("revalidation: storage identity changed")
        return current


def validate_storage(
    config: StorageIdentityConfig, adapter: StorageIdentityAdapter | None = None
) -> StorageSnapshot:
    """Convenience wrapper for :meth:`StorageGuard.validate`."""

    return StorageGuard(config, adapter).validate()


def revalidate_storage(
    config: StorageIdentityConfig,
    prior: StorageSnapshot,
    adapter: StorageIdentityAdapter | None = None,
) -> StorageSnapshot:
    """Convenience wrapper for :meth:`StorageGuard.revalidate`."""

    return StorageGuard(config, adapter).revalidate(prior)


def parse_mountinfo(data: str | bytes) -> tuple[MountInfoRecord, ...]:
    """Strictly parse Linux mountinfo, decoding its four octal escapes."""

    try:
        text = data.decode("utf-8", "strict") if isinstance(data, bytes) else data
        if not isinstance(text, str) or not text or not text.endswith("\n"):
            raise ValueError
        records: list[MountInfoRecord] = []
        seen_ids: set[int] = set()
        for line in text[:-1].split("\n"):
            if not line or "\t" in line or "\r" in line or " - " not in line or line.count(" - ") != 1:
                raise ValueError
            left_text, right_text = line.split(" - ", 1)
            left = left_text.split(" ")
            right = right_text.split(" ")
            if len(left) < 6 or len(right) != 3 or "" in left or "" in right:
                raise ValueError
            mount_id = _decimal(left[0], positive=True)
            parent_id = _decimal(left[1], positive=True)
            major_text, separator, minor_text = left[2].partition(":")
            if not separator or ":" in minor_text:
                raise ValueError
            device = MountIdentity(_decimal(major_text), _decimal(minor_text))
            root = _decode_mountinfo(left[3])
            mount_point = _decode_mountinfo(left[4])
            _decode_mountinfo(left[5])
            for optional_field in left[6:]:
                _validate_optional_field(optional_field)
            filesystem_type = _decode_mountinfo(right[0])
            source = _decode_mountinfo(right[1])
            super_options = _decode_mountinfo(right[2])
            if (
                not _is_absolute_normalized_path(root)
                or not _is_absolute_normalized_path(mount_point)
                or not filesystem_type
                or not filesystem_type.strip()
            ):
                raise ValueError
            if mount_id in seen_ids:
                raise ValueError
            seen_ids.add(mount_id)
            records.append(
                MountInfoRecord(
                    mount_id,
                    parent_id,
                    device,
                    root,
                    mount_point,
                    filesystem_type,
                    source,
                    super_options,
                )
            )
        if not records:
            raise ValueError
        return tuple(records)
    except StorageGuardError:
        raise
    except (UnicodeError, TypeError, ValueError, IndexError) as exc:
        raise StorageIdentityError("identity: malformed mountinfo") from None


def _read_marker(trusted: object, expected_id: str) -> tuple[str, int, os.stat_result]:
    try:
        resolution = trusted.resolve(MARKER_FILENAME, FinalObjectPolicy.REGULAR_FILE)  # type: ignore[attr-defined]
    except PathSafetyError as exc:
        raise StorageMarkerError("marker: unavailable or not a regular file") from None
    try:
        if resolution.fd is None:
            raise StorageMarkerError("marker: unavailable")
        marker_stat = os.fstat(resolution.fd)
        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_size > MARKER_MAX_BYTES:
            raise StorageMarkerError("marker: invalid file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(resolution.fd, MARKER_MAX_BYTES + 1)
            if not chunk:
                break
            total += len(chunk)
            if total > MARKER_MAX_BYTES:
                raise StorageMarkerError("marker: oversized")
            chunks.append(chunk)
        after_stat = os.fstat(resolution.fd)
        if (
            after_stat.st_dev != marker_stat.st_dev
            or after_stat.st_ino != marker_stat.st_ino
            or after_stat.st_size != marker_stat.st_size
        ):
            raise StorageMarkerError("marker: changed while being read")
        try:
            raw = b"".join(chunks).decode("utf-8", "strict")
            value = json.loads(raw, object_pairs_hook=_unique_pairs)
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            raise StorageMarkerError("marker: malformed strict JSON") from None
        if not isinstance(value, dict) or set(value) != {"version", "marker_id"}:
            raise StorageMarkerError("marker: wrong schema")
        if type(value["version"]) is not int or value["version"] != MARKER_VERSION:
            raise StorageMarkerError("marker: wrong version")
        if not isinstance(value["marker_id"], str) or value["marker_id"] != expected_id:
            raise StorageMarkerError("marker: wrong identity")
        return value["marker_id"], value["version"], marker_stat
    except StorageGuardError:
        raise
    except (OSError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise StorageIOError("storage I/O failure while reading marker") from None
    finally:
        try:
            resolution.close()
        except OSError:
            raise StorageIOError("storage I/O failure while closing marker") from None


def _check_observation(config: StorageIdentityConfig, root_stat: os.stat_result, observed: ObservedStorageIdentity) -> None:
    if type(observed) is not ObservedStorageIdentity:
        raise StorageIdentityError("identity: adapter returned invalid data")
    _validate_observed_identity(observed)
    if observed.root_device != root_stat.st_dev or observed.root_inode != root_stat.st_ino:
        raise StorageIdentityError("identity: root descriptor changed")
    if os.major(root_stat.st_dev) != config.device_major or os.minor(root_stat.st_dev) != config.device_minor:
        raise StorageIdentityError("identity: root device mismatch")
    if observed.device != MountIdentity(config.device_major, config.device_minor):
        raise StorageIdentityError("identity: mount device mismatch")
    if observed.marker_device is not None and observed.marker_device != root_stat.st_dev:
        raise StorageIdentityError("identity: marker device mismatch")
    if observed.filesystem_type != config.filesystem_type:
        raise StorageIdentityError("identity: filesystem type mismatch")
    if config.mount_id is not None and observed.mount_id != config.mount_id:
        raise StorageIdentityError("identity: mount identity mismatch")
    if not _under_mount(config.root, observed.mount_point):
        raise StorageIdentityError("identity: mount point does not contain configured root")


def _validate_observed_identity(observed: ObservedStorageIdentity) -> None:
    _validate_observed_int(observed.root_device, "root device", positive=False)
    _validate_observed_int(observed.root_inode, "root inode", positive=True)
    if type(observed.device) is not MountIdentity:
        raise StorageIdentityError("identity: invalid mount device")
    _validate_observed_int(observed.device.major, "device major", positive=False)
    _validate_observed_int(observed.device.minor, "device minor", positive=False)
    _validate_observed_text(observed.filesystem_type, "filesystem type")
    _validate_observed_int(observed.mount_id, "mount ID", positive=True)
    if type(observed.mount_point) is not str or not _is_absolute_normalized_path(observed.mount_point):
        raise StorageIdentityError("identity: invalid mount point")
    if observed.marker_device is not None:
        _validate_observed_int(observed.marker_device, "marker device", positive=False)
    if observed.marker_inode is not None:
        _validate_observed_int(observed.marker_inode, "marker inode", positive=True)
    if observed.marker_size is not None:
        _validate_observed_int(observed.marker_size, "marker size", positive=False)


def _validate_observed_int(value: object, field_name: str, *, positive: bool) -> None:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        raise StorageIdentityError(f"identity: invalid {field_name}")


def _validate_observed_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or not value.strip() or "\x00" in value:
        raise StorageIdentityError(f"identity: invalid {field_name}")


def _is_absolute_normalized_path(value: str) -> bool:
    return (
        type(value) is str
        and value.startswith("/")
        and value == os.path.normpath(value)
        and "//" not in value
        and "\x00" not in value
        and not any(part in {".", ".."} for part in value.split("/"))
    )


def _select_mount(records: tuple[MountInfoRecord, ...], root_path: str) -> MountInfoRecord:
    matches = [record for record in records if _under_mount(root_path, record.mount_point)]
    if not matches:
        raise StorageIdentityError("identity: no matching mount record")
    longest = max(len(record.mount_point) for record in matches)
    selected = [record for record in matches if len(record.mount_point) == longest]
    if len(selected) != 1:
        raise StorageIdentityError("identity: ambiguous mount records")
    return selected[0]


def _under_mount(path: str, mount_point: str) -> bool:
    if mount_point == "/":
        return path.startswith("/")
    return path == mount_point or path.startswith(mount_point.rstrip("/") + "/")


def _decode_mountinfo(value: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            output.append(value[index])
            index += 1
            continue
        code = value[index + 1 : index + 4]
        if len(code) != 3 or code not in {"040", "011", "012", "134"}:
            raise ValueError
        output.append({"040": " ", "011": "\t", "012": "\n", "134": "\\"}[code])
        index += 4
    decoded = "".join(output)
    if "\x00" in decoded:
        raise ValueError
    return decoded


def _validate_optional_field(value: str) -> None:
    if not value or value.count(":") > 1:
        raise ValueError
    if ":" in value:
        tag, field_value = value.split(":", 1)
        if not _valid_mountinfo_component(tag) or not _valid_mountinfo_component(field_value):
            raise ValueError
    elif not _valid_mountinfo_component(value):
        raise ValueError


def _valid_mountinfo_component(value: str) -> bool:
    decoded = _decode_mountinfo(value)
    return bool(decoded)


def _unique_pairs(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _validate_root(value: object) -> None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise StorageConfigError("root: must be an explicit absolute path")
    if not value.startswith("/") or value != os.path.normpath(value) or "//" in value:
        raise StorageConfigError("root: must be an unambiguous absolute path")
    if any(part in {".", ".."} for part in value.split("/")):
        raise StorageConfigError("root: ambiguous path components")


def _validate_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value or not value.strip() or "\x00" in value:
        raise StorageConfigError(f"{field_name}: must be a non-empty string")


def _validate_int(value: object, field_name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise StorageConfigError(f"{field_name}: must be a non-negative integer")


def _decimal(value: str, *, positive: bool = False) -> int:
    if not value or not value.isascii() or not value.isdecimal() or (len(value) > 1 and value[0] == "0"):
        raise ValueError
    result = int(value)
    if positive and result < 1:
        raise ValueError
    return result


__all__ = [
    "MARKER_FILENAME",
    "STORAGE_MARKER_FILENAME",
    "MARKER_MAX_BYTES",
    "MARKER_VERSION",
    "LinuxStorageIdentityAdapter",
    "MountIdentity",
    "MountInfoRecord",
    "ObservedStorageIdentity",
    "StorageConfigError",
    "StorageGuard",
    "StorageGuardError",
    "StorageIOError",
    "StorageIdentityAdapter",
    "StorageIdentityConfig",
    "StorageIdentityError",
    "StorageMarkerError",
    "StorageRevalidationError",
    "StorageSnapshot",
    "parse_mountinfo",
    "revalidate_storage",
    "validate_storage",
]
