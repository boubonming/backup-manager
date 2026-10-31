from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backup_manager.storage_guard import (
    MARKER_FILENAME,
    MARKER_MAX_BYTES,
    MountIdentity,
    ObservedStorageIdentity,
    StorageConfigError,
    StorageGuard,
    StorageGuardError,
    StorageIdentityError,
    StorageIdentityConfig,
    StorageIOError,
    StorageMarkerError,
    StorageRevalidationError,
    parse_mountinfo,
)


class FakeAdapter:
    def __init__(self, observed: ObservedStorageIdentity) -> None:
        self.observed = observed
        self.error: BaseException | None = None

    def observe(self, root_fd: int, root_path: str) -> ObservedStorageIdentity:
        if self.error is not None:
            raise self.error
        return self.observed


def _setup(tmp_path: Path, marker: object = None) -> tuple[Path, StorageIdentityConfig, FakeAdapter]:
    root = tmp_path / "destination"
    root.mkdir()
    marker_id = "destination-2026"
    if marker is None:
        marker = {"version": 1, "marker_id": marker_id}
    if isinstance(marker, bytes):
        (root / MARKER_FILENAME).write_bytes(marker)
    elif isinstance(marker, str):
        (root / MARKER_FILENAME).write_text(marker, encoding="utf-8")
    else:
        (root / MARKER_FILENAME).write_text(json.dumps(marker, separators=(",", ":")), encoding="utf-8")
    root_stat = os.stat(root, follow_symlinks=False)
    observed = ObservedStorageIdentity(
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        filesystem_type="fakefs",
        device=MountIdentity(os.major(root_stat.st_dev), os.minor(root_stat.st_dev)),
        mount_id=42,
        mount_point="/destination",
    )
    config = StorageIdentityConfig(
        root=str(root),
        marker_id=marker_id,
        filesystem_type="fakefs",
        device_major=os.major(root_stat.st_dev),
        device_minor=os.minor(root_stat.st_dev),
        mount_id=42,
    )
    return root, config, FakeAdapter(observed)


def test_valid_identity_and_marker_round_trip_is_non_authorizing(tmp_path: Path):
    root, config, adapter = _setup(tmp_path)
    snapshot = StorageGuard(config, adapter).validate()
    assert snapshot.marker_id == config.marker_id
    assert snapshot.marker_version == 1
    assert snapshot.filesystem_type == "fakefs"
    assert snapshot.device == adapter.observed.device
    assert snapshot.mount_id == 42
    assert snapshot.point_in_time is True
    assert snapshot.non_authorizing is True
    assert snapshot.authorizing is False


@pytest.mark.parametrize(
    "mapping",
    [
        {"root": "relative", "marker_id": "id", "filesystem_type": "x", "device_major": 1, "device_minor": 2},
        {"root": "/tmp/a", "marker_id": " ", "filesystem_type": "x", "device_major": 1, "device_minor": 2},
        {"root": "/tmp/a", "marker_id": "id\x00", "filesystem_type": "x", "device_major": 1, "device_minor": 2},
        {"root": "/tmp/a", "marker_id": "id", "filesystem_type": " ", "device_major": 1, "device_minor": 2},
        {"root": "/tmp/a", "marker_id": "id", "filesystem_type": "x", "device_major": True, "device_minor": 2},
        {"root": "/tmp/a", "marker_id": "id", "filesystem_type": "x", "device_major": 1.0, "device_minor": 2},
        {"root": "/tmp/a", "marker_id": "id", "filesystem_type": "x", "device_major": 1, "device_minor": -1},
        {"root": "/tmp/a", "marker_id": "id", "filesystem_type": "x", "device_major": 1, "device_minor": 2, "extra": 1},
    ],
)
def test_expected_identity_config_rejects_ambiguous_or_malformed_fields(mapping: dict[str, object]):
    with pytest.raises(StorageConfigError):
        StorageIdentityConfig.from_mapping(mapping)


def test_expected_identity_config_rejects_missing_and_unknown_fields():
    with pytest.raises(StorageConfigError):
        StorageIdentityConfig.from_mapping({"root": "/tmp/a"})
    with pytest.raises(StorageConfigError):
        StorageIdentityConfig.from_mapping(
            {"root": "/tmp/a", "marker_id": "id", "filesystem_type": "x", "device_major": 1, "device_minor": 2, "x": 3}
        )


@pytest.mark.parametrize(
    "marker",
    [
        b"",
        {"version": 2, "marker_id": "destination-2026"},
        {"version": 1, "marker_id": "wrong"},
        {"version": 1, "marker_id": "destination-2026", "extra": 1},
        '{"version":1,"marker_id":"destination-2026","marker_id":"other"}',
        b"not utf8: \xff",
        b"[]",
    ],
)
def test_marker_failures_are_stable_public_errors(tmp_path: Path, marker: object):
    _, config, adapter = _setup(tmp_path, marker)
    with pytest.raises(StorageMarkerError) as failure:
        StorageGuard(config, adapter).validate()
    assert str(failure.value).startswith("marker:")


def test_oversized_marker_is_rejected_without_repair(tmp_path: Path):
    root, config, adapter = _setup(tmp_path, b"{" + b"x" * MARKER_MAX_BYTES)
    before = (root / MARKER_FILENAME).read_bytes()
    with pytest.raises(StorageMarkerError, match="marker:"):
        StorageGuard(config, adapter).validate()
    assert (root / MARKER_FILENAME).read_bytes() == before


def test_absent_marker_rejects_a_host_fallback_directory(tmp_path: Path):
    root, config, adapter = _setup(tmp_path)
    (root / MARKER_FILENAME).unlink()
    with pytest.raises(StorageMarkerError, match="marker:"):
        StorageGuard(config, adapter).validate()


def test_symlink_and_nonregular_markers_are_rejected(tmp_path: Path):
    root, config, adapter = _setup(tmp_path)
    marker = root / MARKER_FILENAME
    marker.unlink()
    outside = tmp_path / "outside"
    outside.write_text('{"version":1,"marker_id":"destination-2026"}')
    marker.symlink_to(outside)
    with pytest.raises(StorageMarkerError):
        StorageGuard(config, adapter).validate()
    marker.unlink()
    marker.mkdir()
    with pytest.raises(StorageMarkerError):
        StorageGuard(config, adapter).validate()


def test_identity_mismatches_are_rejected(tmp_path: Path):
    _, config, adapter = _setup(tmp_path)
    for observed in [
        ObservedStorageIdentity(adapter.observed.root_device, adapter.observed.root_inode, "otherfs", adapter.observed.device, 42, "/destination"),
        ObservedStorageIdentity(adapter.observed.root_device, adapter.observed.root_inode, "fakefs", MountIdentity(1, 2), 42, "/destination"),
        ObservedStorageIdentity(adapter.observed.root_device + 1, adapter.observed.root_inode, "fakefs", adapter.observed.device, 42, "/destination"),
    ]:
        adapter.observed = observed
        with pytest.raises(StorageIdentityError, match="identity:"):
            StorageGuard(config, adapter).validate()
    adapter.observed = ObservedStorageIdentity(
        os.stat(config.root, follow_symlinks=False).st_dev,
        adapter.observed.root_inode,
        "fakefs",
        MountIdentity(config.device_major, config.device_minor),
        42,
        "/destination",
    )
    adapter.observed = ObservedStorageIdentity(
        adapter.observed.root_device,
        adapter.observed.root_inode,
        "fakefs",
        MountIdentity(config.device_major, config.device_minor),
        99,
        "/destination",
    )
    with pytest.raises(StorageIdentityError, match="mount identity"):
        StorageGuard(config, adapter).validate()


def test_mountinfo_parser_decodes_escapes_and_rejects_malformed_or_ambiguous_rows():
    row = "36 25 0:42 / /tmp/hello\\040world rw - ext4 /dev/x rw"
    parsed = parse_mountinfo(row + "\n")
    assert parsed[0].mount_point == "/tmp/hello world"
    assert parsed[0].device == MountIdentity(0, 42)
    for bad in [
        "",
        "36 25 bad / /tmp rw - ext4 /dev/x rw",
        "36 25 0:42 / /tmp\\999 rw - ext4 /dev/x rw",
        row + "\n" + row,
    ]:
        with pytest.raises(StorageIdentityError, match="malformed mountinfo"):
            parse_mountinfo(bad)


def test_revalidation_rejects_marker_device_filesystem_mount_and_root_changes(tmp_path: Path):
    root, config, adapter = _setup(tmp_path)
    guard = StorageGuard(config, adapter)
    prior = guard.validate()
    (root / MARKER_FILENAME).write_text('{"version":1,"marker_id":"changed"}', encoding="utf-8")
    with pytest.raises(StorageMarkerError):
        guard.revalidate(prior)
    (root / MARKER_FILENAME).write_text('{"version":1,"marker_id":"destination-2026"}', encoding="utf-8")
    adapter.observed = ObservedStorageIdentity(prior.root_device, prior.root_inode, "changedfs", prior.device, prior.mount_id, prior.mount_point)
    with pytest.raises(StorageRevalidationError):
        guard.revalidate(prior)
    adapter.observed = ObservedStorageIdentity(prior.root_device, prior.root_inode, prior.filesystem_type, prior.device, 99, prior.mount_point)
    with pytest.raises(StorageIdentityError):
        guard.revalidate(prior)
    replacement = tmp_path / "replacement"
    root.rename(replacement)
    root.mkdir()
    (root / MARKER_FILENAME).write_text('{"version":1,"marker_id":"destination-2026"}', encoding="utf-8")
    adapter.observed = ObservedStorageIdentity(os.stat(root, follow_symlinks=False).st_dev, os.stat(root, follow_symlinks=False).st_ino, prior.filesystem_type, prior.device, prior.mount_id, prior.mount_point)
    with pytest.raises(StorageRevalidationError):
        guard.revalidate(prior)


def test_adapter_failures_are_wrapped_and_descriptors_do_not_leak(tmp_path: Path):
    _, config, adapter = _setup(tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    for error in [OSError("io"), RuntimeError("adapter"), KeyError("field"), ValueError("parse")]:
        adapter.error = error
        with pytest.raises(StorageGuardError) as failure:
            StorageGuard(config, adapter).validate()
        assert type(failure.value) is StorageIOError
    adapter.error = None
    assert len(os.listdir("/proc/self/fd")) == before
    for _ in range(20):
        StorageGuard(config, adapter).validate()
    assert len(os.listdir("/proc/self/fd")) == before


def test_missing_root_is_fail_closed(tmp_path: Path):
    root, config, adapter = _setup(tmp_path)
    root.rename(tmp_path / "gone")
    with pytest.raises(StorageGuardError):
        StorageGuard(config, adapter).validate()


def test_fallback_directory_with_absent_or_copied_marker_fails_closed(tmp_path: Path):
    real, config, adapter = _setup(tmp_path)
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    fallback_config = StorageIdentityConfig(
        str(fallback), config.marker_id, config.filesystem_type, config.device_major, config.device_minor, config.mount_id
    )
    with pytest.raises(StorageMarkerError):
        StorageGuard(fallback_config, adapter).validate()
    (fallback / MARKER_FILENAME).write_bytes((real / MARKER_FILENAME).read_bytes())
    adapter.observed = ObservedStorageIdentity(
        os.stat(fallback, follow_symlinks=False).st_dev,
        os.stat(fallback, follow_symlinks=False).st_ino,
        config.filesystem_type,
        MountIdentity(config.device_major + 1, config.device_minor),
        config.mount_id or 42,
        "/destination",
    )
    with pytest.raises(StorageGuardError):
        StorageGuard(fallback_config, adapter).validate()
