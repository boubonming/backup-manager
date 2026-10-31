from __future__ import annotations

import os
from pathlib import Path

import pytest

from backup_manager.paths import open_trusted_root
from backup_manager.storage_guard import (
    MARKER_FILENAME,
    LinuxStorageIdentityAdapter,
    StorageGuard,
    StorageIdentityConfig,
    StorageIOError,
    parse_mountinfo,
)


def test_real_linux_adapter_validates_disposable_temporary_root(tmp_path: Path):
    if not os.path.exists("/proc/self/mountinfo"):
        pytest.skip("Linux mountinfo is unavailable")
    root = tmp_path / "disposable-destination"
    root.mkdir()
    (root / MARKER_FILENAME).write_text('{"version":1,"marker_id":"integration"}', encoding="utf-8")
    adapter = LinuxStorageIdentityAdapter()
    with open_trusted_root(root) as trusted:
        observed = adapter.observe(trusted.fd, str(root))
    config = StorageIdentityConfig(
        root=str(root),
        marker_id="integration",
        filesystem_type=observed.filesystem_type,
        device_major=observed.device.major,
        device_minor=observed.device.minor,
        mount_id=observed.mount_id,
    )
    snapshot = StorageGuard(config, adapter).validate()
    assert snapshot.filesystem_type == observed.filesystem_type
    assert snapshot.device == observed.device
    assert snapshot.mount_id == observed.mount_id
    assert snapshot.non_authorizing is True


def test_real_linux_adapter_rejects_missing_mountinfo_without_touching_root(tmp_path: Path):
    root = tmp_path / "disposable-destination"
    root.mkdir()
    adapter = LinuxStorageIdentityAdapter(str(tmp_path / "does-not-exist"))
    with open_trusted_root(root) as trusted:
        with pytest.raises(StorageIOError, match="storage I/O failure"):
            adapter.observe(trusted.fd, str(root))
    assert list(root.iterdir()) == []


def test_real_mountinfo_contains_strictly_decodable_records():
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        pytest.skip("Linux mountinfo is unavailable")
    parsed = parse_mountinfo(mountinfo.read_bytes())
    assert parsed
    assert all(record.mount_point.startswith("/") for record in parsed)


@pytest.mark.skipif(
    os.environ.get("BACKUP_MANAGER_STORAGE_MOUNT_TESTS") != "1",
    reason="privileged disposable mount tests are opt-in via BACKUP_MANAGER_STORAGE_MOUNT_TESTS=1",
)
def test_opt_in_mount_namespace_checks_are_explicitly_gated():
    pytest.skip("No mount/unmount test is enabled by default; use disposable infrastructure only")
