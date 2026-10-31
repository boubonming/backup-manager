from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
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
    if sys.platform != "linux":
        pytest.skip("missing capability: Linux mount namespaces are required")
    missing_tools = [tool for tool in ("unshare", "mount", "umount", "mktemp") if shutil.which(tool) is None]
    if missing_tools:
        pytest.skip(f"missing capability: required tools unavailable: {', '.join(missing_tools)}")

    probe = subprocess.run(
        [
            "unshare",
            "--mount",
            "--propagation",
            "private",
            "--fork",
            "sh",
            "-c",
            "set -eu; d=$(mktemp -d); trap 'umount \"$d/m\" 2>/dev/null || true; rmdir \"$d/m\" \"$d\"' EXIT; mkdir \"$d/m\"; mount -t tmpfs task5-probe \"$d/m\"; umount \"$d/m\"",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "probe failed").strip().splitlines()[-1]
        pytest.skip(f"missing capability: private mount namespace/CAP_SYS_ADMIN unavailable: {detail}")

    script = textwrap.dedent(
        """
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        from backup_manager.paths import open_trusted_root
        from backup_manager.storage_guard import (
            MARKER_FILENAME,
            LinuxStorageIdentityAdapter,
            StorageGuard,
            StorageGuardError,
            StorageIdentityConfig,
        )

        with tempfile.TemporaryDirectory(prefix="backup-manager-mount-test-") as temp:
            base = Path(temp)
            real = base / "real"
            fallback = base / "fallback"
            real.mkdir()
            fallback.mkdir()
            subprocess.run(["mount", "-t", "tmpfs", "task5-real", str(real)], check=True)
            subprocess.run(["mount", "-t", "tmpfs", "task5-fallback", str(fallback)], check=True)
            try:
                marker = real / MARKER_FILENAME
                marker.write_text('{"version":1,"marker_id":"private-namespace"}', encoding="utf-8")
                shutil.copyfile(marker, fallback / MARKER_FILENAME)
                adapter = LinuxStorageIdentityAdapter()
                with open_trusted_root(real) as trusted:
                    observed = adapter.observe(trusted.fd, str(real))
                config = StorageIdentityConfig(
                    root=str(real),
                    marker_id="private-namespace",
                    filesystem_type=observed.filesystem_type,
                    device_major=observed.device.major,
                    device_minor=observed.device.minor,
                    mount_id=observed.mount_id,
                )
                StorageGuard(config, adapter).validate()
                fallback_config = StorageIdentityConfig(
                    root=str(fallback),
                    marker_id=config.marker_id,
                    filesystem_type=config.filesystem_type,
                    device_major=config.device_major,
                    device_minor=config.device_minor,
                    mount_id=config.mount_id,
                )
                try:
                    StorageGuard(fallback_config, adapter).validate()
                except StorageGuardError:
                    pass
                else:
                    raise AssertionError("copied marker on a different private mount was accepted")
            finally:
                subprocess.run(["umount", "--", str(fallback)], check=True)
                subprocess.run(["umount", "--", str(real)], check=True)
        """
    )
    subprocess.run(
        ["unshare", "--mount", "--propagation", "private", "--fork", sys.executable, "-c", script],
        check=True,
        timeout=30,
    )
