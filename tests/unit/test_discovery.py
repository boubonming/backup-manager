from __future__ import annotations

import os
import socket
import time
import zipfile
from pathlib import Path

import pytest

from backup_manager.discovery import (
    DiscoveryConfigError,
    DiscoveryPolicy,
    DiscoverySourceError,
    DiscoveryCandidate,
    DiscoveryResult,
    discover_completed_files,
)


def _write_zip(path: Path, payload: str = "complete") -> None:
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("payload.txt", payload)


def _discover(root: Path, policy: DiscoveryPolicy, **kwargs: object):
    return discover_completed_files(root, "backups", policy, now_ns=kwargs.pop("now_ns", time.time_ns() + 1_000_000_000), **kwargs)


def test_discovers_completed_zip_with_immutable_non_authorizing_metadata(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    archive = source / "backup.zip"
    _write_zip(archive)
    stat_result = archive.stat()
    result = discover_completed_files(
        root,
        "backups",
        DiscoveryPolicy(include_patterns=("*.zip",), validate_completed_zips=True),
        now_ns=stat_result.st_mtime_ns + 1,
    )
    assert result.healthy is True
    assert result.non_authorizing is True
    assert result.authorizing is False
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.relative_path == "backup.zip"
    assert candidate.device == stat_result.st_dev
    assert candidate.inode == stat_result.st_ino
    assert candidate.byte_length == stat_result.st_size
    assert candidate.mtime_ns == stat_result.st_mtime_ns
    assert candidate.fd is None
    with pytest.raises(AttributeError):
        candidate.relative_path = "other"  # type: ignore[misc]


def test_include_exclude_is_case_sensitive_and_exclude_wins(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    for name in ("backup.zip", "backup.zip.tmp", "backup.part", ".hidden.zip", "BACKUP.ZIP"):
        _write_zip(source / name) if name.endswith(".zip") else (source / name).write_bytes(b"x")
    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",), exclude_patterns=("*.tmp", "*.part", ".*", "backup.zip"), validate_completed_zips=True))
    assert result.candidates == ()
    assert {skip.relative_path for skip in result.skips} >= {"backup.zip", ".hidden.zip"}
    assert all(candidate.relative_path != "backup.zip.tmp" for candidate in result.candidates)


def test_recursion_and_ordering_are_stable_posix_identities(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    (source / "z").mkdir(parents=True)
    (source / "a").mkdir()
    _write_zip(source / "z" / "last.zip")
    _write_zip(source / "a" / "first.zip")
    _write_zip(source / "middle.zip")
    assert [c.relative_path for c in _discover(root, DiscoveryPolicy(include_patterns=("*.zip",))).candidates] == ["middle.zip"]
    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",), recursive=True))
    assert [c.relative_path for c in result.candidates] == ["a/first.zip", "middle.zip", "z/last.zip"]


def test_recursive_exclude_prunes_selected_subtree(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    (source / "selected" / "nested").mkdir(parents=True)
    _write_zip(source / "selected" / "nested" / "ignored.zip")
    _write_zip(source / "kept.zip")
    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",), exclude_patterns=("selected/**",), recursive=True))
    assert [c.relative_path for c in result.candidates] == ["kept.zip"]


def test_recursive_directory_pruning_during_scan_is_unhealthy(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    nested = source / "nested"
    nested.mkdir(parents=True)
    _write_zip(nested / "backup.zip")

    def prune_directory(_path: str, _fd: int) -> None:
        nested.rename(source / "pruned")

    result = _discover(
        root,
        DiscoveryPolicy(include_patterns=("*.zip",), recursive=True),
        between_observations=prune_directory,
    )
    assert result.candidates == ()
    assert result.healthy is False
    assert any(issue.code == "race" and issue.relative_path == "nested" for issue in result.issues)


def test_age_boundary_and_future_mtime_use_injected_now(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    archive = source / "backup.zip"
    archive.write_bytes(b"data")
    os.utime(archive, ns=(9_000_000_000, 9_000_000_000))
    policy = DiscoveryPolicy(include_patterns=("*.zip",), minimum_age_seconds=1)
    assert len(_discover(root, policy, now_ns=10_000_000_000).candidates) == 1
    os.utime(archive, ns=(9_000_000_001, 9_000_000_001))
    result = _discover(root, policy, now_ns=10_000_000_000)
    assert result.candidates == ()
    assert result.skips[0].code == "too_young"
    os.utime(archive, ns=(11_000_000_000, 11_000_000_000))
    assert _discover(root, policy, now_ns=10_000_000_000).skips[0].code == "future_mtime"


def test_empty_source_is_healthy_but_missing_source_is_public_error(tmp_path: Path):
    root = tmp_path / "approved"
    (root / "backups").mkdir(parents=True)
    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",)))
    assert result.healthy is True and result.candidates == ()
    with pytest.raises(DiscoverySourceError):
        discover_completed_files(root, "missing", DiscoveryPolicy())


def test_source_directory_symlink_component_is_rejected(tmp_path: Path):
    root = tmp_path / "approved"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(DiscoverySourceError):
        discover_completed_files(root, "alias", DiscoveryPolicy())


def test_corrupt_and_crc_bad_zip_are_unhealthy(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    (source / "corrupt.zip").write_bytes(b"not a zip")
    crc = source / "crc.zip"
    _write_zip(crc)
    with zipfile.ZipFile(crc) as archive:
        info = archive.infolist()[0]
        data_offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    with crc.open("r+b") as stream:
        stream.seek(data_offset)
        stream.write(b"!")
    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",), validate_completed_zips=True))
    assert result.healthy is False
    assert {issue.relative_path for issue in result.issues} == {"corrupt.zip", "crc.zip"}
    assert all(issue.code == "zip_invalid" for issue in result.issues)


def test_symlink_and_special_objects_never_block_or_become_candidates(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    target = source / "real.zip"
    _write_zip(target)
    (source / "link.zip").symlink_to(target)
    (source / "directory.zip").mkdir()
    fifo = source / "pipe.zip"
    os.mkfifo(fifo)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(source / "socket.zip"))
    except OSError:
        sock.close()
        sock = None
    try:
        result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",), validate_completed_zips=True))
    finally:
        if sock is not None:
            sock.close()
    assert [candidate.relative_path for candidate in result.candidates] == ["real.zip"]
    assert {issue.code for issue in result.issues} >= {"symlink", "special_file"}


@pytest.mark.parametrize("mutation", ["append", "truncate", "replace", "mtime"])
def test_append_truncate_replace_and_mtime_mutations_are_unhealthy(tmp_path: Path, mutation: str):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    archive = source / "backup.zip"
    archive.write_bytes(b"stable")

    def mutate(_path: str, _fd: int) -> None:
        if mutation == "append":
            with archive.open("ab") as stream:
                stream.write(b"changed")
        elif mutation == "truncate":
            archive.write_bytes(b"")
        elif mutation == "replace":
            archive.rename(source / "old.zip")
            archive.write_bytes(b"replacement")
        else:
            os.utime(archive, ns=(archive.stat().st_mtime_ns + 1, archive.stat().st_mtime_ns + 1))

    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",)), between_observations=mutate)
    assert result.candidates == ()
    assert result.healthy is False
    assert result.issues[0].code == "race"


def test_pruning_between_enumeration_and_open_is_a_public_issue(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    archive = source / "backup.zip"
    archive.write_bytes(b"data")

    def prune(_path: str, _parent_fd: int) -> None:
        archive.unlink()

    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",)), before_open=prune)
    assert result.candidates == ()
    assert result.healthy is False
    assert result.issues[0].code == "vanished"


def test_hook_exception_and_descriptor_cleanup_are_stable(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    for index in range(5):
        (source / f"{index}.zip").write_bytes(b"data")
    before = len(os.listdir("/proc/self/fd"))

    def explode(_path: str, _fd: int) -> None:
        raise RuntimeError("secret path should not escape")

    for _ in range(20):
        result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",)), between_observations=explode)
        assert result.healthy is False
        assert all("secret" not in issue.message for issue in result.issues)
    assert len(os.listdir("/proc/self/fd")) <= before + 1


def test_hostile_names_are_not_line_delimited(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    names = ["space name.zip", "ユニコード.zip", "line\nname.zip"]
    for name in names:
        (source / name).write_bytes(b"x")
    result = _discover(root, DiscoveryPolicy(include_patterns=("*.zip",)))
    assert [candidate.relative_path for candidate in result.candidates] == sorted(names)


@pytest.mark.parametrize(
    "root,relative,policy",
    [
        ("relative", "backups", DiscoveryPolicy()),
        ("/tmp/a/../b", "backups", DiscoveryPolicy()),
        ("/tmp/a", "../backups", DiscoveryPolicy()),
        ("/tmp/a", "back\\ups", DiscoveryPolicy()),
        ("/tmp/a", "backups", None),
    ],
)
def test_malformed_inputs_fail_closed(root: object, relative: object, policy: object):
    with pytest.raises(DiscoveryConfigError):
        discover_completed_files(root, relative, policy)


def test_malformed_pattern_and_boolean_values_fail_closed():
    with pytest.raises(DiscoveryConfigError):
        DiscoveryPolicy(include_patterns=("bad\\pattern",))
    with pytest.raises(DiscoveryConfigError):
        DiscoveryPolicy(recursive=1)
    with pytest.raises(DiscoveryConfigError):
        DiscoveryPolicy(minimum_age_seconds=True)
    with pytest.raises(DiscoveryConfigError):
        DiscoveryPolicy(minimum_age_seconds=-1)


def test_public_observation_models_reject_mutable_or_authorizing_shapes():
    candidate = DiscoveryCandidate("file.zip", 1, 2, 3, 4)
    with pytest.raises(DiscoveryConfigError):
        DiscoveryResult([candidate], [], [], True)  # type: ignore[arg-type]
    with pytest.raises(DiscoveryConfigError):
        DiscoveryCandidate("/absolute", 1, 2, 3, 4)
    with pytest.raises(DiscoveryConfigError):
        DiscoveryResult((candidate,), (), (), 1)  # type: ignore[arg-type]
    assert candidate.non_authorizing is True


def test_no_files_are_created_changed_or_deleted(tmp_path: Path):
    root = tmp_path / "approved"
    source = root / "backups"
    source.mkdir(parents=True)
    archive = source / "backup.zip"
    _write_zip(archive)
    before = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*")}
    _discover(root, DiscoveryPolicy(include_patterns=("*.zip",), validate_completed_zips=True))
    after = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*")}
    assert after == before
