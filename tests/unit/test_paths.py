from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from backup_manager.paths import (
    CollisionStatus,
    DestinationOwner,
    FinalObjectPolicy,
    ManagedOwnershipWitness,
    PathGrammarError,
    PathOwnershipError,
    PathSafetyError,
    RootHandle,
    guard_destination_leaf,
    normalize_relative_path,
    open_trusted_root,
    validate_destination_ownership,
    validate_source_destination_separation,
)


def test_normalize_relative_path_preserves_data_and_rejects_escapes():
    assert normalize_relative_path(".//- odd//é/*?/'quote'//name\n ") == "- odd/é/*?/'quote'/name\n "
    assert normalize_relative_path(".") == "."
    with pytest.raises(PathGrammarError, match=r"^path: "):
        normalize_relative_path("a/../b")
    for value in ["", "   ", "/absolute", "\\windows", "C:/drive", "//server/share", "a\\b", "a\x00b"]:
        with pytest.raises(PathGrammarError, match=r"^path: "):
            normalize_relative_path(value)
    with pytest.raises(PathGrammarError, match=r"^relative_path: "):
        normalize_relative_path("../escape", field="relative_path")
    with pytest.raises(PathGrammarError, match=r"^path: must be a string"):
        normalize_relative_path(42)


def test_open_root_and_resolve_expected_final_object_policies(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "file").write_text("safe")
    (root / "bad").write_text("not a directory")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("outside")
    (root / "link").symlink_to(outside, target_is_directory=True)

    with open_trusted_root(root) as trusted:
        with trusted.resolve("nested", FinalObjectPolicy.DIRECTORY) as result:
            assert result.kind == "directory"
            assert stat.S_ISDIR(os.fstat(result.fd).st_mode)
        with trusted.resolve("nested/file", FinalObjectPolicy.REGULAR_FILE) as result:
            assert os.read(result.fd, 32) == b"safe"
        with trusted.resolve("nested/new", FinalObjectPolicy.MISSING_LEAF) as result:
            assert result.kind == "missing"
            assert result.fd is None
            assert not (root / "nested" / "new").exists()
        with pytest.raises(PathSafetyError, match=r"^relative_path: final object is not a directory"):
            trusted.resolve("nested/file", FinalObjectPolicy.DIRECTORY)
        with pytest.raises(PathSafetyError, match=r"^relative_path: final object is not a regular file"):
            trusted.resolve("nested", FinalObjectPolicy.REGULAR_FILE)
        with pytest.raises(PathSafetyError, match=r"^relative_path: symlink component is not allowed"):
            trusted.resolve("link/secret", FinalObjectPolicy.REGULAR_FILE)
        with pytest.raises(PathSafetyError, match=r"^relative_path: intermediate component is not a directory"):
            trusted.resolve("bad/file", FinalObjectPolicy.REGULAR_FILE)


def test_trusted_root_requires_existing_absolute_non_symlink_directory(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    file_path = tmp_path / "file"
    file_path.write_text("not a directory")
    for value in [link, file_path, Path("relative")]:
        with pytest.raises(PathSafetyError, match=r"^root:"):
            open_trusted_root(value)


def test_descriptor_relative_race_rejects_intermediate_replacement(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    safe = root / "safe"
    safe.mkdir()
    (safe / "file").write_text("safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside")
    moved = root / "safe-original"

    def replace_before_open(_parent_fd: int, component: str, index: int) -> None:
        if component == "safe" and index == 0:
            safe.rename(moved)
            safe.symlink_to(outside, target_is_directory=True)

    with open_trusted_root(root) as trusted:
        with pytest.raises(PathSafetyError, match=r"symlink component is not allowed"):
            trusted.resolve(
                "safe/file",
                FinalObjectPolicy.REGULAR_FILE,
                before_component_open=replace_before_open,
            )
    assert (moved / "file").read_text() == "safe"
    assert (safe / "file").read_text() == "outside"


def test_descriptor_relative_race_rejects_final_replacement(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    safe = root / "safe"
    safe.mkdir()
    file_path = safe / "file"
    file_path.write_text("safe")
    outside = tmp_path / "outside"
    outside.write_text("outside")
    old = safe / "file-original"

    def replace_before_open(_parent_fd: int, component: str, index: int) -> None:
        if component == "file" and index == 1:
            file_path.rename(old)
            file_path.symlink_to(outside)

    with open_trusted_root(root) as trusted:
        with pytest.raises(PathSafetyError, match=r"symlink component is not allowed"):
            trusted.resolve(
                "safe/file",
                FinalObjectPolicy.REGULAR_FILE,
                before_component_open=replace_before_open,
            )
    assert old.read_text() == "safe"


def test_held_root_descriptor_does_not_switch_to_replacement(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file").write_text("original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "file").write_text("attacker")

    with open_trusted_root(root) as trusted:
        root.rename(tmp_path / "renamed-root")
        root.mkdir()
        (root / "file").write_text("wrong")
        with trusted.resolve("file", FinalObjectPolicy.REGULAR_FILE) as result:
            assert os.read(result.fd, 32) == b"original"


def test_source_destination_separation_uses_filesystem_identity(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "same-prefix").mkdir()
    (destination / "same-prefix-2").mkdir()

    result = validate_source_destination_separation(source, destination, "same-prefix", "same-prefix-2")
    assert result.allowed is True
    with pytest.raises(PathOwnershipError, match="overlap"):
        validate_source_destination_separation(source, source, ".", ".")
    nested_source = tmp_path / "nested-source"
    nested_destination = nested_source / "destination"
    nested_destination.mkdir(parents=True)
    with pytest.raises(PathOwnershipError, match="overlap"):
        validate_source_destination_separation(nested_source, nested_destination, ".", ".")
    with pytest.raises(PathSafetyError, match="symlink"):
        (destination / "alias").symlink_to(source, target_is_directory=True)
        validate_source_destination_separation(source, destination, ".", "alias")


def test_destination_ownership_rejects_component_aware_overlap_and_names_owner(tmp_path: Path):
    root = tmp_path / "dest"
    root.mkdir()
    for name in ["job-child", "sibling", "other"]:
        (root / name).mkdir()
    (root / "job").mkdir()
    (root / "job" / "child").mkdir()
    assert validate_destination_ownership(
        root, "sibling", [DestinationOwner("job-1", "job"), DestinationOwner("job-2", "job-child")]
    ).allowed
    for proposed, existing in [("job", "job/child"), ("job/child", "job"), ("job", "job")]:
        with pytest.raises(PathOwnershipError, match="conflicting owner 'job'"):
            validate_destination_ownership(root, proposed, [DestinationOwner("job", existing)])
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    (other_root / "job").mkdir()
    assert validate_destination_ownership(other_root, "job", []).allowed


def test_destination_leaf_collision_guard_is_fail_closed_and_side_effect_free(tmp_path: Path):
    root = tmp_path / "dest"
    root.mkdir()
    parent = root / "archives"
    parent.mkdir()
    missing = guard_destination_leaf(root, "archives/new.tar")
    assert missing.status is CollisionStatus.MISSING
    assert missing.allowed is True
    assert not (parent / "new.tar").exists()

    managed = parent / "managed.tar"
    managed.write_text("managed")
    identity = os.stat(managed, follow_symlinks=False)
    witness = ManagedOwnershipWitness("archives/./managed.tar", identity.st_dev, identity.st_ino, "job-1")
    allowed = guard_destination_leaf(root, "archives/managed.tar", witness)
    assert allowed.status is CollisionStatus.MANAGED
    assert allowed.allowed is True

    for filename, make, witness_value in [
        ("file.tar", lambda p: p.write_text("unmanaged"), None),
        ("directory.tar", lambda p: p.mkdir(), None),
        ("symlink.tar", lambda p: p.symlink_to(outside), None),
        ("mismatch.tar", lambda p: p.write_text("changed"), witness),
    ]:
        outside = tmp_path / f"outside-{filename}"
        outside.write_text("outside")
        target = parent / filename
        make(target)
        with pytest.raises(PathOwnershipError, match="collision"):
            guard_destination_leaf(root, f"archives/{filename}", witness_value)
        assert target.is_symlink() or target.exists()


def test_path_operations_do_not_leak_descriptors_on_repeated_failures(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file").write_text("data")
    with open_trusted_root(root) as trusted:
        before = len(os.listdir("/proc/self/fd"))
        for _ in range(50):
            with pytest.raises(PathSafetyError):
                trusted.resolve("file/nope", FinalObjectPolicy.REGULAR_FILE)
        after = len(os.listdir("/proc/self/fd"))
    assert after == before


def test_public_exception_hierarchy_and_immutable_records():
    assert issubclass(PathGrammarError, PathSafetyError)
    assert issubclass(PathOwnershipError, PathSafetyError)
    owner = DestinationOwner("job", "a/b")
    with pytest.raises((AttributeError, TypeError)):
        owner.owner = "changed"
