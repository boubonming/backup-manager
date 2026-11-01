from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from backup_manager.planner import (
    DeleteAction,
    DestinationEntry,
    PlannerValidationError,
    PublishedDestination,
    SourceIdentity,
    SourceObservation,
    Tombstone,
    plan_retention,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def source(path: str, sha: str = _A, length: int = 10, mtime: int = 10) -> SourceIdentity:
    return SourceIdentity(path, sha, length, mtime)


def published(item: SourceIdentity, destination: str | None = None) -> PublishedDestination:
    return PublishedDestination(
        item.source_relative_path,
        item.source_sha256,
        item.source_length,
        item.source_mtime_ns,
        destination or item.source_relative_path,
        item.source_sha256,
        item.source_length,
    )


def test_union_deduplicates_source_and_published_and_orders_pre_epoch_ties():
    old = source("old.zip", _A, 1, -5)
    newest = source("new.zip", _B, 2, 4)
    same = source("same.zip", _C, 3, -1)
    records = (same, old, newest)
    result = plan_retention(
        3,
        records,
        (published(same),),
    )
    assert result.retained == (newest, same, old)
    assert [action.identity for action in result.transfers] == [newest, old]
    assert result.deletions == ()
    assert result.non_authorizing is True
    assert result.deletion_authorized is False


def test_tombstones_prevent_recopy_and_are_never_delete_targets():
    item = source("gone.zip", _A, 4, 8)
    result = plan_retention(1, (item,), (), (Tombstone("gone.zip", _A, 4),))
    assert result.transfers == ()
    assert result.retained == ()
    assert result.deletions == ()
    assert all(action.identity.source_sha256 != _A for action in result.deletions)


def test_keep_count_does_not_copy_old_source_and_proposes_only_owned_deletion():
    newest = source("new.zip", _A, 10, 30)
    old = source("old.zip", _B, 20, 1)
    already = source("already.zip", _C, 30, 20)
    result = plan_retention(1, (old, newest), (published(already),))
    assert result.retained == (newest,)
    assert [action.identity for action in result.transfers] == [newest]
    assert result.deletions == (DeleteAction(already, "already.zip"),)
    assert result.managed_count == 1
    assert result.managed_bytes == 30
    assert any(skip.identity == old and skip.reason == "outside_retained_set" for skip in result.skipped)


def test_empty_unhealthy_incomplete_or_issue_observation_suppresses_deletion():
    existing = source("existing.zip", _A, 10, 10)
    for kwargs in (
        {},
        {"source_present": False},
        {"source_healthy": False},
        {"source_complete": False},
        {"source_issues": ("unresolved",)},
    ):
        result = plan_retention(0, (), (published(existing),), **kwargs)
        assert result.deletions == ()
        assert result.deletion_authorized is False
    healthy = plan_retention(0, (source("new.zip", _B, 1, 20),), (published(existing),))
    assert healthy.deletions == (DeleteAction(existing, "existing.zip"),)


def test_source_observation_wrapper_and_conflicting_publication_metadata_are_fail_closed():
    item = source("new.zip", _A, 4, 10)
    observation = SourceObservation((item,), healthy=True, present=True, complete=True)
    result = plan_retention(1, observation, ())
    assert result.transfers[0].identity == item
    conflicting = PublishedDestination("new.zip", _A, 4, 10, "new.zip", _B, 4)
    result = plan_retention(1, (item,), (conflicting,))
    assert result.blocked is True
    assert result.transfers == ()
    assert result.deletions == ()
    assert any(issue.code == "conflicting_identity" for issue in result.issues)


def test_collisions_and_malformed_inputs_fail_closed():
    item = source("same.zip", _A, 1, 2)
    different = source("same.zip", _B, 1, 3)
    result = plan_retention(1, (different,), (published(item),))
    assert result.blocked is True
    assert result.deletions == ()
    assert any(issue.code == "destination_collision" for issue in result.issues)

    with pytest.raises(PlannerValidationError):
        SourceIdentity("/absolute", _A, 1, 1)
    with pytest.raises(PlannerValidationError):
        SourceIdentity("unsafe/../path", _A, 1, 1)
    with pytest.raises(PlannerValidationError):
        SourceIdentity("x", "not-a-sha", 1, 1)
    with pytest.raises(PlannerValidationError):
        SourceIdentity("x", _A, True, 1)
    with pytest.raises(PlannerValidationError):
        SourceIdentity("x", _A, 1, False)
    with pytest.raises(PlannerValidationError):
        plan_retention(True, (), ())
    with pytest.raises(PlannerValidationError):
        plan_retention(1, [], ())


def test_unmanaged_collision_is_visible_and_untouched():
    item = source("new.zip", _A, 4, 5)
    result = plan_retention(
        1,
        (item,),
        (),
        destination_entries=(DestinationEntry("new.zip", _B, 4, False),),
    )
    assert result.blocked is True
    assert result.transfers == ()
    assert result.deletions == ()
    assert result.unmanaged == (DestinationEntry("new.zip", _B, 4, False),)


def test_results_are_immutable_and_permutations_have_identical_results():
    a = source("a", _A, 1, 0)
    b = source("b", _B, 2, 0)
    left = plan_retention(1, (a, b), ())
    right = plan_retention(1, (b, a), ())
    assert left == right
    with pytest.raises((FrozenInstanceError, AttributeError)):
        left.transfers = ()  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        left.transfers[0].identity.source_relative_path = "x"  # type: ignore[misc]
