from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from backup_manager.planner import (
    DeleteAction,
    DestinationEntry,
    PlannerIssue,
    RetentionPlan,
    PlannerValidationError,
    PublishedDestination,
    SourceIdentity,
    SourceObservation,
    Tombstone,
    TransferAction,
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
    result = plan_retention(1, (item, item), (), (Tombstone("gone.zip", _A, 4),))
    assert result.transfers == ()
    assert result.retained == ()
    assert result.deletions == ()
    assert len(result.skipped) == 1
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
        result = plan_retention(1, (), (published(existing),), **kwargs)
        assert result.deletions == ()
        assert result.deletion_authorized is False
    healthy = plan_retention(1, (source("new.zip", _B, 1, 20),), (published(existing),))
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
        plan_retention(0, (), ())
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


def test_source_same_destination_different_content_blocks_before_keep_selection():
    left = source("same.zip", _A, 1, 2)
    right = source("same.zip", _B, 2, 1)
    for keep_count, records in ((1, (left, right)), (1, (right, left)), (2, (left, right)), (2, (right, left))):
        result = plan_retention(keep_count, records, ())
        assert result.blocked is True
        assert result.transfers == ()
        assert result.deletions == ()
        assert any(issue.relative_path == "same.zip" for issue in result.issues)


def test_collisions_are_global_even_when_conflicting_source_is_not_retained():
    old = source("same.zip", _A, 1, 1)
    newest = source("newest.zip", _C, 1, 2)
    conflicting_publication = PublishedDestination(
        "other.zip", _B, 2, 3, "same.zip", _B, 2
    )
    published_result = plan_retention(1, (old, newest), (conflicting_publication,))
    assert published_result.blocked is True
    assert published_result.transfers == ()
    assert published_result.deletions == ()

    conflicting_unmanaged = DestinationEntry("same.zip", _B, 2, False)
    unmanaged_result = plan_retention(
        1,
        (old, newest),
        (),
        destination_entries=(conflicting_unmanaged,),
    )
    assert unmanaged_result.blocked is True
    assert unmanaged_result.transfers == ()
    assert unmanaged_result.deletions == ()


def test_exact_duplicate_source_observations_are_one_union_member():
    item = source("same.zip", _A, 1, 2)
    result = plan_retention(1, (item, item), ())
    assert result.blocked is False
    assert result.issues == ()
    assert result.retained == (item,)
    assert result.transfers == (TransferAction(item, "same.zip"),)


def test_exact_duplicate_trusted_publications_are_one_union_member():
    item = source("same.zip", _A, 1, 2)
    publication = published(item)
    result = plan_retention(1, (item,), (publication, publication))
    assert result.blocked is False
    assert result.issues == ()
    assert result.transfers == ()
    assert result.deletions == ()
    assert result.managed_count == 1
    assert result.managed_bytes == 1


def test_contradictory_duplicate_source_and_published_records_block():
    source_a = source("same.zip", _A, 1, 2)
    source_observation = source("same.zip", _A, 1, 3)
    source_result = plan_retention(1, (source_a, source_observation), ())
    assert source_result.blocked is True
    assert source_result.transfers == ()
    assert source_result.deletions == ()
    assert any(issue.code == "conflicting_identity" for issue in source_result.issues)

    published_a = published(source_a)
    published_observation = published(source_a, "other.zip")
    published_result = plan_retention(1, (), (published_a, published_observation))
    assert published_result.blocked is True
    assert published_result.transfers == ()
    assert published_result.deletions == ()
    assert any(issue.code == "conflicting_identity" for issue in published_result.issues)

    other = source("other.zip", _B, 2, 1)
    collision_result = plan_retention(1, (), (published(source_a), published(other, "same.zip")))
    assert collision_result.blocked is True
    assert collision_result.transfers == ()
    assert collision_result.deletions == ()
    assert any(issue.code == "destination_collision" for issue in collision_result.issues)


def test_same_timestamp_order_uses_path_then_complete_fingerprint():
    first = source("a.zip", _B, 2, 10)
    second = source("b.zip", _A, 1, 10)
    result = plan_retention(2, (second, first), ())
    assert result.retained == (first, second)


def test_same_path_timestamp_order_uses_sha_then_length_and_blocks_actions():
    sha_first = source("same.zip", _A, 3, 10)
    length_first = source("same.zip", _B, 2, 10)
    length_second = source("same.zip", _B, 7, 10)
    expected = (sha_first, length_first, length_second)

    for records in ((length_second, length_first, sha_first), expected):
        result = plan_retention(3, records, ())
        assert result.blocked is True
        assert result.transfers == ()
        assert result.deletions == ()
        assert result.retained == expected


def test_public_retention_plan_rejects_inconsistent_result_state():
    issue = PlannerIssue("source_missing")
    with pytest.raises(PlannerValidationError):
        RetentionPlan((), (), (), (), (issue,), (), 0, 0, 1, False)

    identity = source("same.zip")
    transfer = TransferAction(identity, "same.zip")
    deletion = DeleteAction(identity, "same.zip")
    with pytest.raises(PlannerValidationError):
        RetentionPlan((transfer,), (), (), (), (issue,), (), 0, 0, 1, True)
    with pytest.raises(PlannerValidationError):
        RetentionPlan((transfer, transfer), (), (), (), (), (), 0, 0, 1, False)
    with pytest.raises(PlannerValidationError):
        RetentionPlan((transfer,), (deletion,), (), (), (), (), 1, 1, 1, False)
    with pytest.raises(PlannerValidationError):
        RetentionPlan((), (), (), (), (), (), 0, 0, 0, False)


def test_retention_plan_rejects_duplicate_deletion_destinations():
    first = source("first.zip", _A, 10)
    second = source("second.zip", _B, 10)
    with pytest.raises(PlannerValidationError):
        RetentionPlan(
            (),
            (DeleteAction(first, "same.zip"), DeleteAction(second, "same.zip")),
            (),
            (),
            (),
            (),
            2,
            20,
            1,
            False,
        )


def test_retention_plan_rejects_managed_aggregate_lower_bounds():
    deletion = DeleteAction(source("deleted.zip", _A, 10), "deleted.zip")

    with pytest.raises(PlannerValidationError):
        RetentionPlan((), (deletion,), (), (), (), (), 0, 0, 1, False)
    with pytest.raises(PlannerValidationError):
        RetentionPlan((), (deletion,), (), (), (), (), 1, 9, 1, False)
    with pytest.raises(PlannerValidationError):
        RetentionPlan((), (), (), (), (), (), 0, 1, 1, False)


@pytest.mark.parametrize("keep_count", [True, False])
def test_retention_plan_rejects_boolean_keep_count(keep_count):
    with pytest.raises(PlannerValidationError):
        RetentionPlan((), (), (), (), (), (), 0, 0, keep_count, False)


def test_planner_purity_does_not_touch_external_services(monkeypatch):
    import builtins
    import os
    import socket
    import sqlite3
    import time

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("planner attempted external access")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(time, "time", forbidden)
    result = plan_retention(1, (source("new.zip"),), ())
    assert result.transfers == (TransferAction(source("new.zip"), "new.zip"),)
