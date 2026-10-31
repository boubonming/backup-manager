"""Descriptor-relative path safety and destination ownership primitives.

The functions in this module deliberately do not create, replace, remove, or write
filesystem objects.  A trusted root is held open while relative components are
traversed, and every component is opened with ``O_NOFOLLOW``.
"""

from __future__ import annotations

import errno
import ntpath
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Final


class PathSafetyError(ValueError):
    """Base class for all expected path-safety and ownership failures."""


class PathGrammarError(PathSafetyError):
    """A supplied root or relative path does not satisfy the path grammar."""


class PathOwnershipError(PathSafetyError):
    """A path overlaps an owned path or collides with an unmanaged object."""


class FinalObjectPolicy(str, Enum):
    """Object type policy for the final component of a relative path."""

    DIRECTORY = "directory"
    REGULAR_FILE = "regular_file"
    MISSING_LEAF = "missing_leaf"
    ANY_EXISTING = "any_existing"


class CollisionStatus(str, Enum):
    """Classification returned by :func:`guard_destination_leaf`."""

    MISSING = "missing"
    MANAGED = "managed"


@dataclass(frozen=True, slots=True)
class PathResolution:
    """An opened path object, or a planned but currently missing leaf.

    Existing objects own a read-only descriptor and must be closed by the caller
    (normally with ``with``).  A missing-leaf result has ``fd is None`` and does
    not authorize a later path-based open.
    """

    relative_path: str
    kind: str
    _fd: int | None

    @property
    def fd(self) -> int | None:
        return self._fd

    def close(self) -> None:
        fd = self._fd
        if fd is not None:
            os.close(fd)
            object.__setattr__(self, "_fd", None)

    def __enter__(self) -> PathResolution:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class RootHandle:
    """A trusted descriptor for one approved directory root."""

    root_path: str
    _fd: int

    @property
    def fd(self) -> int:
        if self._fd < 0:
            raise ValueError("root handle is closed")
        return self._fd

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            object.__setattr__(self, "_fd", -1)

    def __enter__(self) -> RootHandle:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def resolve(
        self,
        relative_path: str,
        policy: FinalObjectPolicy = FinalObjectPolicy.DIRECTORY,
        *,
        before_component_open: Callable[[int, str, int], None] | None = None,
    ) -> PathResolution:
        """Open ``relative_path`` beneath this root without following symlinks.

        ``before_component_open`` is a deterministic test seam.  It is called
        after a non-authorizing ``lstat`` and immediately before the descriptor
        relative open; production callers should leave it unset.
        """

        normalized, components = _relative_parts(relative_path)
        policy = _coerce_policy(policy)
        current = os.dup(self.fd)
        try:
            if not components:
                if policy is FinalObjectPolicy.DIRECTORY or policy is FinalObjectPolicy.ANY_EXISTING:
                    if policy is FinalObjectPolicy.DIRECTORY:
                        _verify_directory(current, "relative_path")
                    return PathResolution(normalized, "directory", current)
                os.close(current)
                if policy is FinalObjectPolicy.MISSING_LEAF:
                    raise PathSafetyError("relative_path: final object already exists")
                raise PathSafetyError("relative_path: final object is not a regular file")

            for index, component in enumerate(components):
                is_final = index == len(components) - 1
                _precheck_component(
                    current,
                    component,
                    "relative_path",
                    is_final=is_final,
                    allow_missing=is_final and policy is FinalObjectPolicy.MISSING_LEAF,
                )
                if before_component_open is not None:
                    before_component_open(current, component, index)

                if not is_final:
                    child = _open_directory_component(current, component, "relative_path")
                    os.close(current)
                    current = child
                    continue

                result = _open_final_component(current, component, policy, normalized)
                os.close(current)
                return result
        except PathSafetyError:
            os.close(current)
            raise
        except OSError as exc:
            os.close(current)
            raise _component_error("relative_path", exc) from None
        raise AssertionError("unreachable")


def normalize_relative_path(value: str, *, field: str = "path") -> str:
    """Return a stable relative POSIX identity, rejecting escapes and ambiguity."""

    _, components = _relative_parts(value, field=field)
    return "." if not components else "/".join(components)


@dataclass(frozen=True, slots=True)
class DestinationOwner:
    """An existing job-owned destination directory under one trusted root."""

    owner: str
    relative_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.owner, str) or not self.owner:
            raise PathGrammarError("owner: must be a non-empty string")
        object.__setattr__(self, "relative_path", normalize_relative_path(self.relative_path, field="relative_path"))


@dataclass(frozen=True, slots=True)
class DestinationOwnershipResult:
    allowed: bool
    relative_path: str
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SeparationResult:
    allowed: bool
    source_identity: tuple[int, int]
    destination_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ManagedOwnershipWitness:
    """Metadata witness proving ownership of one exact regular-file identity."""

    destination_relative_path: str
    device: int
    inode: int
    owner: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "destination_relative_path",
            normalize_relative_path(self.destination_relative_path, field="destination_relative_path"),
        )
        if not isinstance(self.owner, str) or not self.owner:
            raise PathGrammarError("owner: must be a non-empty string")
        if type(self.device) is not int or type(self.inode) is not int:
            raise PathGrammarError("ownership witness: device and inode must be integers")

    @property
    def identity(self) -> tuple[int, int]:
        return (self.device, self.inode)


@dataclass(frozen=True, slots=True)
class CollisionResult:
    allowed: bool
    status: CollisionStatus
    relative_path: str
    identity: tuple[int, int] | None


_DIR_FLAGS: Final[int] = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_PATH_FLAGS: Final[int] = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW


def open_trusted_root(root: str | os.PathLike[str]) -> RootHandle:
    """Open an explicit absolute directory root and hold its descriptor."""

    root_text, components = _root_parts(root)
    current: int | None = None
    try:
        current = os.open("/", _DIR_FLAGS)
        for component in components:
            _precheck_component(current, component, "root", is_final=False)
            child = _open_directory_component(current, component, "root")
            os.close(current)
            current = child
        _verify_directory(current, "root")
        result = RootHandle(root_text, current)
        current = None
        return result
    except PathSafetyError:
        if current is not None:
            os.close(current)
        raise
    except OSError as exc:
        if current is not None:
            os.close(current)
        raise _component_error("root", exc) from None


def validate_source_destination_separation(
    source_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    source_relative_path: str,
    destination_relative_path: str,
) -> SeparationResult:
    """Reject equal or nested effective source/destination directories."""

    with open_trusted_root(source_root) as source, open_trusted_root(destination_root) as destination:
        with source.resolve(source_relative_path, FinalObjectPolicy.DIRECTORY) as source_dir:
            with destination.resolve(destination_relative_path, FinalObjectPolicy.DIRECTORY) as destination_dir:
                source_identity = _identity(source_dir.fd)
                destination_identity = _identity(destination_dir.fd)
                if _is_ancestor(source_dir.fd, destination_dir.fd) or _is_ancestor(
                    destination_dir.fd, source_dir.fd
                ):
                    raise PathOwnershipError("source and destination directories overlap")
                return SeparationResult(True, source_identity, destination_identity)


def validate_destination_ownership(
    destination_root: str | os.PathLike[str],
    proposed_relative_path: str,
    existing: Iterable[DestinationOwner | tuple[str, str] | Mapping[str, object]],
) -> DestinationOwnershipResult:
    """Reject a proposed destination directory overlapping any existing owner."""

    proposed = normalize_relative_path(proposed_relative_path, field="proposed_relative_path")
    owners = tuple(_coerce_owner(item) for item in existing)
    with open_trusted_root(destination_root) as root:
        with root.resolve(proposed, FinalObjectPolicy.DIRECTORY) as proposed_dir:
            proposed_identity = _identity(proposed_dir.fd)
            for owner in owners:
                with root.resolve(owner.relative_path, FinalObjectPolicy.DIRECTORY) as existing_dir:
                    if _is_ancestor(proposed_dir.fd, existing_dir.fd) or _is_ancestor(
                        existing_dir.fd, proposed_dir.fd
                    ):
                        raise PathOwnershipError(
                            f"destination overlap: conflicting owner {owner.owner!r} at {owner.relative_path!r}"
                        )
            return DestinationOwnershipResult(True, proposed, proposed_identity)


def guard_destination_leaf(
    destination_root: str | os.PathLike[str],
    destination_relative_path: str,
    witness: ManagedOwnershipWitness | None = None,
) -> CollisionResult:
    """Classify a destination leaf without creating or opening it for writing."""

    normalized, components = _relative_parts(destination_relative_path, field="destination_relative_path")
    if not components:
        raise PathGrammarError("destination_relative_path: must name a leaf")
    parent = "." if len(components) == 1 else "/".join(components[:-1])
    leaf = components[-1]
    with open_trusted_root(destination_root) as root:
        with root.resolve(parent, FinalObjectPolicy.DIRECTORY) as parent_dir:
            try:
                fd = os.open(leaf, _PATH_FLAGS, dir_fd=parent_dir.fd)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    return CollisionResult(True, CollisionStatus.MISSING, normalized, None)
                raise _component_error("destination_relative_path", exc) from None
            try:
                identity = _identity(fd)
                mode = os.fstat(fd).st_mode
                if stat.S_ISLNK(mode):
                    raise PathOwnershipError("destination collision: final symlink is unmanaged")
                if not stat.S_ISREG(mode):
                    raise PathOwnershipError("destination collision: final object is not a regular file")
                if (
                    witness is not None
                    and witness.destination_relative_path == normalized
                    and witness.identity == identity
                ):
                    return CollisionResult(True, CollisionStatus.MANAGED, normalized, identity)
                raise PathOwnershipError("destination collision: existing object is unmanaged or changed")
            finally:
                os.close(fd)


def _relative_parts(value: object, *, field: str = "path") -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, str):
        raise PathGrammarError(f"{field}: must be a string")
    if not value or not value.strip():
        raise PathGrammarError(f"{field}: must not be empty or whitespace-only")
    if "\x00" in value:
        raise PathGrammarError(f"{field}: NUL is not allowed")
    if "\\" in value:
        raise PathGrammarError(f"{field}: backslash paths are not allowed")
    if os.path.isabs(value) or ntpath.splitdrive(value)[0] or value.startswith("//"):
        raise PathGrammarError(f"{field}: must be a relative POSIX path")
    components = tuple(part for part in value.split("/") if part not in ("", "."))
    if any(part == ".." for part in components):
        raise PathGrammarError(f"{field}: parent component '..' is not allowed")
    return ("." if not components else "/".join(components), components)


def _root_parts(root: str | os.PathLike[str]) -> tuple[str, tuple[str, ...]]:
    try:
        root_text = os.fspath(root)
    except TypeError:
        raise PathGrammarError("root: must be an absolute directory path") from None
    if not isinstance(root_text, str):
        raise PathGrammarError("root: must be an absolute directory path")
    if not root_text or "\x00" in root_text or "\\" in root_text:
        raise PathGrammarError("root: must be an absolute POSIX directory path")
    if not os.path.isabs(root_text) or ntpath.splitdrive(root_text)[0]:
        raise PathGrammarError("root: must be an absolute directory path")
    parts = tuple(part for part in root_text.split("/") if part not in ("", "."))
    if any(part == ".." for part in parts):
        raise PathGrammarError("root: parent component '..' is not allowed")
    return root_text, parts


def _coerce_policy(policy: FinalObjectPolicy | str) -> FinalObjectPolicy:
    try:
        return policy if isinstance(policy, FinalObjectPolicy) else FinalObjectPolicy(policy)
    except (TypeError, ValueError):
        raise PathGrammarError("policy: unknown final-object policy") from None


def _coerce_owner(value: DestinationOwner | tuple[str, str] | Mapping[str, object]) -> DestinationOwner:
    if isinstance(value, DestinationOwner):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return DestinationOwner(value[0], value[1])
    if isinstance(value, Mapping):
        try:
            return DestinationOwner(value["owner"], value["relative_path"])  # type: ignore[arg-type]
        except KeyError:
            raise PathGrammarError("owner: expected owner and relative_path") from None
    raise PathGrammarError("owner: expected DestinationOwner")


def _precheck_component(
    fd: int,
    component: str,
    field: str,
    *,
    is_final: bool,
    allow_missing: bool = False,
) -> None:
    try:
        info = os.stat(component, dir_fd=fd, follow_symlinks=False)
    except OSError as exc:
        if allow_missing and exc.errno == errno.ENOENT:
            return
        raise _component_error(field, exc) from None
    if stat.S_ISLNK(info.st_mode):
        raise PathSafetyError(f"{field}: symlink component is not allowed")
    if not is_final and not stat.S_ISDIR(info.st_mode):
        raise PathSafetyError(f"{field}: intermediate component is not a directory")


def _open_directory_component(fd: int, component: str, field: str) -> int:
    try:
        child = os.open(component, _DIR_FLAGS, dir_fd=fd)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            try:
                if stat.S_ISLNK(os.stat(component, dir_fd=fd, follow_symlinks=False).st_mode):
                    raise PathSafetyError(f"{field}: symlink component is not allowed") from None
            except FileNotFoundError:
                pass
        raise _component_error(field, exc) from None
    try:
        _verify_directory(child, field)
    except BaseException:
        os.close(child)
        raise
    return child


def _open_final_component(fd: int, component: str, policy: FinalObjectPolicy, normalized: str) -> PathResolution:
    if policy is FinalObjectPolicy.MISSING_LEAF:
        try:
            existing_fd = os.open(component, _PATH_FLAGS, dir_fd=fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return PathResolution(normalized, "missing", None)
            raise _component_error("relative_path", exc) from None
        try:
            mode = os.fstat(existing_fd).st_mode
            if stat.S_ISLNK(mode):
                raise PathSafetyError("relative_path: symlink final component is not allowed")
            raise PathSafetyError("relative_path: final object already exists")
        finally:
            os.close(existing_fd)

    flags = _PATH_FLAGS if policy is FinalObjectPolicy.ANY_EXISTING else os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if policy is FinalObjectPolicy.DIRECTORY:
        flags |= os.O_DIRECTORY
    try:
        final_fd = os.open(component, flags, dir_fd=fd)
    except OSError as exc:
        if policy is FinalObjectPolicy.DIRECTORY and exc.errno == errno.ENOTDIR:
            raise PathSafetyError("relative_path: final object is not a directory") from None
        raise _component_error("relative_path", exc) from None
    try:
        mode = os.fstat(final_fd).st_mode
        if stat.S_ISLNK(mode):
            raise PathSafetyError("relative_path: symlink component is not allowed")
        if policy is FinalObjectPolicy.DIRECTORY and not stat.S_ISDIR(mode):
            raise PathSafetyError("relative_path: final object is not a directory")
        if policy is FinalObjectPolicy.REGULAR_FILE and not stat.S_ISREG(mode):
            raise PathSafetyError("relative_path: final object is not a regular file")
        if policy is FinalObjectPolicy.ANY_EXISTING:
            kind = "directory" if stat.S_ISDIR(mode) else "regular_file" if stat.S_ISREG(mode) else "existing"
        else:
            kind = policy.value
        return PathResolution(normalized, kind, final_fd)
    except BaseException:
        os.close(final_fd)
        raise


def _verify_directory(fd: int, field: str) -> None:
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        raise PathSafetyError(f"{field}: final object is not a directory")


def _identity(fd: int | None) -> tuple[int, int]:
    if fd is None:
        raise ValueError("missing path has no identity")
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino)


def _is_ancestor(ancestor_fd: int | None, candidate_fd: int | None) -> bool:
    if ancestor_fd is None or candidate_fd is None:
        return False
    wanted = _identity(ancestor_fd)
    current = os.dup(candidate_fd)
    try:
        for _ in range(4096):
            if _identity(current) == wanted:
                return True
            parent = os.open("..", _DIR_FLAGS, dir_fd=current)
            if _identity(parent) == _identity(current):
                os.close(parent)
                return False
            old = current
            current = parent
            os.close(old)
        return False
    finally:
        os.close(current)


def _component_error(field: str, exc: OSError) -> PathSafetyError:
    if exc.errno in (errno.ELOOP, errno.EMLINK):
        return PathSafetyError(f"{field}: symlink component is not allowed")
    if exc.errno == errno.ENOTDIR:
        return PathSafetyError(f"{field}: intermediate component is not a directory")
    if exc.errno == errno.ENOENT:
        return PathSafetyError(f"{field}: component does not exist")
    return PathSafetyError(f"{field}: filesystem access failed ({exc.errno})")


__all__ = [
    "CollisionResult",
    "CollisionStatus",
    "DestinationOwner",
    "DestinationOwnershipResult",
    "FinalObjectPolicy",
    "ManagedOwnershipWitness",
    "PathGrammarError",
    "PathOwnershipError",
    "PathResolution",
    "PathSafetyError",
    "RootHandle",
    "SeparationResult",
    "guard_destination_leaf",
    "normalize_relative_path",
    "open_trusted_root",
    "validate_destination_ownership",
    "validate_source_destination_separation",
]
