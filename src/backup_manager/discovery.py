"""Safe, descriptor-relative discovery of completed source files.

Discovery is deliberately a point-in-time observation.  A candidate contains
metadata captured from a held descriptor, but it is *not* an authorization to
read or mutate that pathname later.  Callers must open a fresh descriptor and
revalidate identity before any subsequent operation.

Patterns are case-sensitive ``fnmatch`` patterns over the complete
source-subtree-relative POSIX path (for example, ``nested/backup.zip``), not
just the basename.  Excludes are evaluated after includes and always win.
"""

from __future__ import annotations

import errno
import fnmatch
import os
import stat
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field

from .paths import (
    FinalObjectPolicy,
    PathGrammarError,
    PathSafetyError,
    normalize_relative_path,
    open_trusted_root,
)

class DiscoveryError(ValueError):
    """Base for whole-operation configuration and source failures.

    Per-entry I/O, race, callback, and ZIP uncertainty is represented by a
    generic :class:`DiscoveryIssue` instead of being raised.
    """


class DiscoveryConfigError(DiscoveryError):
    """Malformed or ambiguous discovery configuration."""


class DiscoverySourceError(DiscoveryError):
    """The configured source root or source directory cannot be opened safely."""


class _DiscoveryZipError(DiscoveryError):
    """A completed ZIP could not be validated."""


_ISSUE_MESSAGES = {
    "vanished": "directory entry disappeared",
    "permission": "candidate could not be opened safely",
    "symlink": "symlink entries are not eligible",
    "io_error": "filesystem operation failed safely",
    "race": "candidate changed during observation",
    "special_file": "candidate is not a regular file",
    "hook_failure": "discovery callback failed",
    "zip_invalid": "completed ZIP validation failed",
}


def _patterns(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise DiscoveryConfigError(f"{field}: must be a tuple or list of patterns")
    result: list[str] = []
    for pattern in value:
        if not isinstance(pattern, str):
            raise DiscoveryConfigError(f"{field}: patterns must be strings")
        if not pattern or not pattern.strip():
            raise DiscoveryConfigError(f"{field}: patterns must not be empty")
        if "\x00" in pattern or "\\" in pattern:
            raise DiscoveryConfigError(f"{field}: patterns must use safe POSIX text")
        if pattern.startswith("/") or any(part == ".." for part in pattern.split("/")):
            raise DiscoveryConfigError(f"{field}: patterns must not escape the source subtree")
        result.append(pattern)
    if not result:
        raise DiscoveryConfigError(f"{field}: must contain at least one pattern")
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class DiscoveryPolicy:
    """Immutable policy for one descriptor-relative discovery operation."""

    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    recursive: bool
    minimum_age_seconds: int
    validate_completed_zips: bool

    def __init__(
        self,
        include_patterns: object = ("*",),
        exclude_patterns: object = (),
        recursive: object = False,
        minimum_age_seconds: object = 0,
        validate_completed_zips: object = False,
        **aliases: object,
    ) -> None:
        if "include" in aliases:
            if include_patterns != ("*",):
                raise DiscoveryConfigError("include_patterns: ambiguous duplicate value")
            include_patterns = aliases.pop("include")
        if "exclude" in aliases:
            if exclude_patterns != ():
                raise DiscoveryConfigError("exclude_patterns: ambiguous duplicate value")
            exclude_patterns = aliases.pop("exclude")
        if "validate_zips" in aliases:
            if validate_completed_zips is not False:
                raise DiscoveryConfigError("validate_completed_zips: ambiguous duplicate value")
            validate_completed_zips = aliases.pop("validate_zips")
        if "completed_zip_validation" in aliases:
            if validate_completed_zips is not False:
                raise DiscoveryConfigError("validate_completed_zips: ambiguous duplicate value")
            validate_completed_zips = aliases.pop("completed_zip_validation")
        if aliases:
            raise DiscoveryConfigError(f"policy: unknown field '{next(iter(aliases))}'")
        checked_include = _patterns(include_patterns, "include_patterns")
        if isinstance(exclude_patterns, (str, bytes)) or not isinstance(exclude_patterns, (tuple, list)):
            raise DiscoveryConfigError("exclude_patterns: must be a tuple or list of patterns")
        checked_exclude = tuple() if not exclude_patterns else _patterns(exclude_patterns, "exclude_patterns")
        if type(recursive) is not bool:
            raise DiscoveryConfigError("recursive: must be a boolean")
        if type(minimum_age_seconds) is not int or minimum_age_seconds < 0:
            raise DiscoveryConfigError("minimum_age_seconds: must be a non-negative integer")
        if type(validate_completed_zips) is not bool:
            raise DiscoveryConfigError("validate_completed_zips: must be a boolean")
        object.__setattr__(self, "include_patterns", checked_include)
        object.__setattr__(self, "exclude_patterns", checked_exclude)
        object.__setattr__(self, "recursive", recursive)
        object.__setattr__(self, "minimum_age_seconds", minimum_age_seconds)
        object.__setattr__(self, "validate_completed_zips", validate_completed_zips)


@dataclass(frozen=True, slots=True)
class DiscoveryCandidate:
    """Immutable metadata observed from one held regular-file descriptor."""

    relative_path: str
    device: int
    inode: int
    byte_length: int
    mtime_ns: int
    non_authorizing: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        try:
            normalized = normalize_relative_path(self.relative_path, field="relative_path")
        except PathGrammarError:
            raise DiscoveryConfigError("relative_path: must be a safe relative POSIX path") from None
        if (
            type(self.device) is not int
            or type(self.inode) is not int
            or self.device < 0
            or self.inode < 0
        ):
            raise DiscoveryConfigError("candidate identity: device and inode must be non-negative integers")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise DiscoveryConfigError("byte_length: must be a non-negative integer")
        if type(self.mtime_ns) is not int:
            raise DiscoveryConfigError("mtime_ns: must be an integer")
        object.__setattr__(self, "relative_path", normalized)

    @property
    def size(self) -> int:
        """Compatibility spelling for the captured byte length."""
        return self.byte_length

    @property
    def path(self) -> str:
        """Compatibility spelling for the source-relative identity."""
        return self.relative_path

    @property
    def fd(self) -> None:
        """Candidates intentionally never retain an operational descriptor."""
        return None


@dataclass(frozen=True, slots=True)
class DiscoveryIssue:
    """A stable, relative-path-only explanation for an unhealthy observation."""

    code: str
    relative_path: str
    message: str

    def __post_init__(self) -> None:
        try:
            normalized = normalize_relative_path(self.relative_path, field="relative_path")
        except PathGrammarError:
            raise DiscoveryConfigError("relative_path: must be a safe relative POSIX path") from None
        if type(self.code) is not str or self.code not in _ISSUE_MESSAGES:
            raise DiscoveryConfigError("issue code: unknown issue category")
        if self.message != _ISSUE_MESSAGES[self.code]:
            raise DiscoveryConfigError("issue message: must use the generic category message")
        object.__setattr__(self, "relative_path", normalized)


@dataclass(frozen=True, slots=True)
class DiscoverySkip:
    """A benign filter or age exclusion."""

    code: str
    relative_path: str

    def __post_init__(self) -> None:
        try:
            normalized = normalize_relative_path(self.relative_path, field="relative_path")
        except PathGrammarError:
            raise DiscoveryConfigError("relative_path: must be a safe relative POSIX path") from None
        if type(self.code) is not str or not self.code or "\x00" in self.code:
            raise DiscoveryConfigError("skip code: must be non-empty text")
        object.__setattr__(self, "relative_path", normalized)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Immutable discovery observations; never a read or deletion authorization."""

    candidates: tuple[DiscoveryCandidate, ...]
    issues: tuple[DiscoveryIssue, ...]
    skips: tuple[DiscoverySkip, ...]
    healthy: bool
    non_authorizing: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or any(
            not isinstance(candidate, DiscoveryCandidate) for candidate in self.candidates
        ):
            raise DiscoveryConfigError("candidates: must be an immutable tuple of DiscoveryCandidate")
        if type(self.issues) is not tuple or any(
            not isinstance(issue, DiscoveryIssue) for issue in self.issues
        ):
            raise DiscoveryConfigError("issues: must be an immutable tuple of DiscoveryIssue")
        if type(self.skips) is not tuple or any(
            not isinstance(skip, DiscoverySkip) for skip in self.skips
        ):
            raise DiscoveryConfigError("skips: must be an immutable tuple of DiscoverySkip")
        if type(self.healthy) is not bool:
            raise DiscoveryConfigError("healthy: must be a boolean")

    @property
    def authorizing(self) -> bool:
        return False

    @property
    def is_healthy(self) -> bool:
        return self.healthy


@dataclass(frozen=True, slots=True)
class _ObservedStat:
    device: int
    inode: int
    mode: int
    byte_length: int
    mtime_ns: int

    @classmethod
    def from_fd(cls, fd: int) -> _ObservedStat:
        info = os.fstat(fd)
        return cls(info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)

    @classmethod
    def from_dir_entry(cls, parent_fd: int, name: str) -> _ObservedStat:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return cls(info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)

    def identity_matches(self, other: _ObservedStat) -> bool:
        return (
            self.device == other.device
            and self.inode == other.inode
            and stat.S_IFMT(self.mode) == stat.S_IFMT(other.mode)
            and self.byte_length == other.byte_length
            and self.mtime_ns == other.mtime_ns
        )


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _excluded(path: str, patterns: tuple[str, ...]) -> bool:
    """Match entries and make the common ``subtree/**`` spelling prune roots."""
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or (pattern.endswith("/**") and path == pattern[:-3].rstrip("/"))
        for pattern in patterns
    )


def _issue(code: str, path: str, _detail: str = "") -> DiscoveryIssue:
    return DiscoveryIssue(code, path, _ISSUE_MESSAGES[code])


def _errno_code(exc: OSError) -> str:
    if exc.errno == errno.ENOENT:
        return "vanished"
    if exc.errno in (errno.EACCES, errno.EPERM):
        return "permission"
    if exc.errno in (errno.ELOOP, errno.EMLINK):
        return "symlink"
    return "io_error"


def _open_child_directory(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_fd)


def _open_candidate(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    return os.open(name, flags, dir_fd=parent_fd)


def _invoke_hook(hook: Callable[[str], object] | None, path: str) -> None:
    if hook is not None:
        hook(path)


def _validate_zip(fd: int) -> None:
    duplicate = os.dup(fd)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "rb", closefd=True) as stream:
            duplicate = -1
            with zipfile.ZipFile(stream) as archive:
                if archive.testzip() is not None:
                    raise _DiscoveryZipError("zip validation failed")
    except _DiscoveryZipError:
        raise
    except (OSError, EOFError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise _DiscoveryZipError("zip validation failed") from None
    finally:
        if duplicate >= 0:
            os.close(duplicate)


def _validate_root_value(root: object) -> str:
    try:
        text = os.fspath(root)
    except TypeError:
        raise DiscoveryConfigError("approved_source_root: must be an absolute path") from None
    if not isinstance(text, str) or not text or "\x00" in text or "\\" in text:
        raise DiscoveryConfigError("approved_source_root: must be an absolute POSIX path")
    if not os.path.isabs(text):
        raise DiscoveryConfigError("approved_source_root: must be absolute")
    if any(part == ".." for part in text.split("/")):
        raise DiscoveryConfigError("approved_source_root: parent traversal is not allowed")
    return text


def _candidate_for_file(
    parent_fd: int,
    name: str,
    relative_path: str,
    pre_open: _ObservedStat,
    policy: DiscoveryPolicy,
    now_ns: int,
    before_open: Callable[[str], object] | None,
    hook: Callable[[str], object] | None,
    issues: list[DiscoveryIssue],
    skips: list[DiscoverySkip],
) -> DiscoveryCandidate | None:
    try:
        if before_open is not None:
            before_open(relative_path)
    except Exception:
        issues.append(_issue("hook_failure", relative_path, "discovery hook failed"))
        return None
    try:
        fd = _open_candidate(parent_fd, name)
    except OSError as exc:
        issues.append(_issue(_errno_code(exc), relative_path, "candidate could not be opened safely"))
        return None
    try:
        try:
            first = _ObservedStat.from_fd(fd)
        except OSError:
            issues.append(_issue("io_error", relative_path, "candidate metadata could not be read"))
            return None
        if not stat.S_ISREG(first.mode):
            issues.append(_issue("special_file", relative_path, "candidate is not a regular file"))
            return None
        if not pre_open.identity_matches(first):
            issues.append(_issue("race", relative_path, "candidate changed during opening"))
            return None
        age_ns = now_ns - first.mtime_ns
        required_age_ns = policy.minimum_age_seconds * 1_000_000_000
        if age_ns < required_age_ns:
            skips.append(DiscoverySkip("too_young" if age_ns >= 0 else "future_mtime", relative_path))
            return None
        validation_error: str | None = None
        try:
            _invoke_hook(hook, relative_path)
            if policy.validate_completed_zips and relative_path.endswith(".zip"):
                _validate_zip(fd)
        except _DiscoveryZipError:
            validation_error = "zip_invalid"
        except Exception:
            validation_error = "hook_failure"
        try:
            second = _ObservedStat.from_fd(fd)
        except OSError:
            issues.append(_issue("race", relative_path, "candidate disappeared during observation"))
            return None
        if not first.identity_matches(second):
            issues.append(_issue("race", relative_path, "candidate changed during observation"))
            return None
        try:
            current_path_stat = _ObservedStat.from_dir_entry(parent_fd, name)
        except OSError:
            issues.append(_issue("race", relative_path, "candidate pathname disappeared during observation"))
            return None
        if not first.identity_matches(current_path_stat):
            issues.append(_issue("race", relative_path, "candidate pathname changed during observation"))
            return None
        if validation_error is not None:
            issues.append(
                _issue(
                    validation_error,
                    relative_path,
                    "completed-file validation did not produce a stable eligible file",
                )
            )
            return None
        return DiscoveryCandidate(
            relative_path,
            first.device,
            first.inode,
            first.byte_length,
            first.mtime_ns,
        )
    finally:
        os.close(fd)


def _scan_directory(
    directory_fd: int,
    relative_directory: str,
    policy: DiscoveryPolicy,
    now_ns: int,
    before_open: Callable[[str], object] | None,
    hook: Callable[[str], object] | None,
    candidates: list[DiscoveryCandidate],
    issues: list[DiscoveryIssue],
    skips: list[DiscoverySkip],
) -> None:
    try:
        with os.scandir(directory_fd) as entries:
            names = sorted((entry.name for entry in entries), key=lambda value: value)
    except OSError:
        issues.append(
            _issue(
                "io_error",
                relative_directory,
                "directory enumeration failed",
            )
        )
        return
    for name in names:
        relative_path = name if relative_directory == "." else f"{relative_directory}/{name}"
        try:
            pre_open = _ObservedStat.from_dir_entry(directory_fd, name)
        except OSError as exc:
            issues.append(_issue(_errno_code(exc), relative_path, "directory entry disappeared"))
            continue
        if stat.S_ISLNK(pre_open.mode):
            issues.append(_issue("symlink", relative_path, "symlink entries are not eligible"))
            continue
        if stat.S_ISDIR(pre_open.mode):
            if not policy.recursive:
                skips.append(DiscoverySkip("nonrecursive", relative_path))
                continue
            if _excluded(relative_path, policy.exclude_patterns):
                skips.append(DiscoverySkip("excluded", relative_path))
                continue
            try:
                child_fd = _open_child_directory(directory_fd, name)
            except OSError as exc:
                issues.append(_issue(_errno_code(exc), relative_path, "directory could not be opened safely"))
                continue
            try:
                opened = _ObservedStat.from_fd(child_fd)
                if not pre_open.identity_matches(opened) or not stat.S_ISDIR(opened.mode):
                    issues.append(_issue("race", relative_path, "directory changed during opening"))
                    continue
                candidate_start = len(candidates)
                _scan_directory(child_fd, relative_path, policy, now_ns, before_open, hook, candidates, issues, skips)
                directory_race = False
                try:
                    current = _ObservedStat.from_dir_entry(directory_fd, name)
                except OSError:
                    directory_race = True
                    issues.append(_issue("race", relative_path, "directory pathname disappeared during observation"))
                else:
                    if not pre_open.identity_matches(current):
                        directory_race = True
                        issues.append(_issue("race", relative_path, "directory pathname changed during observation"))
                if directory_race:
                    del candidates[candidate_start:]
            except OSError:
                issues.append(_issue("io_error", relative_path, "directory metadata could not be read"))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(pre_open.mode):
            issues.append(_issue("special_file", relative_path, "non-regular entries are not eligible"))
            continue
        if not _matches(relative_path, policy.include_patterns):
            skips.append(DiscoverySkip("not_included", relative_path))
            continue
        if _excluded(relative_path, policy.exclude_patterns):
            skips.append(DiscoverySkip("excluded", relative_path))
            continue
        candidate = _candidate_for_file(
            directory_fd,
            name,
            relative_path,
            pre_open,
            policy,
            now_ns,
            before_open,
            hook,
            issues,
            skips,
        )
        if candidate is not None:
            candidates.append(candidate)


def discover_completed_files(
    approved_source_root: object,
    source_directory: object,
    policy: DiscoveryPolicy,
    *,
    now_ns: object = None,
    before_open: Callable[[str], object] | None = None,
    between_observations: Callable[[str], object] | None = None,
    observation_hook: Callable[[str], object] | None = None,
) -> DiscoveryResult:
    """Discover stable, eligible files beneath one approved source subtree.

    ``between_observations`` is a deterministic test seam and receives only the
    relative path after the first stat and before optional ZIP validation.  It
    is not used by production callers.  An ordinary exception from it makes
    the result unhealthy and is never exposed verbatim.  Process-control
    exceptions propagate to the caller.

    ``before_open`` is a deterministic test seam called with only the relative
    path after non-authorizing enumeration metadata and immediately before the
    descriptor-relative open.  Discovery never lends owned descriptors to
    callback code.
    """
    if not isinstance(policy, DiscoveryPolicy):
        raise DiscoveryConfigError("policy: must be a DiscoveryPolicy")
    if before_open is not None and not callable(before_open):
        raise DiscoveryConfigError("before_open: must be callable")
    if between_observations is not None and observation_hook is not None:
        raise DiscoveryConfigError("observation hook: only one hook may be supplied")
    hook = between_observations if between_observations is not None else observation_hook
    if hook is not None and not callable(hook):
        raise DiscoveryConfigError("observation hook: must be callable")
    if now_ns is None:
        checked_now_ns = time.time_ns()
    elif type(now_ns) is int:
        checked_now_ns = now_ns
    else:
        raise DiscoveryConfigError("now_ns: must be an integer")
    _validate_root_value(approved_source_root)
    try:
        normalized_source = normalize_relative_path(source_directory, field="source_directory")
    except PathGrammarError as exc:
        raise DiscoveryConfigError("source_directory: invalid relative POSIX path") from None

    candidates: list[DiscoveryCandidate] = []
    issues: list[DiscoveryIssue] = []
    skips: list[DiscoverySkip] = []
    try:
        trusted_root = open_trusted_root(approved_source_root)  # type: ignore[arg-type]
    except (PathSafetyError, OSError):
        raise DiscoverySourceError("approved source root could not be opened safely") from None
    try:
        try:
            source = trusted_root.resolve(normalized_source, FinalObjectPolicy.DIRECTORY)
        except (PathSafetyError, OSError):
            raise DiscoverySourceError("source directory could not be opened safely") from None
        with source:
            if source.fd is None:
                raise DiscoverySourceError("source directory could not be opened safely")
            _scan_directory(source.fd, ".", policy, checked_now_ns, before_open, hook, candidates, issues, skips)
    finally:
        trusted_root.close()
    candidates.sort(key=lambda item: item.relative_path)
    issues.sort(key=lambda item: (item.relative_path, item.code))
    skips.sort(key=lambda item: (item.relative_path, item.code))
    return DiscoveryResult(tuple(candidates), tuple(issues), tuple(skips), not issues)


# Short aliases make the intentionally small public surface convenient while
# keeping the fully descriptive function name available to integrations.
DiscoveryConfig = DiscoveryPolicy
discover_files = discover_completed_files
discover = discover_completed_files

__all__ = [
    "DiscoveryCandidate",
    "DiscoveryConfig",
    "DiscoveryConfigError",
    "DiscoveryError",
    "DiscoveryIssue",
    "DiscoveryPolicy",
    "DiscoveryResult",
    "DiscoverySkip",
    "DiscoverySourceError",
    "discover",
    "discover_completed_files",
    "discover_files",
]
