#!/usr/bin/env python3
"""Fetch the subtree an imported skill's pin names, not the repository around it.

`sync_external.py`, which generates an imported skill's directory from its pin and
`--check`s that the two still agree, needs the files under one directory at one
commit. The obvious way to get them, and the way it did get them, is
`codeload.github.com/<repo>/tar.gz/<commit>`: one request, no token, every byte under
the pin.

That does not scale, and the cost is not in the skill. It is in the repository the
skill lives in. Measured against the repositories this catalog already imports from
and the kind it will import from next:

    intel/gpu-ai-skills            383 KB repo ->    248 KB tarball
    intel/scikit-learn-intelex      46 MB repo ->   11.6 MB tarball
    NVIDIA/cuda-quantum            2.0 GB repo ->    113 MB tarball, for one skill

A catalog of a few hundred skills is a catalog sourced from product repositories, not
from a handful of skill repositories: NVIDIA's 343 skills come from 38 upstreams
totalling 7.7 GB. Tarball transport makes every pull request download that and hold
each uncompressed tree in memory while it works.

A blobless single-revision partial clone with a non-cone sparse checkout moves the
subtree and nothing else:

    NVIDIA/cuda-quantum, one subtree -> 3.2 MB transferred, ~50 MB peak RSS

The trade is real and small in the other direction: for a repository that is already
just skills, the tarball is one HTTP GET while this is a git negotiation with
server-side blob filtering, and intel/gpu-ai-skills measures ~2.4 s here against
~0.6 s there. Paying two seconds per upstream to stop downloading gigabytes per
upstream is the right side of that trade, and it is one code path rather than two.

Bytes come from the object store (`ls-tree` + `cat-file`), not from the working tree:
a `.gitattributes` with `text`/`eol` or a clean/smudge filter would otherwise let the
checkout hand back something other than what upstream committed, and what this
repository vendors is a statement about upstream's bytes. `ls-tree` is also where the
mode comes from, so the recorded execute bit does not depend on the umask of whoever
ran the tool.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

# Line-ending translation off on every call, as in bin/intel-skills.mjs: CRLF would
# change the hash of every text file and break every shebang.
GIT_FLAGS = ("-c", "core.autocrlf=false", "-c", "core.eol=lf")

# A tree entry this catalog will vendor. A symlink (120000) or a submodule (160000)
# is refused rather than resolved: install writes files, and a skill that needs more
# than files is a skill whose copy here cannot be verified.
BLOB_MODES = {"100644", "100755"}


class SyncError(Exception):
    """Upstream does not say what skills.yaml claims. A pull request fails on this."""


class SyncUnavailable(Exception):
    """Upstream could not be reached at all.

    Kept apart from SyncError on purpose, and for the same reason --check-links
    only warns on a timeout: a validator that fails a pull request when someone
    else's network hiccups is one contributors learn to ignore. A pinned commit that
    is *gone* is not this -- that is an answer, and the answer is that the pin no
    longer resolves.
    """


# Fragments git prints when the remote answered and the answer was no. Everything
# else -- DNS, proxy, TLS, timeout, reset -- is "could not reach upstream".
ANSWERED = (
    "couldn't find remote ref",
    "not our ref",
    "no such remote ref",
    "repository not found",
    "does not appear to be a git repository",
    "authentication failed",
    "permission denied",
    "access denied",
)


def _git(args: list[str], cwd: Path, *, stdin: bytes | None = None) -> bytes:
    """Run git, and turn its failure into the right one of the two exceptions."""
    try:
        done = subprocess.run(
            ["git", *GIT_FLAGS, *args],
            cwd=cwd,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SyncError(
            "git is not on PATH, and an imported skill is read from its pinned upstream "
            "commit with git; install git to run this gate"
        ) from exc
    if done.returncode == 0:
        return done.stdout
    detail = done.stderr.decode("utf-8", "replace").strip().splitlines()
    last = detail[-1] if detail else f"git {args[0]} exited {done.returncode}"
    lowered = " ".join(detail).lower()
    if any(fragment in lowered for fragment in ANSWERED):
        raise SyncError(last)
    raise SyncUnavailable(last)


@contextlib.contextmanager
def pinned_subtrees(repo: str, commit: str, paths: list[str]) -> Iterator[Path]:
    """Materialize the given subtrees of one commit into a temporary clone.

    The checkout is what makes the blobs local: sparse-checkout narrows the tree to
    the pinned paths, so the promisor fetch pulls those blobs in one request instead
    of one request per file when they are read back.
    """
    work = Path(tempfile.mkdtemp(prefix="intel-skills-upstream-"))
    try:
        _git(["init", "--quiet", "."], work)
        _git(["remote", "add", "origin", repo], work)
        _git(["config", "core.sparseCheckout", "true"], work)
        _git(["config", "core.sparseCheckoutCone", "false"], work)
        patterns = "".join(f"/{path.strip('/')}/*\n" for path in sorted(set(paths)))
        (work / ".git" / "info" / "sparse-checkout").write_text(patterns, encoding="utf-8")
        try:
            _git(
                ["fetch", "--quiet", "--filter=blob:none", "--depth", "1", "origin", commit],
                work,
            )
        except SyncError as exc:
            raise SyncError(
                f"{repo} at {commit[:12]}: {exc} — the pinned commit is not there"
            ) from exc
        _git(["checkout", "--quiet", "FETCH_HEAD"], work)
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _read_blobs(work: Path, oids: list[str]) -> list[bytes]:
    """Read blobs by object id in one `cat-file --batch` conversation."""
    if not oids:
        return []
    stdout = _git(["cat-file", "--batch"], work, stdin=("\n".join(oids) + "\n").encode())
    blobs: list[bytes] = []
    offset = 0
    for oid in oids:
        end = stdout.index(b"\n", offset)
        header = stdout[offset:end].decode("utf-8", "replace").split()
        if len(header) != 3 or header[1] != "blob":
            raise SyncError(f"{oid}: git cat-file did not return a blob ({header})")
        size = int(header[2])
        start = end + 1
        blobs.append(stdout[start : start + size])
        offset = start + size + 1  # trailing newline git adds after the payload
    return blobs


def subtree_files(work: Path, paths: list[str]) -> dict[str, tuple[bytes, int]]:
    """Every file under the given subtrees, keyed by repository-relative path.

    The value is the committed bytes and the tree mode, both straight from the
    object store, so neither depends on how the checkout was written to disk.
    """
    listing = _git(
        [
            "ls-tree",
            "-r",
            "-z",
            "--format=%(objectmode) %(objectname) %(path)",
            "FETCH_HEAD",
            "--",
            *[path.strip("/") for path in sorted(set(paths))],
        ],
        work,
    )
    entries: list[tuple[str, str, str]] = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        mode, oid, relative = record.decode("utf-8").split(" ", 2)
        if mode not in BLOB_MODES:
            raise SyncError(
                f"{relative}: tree entry has mode {mode}, which is not a regular file. "
                "A symlink or submodule under a pinned skill cannot be installed as "
                "files, so it cannot be locked either"
            )
        entries.append((mode, oid, relative))
    blobs = _read_blobs(work, [oid for _, oid, _ in entries])
    return {
        relative: (blob, int(mode[-3:], 8))
        for (mode, _, relative), blob in zip(entries, blobs)
    }


def root_files(work: Path, wanted: set[str]) -> dict[str, bytes]:
    """Root-level files whose name (upper-cased) is in `wanted`.

    For the licence: it sits beside the repository, not inside the skill, so it is
    outside every sparse pattern. There are one or two of them, so reading them
    through the promisor one at a time costs less than widening the checkout.
    """
    listing = _git(["ls-tree", "-z", "--format=%(objectmode) %(objectname) %(path)",
                    "FETCH_HEAD"], work)
    found: dict[str, bytes] = {}
    oids: list[str] = []
    names: list[str] = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        mode, oid, relative = record.decode("utf-8").split(" ", 2)
        if mode in BLOB_MODES and relative.upper() in wanted:
            names.append(relative)
            oids.append(oid)
    for name, blob in zip(names, _read_blobs(work, oids)):
        found[name] = blob
    return found
