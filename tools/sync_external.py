#!/usr/bin/env python3
"""Vendor every imported skill from the upstream commit `skills.yaml` pins.

An imported skill is a copy in this repository, not a link out of it. It used to be
a link: `skills/<name>/SKILL.md` was a generated pointer that named an upstream
repository, and `install` fetched the real files at install time. That is a smaller
diff and a worse repository, in four ways that all had the same shape -- the
directory looked like a skill and was not one:

  - `README` said a skill directory is self-contained. For two thirds of the catalog
    it held a stub naming a procedure it did not carry, and anyone who cloned the
    repository and pointed an agent at `skills/<name>/SKILL.md` -- the documented way
    to use a skill without the installer -- got the stub. Nothing failed;
  - `install` needed `git`, network access, and a hash manifest of somebody else's
    bytes, so the one command a new user runs first could fail for reasons that had
    nothing to do with this repository;
  - upstream's own `scripts/` sat outside the tree, so the content scan in
    `validate_skills.py` -- the check that reads what a skill would make an agent do
    -- never read the files an agent would actually run;
  - `.source.json` was documented in six places and shipped by nothing, because a
    pointer has no upstream files to record provenance for.

So the files are here, and `install` is a copy. What does not change is where the
decision lives: `skills.yaml` carries the pin (`external-repo`, `external-commit`,
`external-path`, `external-license`), a pull request that moves a pin is the unit a
reviewer reads, and this script is what makes the tree agree with it.

    python3 tools/sync_external.py --check            # network, CI gate
    python3 tools/sync_external.py --write            # re-vendor in place
    python3 tools/sync_external.py --write linux-perf # one skill

`--check` re-fetches the pinned commit, regenerates what `--write` would produce, and
byte-compares. That is what turns "someone edited an import" and "upstream rewrote
the commit under the pin" into a failing step rather than a discovery months later.
Upstream being unreachable warns instead: a gate that fails when someone else's
server hiccups is one contributors learn to ignore.

Bytes are upstream's, with one exception, applied mechanically and recorded in
`.source.json`. Upstream documents its commands with the path they have in the
repository they live in -- `plugins/intel-gpu-ai-skills/skills/<name>/scripts/x.sh`
-- and that path exists in no install: the file lands beside `SKILL.md` under
`~/.claude/skills/<name>/`. An agent reading the body runs the documented path and
gets ENOENT. Removing the upstream prefix changes no guidance and no claim, it makes
the documented path resolve where the file is, and because the rule is derived from
the pin rather than hand-applied, `--check` reproduces it exactly.

Why the source is an anonymous git fetch and not the GitHub API: it needs no token
and has no rate limit worth hitting, so the gate runs from a fork and from CI without
a credential. tools/upstream_git.py fetches the pinned subtree rather than the
repository around it, plus the root licence file, which is what lets this script
verify the licence claim against what upstream actually grants.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# The two exceptions are defined beside the fetch that raises them, and re-exported
# here because callers of this module have always imported them from it.
from upstream_git import (
    SyncError,
    SyncUnavailable,
    pinned_subtrees,
    root_files,
    subtree_files,
)

# EXTERNAL_KEYS, IMPORT_MANIFEST and REWRITE_NOTE come from the validator: it is the
# module that has to recognise what this one writes, so the strings have a single
# definition.
# external-license is the one field that cannot be derived from the pin -- a licence is
# a legal claim about someone else's bytes, so a human declares it and this script
# verifies the declaration against upstream rather than trusting it.
from validate_skills import (
    EXTERNAL_KEYS,
    IMPORT_MANIFEST,
    NOT_VENDORED_DIRS,
    REPO_ROOT,
    REWRITE_NOTE,
    SKILLS_DIR,
    Report,
    parse_catalog,
    split_frontmatter,
)

# Where a repository states its terms, matched case-insensitively on the file name.
LICENSE_FILES = {"LICENSE", "LICENSE.MD", "LICENSE.TXT", "COPYING", "COPYRIGHT.MD"}

# The phrase each licence uses to name itself in its own text. This table exists
# because searching a licence for its SPDX identifier fails in both directions, and
# both failures were observed against the repositories this script imports from:
#
#   - a licence text does not contain its identifier. The Apache 2.0 file says
#     "Apache License, Version 2.0" and never "Apache-2.0", so a substring search
#     rejects a correctly declared Apache or BSD import -- intel/gpu-ai-skills is
#     Apache-2.0 and `external-license: Apache-2.0` was refused;
#   - searched in prose the identifier matches words. "MIT" occurs inside "limit",
#     "submit" and "permitted", so `external-license: MIT` was accepted for that
#     same Apache-2.0 repository.
#
# A licence is the one field here a human declares, so the check on it has to be the
# strict one: an identifier this table does not know fails rather than passes, and
# adding one means writing down how that licence names itself.
SPDX_MARKERS = {
    "MIT": (r"MIT License",),
    "Apache-2.0": (r"Apache License,?\s+Version 2\.0",),
    "BSD-2-Clause": (r"BSD 2-Clause",),
    # A 3-clause text often carries no "BSD 3-Clause" title; what separates it from
    # the 2-clause form is the advertising clause itself.
    "BSD-3-Clause": (r"BSD 3-Clause", r"[Nn]either the name of"),
    "GPL-2.0": (r"GNU GENERAL PUBLIC LICENSE\s+Version 2",),
    "GPL-3.0": (r"GNU GENERAL PUBLIC LICENSE\s+Version 3",),
    "LGPL-2.1": (r"GNU LESSER GENERAL PUBLIC LICENSE\s+Version 2\.1",),
    "LGPL-3.0": (r"GNU LESSER GENERAL PUBLIC LICENSE\s+Version 3",),
    "MPL-2.0": (r"Mozilla Public License Version 2\.0",),
    "ISC": (r"ISC License",),
}

# Upstream stating its own licence in machine-readable form outranks the table: it is
# the author's declaration about their own file rather than our inference from a text.
SPDX_DECLARATION_RE = re.compile(r"SPDX-License-Identifier:\s*([^\s\"']+)")

# How the notice is written into each kind of file. A file type not here is not
# rewritten silently -- see rewrite_paths.
COMMENT_STYLES = {
    ".md": lambda note: f"<!-- {note} -->",
    ".sh": lambda note: f"# {note}",
    ".py": lambda note: f"# {note}",
}


def upstream_root(entry: dict[str, str], name: str) -> str:
    """The directory upstream keeps its skills in, derived from the pin.

    `external-path` is `<root>/<name>`, and `<root>` is the prefix upstream's own
    documentation uses when it names a file: `plugins/intel-gpu-ai-skills/skills` for
    one of the two upstreams here, plain `skills` for the other. Derived rather than
    configured, so a new upstream needs no entry anywhere.
    """
    path = entry["external-path"].strip("/")
    head, _, tail = path.rpartition("/")
    if tail != name:
        raise SyncError(
            f"{name}: external-path is {path!r}, whose last segment is {tail!r}. An "
            "import keeps upstream's own name, so the pinned path has to end in it"
        )
    return head


def rewrite_paths(
    name: str, root: str, names: set[str], relative: str, blob: bytes
) -> tuple[bytes, bool]:
    """Make upstream's repository-relative paths resolve where install writes them.

    Only a path naming a skill this catalog carries is touched, and only with the
    upstream root the pin derives. `<root>/<this skill>/x` becomes `x`, which is
    where the file sits both here and after install; `<root>/<sibling>/x` becomes
    `<sibling>/x`, which is what the sibling is called in both layouts. Everything
    else, including upstream prose *about* its own layout, is left alone: the needle
    is the root followed by a name, so a sentence ending at the root does not match.

    A file this cannot annotate is not rewritten quietly -- it fails, because a
    modified file has to be able to say that it was modified.
    """
    if not root:
        return blob, False
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return blob, False

    out = text
    for other in sorted(names, key=len, reverse=True):
        out = out.replace(f"{root}/{other}/", "" if other == name else f"{other}/")
    if out == text:
        return blob, False

    comment = COMMENT_STYLES.get(Path(relative).suffix.lower())
    if comment is None:
        raise SyncError(
            f"{name}: {relative} names an upstream path that does not resolve after "
            f"install, but tools/sync_external.py has no comment syntax for a "
            f"{Path(relative).suffix!r} file, so the rewrite could not be recorded in "
            "the file itself. Add one to COMMENT_STYLES"
        )
    return insert_note(out, relative, comment(REWRITE_NOTE)).encode("utf-8"), True


def insert_note(text: str, relative: str, note: str) -> str:
    """Put `note` where a reader of that file sees it first.

    After the frontmatter for a Markdown skill -- nothing may precede it, an agent
    parses it as the file's first bytes -- and after a shebang for a script, for the
    same reason. Otherwise the first line.
    """
    lines = text.split("\n")
    at = 0
    if relative.endswith(".md") and lines and lines[0].strip() == "---":
        closing = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if closing is not None:
            at = closing + 1
    elif lines and lines[0].startswith("#!"):
        at = 1
    lines.insert(at, note)
    return "\n".join(lines)


def verify_license(
    name: str,
    entry: dict[str, str],
    front: dict,
    content: bytes,
    files: dict[str, bytes],
    skill_md: str,
) -> None:
    """The declared licence must be the one upstream actually grants.

    `external-license` is the only field a human writes rather than derives from the
    pin, and it is a legal statement about someone else's work, so it is the field
    with the most to lose from a check that merely looks satisfied. Two sources are
    accepted, in order of authority: upstream declaring the licence itself (SKILL.md
    frontmatter or an SPDX-License-Identifier line), then the licence file at the
    pinned commit matched on how that licence names itself -- see SPDX_MARKERS for
    why the identifier is never searched for directly.

    Failing closed is deliberate. A licence this script cannot identify is not a
    licence it may wave through -- and since the import republishes upstream's bytes
    rather than pointing at them, it is this repository doing the republishing.
    """
    licence = entry["external-license"]
    text = content.decode("utf-8", "replace")
    declared = front.get("license")
    if not isinstance(declared, str) or not declared.strip():
        match = SPDX_DECLARATION_RE.search(text)
        declared = match.group(1).strip("\"'") if match else None
    if declared:
        if declared.strip().lower() == licence.lower():
            return
        raise SyncError(
            f"{name}: {skill_md} declares license {declared!r} at the pinned commit, but "
            f"skills.yaml pins external-license {licence!r}. Upstream's own statement is "
            "what this repository republishes, so the pin follows it"
        )

    # SPDX identifiers are case-insensitive, so `mit` names the same licence as `MIT`
    # and must not fall through to the "cannot verify this" branch.
    markers = next(
        (m for spdx, m in SPDX_MARKERS.items() if spdx.lower() == licence.strip().lower()),
        None,
    )
    if markers is None:
        raise SyncError(
            f"{name}: external-license {licence!r} is not an identifier this script can "
            f"verify. Upstream states no licence in {skill_md}, so the claim has to be "
            "checked against a licence file — add how that licence names itself to "
            "SPDX_MARKERS in tools/sync_external.py, or use the SPDX id of the licence "
            "the file actually carries"
        )
    licence_texts = [
        candidate_text.decode("utf-8", "replace")
        for candidate, candidate_text in files.items()
        if candidate.upper() in LICENSE_FILES
    ]
    if not licence_texts:
        raise SyncError(
            f"{name}: the pinned commit carries no licence file and {skill_md} states no "
            f"licence, so nothing there supports external-license {licence!r}. An "
            "import republishes someone else's terms; guessing them is not an option"
        )
    for marker in markers:
        if any(re.search(marker, candidate, re.I | re.S) for candidate in licence_texts):
            return
    raise SyncError(
        f"{name}: skills.yaml declares external-license {licence!r}, but no licence file "
        f"at the pinned commit names that licence and {skill_md} states none either. "
        f"Check what {entry['external-repo']} actually grants at "
        f"{entry['external-commit'][:12]} and pin that"
    )


def vendored(
    name: str,
    entry: dict[str, str],
    files: dict[str, tuple[bytes, int]],
    licences: dict[str, bytes],
    names: set[str],
) -> dict[str, tuple[bytes, int]]:
    """Every file this import should hold, keyed by path inside the skill directory.

    Fails rather than degrades on every disagreement it can see. A pin that resolves
    to a directory with no SKILL.md, or to a skill upstream calls something else, is
    not a pin to that skill.
    """
    root = upstream_root(entry, name)
    prefix = f"{entry['external-path'].strip('/')}/"
    subtree = {
        relative[len(prefix) :]: value
        for relative, value in files.items()
        if relative.startswith(prefix)
    }
    if "SKILL.md" not in subtree:
        raise SyncError(
            f"{name}: {prefix}SKILL.md is not in {entry['external-repo']} at "
            f"{entry['external-commit'][:12]} — external-path names no skill"
        )

    content = subtree["SKILL.md"][0]
    report = Report()
    front, _ = split_frontmatter(content.decode("utf-8"), f"{prefix}SKILL.md", report)
    if report.errors:
        raise SyncError(f"{name}: upstream frontmatter is unparsable: {report.errors[0]}")
    if front.get("name") != name:
        raise SyncError(
            f"{name}: upstream calls this skill {front.get('name')!r}. An import keeps "
            "the upstream name — a local rename gives the same skill two identities and "
            "an agent a name no other catalog resolves"
        )
    description = front.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SyncError(f"{name}: upstream SKILL.md has no description to route on")

    verify_license(name, entry, front, content, licences, f"{prefix}SKILL.md")

    out: dict[str, tuple[bytes, int]] = {}
    changed: list[str] = []
    for relative, (blob, mode) in sorted(subtree.items()):
        if relative.split("/")[0] in NOT_VENDORED_DIRS:
            raise SyncError(
                f"{name}: upstream ships {relative}, and {relative.split('/')[0]}/ is "
                "where this repository keeps its own eval and measurement data. Renaming "
                "either would hide one behind the other"
            )
        rewritten, was_changed = rewrite_paths(name, root, names, relative, blob)
        if was_changed:
            changed.append(relative)
        out[relative] = (rewritten, mode)

    out[IMPORT_MANIFEST] = (manifest(entry, changed), 0o644)
    return out


def manifest(entry: dict[str, str], changed: list[str]) -> bytes:
    """`.source.json` — what this skill is a copy of, and how it differs.

    The four fields CONTRIBUTING documents, plus the modification record when there is
    one: Apache-2.0 section 4(b) asks a redistributor to say which files it changed,
    and a reviewer asked to approve a copy of someone else's work should not have to
    diff it against upstream to find out.
    """
    data: dict[str, object] = {
        "repo": entry["external-repo"].rstrip("/").removesuffix(".git"),
        "path": entry["external-path"].strip("/"),
        "commit": entry["external-commit"],
        "license": entry["external-license"],
    }
    if changed:
        data["modified-files"] = changed
        data["modification"] = (
            "Upstream repository-relative paths rewritten to skill-relative paths, so "
            "the commands in the body resolve where this skill installs. Generated by "
            "tools/sync_external.py; no other change."
        )
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def on_disk(skill_dir: Path) -> dict[str, bytes]:
    """The bytes the skill directory holds now, keyed the same way `vendored` keys.

    This repository's own evals/ and perf/ are excluded: they are not upstream's and
    are not part of the comparison. Everything else under the directory is, which is
    what makes a file nobody vendored show up as drift rather than survive quietly.
    """
    out: dict[str, bytes] = {}
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(skill_dir).as_posix()
        if relative.split("/")[0] in NOT_VENDORED_DIRS:
            continue
        out[relative] = path.read_bytes()
    return out


def _git(args: list[str]) -> str:
    done = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return done.stdout if done.returncode == 0 else ""


def index_modes() -> dict[str, int]:
    """The file modes git records under skills/, read from the index and not from disk.

    A Windows working copy has no execute bit, so a script vendored there is committed
    non-executable unless something sets the mode explicitly -- and an installed
    `scripts/x.sh` that is not executable is a skill whose first command fails on the
    machine it was written for. The index is where the mode this repository publishes
    actually lives, so it is what `--write` sets and what `--check` compares.
    """
    modes = {}
    for line in _git(["ls-files", "--stage", "--", "skills"]).splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4:
            modes[fields[3]] = 0o755 if fields[0] == "100755" else 0o644
    return modes


def set_modes(skill_dir: Path, wanted: dict[str, int]) -> list[str]:
    """Stage each vendored file with the mode upstream gives it.

    `git update-index --add --chmod` is the only way to record an execute bit from a
    filesystem that has none, and it is why this tool touches the index at all. On a
    filesystem that does have one the chmod above has already done the job and this
    agrees with it.
    """
    notes = []
    current = index_modes()
    for relative, mode in sorted(wanted.items()):
        tracked = (skill_dir / relative).relative_to(REPO_ROOT).as_posix()
        if current.get(tracked) == mode:
            continue
        flag = "+x" if mode == 0o755 else "-x"
        if _git(["update-index", "--add", f"--chmod={flag}", "--", tracked]) == "":
            # An empty stdout is what success looks like here; a failure is reported
            # by the mode still disagreeing on the next run, which --check will say.
            pass
        notes.append(f"mode {flag} {relative}")
    return notes


def write_tree(skill_dir: Path, wanted: dict[str, tuple[bytes, int]]) -> list[str]:
    """Make the directory hold exactly `wanted`, and report what changed.

    Stale files are removed rather than left: an import that drops a reference file
    upstream would otherwise keep shipping ours, and a file no upstream commit
    contains is a file nobody reviews.
    """
    notes: list[str] = []
    current = on_disk(skill_dir) if skill_dir.is_dir() else {}
    for relative in sorted(set(current) - set(wanted)):
        (skill_dir / relative).unlink()
        notes.append(f"removed {relative}")
    for relative, (blob, mode) in sorted(wanted.items()):
        target = skill_dir / relative
        if current.get(relative) != blob:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            notes.append(f"{'updated' if relative in current else 'added'} {relative}")
        if mode == 0o755:
            target.chmod(0o755)
    for directory in sorted(
        (p for p in skill_dir.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)
    ):
        if directory.name not in NOT_VENDORED_DIRS and not any(directory.iterdir()):
            directory.rmdir()
    notes += set_modes(skill_dir, {rel: mode for rel, (_, mode) in wanted.items()})
    return notes


def differences(
    skill_dir: Path, wanted: dict[str, tuple[bytes, int]], current: dict[str, bytes]
) -> list[str]:
    """Every way the tree disagrees with the pin, in the words a fix needs."""
    out = []
    for relative in sorted(set(wanted) - set(current)):
        out.append(f"{relative} is missing")
    for relative in sorted(set(current) - set(wanted)):
        out.append(f"{relative} is not part of the pinned upstream skill")
    modes = index_modes()
    for relative in sorted(set(wanted) & set(current)):
        if wanted[relative][0] != current[relative]:
            out.append(f"{relative} differs from upstream at the pinned commit")
            continue
        tracked = (skill_dir / relative).relative_to(REPO_ROOT).as_posix()
        recorded = modes.get(tracked)
        if recorded is not None and recorded != wanted[relative][1]:
            out.append(
                f"{relative} is committed mode {recorded:o}, upstream has "
                f"{wanted[relative][1]:o}"
            )
    return out


def external_entries(
    catalog: dict[str, dict[str, str]], report: Report
) -> dict[str, dict[str, str]]:
    """Catalog entries carrying a pin, with the pin's fields all present."""
    entries = {}
    for name, entry in sorted(catalog.items()):
        if not any(key in entry for key in EXTERNAL_KEYS):
            continue
        missing = [key for key in EXTERNAL_KEYS if not entry.get(key)]
        if missing:
            report.error(f"skills.yaml: {name}: incomplete pin, missing {missing}")
            continue
        entries[name] = entry
    return entries


def fetch(repo: str, commit: str, paths: list[str]) -> tuple[dict[str, tuple[bytes, int]], dict[str, bytes]]:
    """The files under the pinned subtrees, plus any root-level licence file.

    Only the pinned subtrees are fetched: the licence lives at the root and is read
    through the promisor, and the repository around them is not downloaded at all.
    """
    with pinned_subtrees(repo, commit, paths) as work:
        files = subtree_files(work, paths)
        licences = root_files(work, LICENSE_FILES)
    if not files:
        raise SyncError(
            f"{repo} at {commit[:12]}: none of {sorted(paths)} exists at the pinned commit"
        )
    return files, licences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write", action="store_true", help="re-vendor each imported skill in place"
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail if a vendored skill differs from the upstream commit it pins",
    )
    parser.add_argument("names", nargs="*", help="limit to these skills (default: all)")
    args = parser.parse_args()

    report = Report()
    catalog = parse_catalog(report)
    entries = external_entries(catalog, report)
    if report.errors:
        for error in report.errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    everything = set(catalog)
    if args.names:
        unknown = [name for name in args.names if name not in entries]
        if unknown:
            print(f"FAIL not imported skills: {unknown}", file=sys.stderr)
            return 1
        entries = {name: entries[name] for name in args.names}
    if not entries:
        print("OK no imported skills")
        return 0

    # One fetch per (repo, commit) carrying every subtree pinned there. Several
    # imports usually share an upstream — twenty of them share one gpu-ai-skills pin
    # — so the paths are collected before anything is fetched: fetching per skill
    # would repeat the negotiation twenty times, and fetching one subtree at a time
    # would repeat it once per import as well.
    pinned: dict[tuple[str, str], list[str]] = {}
    for entry in entries.values():
        key = (entry["external-repo"], entry["external-commit"])
        pinned.setdefault(key, []).append(entry["external-path"].strip("/"))

    trees: dict[tuple[str, str], tuple[dict[str, tuple[bytes, int]], dict[str, bytes]]] = {}
    failures = 0
    warnings = 0
    for name, entry in entries.items():
        key = (entry["external-repo"], entry["external-commit"])
        where = f"skills/{name}/"
        try:
            if key not in trees:
                trees[key] = fetch(*key, pinned[key])
            files, licences = trees[key]
            wanted = vendored(name, entry, files, licences, everything)
        except SyncUnavailable as exc:
            print(f"WARN {name}: upstream unreachable, not checked ({exc})")
            warnings += 1
            continue
        except SyncError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            failures += 1
            continue

        skill_dir = SKILLS_DIR / name
        if args.write:
            skill_dir.mkdir(parents=True, exist_ok=True)
            notes = write_tree(skill_dir, wanted)
            if notes:
                print(f"WROTE {where} ({', '.join(notes)})")
            else:
                print(f"OK   {where} unchanged")
            continue

        if not skill_dir.is_dir():
            print(
                f"FAIL {where}: missing — run: python3 tools/sync_external.py --write {name}",
                file=sys.stderr,
            )
            failures += 1
            continue
        drift = differences(skill_dir, wanted, on_disk(skill_dir))
        if drift:
            print(
                f"FAIL {where}: does not match {entry['external-repo']} at "
                f"{entry['external-commit'][:12]}: "
                + "; ".join(drift[:4])
                + (f" (+{len(drift) - 4} more)" if len(drift) > 4 else "")
                + f". Either the copy was edited here or the pinned commit no longer "
                f"holds what it claims. Re-vendor with: python3 "
                f"tools/sync_external.py --write {name}",
                file=sys.stderr,
            )
            failures += 1
        else:
            print(f"OK   {where} {len(wanted)} file(s)")

    if failures:
        print(f"\n{failures} imported skill(s) out of sync", file=sys.stderr)
        return 1
    if warnings:
        print(f"\n{warnings} imported skill(s) not checked — upstream was unreachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
