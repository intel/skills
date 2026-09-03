#!/usr/bin/env python3
"""The digest that binds a measurement to the skill bytes it measured.

`perf/hw-results.json` identifies its subject by `skill_name` and `skill_version`.
Two things silently detach the evidence from the artifact it describes:

  - editing SKILL.md leaves the results advertising a measurement of instructions
    that no longer exist, and nothing in the repository can tell;
  - a run can measure a *different copy* of the same skill. Not hypothetical: the
    Intel material these skills were adapted from carries its own
    dpnp-quickstart/SKILL.md under the same name, and the copy here was rewritten on
    the way in -- so every run taken before that describes bytes that are not the
    ones here.

A hash over everything the skill ships cannot fix this: it would cover perf/
itself, so recording the digest inside perf/hw-results.json would change the hash
meant to certify it. This digest covers only the files that determine what the
agent does -- SKILL.md, references/, scripts/, assets/ -- and excludes evals/ and
perf/, which are evidence about the skill rather than instructions to the agent.

Version is deliberately outside the payload. A version bump with identical
instructions does not invalidate a measurement; an unversioned SKILL.md edit must.
The exclusion list is a deny-list, so a new directory of instructions lands in the
digest automatically rather than escaping it.

Print the digest for a skill:

    python3 tools/behavior_digest.py skills/dpnp-quickstart
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Evidence about the skill, not instructions to the agent. perf/ must be excluded
# for the digest to be recordable inside it at all; evals/ is excluded for the same
# reason one step removed -- importing a result rewrites it.
EXCLUDED_DIRS = {"evals", "perf"}

DIGEST_FIELD = "skill_behavior_sha256"
DIGEST_LEN = 64


def behavior_files(skill_dir: Path) -> list[Path]:
    """Every shipped file that determines agent behavior, in a stable order."""
    files = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(skill_dir)
        if relative.parts[0] in EXCLUDED_DIRS:
            continue
        files.append(path)
    return files


def behavior_digest(skill_dir: Path) -> str:
    """sha256 over the behavior files of skill_dir.

    The relative path and the byte length go into the hash alongside the content, so
    renaming a reference file or splitting one file into two changes the digest. A
    digest over concatenated content alone would not see either.
    """
    digest = hashlib.sha256()
    for path in behavior_files(skill_dir):
        relative = path.relative_to(skill_dir).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} skills/<name>", file=sys.stderr)
        return 2
    skill_dir = Path(argv[1])
    if not (skill_dir / "SKILL.md").is_file():
        print(f"{skill_dir}: no SKILL.md there", file=sys.stderr)
        return 1
    print(behavior_digest(skill_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
