#!/usr/bin/env python3
"""Structural gate for every skill in skills/.

Keyless and offline: no model call, no container, no credential. That is what
lets it block every pull request, including one from a fork.

Checks, in order of what breaks if they fail:

  1. Frontmatter carries name and description — what the agentskills.io spec
     requires, and nothing more. The licence is a fact about the skill, not a field
     an author has to remember: it is stated in skills.yaml, which is where every
     other fact this repository keeps about a skill already lives. license stays
     optional in frontmatter, and if it is there it must agree with the catalog —
     an imported file may already carry upstream's own value, and rewriting it
     would be republishing somebody else's text under terms they did not choose.
     Any other key is passed through untouched — the spec is the floor for what a
     loader reads, not a ceiling on what a file may carry.
  2. name matches the directory exactly, so an agent resolving by name finds it.
  3. description is present and under the length limit. Under progressive
     disclosure this is the only text in context when the agent decides whether
     to open the skill, so an empty or vocabulary-free description makes the
     skill unreachable no matter how good the body is.
  4. The description names every product skills.yaml says the skill covers. Same
     reason as 3, one step further: a skill whose description omits its own
     product is unreachable *for that product*, and it fails silently — nothing
     errors, the agent simply never selects it.
  5. The files a skill ships carry none of the five content shapes no skill has a
     legitimate reason to contain. This is the one check whose subject is what the
     text does rather than whether it is well-formed.
  6. The body structure is the author's. No section is required of any skill: a
     document that has to fill fixed headings grows padding under the ones it has
     nothing to say about, and padding is what an agent reads instead of the answer.
  7. Every path mentioned in SKILL.md exists, and every relative markdown link
     resolves. An agent follows a pointer and improvises when it 404s.
  8. Every file a skill ships is mentioned by path, wherever in the skill it sits.
     A file nothing points at is either unreachable or an undeclared instruction.
     For an imported skill — one shipping .source.json — this warns instead of
     failing: the only way to satisfy it would be to edit another team's body, and
     an edited import no longer matches the text their measurements describe. The
     content scan in 5 does not relax; it reads every file either way.
  9. SKILL.md length. Target ~150 lines, warn at 250, fail at 500.
 10. skills.yaml and the skills/ tree agree, in both directions.
 11. Status preconditions. There are two statuses: 'published', which every entry
     is unless it says otherwise, and 'validated', which needs the three perf/
     files, references/official-sources.md, intel-hw-validated-on, and a capability
     suite. A validated skill's perf/hw-results.json must also carry the digest of
     the behavior files it measured, and that digest must be the current one --
     otherwise the numbers hold a status up for bytes that are no longer here.
 12. An imported skill and its pin agree. skills.yaml carries the upstream
     repository, commit, path, and licence; the skill ships .source.json stating the
     same four, NOTICE names the upstream it republishes, and any file .source.json
     records as modified carries the notice saying so. Checked offline by comparing
     the copy's own claims against the catalog's; the byte comparison against
     upstream itself needs the network and lives in tools/sync_external.py --check.
 13. evaluation/harbor/suites.json, the tasks/ tree, and skills.yaml agree: every
     suite names a real skill, every 'covers' resolves to a declared capability,
     and a task marked implemented is actually runnable. A task no suite claims is
     still checked for structure and still run by the oracle smoke job; what it
     lacks is a capability to count toward, which is a maintainer's judgement and
     warns until 'validated' claims the measurement exists.

Everything above is offline. `--check-links` adds the guide's "link check": it
resolves the http(s) links in each SKILL.md, and for an imported skill the upstream
repository and its pinned commit, so a moved or deleted target is caught before a
reader follows a dead reference. It is opt-in and a separate CI step because it is
the one check whose result depends on someone else's server being up. A dead link in
an imported body warns rather than fails, for the same reason 8 does: the fix belongs
upstream, and an edited import no longer matches its pin. The pinned commit itself
still fails, because that one is this repository's own claim rather than upstream's.

Needs Python 3.11 or newer for tomllib. Nothing else outside the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

# Said here rather than left to `import tomllib`, which on 3.10 fails as
# ModuleNotFoundError -- a message that reads like a missing dependency in a tool whose
# whole claim is that it needs none.
if sys.version_info < (3, 11):
    sys.exit(
        "tools/validate_skills.py needs Python 3.11 or newer (tomllib); "
        f"this is {sys.version.split()[0]}"
    )

import tomllib  # noqa: E402  -- after the version guard, deliberately
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from behavior_digest import DIGEST_FIELD, behavior_digest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
CATALOG_PATH = REPO_ROOT / "skills.yaml"
NOTICE_PATH = REPO_ROOT / "NOTICE"
SUITES_PATH = REPO_ROOT / "evaluation" / "harbor" / "suites.json"
TASKS_DIR = REPO_ROOT / "evaluation" / "harbor" / "tasks"
INSTRUCTIONS_DIR = REPO_ROOT / "evaluation" / "harbor" / "instructions"

TASK_STATUSES = ("implemented", "planned")
TASK_ROLES = ("smoke", "discriminating")
# Files Harbor needs to build, solve, and score a task. Missing any one of them
# turns into a runtime error minutes into a container build, not a clear message.
TASK_REQUIRED = ("task.toml", "instruction.md", "environment/Dockerfile")
TASK_REQUIRED_DIRS = ("solution", "tests")

REQUIRED_TOP_LEVEL_KEYS = {"name", "description"}

# The licences this repository publishes. Not a constant to compare a skill against:
# a skill written here is Apache-2.0, and an imported one keeps the terms it arrived
# under. What the list is for is that adding a third licence is a line in a pull
# request somebody reviews, rather than a string that appears in one catalog entry.
PUBLISHED_LICENSES = {"Apache-2.0", "MIT"}
DEFAULT_LICENSE = "Apache-2.0"

SKILL_TYPES = {"tool-skill", "problem-skill"}

# Two statuses, and one of them is the default. Four statuses described a review
# pipeline rather than the skill: 'draft' and 'incubating' differed by whether a
# human had looked at the skill, which is what merging it already means, and
# 'promoted' by an editorial judgement no check can make. What is left is the one
# distinction a reader acts on — whether hardware measurements stand behind this
# skill — and it is the only one the validator can hold anybody to.
STATUSES = ("published", "validated")
DEFAULT_STATUS = "published"

# No section is required of any skill. The body structure is the author's: a
# document that has to fill fixed headings grows padding under the ones it has
# nothing to say about, and padding is what an agent reads instead of the answer.
# What the body is held to instead is length, and that every path in it resolves —
# see check_paths.

# Where a skill records the upstream documentation its claims come from. Optional
# until the skill claims 'validated', at which point it is required: a measurement
# published under Intel's name has to be traceable to the sources it rests on, and
# 'published' makes no such claim.
OFFICIAL_SOURCES = "references/official-sources.md"

# The validated checklist names these three by name: both arms, the human-readable
# summary, and the config that makes the run repeatable. A perf/ directory holding
# only a summary is a number with no measurement behind it.
PERF_REQUIRED = ("hw-results.json", "summary.md", "benchmark_config.json")

# A fenced block or an HTML comment is not prose: a path or a URL inside one is a
# command argument rather than a reference, and the checks that read the body strip
# both out first.
#
# The leading [ \t]* is not decoration. An Implementation Guide is a numbered list,
# and a code block belonging to step 3 is indented under it — so most fences in a
# real skill are not flush left. Anchoring at column zero left those blocks in the
# "prose" this module derives links from, which made --check-links probe the
# install-command URLs inside them: a conda channel root answers 404 to a GET while
# serving repodata beneath it, so the check failed a pull request over a command that
# was correct. See INLINE_CODE_RE for the same reasoning one indirection down.
FENCE_RE = re.compile(r"^[ \t]*```.*?^[ \t]*```", re.MULTILINE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
UPSTREAM_URL_RE = re.compile(r"https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

# The pin an imported skill carries in skills.yaml, and the file the copy carries
# beside SKILL.md saying the same thing. Both live here rather than in
# tools/sync_external.py because the generator imports this module, and two copies
# of a name is how a check and the thing it checks drift apart.
EXTERNAL_KEYS = ("external-repo", "external-commit", "external-path", "external-license")
IMPORT_MANIFEST = ".source.json"
# The keys .source.json is documented to carry, in CONTRIBUTING.md, mapped to the
# catalog key each one must agree with.
MANIFEST_KEYS = {
    "repo": "external-repo",
    "commit": "external-commit",
    "path": "external-path",
    "license": "external-license",
}
# The notice tools/sync_external.py writes into a file it rewrites. Apache-2.0
# section 4(b) asks a modified file to say so; this is that sentence, and the check
# that it is present reads the same constant the generator writes, wrapped in
# whatever comment syntax that file type uses.
REWRITE_NOTE = (
    "Modified by intel/skills: upstream repository-relative paths rewritten to "
    "resolve where this skill installs. Provenance: .source.json"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DESCRIPTION_MAX = 1024
LINES_WARN = 250
LINES_FAIL = 500

# Shortest product token worth requiring in a description. Below this a token is
# too generic to prove anything: 'ai' or 'go' appears in ordinary English prose.
PRODUCT_TOKEN_MIN = 3

# Content shapes a skill has no legitimate reason to carry. A skill is text an
# agent obeys and scripts it may run, so this deny-list is short on purpose: each
# entry is a shape with no honest use in a skill, because a check that fires on
# correct content is a check someone will switch off, and then it protects nothing.
# Anything subtler than these belongs to review, not to a regular expression.
DANGEROUS_CONTENT = (
    (
        "curl-pipe-shell",
        re.compile(r"\b(?:curl|wget)\b[\s\S]{0,160}\|\s*(?:sudo\s+)?(?:sh|bash)\b"),
        "piping a download straight into a shell runs code nobody reviewed, and an "
        "agent will repeat it verbatim — install from a package manager, or download, "
        "verify, then run",
    ),
    (
        "destructive-root-delete",
        re.compile(r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*f|-rf|-fr)\s+(?:/|\$HOME|~)(?:\s|$)"),
        "a recursive delete rooted at /, $HOME, or ~ destroys the user's machine when "
        "the agent runs it with a variable one step off",
    ),
    (
        "prompt-injection",
        re.compile(
            r"\b(?:ignore|override|bypass)\b.{0,80}\b(?:previous|prior|system|developer)\b"
            r".{0,80}\binstructions?\b",
            re.IGNORECASE,
        ),
        "a skill is loaded into the agent's own context, so text telling it to "
        "disregard its instructions is an injection whatever the intent behind it",
    ),
    (
        "secret-exfiltration",
        re.compile(
            r"\b(?:exfiltrate|leak|steal|upload|send)\b.{0,80}"
            r"\b(?:secret|token|credential|api[_-]?key|password)\b",
            re.IGNORECASE,
        ),
        "no skill needs to move a credential anywhere; if one genuinely must read a "
        "token, say which variable and stop there",
    ),
    (
        "disable-security-controls",
        re.compile(
            r"\b(?:disable|turn off|bypass)\b.{0,80}"
            r"\b(?:firewall|antivirus|security|selinux|app(?:armor)?|secret scanning)\b",
            re.IGNORECASE,
        ),
        "telling an agent to switch off a protection is advice it will follow without "
        "the context a human had for deciding it was safe",
    ),
)

# Extensions worth reading for the scan above. A skill ships prose, scripts, and
# data; anything else in it is caught by the mention rules instead.
CONTENT_SCAN_SUFFIXES = {".md", ".txt", ".py", ".sh", ".json", ".yaml", ".yml", ".toml"}

# A skill-local path mentioned in prose or a code fence: references/x.md,
# scripts/y.py. The lookbehind keeps a longer path that merely ends this way
# (skills/linux-perf/references/flow-a.md) from being read as skill-local.
MENTION_RE = re.compile(r"(?<![\w/-])(?:references|scripts|assets)/[A-Za-z0-9_./-]+")

# Directories inside a skill that the mention and content-safety rules skip. An
# eval case has to be able to name the phrase it tests for, and perf/ is generated
# measurement data rather than text an agent loads.
SCAN_EXCLUDE_DIRS = {"evals", "perf"}
# The same two directories, under the name the reason goes by in
# tools/sync_external.py: they are this repository's own evidence about an imported
# skill, so they are not part of the copy and are not compared against upstream.
NOT_VENDORED_DIRS = SCAN_EXCLUDE_DIRS
# Files that may ship without being mentioned. SKILL.md is the document doing the
# mentioning. .source.json records where an import came from, and an imported
# SKILL.md keeps upstream's text as it was — text that cannot refer to an artifact
# this repository invented.
MENTION_EXEMPT = {"SKILL.md", ".source.json"}

# A markdown link target: the part inside the parentheses. Read from prose with
# fences and inline code removed, because a C++ lambda — [&](const auto& r) — is
# this shape too, and a skill teaching oneTBB is full of them.
LINK_TARGET_RE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)")
URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

# Links for --check-links. Trailing markdown and sentence punctuation is stripped
# afterwards; a URL is allowed to end in a closing paren of its own (rare, but
# Wikipedia-style targets exist), so only unbalanced trailers come off.
URL_RE = re.compile(r"https?://[^\s<>\"'`\]\\]+")
# A URL inside code is excluded. `conda install -c https://software.repos.intel.com/
# python/conda` is an argument to a command, and a package-channel root answers 404
# to a GET while serving the repodata beneath it — so probing it fails a pull
# request over documentation that is correct. The check is for links a reader
# follows, which are the ones in prose.
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
LINK_TIMEOUT = 15
LINK_WORKERS = 8
# A 429 is retried rather than accepted as 'unknown'. Rate limiting only warns, which
# is the right call for someone else's outage — but a host that rate-limits us instead
# of answering hides whatever the real answer was, and a gate that reports 'reachable'
# because it was throttled is worse than one that reports nothing. Documentation hosts
# do this readily: eight workers over ~100 URLs concentrated on a handful of hosts is
# enough to trigger it.
LINK_RETRIES = 3
LINK_RETRY_CAP = 8.0
# Some documentation hosts answer a bare urllib request with 403. This is the
# minimum that makes them behave; it is not an attempt to look like a browser.
LINK_HEADERS = {
    "User-Agent": "intel-skills-link-check/1.0 (+https://github.com/intel/skills)",
    "Accept": "*/*",
}


def has_files(directory: Path) -> bool:
    """True when the directory exists and holds at least one file.

    Emptiness matters, not existence: git does not track empty directories, so a
    leftover empty references/ in a working copy would make the local run
    disagree with the run on a fresh CI clone.
    """
    return directory.is_dir() and any(p.is_file() for p in directory.rglob("*"))


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def split_frontmatter(text: str, where: str, report: Report) -> tuple[dict, str]:
    """Parse the leading YAML block. Flat keys plus a one-level metadata map.

    Deliberately not a full YAML parser: the spec's metadata values are flat
    strings, so anchors and nested maps are out of scope and rejected loudly
    rather than silently half-parsed.
    """
    if not text.startswith("---\n"):
        report.error(f"{where}: must begin with a '---' frontmatter block")
        return {}, text

    end = text.find("\n---", 4)
    if end == -1:
        report.error(f"{where}: frontmatter block is not closed with '---'")
        return {}, text

    block = text[4:end]
    body = text[end + 4 :]

    data: dict[str, object] = {}
    metadata: dict[str, str] = {}
    current_key: str | None = None
    in_metadata = False

    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indented = line[0] in " \t"
        stripped = line.strip()

        if indented:
            if in_metadata and ":" in stripped:
                key, _, value = stripped.partition(":")
                metadata[key.strip()] = value.strip().strip("\"'")
            elif current_key in data and isinstance(data.get(current_key), str):
                # Continuation of a folded block scalar (description: >-).
                joiner = " " if data[current_key] else ""
                data[current_key] = f"{data[current_key]}{joiner}{stripped}"
            continue

        if ":" not in stripped:
            report.error(f"{where}: unparsable frontmatter line: {stripped!r}")
            continue

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        current_key = key
        in_metadata = key == "metadata"

        if in_metadata:
            if value:
                report.error(f"{where}: metadata must be a block, not {value!r}")
            continue

        if value in (">-", ">", "|", "|-"):
            data[key] = ""
        else:
            data[key] = value.strip("\"'")

    if metadata or in_metadata:
        data["metadata"] = metadata
    return data, body


def check_frontmatter(front: dict, skill_dir: Path, report: Report) -> str:
    where = f"skills/{skill_dir.name}/SKILL.md"

    # Two keys are required — the two the agentskills.io spec requires. A key beyond
    # them is passed through untouched: the spec is the floor for what a loader reads,
    # not a ceiling on what a file may carry, and rejecting an extra key sends the
    # author editing frontmatter that no agent behaviour depends on.
    missing = REQUIRED_TOP_LEVEL_KEYS - set(front)
    if missing:
        report.error(f"{where}: missing required keys: {sorted(missing)}")

    name = front.get("name")
    if isinstance(name, str):
        if not NAME_RE.match(name):
            report.error(f"{where}: name must be kebab-case, got {name!r}")
        if name != skill_dir.name:
            report.error(
                f"{where}: name {name!r} does not match directory {skill_dir.name!r}"
            )
        if len(name) > 64:
            report.error(f"{where}: name exceeds 64 characters")

    description = front.get("description")
    if isinstance(description, str):
        if not description.strip():
            report.error(f"{where}: description is empty — the skill is unreachable")
        elif len(description) > DESCRIPTION_MAX:
            report.error(
                f"{where}: description is {len(description)} chars, limit is {DESCRIPTION_MAX}"
            )

    metadata = front.get("metadata") or {}
    if not isinstance(metadata, dict):
        report.error(f"{where}: metadata must be a map of flat string values")
        metadata = {}

    for key, value in metadata.items():
        if not isinstance(value, str):
            report.error(f"{where}: metadata.{key} must be a string")

    skill_type = metadata.get("intel-skill-type", "tool-skill")
    if skill_type not in SKILL_TYPES:
        report.error(
            f"{where}: metadata.intel-skill-type must be one of {sorted(SKILL_TYPES)}, "
            f"got {skill_type!r}"
        )

    # The licence is not checked here: it is a fact about the skill rather than a
    # property of this file, and the catalog is what states it. See check_licenses,
    # which needs skills.yaml and so runs once both are read.
    return skill_type


def shipped_files(skill_dir: Path) -> list[Path]:
    """The files a skill ships that an agent may load, sorted.

    Everything under the skill directory except SCAN_EXCLUDE_DIRS. Not a fixed list
    of subdirectory names: a file at the skill's root is loaded the same as one
    under references/, so a rule that names three directories leaves the shortest
    path into the repository — skills/x/notes.md — checked by nothing.
    """
    out = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(skill_dir)
        if rel.parts[0] in SCAN_EXCLUDE_DIRS:
            continue
        out.append(path)
    return out


def check_paths(body: str, skill_dir: Path, report: Report) -> None:
    where = f"skills/{skill_dir.name}/SKILL.md"

    mentioned = {m.rstrip(".,;:)`") for m in MENTION_RE.findall(body)}
    for rel in sorted(mentioned):
        if not (skill_dir / rel).is_file():
            report.error(f"{where}: points at {rel}, which does not exist")

    # A relative markdown link is a path the reader is told to follow, whatever
    # directory it names, so it is checked by resolution rather than by prefix.
    # Links leaving the repository are somebody else's to keep alive.
    prose = INLINE_CODE_RE.sub("", COMMENT_RE.sub("", FENCE_RE.sub("", body)))
    repo_root = REPO_ROOT.resolve()
    for target in dict.fromkeys(LINK_TARGET_RE.findall(prose)):
        rel = target.split("#", 1)[0].strip()
        if not rel or rel.startswith(("/", "<")) or URL_SCHEME_RE.match(rel):
            continue
        resolved = (skill_dir / rel).resolve()
        if repo_root not in resolved.parents:
            continue
        if not resolved.exists():
            report.error(f"{where}: links to {rel}, which does not exist")

    # An unmentioned file is either unreachable or an undeclared instruction, so for
    # a skill written here this fails. For an import it warns: the body belongs to
    # another team, the only way to satisfy the rule would be to edit their text, and
    # an edited import no longer matches the file their measurements describe — the
    # thing MAINTAINERS.md asks an importer not to do. Warning still surfaces the
    # file, and check_content_safety reads it regardless of who wrote the body.
    imported = (skill_dir / ".source.json").is_file()
    for path in shipped_files(skill_dir):
        rel = path.relative_to(skill_dir).as_posix()
        if rel in MENTION_EXEMPT:
            continue
        if rel not in body:
            note = (
                f"{where}: ships {rel} but never mentions it by path — "
                "an unmentioned file is unreachable or an undeclared instruction"
            )
            if imported:
                report.warn(f"{note} (imported: upstream's body is kept as it is)")
            else:
                report.error(note)


def check_content_safety(skill_dir: Path, report: Report) -> None:
    """Scan the files a skill ships for the shapes in DANGEROUS_CONTENT.

    Every other check here asks whether a skill is well-formed. This one asks what
    its text would make an agent do, which is a different question and the reason a
    catalog of instructions cannot rely on review alone: a reviewer reads the diff,
    and the risk is in the file that ships.

    Scope is everything the skill ships, because everything it ships is loadable.
    evals/ and perf/ are deliberately out: an eval case is allowed to name a phrase
    precisely because a skill must not contain it, and scanning the case would fail
    the skill for testing the rule.
    """
    for path in shipped_files(skill_dir):
        if path.suffix.lower() not in CONTENT_SCAN_SUFFIXES:
            continue
        where = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for content_id, pattern, why in DANGEROUS_CONTENT:
            match = pattern.search(text)
            if match is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            report.error(f"{where}:{line}: {content_id} — {why}")


def extract_urls(body: str) -> list[str]:
    """The http(s) links in a SKILL.md, cleaned of trailing markdown punctuation.

    Fenced blocks, inline code and HTML comments come out first: a URL there is a
    command argument or a machine endpoint rather than a link, and see INLINE_CODE_RE
    for why probing those produces failures only a worse document can fix.
    """
    prose = INLINE_CODE_RE.sub(" ", COMMENT_RE.sub(" ", FENCE_RE.sub(" ", body)))
    urls: list[str] = []
    for raw in URL_RE.findall(prose):
        url = raw
        while url and url[-1] in ".,;:!?)":
            # A closing paren is only punctuation when it has no opener inside the
            # URL — otherwise it belongs to the target, as in .../Foo_(bar).
            if url[-1] == ")" and url.count("(") >= url.count(")"):
                break
            url = url[:-1]
        if url and url not in urls:
            urls.append(url)
    return urls


def link_targets(body: str, source: dict | None) -> list[tuple[str, str, bool]]:
    """(url, what it is, came from an imported body) triples to probe for one skill.

    For an imported skill the pinned commit is probed as well as the repository, from
    .source.json rather than from the body: the body is upstream's own text and says
    nothing about where this copy came from. The two fail differently, and the
    difference is the whole point of pinning — an unreachable repository means the
    provenance leads nowhere, while a reachable repository with an unreachable commit
    means the SHA was rewritten out of history, so nothing can be re-verified against
    what we shipped.

    The third element separates those two origins. A link in an imported body is
    upstream's, and the pin is ours, so a dead one is a different kind of finding
    even though both are dead URLs.
    """
    imported = source is not None
    targets = [(url, "link", imported) for url in extract_urls(body)]
    if not source:
        return targets

    repo = str(source.get("repo", "")).rstrip("/").removesuffix(".git")
    commit = str(source.get("commit", ""))
    if UPSTREAM_URL_RE.fullmatch(repo) and COMMIT_RE.match(commit):
        targets.append((f"{repo}/commit/{commit}", "pinned commit", False))
    return targets


def retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
    """How long to wait before retrying a throttled request.

    Retry-After comes in two forms and only the seconds form is honoured. The HTTP
    date form is what a host sends when the wait is minutes, which is longer than a
    CI step should hold a pull request open for — exponential backoff and then a
    warning is the better answer there.
    """
    header = exc.headers.get("Retry-After") if exc.headers else None
    try:
        wait = float(header)
    except (TypeError, ValueError):
        wait = 2.0**attempt
    return min(max(wait, 0.0), LINK_RETRY_CAP)


def probe_url(url: str) -> tuple[str, str]:
    """Reachability of one URL: ('ok'|'dead'|'unknown', detail).

    'dead' is reserved for an answer from the server that the target does not
    exist. Everything else — a 5xx, DNS trouble, a timeout — is 'unknown', because a
    validator that fails a pull request when someone else's CDN hiccups teaches
    contributors to ignore it.

    HEAD first, GET on rejection: some hosts answer HEAD with 403 or 405 while
    serving the page. The GET is not read, only opened.

    A 429 is retried with backoff before it is allowed to become 'unknown'. It is the
    one 'unknown' a host returns *instead of* the real answer rather than alongside
    it, so accepting it on the first try makes the verdict depend on how busy someone
    else's CDN was — the same commit passing and failing minutes apart, with a dead
    link the throttled run never got far enough to see.
    """
    for attempt in range(LINK_RETRIES + 1):
        throttled = None
        for method in ("HEAD", "GET"):
            request = urllib.request.Request(url, method=method, headers=LINK_HEADERS)
            try:
                with urllib.request.urlopen(request, timeout=LINK_TIMEOUT) as response:
                    return "ok", f"{response.status}"
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 410):
                    return "dead", f"HTTP {exc.code}"
                if exc.code == 429:
                    throttled = exc
                    break
                if method == "HEAD" and exc.code in (403, 405, 400, 501):
                    continue
                return "unknown", f"HTTP {exc.code}"
            except urllib.error.URLError as exc:
                return "unknown", f"{type(exc.reason).__name__}: {exc.reason}"
            except (TimeoutError, OSError) as exc:
                return "unknown", f"{type(exc).__name__}: {exc}"
        if throttled is None:
            return "unknown", "no method accepted"
        if attempt < LINK_RETRIES:
            time.sleep(retry_after(throttled, attempt))
    return "unknown", f"HTTP 429 after {LINK_RETRIES + 1} attempts"


def check_links(
    targets: list[tuple[str, str, str, bool]], report: Report
) -> None:
    """Probe every collected link.

    targets are (where, url, kind, from an imported body) quadruples. A dead link
    warns rather than fails when it came from an imported body, matching check 8: the
    body is upstream's, editing it here would break the byte-compare against the pin
    that tools/sync_external.py --check enforces, and the repair has to land upstream
    and arrive through a moved pin. The warning names that so it is actionable rather
    than noise. Everything this repository wrote — its own skills, and the pin itself
    — still fails.
    """
    if not targets:
        return
    unique = sorted({url for _, url, _, _ in targets})
    with ThreadPoolExecutor(max_workers=LINK_WORKERS) as pool:
        results = dict(zip(unique, pool.map(probe_url, unique)))

    for where, url, kind, upstream_body in targets:
        verdict, detail = results[url]
        if verdict == "dead":
            note = f"{where}: {kind} {url} is gone ({detail})"
            if upstream_body:
                report.warn(
                    f"{note} (imported: upstream's body is kept as it is — fix it "
                    "upstream, then move external-commit)"
                )
            else:
                report.error(note)
        elif verdict == "unknown":
            report.warn(f"{where}: {kind} {url} was not reachable ({detail})")

    checked = len(unique)
    dead = {url for url, (verdict, _) in results.items() if verdict == "dead"}
    ours = {url for _, url, _, imported in targets if url in dead and not imported}
    summary = f"link check: {checked} URL(s), {len(dead)} gone"
    if dead - ours:
        summary += f", {len(dead - ours)} of them only in imported bodies (warned)"
    print(summary)


def check_length(text: str, skill_dir: Path, report: Report) -> None:
    where = f"skills/{skill_dir.name}/SKILL.md"
    lines = len(text.splitlines())
    if lines > LINES_FAIL:
        report.error(f"{where}: {lines} lines exceeds the {LINES_FAIL}-line limit")
    elif lines > LINES_WARN:
        report.warn(
            f"{where}: {lines} lines is over the {LINES_WARN}-line soft limit — "
            "move detail into references/ with a lazy-load hint"
        )


def parse_catalog(report: Report) -> dict[str, dict[str, str]]:
    """Read skills.yaml. Flat list of '- key: value' records, no nesting."""
    if not CATALOG_PATH.is_file():
        report.error("skills.yaml: missing — every skill needs a registry entry")
        return {}

    entries: list[dict[str, str]] = []
    for raw in CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            entries.append({})
            stripped = stripped[2:]
        if not entries:
            report.error(f"skills.yaml: content before the first '-' entry: {stripped!r}")
            continue
        if ":" not in stripped:
            report.error(f"skills.yaml: unparsable line: {stripped!r}")
            continue
        key, _, value = stripped.partition(":")
        entries[-1][key.strip()] = value.strip().strip("\"'")

    catalog: dict[str, dict[str, str]] = {}
    for entry in entries:
        name = entry.get("name")
        if not name:
            report.error(f"skills.yaml: entry without a name: {entry}")
            continue
        if name in catalog:
            report.error(f"skills.yaml: duplicate entry for {name!r}")
        catalog[name] = entry
    return catalog


def check_perf_subject(name: str, skill_dir: Path, report: Report) -> None:
    """A validated claim must name the bytes it measured, and they must be these.

    `skill_name` plus `skill_version` cannot identify what was measured: an edit to
    SKILL.md under the same version leaves the numbers describing instructions that
    no longer exist, and a run against another copy of the same skill is
    indistinguishable from a run against this one. See tools/behavior_digest.py.

    A mismatch is an error rather than a warning because the status is the claim: a
    skill is `validated` on the strength of those numbers, and once they describe
    different bytes the status is unsupported. Withdraw it back to `published` or
    re-measure -- editing the digest by hand is not one of the two.
    """
    results_path = skill_dir / "perf" / "hw-results.json"
    where = f"skills/{name}/perf/hw-results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.error(f"{where}: unreadable ({exc})")
        return
    if not isinstance(results, dict):
        report.error(f"{where}: must be a JSON object")
        return

    recorded = results.get(DIGEST_FIELD)
    if recorded is None:
        report.error(
            f"{where}: missing {DIGEST_FIELD} — a validated status has to name the "
            f"skill bytes it measured (get it with: "
            f"python3 tools/behavior_digest.py skills/{name})"
        )
        return
    if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
        report.error(f"{where}: {DIGEST_FIELD} must be a lowercase sha256 digest")
        return

    current = behavior_digest(skill_dir)
    if recorded != current:
        report.error(
            f"{where}: {DIGEST_FIELD} does not match the current skill "
            f"(measured {recorded[:12]}, current {current[:12]}) — SKILL.md, "
            "references/, scripts/ or assets/ changed after the measurement, so the "
            "numbers describe bytes that are no longer here. Re-measure, or move the "
            "status back to 'published' until you can."
        )


def check_catalog(
    catalog: dict[str, dict[str, str]], skill_types: dict[str, str], report: Report
) -> None:
    for name in sorted(set(skill_types) - set(catalog)):
        report.error(f"skills.yaml: no entry for skills/{name}/")
    for name in sorted(set(catalog) - set(skill_types)):
        report.error(f"skills.yaml: entry {name!r} has no skills/{name}/ directory")

    for name, entry in sorted(catalog.items()):
        if name not in skill_types:
            continue
        skill_dir = SKILLS_DIR / name

        # Absent means 'published'. A field whose only legal value in a first pull
        # request is the one the validator would have assumed is a field that
        # teaches an author nothing and fails their build once.
        status = entry.get("status") or DEFAULT_STATUS
        if status not in STATUSES:
            report.error(
                f"skills.yaml: {name}: status must be one of {list(STATUSES)}, got {status!r}"
            )
        if not entry.get("maintainer"):
            report.error(f"skills.yaml: {name}: maintainer is required")

        has_perf = has_files(skill_dir / "perf")

        # evals/evals.json is not required at any status. It records what a correct
        # answer contains, which is useful and voluntary; a skill whose value is a
        # workflow rather than a set of facts has nothing to put in it, and requiring
        # the file from such a skill produces cases written to satisfy the check.
        if status == "validated":
            if not has_perf:
                report.error(f"skills/{name}/: status {status!r} requires perf/")
            else:
                for filename in PERF_REQUIRED:
                    if not (skill_dir / "perf" / filename).is_file():
                        report.error(
                            f"skills/{name}/: status {status!r} requires perf/{filename}"
                        )
                check_perf_subject(name, skill_dir, report)
        if status == "validated" and not (skill_dir / OFFICIAL_SOURCES).is_file():
            report.error(
                f"skills/{name}/: status {status!r} requires {OFFICIAL_SOURCES} — "
                "a validated claim has to name the upstream documentation behind it"
            )

        # intel-source-ledger is the catalog's pointer at that same file. Empty is a
        # legitimate value below 'validated'; a value naming a file that is not
        # there is not, because the catalog is what a reader searches instead of
        # opening every skill.
        ledger = entry.get("intel-source-ledger") or ""
        if ledger:
            if not (REPO_ROOT / ledger).is_file():
                report.error(
                    f"skills.yaml: {name}: intel-source-ledger points at {ledger!r}, "
                    "which does not exist"
                )
            elif ledger != f"skills/{name}/{OFFICIAL_SOURCES}":
                report.error(
                    f"skills.yaml: {name}: intel-source-ledger must be "
                    f"'skills/{name}/{OFFICIAL_SOURCES}', got {ledger!r}"
                )
        elif status == "validated":
            report.error(
                f"skills.yaml: {name}: status {status!r} requires intel-source-ledger"
            )
        if status == "validated" and not entry.get("intel-hw-validated-on"):
            report.error(
                f"skills.yaml: {name}: status {status!r} requires intel-hw-validated-on"
            )


def effective_license(entry: dict[str, str]) -> str:
    """What a skill is published under, according to the catalog.

    'license' for a skill this repository publishes on its own terms, the pin's
    'external-license' for an import, and Apache-2.0 for an entry that says neither,
    which is the repository's own licence and what a skill written here inherits.
    """
    return entry.get("license") or entry.get("external-license") or DEFAULT_LICENSE


def check_licenses(
    catalog: dict[str, dict[str, str]],
    skill_types: dict[str, str],
    fronts: dict[str, dict],
    report: Report,
) -> None:
    """The catalog states the licence; frontmatter may repeat it but may not contradict it.

    Two failures, and neither is a missing field. One is a licence this repository does
    not publish, which is a decision for review rather than a value in a catalog entry.
    The other is a SKILL.md whose own 'license' disagrees with the catalog: an import
    arrives carrying upstream's value, and if the catalog says something else then one
    of the two is republishing that text under terms nobody granted. An entry that
    states no licence is not an error — Apache-2.0 is what this repository publishes
    under, and a skill written here does not restate it.
    """
    for name, entry in sorted(catalog.items()):
        if name not in skill_types:
            continue  # a catalog entry with no directory is check_catalog's to report
        declared = effective_license(entry)
        if declared not in PUBLISHED_LICENSES:
            report.error(
                f"skills.yaml: {name}: license {declared!r} is not one this repository "
                f"publishes ({', '.join(sorted(PUBLISHED_LICENSES))}). Adding one is a "
                "line in PUBLISHED_LICENSES, reviewed with the skill that needs it"
            )
        stated = (fronts.get(name) or {}).get("license")
        if stated and stated != declared:
            report.error(
                f"skills/{name}/SKILL.md: license is {stated!r} but skills.yaml says "
                f"{declared!r}. The catalog is where the licence is recorded, so the "
                "two cannot differ — and neither one may be edited to match the other "
                "without knowing which is true"
            )


def read_source(skill_dir: Path, report: Report) -> dict | None:
    """The skill's .source.json, or None when it ships none.

    Unparsable is an error and not a missing file: a provenance record nothing can
    read is worse than no record, because the directory still looks like an import.
    """
    path = skill_dir / IMPORT_MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        report.error(f"skills/{skill_dir.name}/{IMPORT_MANIFEST}: not valid JSON — {exc}")
        return None
    if not isinstance(data, dict):
        report.error(
            f"skills/{skill_dir.name}/{IMPORT_MANIFEST}: must be a JSON object stating "
            f"{sorted(MANIFEST_KEYS)}"
        )
        return None
    return data


def check_imports(
    catalog: dict[str, dict[str, str]],
    sources: dict[str, dict | None],
    report: Report,
) -> None:
    """An imported skill and the pin it was copied from must agree.

    The pin lives in skills.yaml, the copy lives under skills/, and .source.json is
    the copy's own statement of where it came from. What can go wrong offline is the
    two drifting apart: the commit is bumped in the catalog and the copy is left as it
    was, or the copy is edited by hand and now claims a commit, a path or a licence
    the catalog does not pin. Either leaves a directory that works — an agent reads
    real content — while the provenance beside it describes something else, and that
    is the failure a link check cannot see.

    Whether the bytes still match the pinned commit needs upstream, so it is
    `tools/sync_external.py --check` in its own CI step, alongside the link check and
    for the same reason. This function only compares the repository against itself,
    which is what makes it part of the keyless gate.
    """
    notice = NOTICE_PATH.read_text(encoding="utf-8") if NOTICE_PATH.is_file() else ""
    for name in sorted(set(catalog) | set(sources)):
        entry = catalog.get(name) or {}
        source = sources.get(name)
        pinned = [key for key in EXTERNAL_KEYS if entry.get(key)]
        where = f"skills/{name}/{IMPORT_MANIFEST}"

        if not pinned and source is None:
            continue
        if not pinned:
            report.error(
                f"{where}: this skill says it is a copy of {source.get('repo')!r}, but "
                f"skills.yaml pins nothing for {name}. The catalog is where an import is "
                f"reviewed, so add {list(EXTERNAL_KEYS)} there — or delete "
                f"{IMPORT_MANIFEST} if the skill was written here"
            )
            continue
        missing = [key for key in EXTERNAL_KEYS if not entry.get(key)]
        if missing:
            report.error(
                f"skills.yaml: {name}: incomplete pin, missing {missing}. The copy under "
                "skills/ is produced from those fields (python3 tools/sync_external.py "
                "--write), so without them there is nothing to reproduce it from and "
                "nothing to check it against"
            )
            continue
        if source is None:
            report.error(
                f"skills/{name}/: skills.yaml pins an upstream for this skill but it "
                f"ships no {IMPORT_MANIFEST}, so nothing in the installed skill says "
                f"whose work it is. Re-vendor it: python3 tools/sync_external.py "
                f"--write {name}"
            )
            continue

        repo = entry["external-repo"].rstrip("/").removesuffix(".git")
        commit = entry["external-commit"]
        path = entry["external-path"].strip("/")
        if not repo.startswith("https://github.com/"):
            report.error(
                f"skills.yaml: {name}: external-repo must be an https github.com URL, "
                f"got {entry['external-repo']!r}"
            )
        if not COMMIT_RE.match(commit):
            report.error(
                f"skills.yaml: {name}: external-commit must be a full lowercase "
                f"40-character SHA, got {commit!r} — a branch or tag names bytes that can "
                "change under a copy nobody re-reviews"
            )
        # An import keeps upstream's own name, and the pin is what says which upstream
        # directory it is: a path whose last segment is something else pins a different
        # skill than the one it sits in.
        if path.rpartition("/")[2] != name:
            report.error(
                f"skills.yaml: {name}: external-path is {path!r}, whose last segment is "
                f"{path.rpartition('/')[2]!r}. An import keeps the upstream name, so the "
                "pinned path has to end in it"
            )

        expected = {
            "repo": repo,
            "commit": commit,
            "path": path,
            "license": entry["external-license"],
        }
        for key, catalog_key in sorted(MANIFEST_KEYS.items()):
            stated = source.get(key)
            if stated is None:
                report.error(
                    f"{where}: no {key!r}. An import records all of "
                    f"{sorted(MANIFEST_KEYS)}, so the installed skill carries its own "
                    "provenance and does not depend on this catalog to be readable"
                )
            elif str(stated).rstrip("/").removesuffix(".git") != expected[key]:
                report.error(
                    f"{where}: {key} is {stated!r} but skills.yaml pins {catalog_key} "
                    f"{expected[key]!r}. Re-vendor after changing a pin: python3 "
                    f"tools/sync_external.py --write {name}"
                )

        # Both licences this repository publishes require the redistributor to pass
        # the terms and the attribution on. NOTICE is where that is done once for the
        # whole repository, so an import whose upstream is not named there is text
        # republished with nothing saying whose it is.
        if repo and repo not in notice:
            report.error(
                f"NOTICE: does not name {repo}, which skills/{name}/ is a copy of. "
                "Redistributing someone else's work means carrying its attribution; "
                "add the upstream, its licence and its commit to the third-party "
                "section"
            )

        # Apache-2.0 section 4(b) and honesty both: a file listed as modified has to
        # exist and has to say so in itself, or the record is a claim about bytes a
        # reader cannot find.
        modified = source.get("modified-files") or []
        if not isinstance(modified, list):
            report.error(f"{where}: modified-files must be a list of paths")
            continue
        for relative in modified:
            target = SKILLS_DIR / name / str(relative)
            if not target.is_file():
                report.error(
                    f"{where}: modified-files names {relative!r}, which this skill does "
                    "not ship"
                )
            elif REWRITE_NOTE not in target.read_text(encoding="utf-8", errors="replace"):
                report.error(
                    f"skills/{name}/{relative}: {IMPORT_MANIFEST} records this file as "
                    "modified, but the file itself does not say so. A redistributed file "
                    "carries the notice that it was changed — re-vendor it: python3 "
                    f"tools/sync_external.py --write {name}"
                )


def check_trigger_vocabulary(
    catalog: dict[str, dict[str, str]],
    descriptions: dict[str, str],
    imported: set[str],
    report: Report,
) -> None:
    """Every product the catalog claims must appear in the skill's description.

    `intel-products` in skills.yaml is what a reader searches; `description` is the
    only text in the agent's context when it decides whether to open the skill. When
    the two disagree the catalog promises coverage the agent can never route to, and
    nothing errors — the skill is simply never selected for that product.

    This is the one rule about description *content* that can be checked offline,
    because the catalog already states the answer. Whether the rest of the wording
    would trigger stays with review and with the description test in the guide.

    On an imported skill the same gap is reported as a warning, because the description
    is upstream's own text and this repository does not edit it: the only ways to
    satisfy an error would be to drop the product from the catalog or to rewrite
    someone else's frontmatter, and the first is what already happened -- an import
    shipped with `intel-products: ""` to get past the check. The finding is still true
    and still worth seeing, so it stays visible; closing it means asking upstream to
    name the product, not editing our copy.
    """
    for name, entry in sorted(catalog.items()):
        if name not in descriptions:
            continue  # a missing directory is check_catalog's error to report
        where = f"skills/{name}/SKILL.md"
        generated = name in imported
        unreachable = report.warn if generated else report.error
        products = [p.strip() for p in (entry.get("intel-products") or "").split(",")]
        products = [p for p in products if p]
        if not products:
            report.warn(
                f"skills.yaml: {name}: intel-products is empty, so nothing checks that "
                "the description carries the vocabulary a user would type"
            )
            continue

        haystack = descriptions[name].lower()
        for product in products:
            # Length floor first, and before the literal match as well as the split
            # one: a two-character name is a substring of ordinary English, so
            # accepting it would report a pass this check did not earn.
            parts = [
                part
                for part in re.split(r"[-_\s]+", product.lower())
                if len(part) >= PRODUCT_TOKEN_MIN
            ]
            if not parts:
                report.warn(
                    f"{where}: intel-products lists {product!r}, too short to look for "
                    "in prose — check by hand that the description would trigger"
                )
                continue
            # A product written as one token may be spelled apart in prose —
            # 'linux-perf' as "Linux ... perf". Accept that, but only when every
            # part long enough to mean something is present.
            if product.lower() in haystack or all(part in haystack for part in parts):
                continue
            unreachable(
                f"{where}: description never names {product!r}, which skills.yaml says "
                "this skill covers — under progressive disclosure the description is "
                "the only text in context, so the skill is unreachable for that "
                "product and fails silently"
                + (
                    ". The description of an imported skill is upstream's own text, so "
                    "this is fixed upstream or by not claiming the product here"
                    if generated
                    else ""
                )
            )


def check_suites(
    catalog: dict[str, dict[str, str]],
    skill_types: dict[str, str],
    imported: set[str],
    report: Report,
) -> None:
    """Cross-check evaluation/harbor/suites.json, tasks/, and skills.yaml.

    Three places name the same tasks, and nothing but this function stops them
    from disagreeing. Each direction has a way of failing quietly:

      a suite for a skill that does not exist    evaluates nothing, reports fine
      a skill with no suite                      reaches 'validated' unmeasured
      a task directory in no suite               runs in the oracle job, counted
                                                 toward no capability
      a 'covers' id no capability declares       coverage arithmetic silently
                                                 rounds a gap down to zero
      status 'implemented' with no directory     --include-task-name matches
                                                 nothing and the job still passes

    Portfolio minimums in policy — the existence of a suite at all, and a suite
    entry for a task on disk — are enforced as errors only for a skill claiming
    'validated'. Below that a gap is legitimate and reported as a warning: a suite
    is how a skill gets measured later, not a condition of merging it.
    """
    if not SUITES_PATH.is_file():
        report.error(f"{SUITES_PATH.relative_to(REPO_ROOT).as_posix()} is missing")
        return
    try:
        suites_doc = json.loads(SUITES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error(f"suites.json is not valid JSON: {exc}")
        return

    where = "suites.json"
    policy = suites_doc.get("policy") or {}
    suites = suites_doc.get("suites")
    if not isinstance(suites, list) or not suites:
        report.error(f"{where}: 'suites' must be a non-empty list")
        return

    required_classes = set(policy.get("required_capability_classes") or [])
    min_tasks = policy.get("minimum_tasks_per_skill") or 0
    min_discriminating = policy.get("minimum_discriminating_tasks_per_skill") or 0

    claimed_tasks: dict[str, str] = {}  # task name -> skill that claims it

    for suite in suites:
        skill = suite.get("skill")
        if not skill:
            report.error(f"{where}: a suite has no 'skill'")
            continue
        if skill not in skill_types:
            report.error(f"{where}: suite {skill!r} has no skills/{skill}/ directory")
            continue

        capabilities = {
            capability.get("id"): capability for capability in suite.get("capabilities") or []
        }
        for capability_id, capability in capabilities.items():
            klass = capability.get("class")
            if required_classes and klass not in required_classes:
                report.error(
                    f"{where}: {skill}: capability {capability_id!r} has class {klass!r}, "
                    f"not one of {sorted(required_classes)}"
                )

        tasks = suite.get("tasks") or []
        if suite.get("evaluated_here") is False or skill in imported:
            if tasks or capabilities:
                report.error(
                    f"{where}: {skill}: an entry evaluated elsewhere must declare no "
                    "tasks and no capabilities"
                )
            continue

        implemented: list[str] = []
        discriminating = 0
        covered_classes: set[str] = set()

        for task in tasks:
            name = task.get("name")
            if not name:
                report.error(f"{where}: {skill}: a task has no 'name'")
                continue
            if name in claimed_tasks:
                report.error(
                    f"{where}: task {name!r} is claimed by both {claimed_tasks[name]!r} "
                    f"and {skill!r}; a reward can only count for one skill"
                )
            claimed_tasks[name] = skill

            status = task.get("status")
            role = task.get("role")
            if status not in TASK_STATUSES:
                report.error(
                    f"{where}: {skill}/{name}: status must be one of "
                    f"{list(TASK_STATUSES)}, got {status!r}"
                )
            if role not in TASK_ROLES:
                report.error(
                    f"{where}: {skill}/{name}: role must be one of {list(TASK_ROLES)}, "
                    f"got {role!r}"
                )

            covers = task.get("covers") or []
            if not covers:
                report.error(f"{where}: {skill}/{name}: 'covers' is empty")
            for capability_id in covers:
                if capability_id not in capabilities:
                    report.error(
                        f"{where}: {skill}/{name}: covers {capability_id!r}, which no "
                        f"capability in this suite declares"
                    )

            task_dir = TASKS_DIR / name
            if status == "planned":
                if task_dir.exists():
                    report.error(
                        f"{where}: {skill}/{name}: status is 'planned' but "
                        f"evaluation/harbor/tasks/{name}/ exists — set it to 'implemented'"
                    )
                continue

            implemented.append(name)
            if role == "discriminating":
                discriminating += 1
            for capability_id in covers:
                if capability_id in capabilities:
                    covered_classes.add(capabilities[capability_id].get("class"))

            if not task_dir.is_dir():
                report.error(
                    f"{where}: {skill}/{name}: status is 'implemented' but "
                    f"evaluation/harbor/tasks/{name}/ does not exist"
                )
                continue
            check_task_dir(task_dir, skill, set(covers), set(skill_types), report)

        treatment = INSTRUCTIONS_DIR / f"use-{skill}.md"
        if implemented and not treatment.is_file():
            report.error(
                f"{where}: {skill}: {treatment.relative_to(REPO_ROOT).as_posix()} is "
                "missing — the skill arms would have no treatment instruction"
            )

        status = (catalog.get(skill) or {}).get("status")
        promoting = status == "validated"
        shortfalls = []
        if len(implemented) < min_tasks:
            shortfalls.append(f"{len(implemented)} implemented task(s), policy wants {min_tasks}")
        if discriminating < min_discriminating:
            shortfalls.append(
                f"{discriminating} discriminating task(s), policy wants {min_discriminating}"
            )
        missing_classes = sorted(required_classes - covered_classes)
        if missing_classes:
            shortfalls.append(f"no implemented task covers {', '.join(missing_classes)}")
        for shortfall in shortfalls:
            message = f"{where}: {skill}: {shortfall}"
            if promoting:
                report.error(f"{message} — required at status {status!r}")
            else:
                report.warn(message)

    # A capability suite is what a skill needs to be measured, not what it needs to
    # be merged. Writing one is the work of a maintainer with the hardware and the
    # portfolio in view, and demanding it in a first pull request asks the author to
    # design an evaluation for a skill nobody has reviewed yet. It becomes a hard
    # requirement exactly where a measurement is being claimed: at 'validated'.
    for name in sorted(set(skill_types) - {suite.get("skill") for suite in suites}):
        status = (catalog.get(name) or {}).get("status")
        message = f"{where}: no suite for skills/{name}/ — the skill cannot be measured"
        if status == "validated":
            report.error(f"{message}, which status {status!r} claims it has been")
        else:
            report.warn(f"{message} until one is added")

    # A task no suite lists is the same case one step down: the task is the
    # contributor's, the capability it counts toward is the maintainer's. It is still
    # checked for structure here and still run by the oracle smoke job, so the author
    # gets the feedback that matters — is it well-formed, is it solvable — without
    # having to design a suite first.
    if TASKS_DIR.is_dir():
        on_disk = {p.name for p in TASKS_DIR.iterdir() if p.is_dir()}
        for name in sorted(on_disk - set(claimed_tasks)):
            task_dir = TASKS_DIR / name
            declared = ((read_task(task_dir) or {}).get("metadata") or {}).get("skill")
            status = (catalog.get(declared) or {}).get("status")
            message = (
                f"evaluation/harbor/tasks/{name}/: no suite lists this task, so its "
                "reward counts toward no capability"
            )
            if status == "validated":
                report.error(f"{message} — required at status {status!r}")
            else:
                report.warn(f"{message} until a maintainer adds one")
            check_task_dir(task_dir, None, None, set(skill_types), report)


def read_task(task_dir: Path) -> dict | None:
    """The task's own task.toml, or None if it is absent or unparseable."""
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return None
    try:
        return tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return None


def check_task_dir(
    task_dir: Path,
    skill: str | None,
    covers: set[str] | None,
    known_skills: set[str],
    report: Report,
) -> None:
    """The task on disk must be runnable, and agree with whatever claims it.

    skill and covers are what suites.json says about the task, or None when no
    suite lists it. In that case the task's own declaration is all there is, so it
    is checked for being well-formed rather than for agreeing with a second source.
    """
    name = task_dir.name
    where = f"evaluation/harbor/tasks/{name}"
    for relative in TASK_REQUIRED:
        if not (task_dir / relative).is_file():
            report.error(f"{where}/{relative} is missing")
    for relative in TASK_REQUIRED_DIRS:
        if not has_files(task_dir / relative):
            report.error(f"{where}/{relative}/ is missing or empty")

    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return
    try:
        task = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        report.error(f"{where}/task.toml is not valid TOML: {exc}")
        return

    if not task.get("schema_version"):
        report.error(f"{where}/task.toml: schema_version is required")

    metadata = task.get("metadata") or {}
    declared_skill = metadata.get("skill")
    if skill is None:
        if not declared_skill:
            report.error(
                f"{where}/task.toml: [metadata] skill is required — it is what says "
                "which skill the task exercises"
            )
        elif declared_skill not in known_skills:
            report.error(
                f"{where}/task.toml: [metadata] skill is {declared_skill!r}, which is "
                "not a skill under skills/"
            )
    elif declared_skill != skill:
        report.error(
            f"{where}/task.toml: [metadata] skill is {declared_skill!r} but suites.json "
            f"lists this task under {skill!r}"
        )

    declared_covers = metadata.get("covers")
    if isinstance(declared_covers, str):
        report.error(
            f"{where}/task.toml: [metadata] covers must be a list of capability ids, "
            "not a string"
        )
    elif covers is None:
        if not declared_covers:
            report.error(f"{where}/task.toml: [metadata] covers is empty")
    elif set(declared_covers or []) != covers:
        report.error(
            f"{where}/task.toml: [metadata] covers {sorted(declared_covers or [])} "
            f"disagrees with suites.json {sorted(covers)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="also resolve the http(s) links in each SKILL.md, and an imported skill's "
        "upstream repository and pinned commit. Needs network; a 404 fails, a timeout "
        "warns, and a 404 in an imported body warns because the fix belongs upstream.",
    )
    args = parser.parse_args()

    report = Report()
    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    if not skill_dirs:
        print("FAIL no skills found under skills/", file=sys.stderr)
        return 1

    skill_types: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    fronts: dict[str, dict] = {}
    sources: dict[str, dict | None] = {}
    links: list[tuple[str, str, str, bool]] = []
    for skill_dir in skill_dirs:
        sources[skill_dir.name] = read_source(skill_dir, report)
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            report.error(f"skills/{skill_dir.name}/: SKILL.md is missing")
            continue

        text = skill_md.read_text(encoding="utf-8")
        front, body = split_frontmatter(text, f"skills/{skill_dir.name}/SKILL.md", report)
        skill_type = check_frontmatter(front, skill_dir, report)
        skill_types[skill_dir.name] = skill_type
        fronts[skill_dir.name] = front
        description = front.get("description")
        descriptions[skill_dir.name] = description if isinstance(description, str) else ""
        check_paths(body, skill_dir, report)
        check_content_safety(skill_dir, report)
        check_length(text, skill_dir, report)
        if args.check_links:
            where = f"skills/{skill_dir.name}/SKILL.md"
            links += [
                (where, url, kind, upstream_body)
                for url, kind, upstream_body in link_targets(
                    body, sources[skill_dir.name]
                )
            ]
            # references/ as well as SKILL.md. The guide asks for "the links in the
            # references table", and for a skill of any size those links are one
            # indirection away, in the file the table points at — official-sources.md
            # is nothing but links. On an imported skill these files are upstream's
            # too: sync_external.py vendors the whole directory, not just SKILL.md.
            upstream_body = sources[skill_dir.name] is not None
            for path in sorted((skill_dir / "references").glob("*.md")):
                rel = path.relative_to(REPO_ROOT).as_posix()
                links += [
                    (rel, url, "link", upstream_body)
                    for url in extract_urls(path.read_text(encoding="utf-8"))
                ]

    catalog = parse_catalog(report)
    imported = {name for name, source in sources.items() if source is not None}
    check_catalog(catalog, skill_types, report)
    check_licenses(catalog, skill_types, fronts, report)
    check_imports(catalog, sources, report)
    check_trigger_vocabulary(catalog, descriptions, imported, report)
    check_suites(catalog, skill_types, imported, report)
    if args.check_links:
        check_links(links, report)

    for warning in report.warnings:
        print(f"WARN {warning}")
    for error in report.errors:
        print(f"FAIL {error}", file=sys.stderr)

    if report.errors:
        print(
            f"\n{len(report.errors)} error(s) across {len(skill_dirs)} skill(s)",
            file=sys.stderr,
        )
        return 1
    if report.warnings and args.strict:
        print(f"\n{len(report.warnings)} warning(s) with --strict", file=sys.stderr)
        return 1

    print(f"OK {len(skill_dirs)} skill(s): {', '.join(sorted(skill_types))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
