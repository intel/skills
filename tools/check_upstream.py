#!/usr/bin/env python3
"""Notice when an imported skill's upstream has moved past its pin, and propose the move.

`sync_external.py --check` proves the copy matches its pin; nothing proved the pin still
matches upstream, and a stale pin keeps every check green.

What is compared is the tree object id of each `external-path` at its own pin against the
same path at the tip of upstream's default branch, not upstream's HEAD: a skill whose
directory did not change is not proposed, however far upstream moved. No file content is
fetched.

Each changed skill gets its own pull request, which moves only that skill's
`external-commit` and re-vendors it with `sync_external.py --write`, so any subset can be
merged and the rest closed. The branch is named by the directory's tree id: a closed one
is not proposed again, and a newer change supersedes an open one. No other line states a
commit, so two of these pull requests never edit the same line.

    python3 tools/check_upstream.py                  # survey every pin, change nothing
    python3 tools/check_upstream.py --json           # the same survey, as data
    python3 tools/check_upstream.py --update         # move the pins and re-vendor, no git
    python3 tools/check_upstream.py --open-pr        # per skill: branch, commit, push, open
    python3 tools/check_upstream.py --open-pr --dry-run   # print what that would run
    python3 tools/check_upstream.py --open-pr --remote fork --against intel   # from a fork
    python3 tools/check_upstream.py --update xpu-system-setup   # one skill, or one upstream

`--remote` is pushed to and `--against` is opened against; it defaults to `--remote`.

Fails on a pin that no longer resolves or a pinned path that is gone upstream; an
upstream that merely moved does not fail, and one that cannot be reached warns.

`--open-pr` needs `git`, `gh`, and a token that may push and open a pull request. With
the workflow's `GITHUB_TOKEN`, its checks wait for a maintainer to approve them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from sync_external import external_entries

# BROKEN lives in upstream_git so --mutate can also break the parse done there.
from upstream_git import (
    BROKEN,
    SyncError,
    SyncUnavailable,
    commit_trees,
    default_head,
    parse_symref,
    subtree_hashes,
)
from validate_skills import (
    CATALOG_PATH,
    REPO_ROOT,
    Report,
    parse_catalog,
)

TOOL = "tools/check_upstream.py"
BRANCH_RE = re.compile(r"^sync/(.+)-([0-9a-f]{12})$")

# Each silent way this detector can break, and --self-test must catch every one.
MUTATIONS = {
    "M1": "set_pin moves every entry pinned at that commit, not the one skill",
    "M2": "set_pin moves the first entry pinned at that commit, whichever skill it is",
    "M3": "classify never finds a changed directory",
    "M4": "classify treats a pinned path that is gone upstream as unchanged",
    "M5": "groups keys by skill instead of by upstream repository",
    "M6": "parse_symref reads the commit off the symbolic-ref line",
    "M7": "branch_name leaves the tree id out of the branch",
    "M8": "parse_remote_url accepts a remote that is not a github repository",
    "M9": "head_ref leaves the fork's owner off a branch pushed somewhere else",
    "M10": "groups gives every skill of an upstream the first skill's pin",
    "M11": "superseded matches a branch by prefix, so it closes another skill's pull request",
}


def upstream_slug(repo: str) -> str:
    """`intel/gpu-ai-skills` from the pin's URL: how a message names an upstream."""
    return "/".join(repo.rstrip("/").removesuffix(".git").rsplit("/", 2)[-2:])


def branch_name(skill: str, tree: str) -> str:
    """The branch for this version of one skill, so a rerun finds its own pull request."""
    if BROKEN.which == "M7":
        return f"sync/{skill}"
    return f"sync/{skill}-{tree[:12]}"


def groups(entries: dict[str, dict[str, str]]) -> dict[str, dict]:
    """Every pinned skill, grouped by upstream so each upstream is asked and fetched once."""
    out: dict[str, dict] = {}
    for name, entry in entries.items():
        repo = entry["external-repo"].rstrip("/").removesuffix(".git")
        key = name if BROKEN.which == "M5" else repo
        group = out.setdefault(key, {"repo": repo, "skills": {}, "pins": {}})
        commit = entry["external-commit"]
        if BROKEN.which == "M10" and group["pins"]:
            commit = next(iter(group["pins"].values()))
        group["skills"][entry["external-path"].strip("/")] = name
        group["pins"][name] = commit
    return out


def paths_by_pin(group: dict) -> dict[str, list[str]]:
    """The pinned paths of one upstream, keyed by the commit each is pinned at."""
    out: dict[str, list[str]] = {}
    for path, name in sorted(group["skills"].items()):
        out.setdefault(group["pins"][name], []).append(path)
    return out


def classify(group: dict, before: dict[str, str], after: dict[str, str]) -> dict:
    """What each pinned directory at its pin and at upstream's tip say about the pins."""
    paths = sorted(group["skills"])
    gone = sorted(path for path in paths if path not in after)
    changed = sorted(
        group["skills"][path]
        for path in paths
        if path in after and before.get(path) != after[path]
    )
    if BROKEN.which == "M3":
        changed = []
    if BROKEN.which == "M4":
        gone = []
    if gone:
        return {
            "state": "broken",
            "changed": changed,
            "gone": gone,
            "detail": "no longer a directory upstream: " + ", ".join(gone),
        }
    return {"state": "stale" if changed else "identical", "changed": changed, "gone": []}


def survey(group: dict) -> dict:
    """One upstream's state, carrying everything its pull requests would need."""
    repo = group["repo"]
    paths = sorted(group["skills"])
    record = {
        "repo": repo,
        "skills": dict(sorted(group["skills"].items())),
        "pins": dict(sorted(group["pins"].items())),
        "trees": {},
    }
    try:
        default_branch, head = default_head(repo)
        record |= {"default-branch": default_branch, "head": head}
        if set(group["pins"].values()) == {head}:
            return record | {"state": "current", "changed": [], "gone": []}
        before: dict[str, str] = {}
        with commit_trees(repo, [*group["pins"].values(), head]) as work:
            for commit, pinned_paths in paths_by_pin(group).items():
                before |= subtree_hashes(work, commit, pinned_paths)
            after = subtree_hashes(work, head, paths)
    except SyncUnavailable as exc:
        return record | {"state": "unreachable", "changed": [], "gone": [], "detail": str(exc)}
    except SyncError as exc:
        return record | {"state": "broken", "changed": [], "gone": [], "detail": str(exc)}

    absent = sorted(set(paths) - set(before))
    if absent:
        return record | {
            "state": "broken",
            "changed": [],
            "gone": absent,
            "detail": "the pinned commit itself does not hold " + ", ".join(absent),
        }
    record["trees"] = {group["skills"][path]: tree for path, tree in sorted(after.items())}
    return record | classify(group, before, after)


def proposals(record: dict) -> list[dict]:
    """One pull request per changed skill of a stale upstream."""
    changed = record["changed"]
    return [
        {
            "repo": record["repo"],
            "default-branch": record["default-branch"],
            "head": record["head"],
            "skill": name,
            "pinned": record["pins"][name],
            "branch": branch_name(name, record["trees"][name]),
            "siblings": [other for other in changed if other != name],
        }
        for name in changed
    ]


def set_pin(text: str, skill: str, old: str, new: str) -> str:
    """Move one skill's `external-commit` in skills.yaml, and no other line."""
    if BROKEN.which == "M1":
        return text.replace(old, new)
    if BROKEN.which == "M2":
        return text.replace(old, new, 1)
    lines = text.split("\n")
    if f"- name: {skill}" not in lines:
        raise ValueError(f"no entry for {skill}")
    start = lines.index(f"- name: {skill}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("- ")), len(lines)
    )
    hits = [i for i in range(start, end) if lines[i] == f'  external-commit: "{old}"']
    if len(hits) != 1:
        raise ValueError(f"{skill} has {len(hits)} external-commit line(s) at {old[:12]}")
    lines[hits[0]] = f'  external-commit: "{new}"'
    return "\n".join(lines)


def bump(proposal: dict) -> None:
    """Move one skill's pin, then re-vendor it from the new commit."""
    skill, old, new = proposal["skill"], proposal["pinned"], proposal["head"]
    text = CATALOG_PATH.read_bytes().decode("utf-8")
    try:
        moved = set_pin(text, skill, old, new)
    except ValueError as exc:
        sys.exit(
            f"FAIL skills.yaml: {exc}, so this run has nothing to move there. Either it was "
            "edited by hand between the survey and now, or the pin is written some way this "
            "tool does not recognise"
        )
    CATALOG_PATH.write_bytes(moved.encode("utf-8"))
    print(f"     skills.yaml: {skill} {old[:12]} -> {new[:12]}")

    sys.stdout.flush()
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "sync_external.py"), "--write", skill],
        cwd=REPO_ROOT,
        check=False,
    )
    if done.returncode != 0:
        sys.exit(
            f"FAIL sync_external.py --write refused {new[:12]}. The pin moved in skills.yaml "
            "and the tree was not re-vendored, so nothing here is in a state to propose: read "
            "the failure above, and revert skills.yaml"
        )
    print(f"     re-vendored {skill} from {new[:12]}")


def commit_subject(proposal: dict) -> str:
    return (
        f"chore: move {proposal['skill']} to "
        f"{upstream_slug(proposal['repo'])}@{proposal['head'][:12]}"
    )


def compare_url(proposal: dict) -> str:
    return f"{proposal['repo']}/compare/{proposal['pinned'][:12]}...{proposal['head'][:12]}"


def pr_body(proposal: dict) -> str:
    """What a reviewer needs that the diff does not say. ASCII only: it crosses a console."""
    skill = proposal["skill"]
    lines = [
        f"`{skill}` has changed in `{upstream_slug(proposal['repo'])}` on "
        f"`{proposal['default-branch']}` and this repository's copy of it has not. Opened by "
        f"`{TOOL}`; the bytes are upstream's at the new commit, written by "
        "`tools/sync_external.py --write`.",
        "",
        f"- pin: `{proposal['pinned']}` -> `{proposal['head']}`",
        f"- upstream diff: {compare_url(proposal)}",
    ]
    if proposal["siblings"]:
        lines.append(
            "- also changed upstream, each in its own pull request: "
            + ", ".join(f"`{name}`" for name in proposal["siblings"])
        )
    lines += [
        "",
        f"Merge or close this one on its own. Closing it declines this version of `{skill}`: "
        "it is not proposed again, and the next change upstream is.",
        "",
        "What this does not decide is whether the new text should be merged. An import is "
        "somebody else's document, so a change in it is a change somebody made there: the "
        "checks below answer the structural half, and the diff is worth reading.",
        "",
        "If the checks are waiting for approval, this pull request was opened with "
        "`GITHUB_TOKEN`: select **Approve workflows to run** in the merge box to start them.",
    ]
    return "\n".join(lines) + "\n"


def git(*args: str) -> str:
    """Run git in this repository, or exit naming what failed."""
    done = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        detail = done.stderr.strip() or done.stdout.strip()
        sys.exit(f"FAIL git {args[0]}: {detail}")
    return done.stdout.strip()


def gh(*args: str, stdin: str | None = None) -> str:
    """Run the GitHub CLI, or exit naming what failed."""
    try:
        done = subprocess.run(
            ["gh", *args], cwd=REPO_ROOT, input=stdin, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        sys.exit(
            "FAIL gh is not on PATH. Opening a pull request needs it; the survey and "
            f"--update do not, and `{TOOL} --open-pr --dry-run` prints what it would run"
        )
    if done.returncode != 0:
        sys.exit(f"FAIL gh {args[0]}: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


def base_branch(remote: str) -> str:
    """This repository's default branch, as the remote states it."""
    done = subprocess.run(
        ["git", "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    ref = done.stdout.strip()
    return ref.split("/", 1)[1] if done.returncode == 0 and "/" in ref else "main"


def parse_remote_url(url: str, remote: str) -> str:
    """`owner/name` for a github remote, in either the https or the ssh spelling."""
    trimmed = url.strip().removesuffix("/").removesuffix(".git")
    for prefix in ("https://github.com/", "git@github.com:", "ssh://git@github.com/"):
        if trimmed.startswith(prefix):
            slug = trimmed[len(prefix) :]
            if slug.count("/") == 1 and all(slug.split("/")):
                return slug
    if BROKEN.which == "M8":
        return "intel/skills"
    raise SystemExit(
        f"FAIL remote {remote!r} is {url.strip()!r}, which is not a github repository. The "
        "pull request is opened against the repository that remote names, so it has to be "
        "one"
    )


def remote_slug(remote: str) -> str:
    """The repository `gh` acts on, from the remote: bare `gh` may pick another clone remote."""
    return parse_remote_url(git("remote", "get-url", remote), remote)


def head_ref(push_slug: str, target_slug: str, branch: str) -> str:
    """The `--head` for `gh pr create`: a branch on a fork needs the fork's owner."""
    if push_slug == target_slug or BROKEN.which == "M9":
        return branch
    return f"{push_slug.split('/')[0]}:{branch}"


def superseded(branch: str, skill: str, open_prs: list[dict]) -> list[int]:
    """Open pull requests for an older version of this skill."""
    if BROKEN.which == "M11":
        return [
            pull["number"]
            for pull in open_prs
            if pull["headRefName"].startswith(f"sync/{skill}-") and pull["headRefName"] != branch
        ]
    return [
        pull["number"]
        for pull in open_prs
        if (match := BRANCH_RE.match(pull["headRefName"]))
        and match.group(1) == skill
        and pull["headRefName"] != branch
    ]


def dry_run(proposal: dict, remote: str, against: str, base: str) -> None:
    """Print the run without doing any of it, including the body a reviewer would read."""
    branch, skill = proposal["branch"], proposal["skill"]
    target = remote_slug(against)
    head = head_ref(remote_slug(remote), target, branch)
    for line in (
        f"git switch --create {branch} {against}/{base}",
        f"{TOOL} --update {skill}",
        f"git add -- skills/{skill} skills.yaml && git commit -m {commit_subject(proposal)!r}",
        f"git push {remote} HEAD:refs/heads/{branch}",
        f"gh pr create --repo {target} --base {base} --head {head} "
        f"--title {commit_subject(proposal)!r}",
        f"gh pr close <each open sync/{skill}-* of an older tree> --comment 'Superseded ...'",
    ):
        print(f"     would run: {line}")
    print("     not asked here: whether that pull request already exists, which is a gh call")
    print("".join(f"     | {line}\n" for line in pr_body(proposal).splitlines()), end="")


def propose(proposal: dict, remote: str, against: str) -> None:
    """Branch, re-vendor, commit, push, and open the pull request for one changed skill."""
    branch, skill = proposal["branch"], proposal["skill"]
    if dirty := git("status", "--porcelain"):
        sys.exit(
            "FAIL the working tree has uncommitted changes, and a pull request from it would "
            f"carry them: {dirty.splitlines()[0]}"
        )
    target = remote_slug(against)
    listed = gh(
        "pr", "list", "--repo", target, "--state", "all", "--head", branch,
        "--json", "number,state",
    )
    if same := json.loads(listed):
        print(f"SKIP {skill}: #{same[0]['number']} ({same[0]['state'].lower()}) is {branch}")
        return
    open_prs = json.loads(
        gh(
            "pr", "list", "--repo", target, "--state", "open", "--limit", "500",
            "--json", "number,headRefName",
        )
    )

    # Off the target's default branch: a stale fork base would show up as deletions.
    base = base_branch(against)
    git("fetch", "--quiet", against)
    git("switch", "--quiet", "--create", branch, f"{against}/{base}")
    print(f"     branched {branch} off {against}/{base}")
    bump(proposal)
    git("add", "--", f"skills/{skill}", "skills.yaml")
    git("commit", "--quiet", "-m", commit_subject(proposal), "-m", compare_url(proposal))
    git("push", "--quiet", remote, f"HEAD:refs/heads/{branch}")
    url = gh(
        "pr", "create",
        "--repo", target,
        "--base", base,
        "--head", head_ref(remote_slug(remote), target, branch),
        "--title", commit_subject(proposal),
        "--body-file", "-",
        stdin=pr_body(proposal),
    )
    print(f"OPENED {url}")
    for number in superseded(branch, skill, open_prs):
        gh(
            "pr", "close", str(number), "--repo", target,
            "--comment", f"Superseded by {url}: `{skill}` changed again upstream.",
        )
        print(f"CLOSED #{number}, superseded")


def render(records: list[dict]) -> None:
    """One line per upstream, plus one per skill a stale one would propose."""
    for record in records:
        where = upstream_slug(record["repo"])
        count = len(record["skills"])
        if record["state"] == "current":
            print(f"OK   {where}: {record['default-branch']} is at the pin, {count} skill(s)")
        elif record["state"] == "identical":
            print(
                f"OK   {where}: {record['default-branch']} moved to {record['head'][:12]} and "
                f"none of the {count} pinned directories changed, nothing to update"
            )
        elif record["state"] == "stale":
            print(
                f"NEW  {where}: {record['default-branch']} at {record['head'][:12]}, "
                f"{len(record['changed'])} of {count} skill(s) changed"
            )
            for proposal in proposals(record):
                print(
                    f"     {proposal['skill']}: {proposal['pinned'][:12]} -> "
                    f"{proposal['head'][:12]}, branch {proposal['branch']}"
                )
        elif record["state"] == "unreachable":
            print(f"WARN {where}: not checked, upstream unreachable ({record['detail']})")
        else:
            print(f"FAIL {where}: {record['detail']}", file=sys.stderr)


def self_test() -> int:
    """Assert the detector against this catalog's real pins, offline. Each --mutate fails it."""
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(message)

    report = Report()
    entries = external_entries(parse_catalog(report), report)
    grouped = groups(entries)
    check(not report.errors, f"the catalog's pins parse: {'; '.join(report.errors) or 'clean'}")
    check(bool(grouped), f"{len(grouped)} group(s) carry a pin")
    check(
        len({group["repo"] for group in grouped.values()}) == len(grouped),
        "one group per upstream repository, so one ls-remote and one fetch per upstream",
    )
    check(
        all(
            path.rsplit("/", 1)[-1] == name
            for group in grouped.values()
            for path, name in group["skills"].items()
        ),
        "every pinned path ends in the skill it is imported as",
    )

    subject = max(grouped.values(), key=lambda group: (len(group["skills"]), group["repo"]))
    names = sorted(subject["pins"])
    last = names[-1]
    moved_entries = {
        name: entry | {"external-commit": "f" * 40} if name == last else entry
        for name, entry in entries.items()
    }
    split = groups(moved_entries)[subject["repo"]]
    last_path = next(path for path, name in split["skills"].items() if name == last)
    check(
        all(split["pins"][name] == subject["pins"][name] for name in names if name != last)
        and paths_by_pin(split)["f" * 40] == [last_path],
        f"one upstream at several pins is one group that fetches each ({last} moved alone)",
    )

    catalog = CATALOG_PATH.read_bytes().decode("utf-8")
    old, new = subject["pins"][last], "0" * 40
    moved = set_pin(catalog, last, old, new)
    lines = catalog.split("\n")
    differing = [i for i, (was, now) in enumerate(zip(lines, moved.split("\n"))) if was != now]
    check(
        [moved.split("\n")[i] for i in differing] == [f'  external-commit: "{new}"'],
        f"moving {last} rewrites exactly one line of skills.yaml ({len(differing)} changed)",
    )
    start = lines.index(f"- name: {last}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("- ")), len(lines))
    check(
        all(start < i < end for i in differing),
        f"and that line is in the {last} entry, not in another skill pinned at the same commit",
    )
    pins = {pin for group in grouped.values() for pin in group["pins"].values()}
    prose = [
        line.strip()
        for line in lines
        if any(pin[:12] in line for pin in pins) and not line.startswith("  external-commit: ")
    ]
    check(
        not prose,
        "no line but an entry's external-commit names a pin, so two pull requests for one "
        f"upstream never edit the same line: {prose[:1] or 'none'}",
    )

    paths = sorted(subject["skills"])
    same = {path: f"tree-{path}" for path in paths}
    check(
        classify(subject, same, same)["state"] == "identical",
        "an upstream that moved without touching a pinned directory needs no update",
    )
    one = paths[0]
    check(
        classify(subject, same, same | {one: "moved"})
        == {"state": "stale", "changed": [subject["skills"][one]], "gone": []},
        f"a changed directory is reported, and as the skill it is imported as "
        f"({subject['skills'][one]})",
    )
    check(
        classify(subject, same, {p: h for p, h in same.items() if p != one})["state"] == "broken",
        "a pinned path that is gone upstream fails rather than reporting an update",
    )

    head = "1234567890abcdef1234567890abcdef12345678"
    tree = "abcdef1234567890abcdef1234567890abcdef12"
    branch = branch_name(last, tree)
    check(
        branch == f"sync/{last}-{tree[:12]}",
        "the branch names the skill and its tree, so a closed one is not proposed again",
    )
    open_prs = [
        {"number": 1, "headRefName": f"sync/{last}-{'9' * 12}"},
        {"number": 2, "headRefName": f"sync/{last}-extra-{'9' * 12}"},
        {"number": 3, "headRefName": branch},
    ]
    check(
        superseded(branch, last, open_prs) == [1],
        "a newer version supersedes only this skill's older pull request, not a sibling's",
    )

    listing = f"ref: refs/heads/main\tHEAD\n{head}\tHEAD\n"
    check(
        parse_symref(listing, "x") == ("main", head),
        "ls-remote --symref is read as a branch and the commit at its tip",
    )
    try:
        parse_symref(f"{head}\tHEAD\n", "x")
        check(False, "a listing with no symbolic ref for HEAD fails")
    except SyncError:
        check(True, "a listing with no symbolic ref for HEAD fails")

    check(
        all(
            parse_remote_url(url, "r") == "intel/skills"
            for url in (
                "https://github.com/intel/skills",
                "https://github.com/intel/skills.git",
                "git@github.com:intel/skills.git",
            )
        ),
        "the remote a pull request is opened against is read off its url, in both spellings",
    )
    for url in ("../skills.git", "git@example.com:someone/skills.git"):
        try:
            parse_remote_url(url, "r")
            caught = False
        except SystemExit:
            caught = True
        check(caught, f"{url} is not a github repository and fails rather than being guessed")
    check(
        head_ref("intel/skills", "intel/skills", "sync/x-1") == "sync/x-1",
        "a branch pushed to the repository it is proposed to is named by itself",
    )
    check(
        head_ref("someone/skills", "intel/skills", "sync/x-1") == "someone:sync/x-1",
        "and one pushed to a fork carries the fork's owner, or it names a branch of this "
        "repository instead",
    )

    record = {
        **subject, "head": head, "default-branch": "main", "changed": [names[0], last],
        "trees": {name: tree for name in names},
    }
    body = pr_body(proposals(record)[1])
    check(
        head in body and old in body and last in body and names[0] in body,
        "the pull request body names both commits, its skill, and the others changed with it",
    )

    print(f"\n{'FAIL' if failures else 'PASS'} --self-test: {len(failures)} failure(s)")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Every flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="the survey, as JSON")
    mode.add_argument(
        "--update",
        action="store_true",
        help="move the pin of every skill that changed upstream, and re-vendor. No git, no PR",
    )
    mode.add_argument(
        "--open-pr",
        action="store_true",
        help="one branch, commit, push and pull request per skill that changed upstream",
    )
    mode.add_argument("--self-test", action="store_true", help="assert against this catalog")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --open-pr: print the commands and the body, change nothing",
    )
    parser.add_argument("--remote", default="origin", help="the remote to push to (origin)")
    parser.add_argument(
        "--against",
        help="the remote whose repository the pull request is opened against (--remote)",
    )
    parser.add_argument(
        "names", nargs="*", help="limit to these upstreams or skills (default: all)"
    )
    parser.add_argument("--mutate", choices=sorted(MUTATIONS), help=argparse.SUPPRESS)
    return parser


def wanted_by(names: set[str], group: dict) -> set[str]:
    """The skills of one upstream that the command line asks for."""
    skills = set(group["pins"])
    if not names or names & {group["repo"].lower(), upstream_slug(group["repo"]).lower()}:
        return skills
    return skills & names


def main() -> int:
    args = build_parser().parse_args()
    BROKEN.which = args.mutate
    if BROKEN.which:
        print(f"# mutation {BROKEN.which}: {MUTATIONS[BROKEN.which]}")

    if args.self_test:
        return self_test()

    report = Report()
    grouped = groups(external_entries(parse_catalog(report), report))
    for error in report.errors:
        print(f"FAIL {error}", file=sys.stderr)
    if report.errors:
        return 1
    if not grouped:
        print("OK   no imported skills")
        return 0

    names = {name.rstrip("/").removesuffix(".git").lower() for name in args.names}
    chosen = {key: group for key, group in sorted(grouped.items()) if wanted_by(names, group)}
    if not chosen:
        sys.exit(f"FAIL no pinned upstream or skill matches {', '.join(args.names)}")

    records = [survey(group) for group in chosen.values()]
    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    render(records)
    against = args.against or args.remote
    for record, group in zip(records, chosen.values()):
        if record["state"] != "stale":
            continue
        for proposal in proposals(record):
            if proposal["skill"] not in wanted_by(names, group):
                continue
            if args.update:
                bump(proposal)
            elif args.open_pr and args.dry_run:
                dry_run(proposal, args.remote, against, base_branch(against))
            elif args.open_pr:
                propose(proposal, args.remote, against)
    return 1 if any(record["state"] == "broken" for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
