#!/usr/bin/env python3
"""Notice when an imported skill's upstream has moved past its pin, and propose the move.

`sync_external.py --check` answers "is the copy still the pinned commit". Nothing
answered the other half: the pin is a commit, upstream keeps committing, and a copy that
matches its pin perfectly can be a year behind. That is a silent failure -- every check
in this repository stays green while what it publishes drifts away from what upstream
maintains.

What is compared is the *pinned directory*, not the repository around it: the tree object
id of `external-path` at the pin against the same path at the tip of upstream's default
branch. Upstream's HEAD moving is not news here -- most commits upstream touch none of
the skills this catalog imports, and treating every one of them as an update would open a
pull request that moves 23 pins, rewrites 23 `.source.json` files and changes no skill
text at all. A directory hash answers the question that matters, and answers it for
13.75 KiB of transfer: no file content is fetched and nothing is checked out.

When a pinned directory did change, the update is mechanical and this tool applies it --
move the commit everywhere the repository writes it down (`skills.yaml`, including the
comment introducing each import group, and `NOTICE`), re-vendor with
`sync_external.py --write`, then open one pull request per upstream.

What it deliberately does not decide is whether the new bytes should be merged. That is
the pull request's own business: its checks answer the structural half, and a reviewer
reads the diff -- an import is somebody else's document, so a change in it is a change
somebody made there and may be one this catalog does not want.

    python3 tools/check_upstream.py                  # survey every pin, report, change nothing
    python3 tools/check_upstream.py --json           # the same survey, for a workflow to act on
    python3 tools/check_upstream.py --update         # move the pins and re-vendor, no git
    python3 tools/check_upstream.py --open-pr        # branch, commit, push, open the pull request
    python3 tools/check_upstream.py --open-pr --dry-run   # print what that would run
    python3 tools/check_upstream.py --open-pr --remote fork --against intel   # from a fork

The workflow pushes to the repository it runs in, where the branch and the pull request
live in the same place. A maintainer running this by hand pushes to their fork and opens
against this repository, so the two are separate: `--remote` is pushed to, `--against`
is opened against, and it defaults to `--remote`.

Exit status is about this repository, not about upstream: a pin that no longer resolves,
or a pinned path that is no longer a directory upstream, fails. An upstream that has
merely moved ahead does not -- that is what the pull request is for -- and an upstream
that cannot be reached warns, for the same reason the link check does.

`--open-pr` needs `git` and `gh`, and a token that may push a branch and open a pull
request. A pull request opened with the workflow's own `GITHUB_TOKEN` carries no checks:
GitHub does not start workflow runs for events that token causes. The pull request says
so in its body, and the fix is a token belonging to an account -- see
.github/workflows/upstream-sync.yml.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from sync_external import external_entries

# BROKEN is defined beside the fetch rather than here: --mutate has to be able to break
# the parse of upstream's default branch too, and that lives in the transport module.
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
    NOTICE_PATH,
    REPO_ROOT,
    Report,
    parse_catalog,
)

TOOL = "tools/check_upstream.py"

# Every way this detector can break is silent -- an upstream that moved and was not
# reported reads exactly like an upstream nobody has touched -- so each break is
# available as a mutation and --self-test has an assertion that must catch it.
MUTATIONS = {
    "M1": "rewrite_sha moves the full commit id but not the twelve characters used in prose",
    "M2": "rewrite_sha replaces every commit id it finds, not this upstream's",
    "M3": "classify never finds a changed directory",
    "M4": "classify treats a pinned path that is gone upstream as unchanged",
    "M5": "groups keys by skill instead of by upstream repository",
    "M6": "parse_symref reads the commit off the symbolic-ref line",
    "M7": "branch_name leaves the target commit out of the branch",
    "M8": "parse_remote_url accepts a remote that is not a github repository",
    "M9": "head_ref leaves the fork's owner off a branch pushed somewhere else",
}


def upstream_slug(repo: str) -> str:
    """`intel/gpu-ai-skills` from the pin's URL: how a message names an upstream."""
    return "/".join(repo.rstrip("/").removesuffix(".git").rsplit("/", 2)[-2:])


def branch_name(repo: str, head: str) -> str:
    """The branch a pull request for this move goes on.

    Derived from the target commit rather than from the date or the run number, so a
    second run proposes the same branch, finds its own pull request, and stops -- which
    is the whole of this bot's idempotence.
    """
    slug = upstream_slug(repo).rsplit("/", 1)[-1]
    if BROKEN.which == "M7":
        return f"sync/{slug}"
    return f"sync/{slug}-{head[:12]}"


def groups(report: Report) -> dict[str, dict]:
    """Every pin in the catalog, grouped by the upstream repository it names."""
    out: dict[str, dict] = {}
    for name, entry in external_entries(parse_catalog(report), report).items():
        repo = entry["external-repo"].rstrip("/").removesuffix(".git")
        key = name if BROKEN.which == "M5" else repo
        group = out.setdefault(
            key, {"repo": repo, "pinned": entry["external-commit"], "skills": {}}
        )
        if group["pinned"] != entry["external-commit"]:
            report.error(
                f"skills.yaml: {repo} is pinned at both {group['pinned'][:12]} and "
                f"{entry['external-commit'][:12]}. One upstream is one pin here: a second "
                "commit fetches the same repository twice and splits one upstream move "
                "across two reviews"
            )
            continue
        group["skills"][entry["external-path"].strip("/")] = name
    return out


def classify(group: dict, before: dict[str, str], after: dict[str, str]) -> dict:
    """What two revisions of one upstream's pinned directories say about the pin."""
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
    """One upstream's state, carrying everything a pull request for it would need."""
    repo, pinned = group["repo"], group["pinned"]
    paths = sorted(group["skills"])
    record = {"repo": repo, "pinned": pinned, "skills": dict(sorted(group["skills"].items()))}
    try:
        default_branch, head = default_head(repo)
        record |= {
            "default-branch": default_branch,
            "head": head,
            "branch": branch_name(repo, head),
        }
        if head == pinned:
            return record | {"state": "current", "changed": [], "gone": []}
        with commit_trees(repo, [pinned, head]) as work:
            before = subtree_hashes(work, pinned, paths)
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
    return record | classify(group, before, after)


def rewrite_sha(text: str, old: str, new: str) -> str:
    """Move a pin's commit everywhere one file writes it down.

    Both the object id and the twelve characters this repository uses for it in prose:
    `skills.yaml` states the pin in every entry of an upstream and again in the comment
    that introduces the group, and `NOTICE` repeats it for each upstream republished
    here. A commit id belongs to one upstream, so replacing the string cannot reach
    another upstream's pin -- asserted against the real catalog in --self-test rather
    than assumed.
    """
    if BROKEN.which == "M1":
        return text.replace(old, new)
    if BROKEN.which == "M2":
        return re.sub(r"\b[0-9a-f]{40}\b", new, text).replace(old[:12], new[:12])
    return text.replace(old, new).replace(old[:12], new[:12])


def bump(record: dict) -> None:
    """Move the pin in every file that states it, then re-vendor from the new commit.

    Progress is printed here rather than returned so that it interleaves with what the
    generator prints: the two are one operation, and reading it afterwards should say
    which pin moved before saying which files it rewrote.
    """
    for path in (CATALOG_PATH, NOTICE_PATH):
        text = path.read_bytes().decode("utf-8")
        moved = rewrite_sha(text, record["pinned"], record["head"])
        if moved == text:
            sys.exit(
                f"FAIL {path.name} states no pin at {record['pinned'][:12]}, so this run has "
                "nothing to move there. Either it was edited by hand between the survey and "
                "now, or the pin is written some way this tool does not recognise"
            )
        path.write_bytes(moved.encode("utf-8"))
        print(f"     {path.name}: {record['pinned'][:12]} -> {record['head'][:12]}")

    names = sorted(record["skills"].values())
    # The generator writes to this stdout, so what it says lands after what led to it.
    sys.stdout.flush()
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "sync_external.py"), "--write", *names],
        cwd=REPO_ROOT,
        check=False,
    )
    if done.returncode != 0:
        sys.exit(
            f"FAIL sync_external.py --write refused {record['head'][:12]}. The pin moved in "
            "the two files above and the tree was not re-vendored, so nothing here is in a "
            "state to propose: read the failure above, and revert both files"
        )
    print(f"     re-vendored {len(names)} skill(s) from {record['head'][:12]}")


def commit_subject(record: dict) -> str:
    return f"chore: move the {upstream_slug(record['repo'])} pin to {record['head'][:12]}"


def compare_url(record: dict) -> str:
    return f"{record['repo']}/compare/{record['pinned'][:12]}...{record['head'][:12]}"


def pr_body(record: dict) -> str:
    """What a reviewer needs to know that the diff does not say.

    ASCII only: this text is printed by --dry-run and handed to `gh` on a pipe, and both
    of those go through the console encoding on the machine running the tool.
    """
    changed = record["changed"]
    untouched = sorted(set(record["skills"].values()) - set(changed))
    lines = [
        f"`{upstream_slug(record['repo'])}` has moved on `{record['default-branch']}` and "
        f"this repository's copy of it has not. Opened by `{TOOL}`; the bytes are "
        "upstream's at the new commit, written by `tools/sync_external.py --write`.",
        "",
        f"- pin: `{record['pinned']}` -> `{record['head']}`",
        f"- upstream diff: {compare_url(record)}",
        f"- changed upstream: {', '.join(f'`{name}`' for name in changed)}",
    ]
    if untouched:
        lines.append(
            "- unchanged upstream, re-vendored because the pin moved: "
            + ", ".join(f"`{name}`" for name in untouched)
        )
    lines += [
        "",
        "What this does not decide is whether the new text should be merged. An import is "
        "somebody else's document, so a change in it is a change somebody made there: the "
        "checks below answer the structural half, and the diff is worth reading.",
        "",
        "If no checks are listed, this pull request was opened with `GITHUB_TOKEN`, which "
        "does not start workflow runs. Push to the branch, or close and reopen it, to get "
        "them.",
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
    """Which repository `gh` is to act on, taken from the remote rather than guessed.

    `gh` resolves a bare `pr list` from whatever remotes the checkout has, which is one
    repository in CI and several in a maintainer's clone -- there it can answer about, or
    open against, a repository nobody asked for.
    """
    return parse_remote_url(git("remote", "get-url", remote), remote)


def head_ref(push_slug: str, target_slug: str, branch: str) -> str:
    """How the branch is named to the repository the pull request is opened against.

    A branch pushed to a fork is not a ref of the target repository, so `gh` has to be
    told whose it is. Without the owner the request either names a branch of the target
    that does not exist, or -- worse -- one that does and was never part of this move.
    """
    if push_slug == target_slug or BROKEN.which == "M9":
        return branch
    return f"{push_slug.split('/')[0]}:{branch}"


def already_proposed(branch: str, slug: str) -> str | None:
    """Whether this move, or an earlier one for the same upstream, is already open."""
    listed = gh(
        "pr", "list", "--repo", slug, "--state", "all", "--head", branch,
        "--json", "number,state",
    )
    if same := json.loads(listed):
        return f"#{same[0]['number']} ({same[0]['state'].lower()}) already proposes {branch}"
    prefix = f"{branch.rsplit('-', 1)[0]}-"
    open_prs = json.loads(
        gh("pr", "list", "--repo", slug, "--state", "open", "--json", "number,headRefName")
    )
    others = [
        pull
        for pull in open_prs
        if pull["headRefName"].startswith(prefix) and pull["headRefName"] != branch
    ]
    if others:
        return (
            f"#{others[0]['number']} is an open update for the same upstream and is itself "
            "behind now; merge or close it and the next run proposes the newer commit"
        )
    return None


def dry_run(record: dict, remote: str, against: str, base: str) -> None:
    """Print the run without doing any of it, including the body a reviewer would read."""
    branch = record["branch"]
    target = remote_slug(against)
    head = head_ref(remote_slug(remote), target, branch)
    for line in (
        f"git switch --create {branch} {against}/{base}",
        f"{TOOL} --update {upstream_slug(record['repo'])}",
        f"git add -- skills skills.yaml NOTICE && git commit -m {commit_subject(record)!r}",
        f"git push {remote} HEAD:refs/heads/{branch}",
        f"gh pr create --repo {target} --base {base} --head {head} "
        f"--title {commit_subject(record)!r}",
    ):
        print(f"     would run: {line}")
    print("     not asked here: whether that pull request already exists, which is a gh call")
    print("".join(f"     | {line}\n" for line in pr_body(record).splitlines()), end="")


def propose(record: dict, remote: str, against: str) -> None:
    """Branch, re-vendor, commit, push, and open the pull request for one moved upstream."""
    branch = record["branch"]
    if dirty := git("status", "--porcelain"):
        sys.exit(
            "FAIL the working tree has uncommitted changes, and a pull request from it would "
            f"carry them: {dirty.splitlines()[0]}"
        )
    target = remote_slug(against)
    if existing := already_proposed(branch, target):
        print(f"SKIP {upstream_slug(record['repo'])}: {existing}")
        return

    # Off the target's default branch, not the fork's: a fork can be behind, and a pull
    # request branched off a stale base carries whatever it is missing as a deletion.
    base = base_branch(against)
    git("fetch", "--quiet", against)
    git("switch", "--quiet", "--create", branch, f"{against}/{base}")
    print(f"     branched {branch} off {against}/{base}")
    bump(record)
    git("add", "--", "skills", "skills.yaml", "NOTICE")
    git(
        "commit",
        "--quiet",
        "-m",
        commit_subject(record),
        "-m",
        f"{len(record['changed'])} of {len(record['skills'])} imported skill(s) changed "
        f"upstream. Diff: {compare_url(record)}",
    )
    git("push", "--quiet", remote, f"HEAD:refs/heads/{branch}")
    url = gh(
        "pr", "create",
        "--repo", target,
        "--base", base,
        "--head", head_ref(remote_slug(remote), target, branch),
        "--title", commit_subject(record),
        "--body-file", "-",
        stdin=pr_body(record),
    )
    print(f"OPENED {url}")


def render(records: list[dict]) -> None:
    """One line per upstream, plus what a stale one would carry into a pull request."""
    for record in records:
        where = upstream_slug(record["repo"])
        count = len(record["skills"])
        if record["state"] == "current":
            print(f"OK   {where}: {record['default-branch']} is at the pin, {count} skill(s)")
        elif record["state"] == "identical":
            print(
                f"OK   {where}: {record['default-branch']} moved to {record['head'][:12]} and "
                f"none of the {count} pinned directories changed — nothing to update"
            )
        elif record["state"] == "stale":
            print(
                f"NEW  {where}: {record['default-branch']} moved {record['pinned'][:12]} -> "
                f"{record['head'][:12]}, {len(record['changed'])} of {count} skill(s) changed"
            )
            print(f"     changed: {', '.join(record['changed'])}")
            print(f"     branch:  {record['branch']}")
        elif record["state"] == "unreachable":
            print(f"WARN {where}: not checked, upstream unreachable ({record['detail']})")
        else:
            print(f"FAIL {where}: {record['detail']}", file=sys.stderr)


def self_test() -> int:
    """Assert the detector against this repository's real pins, offline.

    Every number below is a property of the catalog as it stands rather than a fixture,
    so a pin moved by hand or a rewrite that reaches too far shows up here. --mutate
    M1..M9 breaks one thing each, and each of them must turn one of these lines red.
    """
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(message)

    report = Report()
    grouped = groups(report)
    check(not report.errors, f"the catalog's pins parse: {'; '.join(report.errors) or 'clean'}")
    check(bool(grouped), f"{len(grouped)} group(s) carry a pin")
    check(
        len({group["repo"] for group in grouped.values()}) == len(grouped),
        "one group per upstream repository, so one fetch and one pull request per move",
    )
    check(
        all(
            path.rsplit("/", 1)[-1] == name
            for group in grouped.values()
            for path, name in group["skills"].items()
        ),
        "every pinned path ends in the skill it is imported as",
    )
    pins = [group["pinned"] for group in grouped.values()]
    check(
        len(set(pins)) == len(pins),
        "no two upstreams share a commit id, which is what makes moving one a textual "
        "substitution",
    )

    subject = max(grouped.values(), key=lambda group: (len(group["skills"]), group["repo"]))
    old, new = subject["pinned"], "0" * 40
    catalog = CATALOG_PATH.read_bytes().decode("utf-8")
    moved = rewrite_sha(catalog, old, new)
    check(
        moved.count(new) == catalog.count(old) and old not in moved,
        f"moving {upstream_slug(subject['repo'])} rewrites its {catalog.count(old)} pin line(s)",
    )
    check(
        old[:12] not in moved and moved.count(new[:12]) >= catalog.count(old[:12]),
        f"and the {catalog.count(old[:12]) - catalog.count(old)} place(s) that name it in "
        "prose, so no comment is left describing the commit before the move",
    )
    others = [pin for pin in pins if pin != old]
    check(
        all(moved.count(pin) == catalog.count(pin) for pin in others),
        f"while {len(others)} other upstream pin(s) are untouched",
    )
    touched = [
        line
        for line in catalog.splitlines()
        if old in line or old[:12] in line
    ]
    differing = [
        was
        for was, now in zip(catalog.splitlines(), moved.splitlines())
        if was != now
    ]
    check(
        differing == touched,
        f"and the {len(differing)} changed line(s) are exactly the ones naming that commit",
    )
    notice = NOTICE_PATH.read_bytes().decode("utf-8")
    check(
        old not in rewrite_sha(notice, old, new) and new in rewrite_sha(notice, old, new),
        "NOTICE, which repeats the pin for each upstream it republishes, moves too",
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
    slug = upstream_slug(subject["repo"]).rsplit("/", 1)[-1]
    check(
        branch_name(subject["repo"], head) == f"sync/{slug}-{head[:12]}",
        "the branch names the commit it moves to, so a second run finds its own pull request",
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
    try:
        parse_remote_url("/tmp/somewhere.git", "r")
        check(False, "a remote that is not a github repository fails rather than being guessed")
    except SystemExit:
        check(True, "a remote that is not a github repository fails rather than being guessed")
    check(
        head_ref("intel/skills", "intel/skills", "sync/x-1") == "sync/x-1",
        "a branch pushed to the repository it is proposed to is named by itself",
    )
    check(
        head_ref("someone/skills", "intel/skills", "sync/x-1") == "someone:sync/x-1",
        "and one pushed to a fork carries the fork's owner, or it names a branch of this "
        "repository instead",
    )

    body = pr_body(
        {**subject, "head": head, "default-branch": "main", "changed": [subject["skills"][one]]}
    )
    check(
        head in body and old in body and subject["skills"][one] in body,
        "the pull request body names both commits and every skill that changed",
    )

    print(f"\n{'FAIL' if failures else 'PASS'} --self-test: {len(failures)} failure(s)")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Every flag, kept out of main() so what main() does is legible in one screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="the survey, as JSON")
    mode.add_argument(
        "--update",
        action="store_true",
        help="move the pin of every upstream that moved, and re-vendor. No git, no PR",
    )
    mode.add_argument(
        "--open-pr",
        action="store_true",
        help="one branch, commit, push and pull request per upstream that moved",
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
    parser.add_argument("repos", nargs="*", help="limit to these upstreams (default: all)")
    parser.add_argument("--mutate", choices=sorted(MUTATIONS), help=argparse.SUPPRESS)
    return parser


def selected(grouped: dict[str, dict], repos: list[str]) -> list[dict]:
    """The groups named on the command line, or all of them."""
    if not repos:
        return [grouped[key] for key in sorted(grouped)]
    wanted = {repo.rstrip("/").removesuffix(".git").lower() for repo in repos}
    chosen = [
        group
        for key, group in sorted(grouped.items())
        if wanted & {group["repo"].lower(), upstream_slug(group["repo"]).lower()}
    ]
    if not chosen:
        sys.exit(f"FAIL no pinned upstream matches {', '.join(repos)}")
    return chosen


def main() -> int:
    args = build_parser().parse_args()
    BROKEN.which = args.mutate
    if BROKEN.which:
        print(f"# mutation {BROKEN.which}: {MUTATIONS[BROKEN.which]}")

    if args.self_test:
        return self_test()

    report = Report()
    grouped = groups(report)
    for error in report.errors:
        print(f"FAIL {error}", file=sys.stderr)
    if report.errors:
        return 1
    if not grouped:
        print("OK   no imported skills")
        return 0

    records = [survey(group) for group in selected(grouped, args.repos)]
    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    render(records)
    against = args.against or args.remote
    stale = [record for record in records if record["state"] == "stale"]
    for record in stale:
        if args.update:
            bump(record)
        elif args.open_pr and args.dry_run:
            dry_run(record, args.remote, against, base_branch(against))
        elif args.open_pr:
            propose(record, args.remote, against)
    return 1 if any(record["state"] == "broken" for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
