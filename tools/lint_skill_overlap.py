#!/usr/bin/env python3
"""Report skill pairs that do the same things without saying which one to use.

    python3 tools/lint_skill_overlap.py                      # every pair, ranked
    python3 tools/lint_skill_overlap.py --show vllm-xpu-run  # one skill's signature
    python3 tools/lint_skill_overlap.py --max-overlap 0.65 --min-shared 8 --advisory

Two skills driving the same tool with the same flags compete for the same request. That
is fine, and this catalog is full of such pairs — as long as one of them says which is
which. `vllm-xpu-bench` and `vllm-xpu-run` name each other; `torch-xpu-run` and
`torch-xpu-profile` never have, and an agent holding both has nothing to route on.

So this compares not prose but what each skill *does*: the commands, flags, environment
variables, API calls and endpoint paths in its code — fenced blocks, inline spans and
bundled `.sh`/`.py`/`.md` — then subtracts the pairs where a hand-off is written down.
The fence's language tag picks the extractor, and that split is load-bearing: read a
Python fence as shell and `import` becomes a high-frequency command.

PLATFORM strips what every skill here does anyway: `docker`, `/dev/dri`, `--group-add`.
It is written and reviewed by hand, never derived from frequency — `cmd:vllm` is frequent
*because* it is the tool's identity, which is the one thing worth comparing.

What it cannot tell you: a skill that is prose only, or a restatement in different words
with no shared commands. That was measured, not assumed. If a keyless check ever claims
to catch a paraphrase, it is lying. This one claims something smaller and checkable.

Which is why the name and the description are also read, but only to order a queue of the
pairs the action axis cannot judge — a skill with no code still has both. They decide
nothing: the two axes agree on 2 of the 14 pairs at the top of this tree, a shared-name
precondition would drop 4 of those 14, and the near-verbatim fixture shares 0.05 of its
description with the skill it copies while sharing 1.0000 of its actions. As a reading
order they are worth the ~0.2s they cost, and they are the only affordable way to pick the
few pairs anything downstream that pays per pair should look at.

A high score is not a defect by itself; a high score nobody documented is. Which of two
overlapping skills should win is a judgement about the catalog, not about the bytes, so CI
runs `--advisory`: findings are annotated on the pull request and the merge is not blocked.
`--self-test` does block, because a detector that has stopped detecting is not a judgement
call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
# --self-test reads the threshold out of the workflow rather than keeping a second copy.
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validate.yml"
GATE_THRESHOLD = re.compile(r"--max-overlap\s+([0-9.]+)")

from validate_skills import Report, split_frontmatter  # noqa: E402

# Shell and language plumbing, present in a skill about anything.
GENERIC_CMD = {
    "echo", "cat", "ls", "cd", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "ln",
    "grep", "egrep", "sed", "awk", "head", "tail", "sort", "uniq", "wc", "tee", "tr",
    "cut", "xargs", "find", "diff", "tar", "unzip", "gzip", "chmod", "chown", "chgrp",
    "curl", "wget", "jq", "yq", "git", "ssh", "scp", "rsync", "sudo", "su", "env",
    "export", "source", "which", "whereis", "type", "printf", "read", "sleep", "time",
    "watch", "nohup", "kill", "pkill", "ps", "top", "df", "du", "free", "uname",
    "nproc", "lscpu", "date", "hostname", "whoami", "id", "apt", "apt-get", "yum",
    "dnf", "zypper", "brew", "make", "cmake", "ninja", "gcc", "g++", "cc", "ld",
    "pkg-config", "python", "python3", "pip", "pip3", "uv", "uvx", "pipx", "conda",
    "mamba", "bash", "sh", "zsh", "npm", "npx", "node", "pytest", "tee", "seq",
    "true", "false", "exit", "set", "unset", "test", "eval", "exec", "trap", "wait",
    "mktemp", "basename", "dirname", "realpath", "readlink", "stat", "md5sum",
    "sha256sum", "base64", "openssl", "systemctl", "journalctl", "dmesg", "lsmod",
    "modprobe", "lspci", "lsusb", "ip", "ping", "netstat", "ss", "nc", "telnet",
}
# Python heads whose attributes are plumbing, not a subject-matter API. The second half is
# conventional local variable names: `p.add_argument` and `cfg.get` say nothing about a
# subject, and they were the majority of one reported pair's shared "actions".
GENERIC_PY_HEAD = {
    "os", "sys", "json", "re", "time", "datetime", "math", "random", "subprocess",
    "pathlib", "argparse", "logging", "shutil", "glob", "csv", "tempfile", "textwrap",
    "collections", "itertools", "functools", "typing", "dataclasses", "warnings",
    "traceback", "unittest", "pytest", "print", "str", "int", "float", "list", "dict",
    "set", "tuple", "self", "cls", "args", "kwargs", "f", "fp", "file", "line", "lines",
    "result", "results", "response", "resp", "data", "out", "output", "text", "s", "x",
    "y", "z", "i", "j", "k", "n", "df", "ax", "plt", "np", "pd",
    "p", "parser", "ap", "opts", "options", "ns", "cfg", "conf", "config", "ctx", "buf",
    "req", "client", "session", "logger", "log", "url", "path", "paths", "name", "names",
    "key", "keys", "value", "values", "item", "items", "row", "rows", "col", "cols",
    "urllib", "socket", "http", "requests", "e", "err", "exc", "tmp", "obj", "arr",
}
# A dotted pair whose tail is one of these is a filename or a host, not a call.
NOT_AN_API = {
    "json", "txt", "md", "yaml", "yml", "py", "sh", "csv", "tsv", "html", "xml", "log",
    "cfg", "ini", "toml", "conf", "out", "err", "dat", "bin", "so", "whl", "tar", "gz",
    "zip", "png", "jpg", "svg", "pdf", "xlsx", "com", "org", "io", "co", "net", "ai",
    "dev", "gov", "edu", "cn", "ru", "de", "xyz", "sha256", "lock", "env",
}
GENERIC_MOD = {
    "os", "sys", "json", "re", "time", "datetime", "math", "random", "subprocess",
    "pathlib", "argparse", "logging", "shutil", "glob", "csv", "tempfile", "textwrap",
    "collections", "itertools", "functools", "typing", "dataclasses", "warnings",
    "traceback", "unittest", "pytest", "statistics", "urllib", "socket", "shlex",
    "__future__", "annotations", "contextlib", "io", "copy", "abc", "enum", "string",
}
# Arguments that address the input rather than describe the method: every serving skill
# names a model and a port, none of them differ by it. `--dtype`, `--max-model-len` and
# `--kv-cache-dtype` are deliberately *not* here — those are choices about how to run.
GENERIC_SERVING = {"flag:--model", "flag:--port", "flag:--host", "path:/dev/null"}
# `PASS=0` and `ZE_AFFINITY_MASK=0` are the same shape, so the words that are status
# rather than configuration have to be named. Otherwise two preflight skills "share" them.
GENERIC_ENV = {
    "PASS", "FAIL", "WARN", "OK", "INFO", "ERROR", "DEBUG", "SKIP", "TRACE", "NOTE",
    "EOF", "END", "YES", "TRUE", "FALSE", "NONE", "ALL", "USAGE", "STATUS", "RESULT",
    "COUNT", "TOTAL", "LINE", "ARGS", "OUT", "DIR", "TMP", "RED", "GREEN", "YELLOW",
    "NC", "BOLD", "RESET",
    "PATH", "HOME", "USER", "PWD", "SHELL", "TERM", "LANG", "LC_ALL", "TMPDIR", "TZ",
    "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONIOENCODING", "LD_LIBRARY_PATH", "CC", "CXX",
    "CFLAGS", "CXXFLAGS", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy",
    "https_proxy", "no_proxy",
}

# Intel GPU device plumbing, shared by the 28 of 33 skills that touch a GPU at all.
DEVICE_PLUMBING = {
    "cmd:xpu-smi", "cmd:clinfo", "cmd:sycl-ls", "cmd:intel_gpu_top", "cmd:hwinfo",
    "env:ZE_AFFINITY_MASK", "env:ZE_FLAT_DEVICE_HIERARCHY", "env:ONEAPI_DEVICE_SELECTOR",
    "env:SYCL_DEVICE_FILTER", "env:SYCL_CACHE_PERSISTENT", "env:LIBVA_DRIVER_NAME",
    "env:ZES_ENABLE_SYSMAN", "env:OverrideGpuAddressSpace", "env:NEOReadDebugKeys",
    "path:/dev/dri", "path:/dev/dri/renderD128", "path:/dev/dri/card0",
    "path:/sys/class/drm", "path:/sys/bus/pci", "path:/proc/cpuinfo", "path:/proc/meminfo",
    "flag:--device", "flag:--group-add", "flag:--gpus",
}

# Container boilerplate: how a skill is run, never what it is for.
CONTAINER_PLUMBING = {
    "cmd:docker", "cmd:podman", "cmd:docker-compose", "cmd:nerdctl",
    "flag:--rm", "flag:--it", "flag:--interactive", "flag:--tty", "flag:--name",
    "flag:--volume", "flag:--mount", "flag:--workdir", "flag:--user", "flag:--env",
    "flag:--env-file", "flag:--entrypoint", "flag:--network", "flag:--net",
    "flag:--ipc", "flag:--shm-size", "flag:--privileged", "flag:--publish",
    "flag:--detach", "flag:--restart", "flag:--add-host", "flag:--cap-add",
    "flag:--security-opt", "flag:--pull", "flag:--platform", "flag:--build-arg",
    "flag:--tag", "flag:--file", "flag:--no-cache", "flag:--quiet", "flag:--help",
    "flag:--version", "flag:--verbose", "flag:--output", "flag:--input", "flag:--force",
    "flag:--yes", "flag:--dry-run", "flag:--all", "flag:--upgrade", "flag:--index-url",
    "flag:--extra-index-url", "flag:--pre", "flag:--user", "flag:--editable",
    "flag:--no-deps", "flag:--requirement",
}

PLATFORM = (
    {f"cmd:{name}" for name in GENERIC_CMD}
    | {f"mod:{name}" for name in GENERIC_MOD}
    | {f"env:{name}" for name in GENERIC_ENV}
    | GENERIC_SERVING
    | DEVICE_PLUMBING
    | CONTAINER_PLUMBING
)

# Frequent because they are a tool's identity. --self-test requires every element at
# document frequency 10 or more to be in PLATFORM or here, so a new frequent element gets
# classified by a person instead of quietly diluting every score.
IDENTITY = {
    "cmd:vllm", "cmd:python", "api:torch.xpu", "mod:torch", "mod:dpnp", "mod:vllm",
    "mod:transformers", "mod:numpy", "mod:intel_extension_for_pytorch", "mod:ipex",
    "path:/v1/chat/completions", "path:/v1/completions", "path:/v1/models",
}


class Broken:
    """Which deliberate break --mutate is running, readable from every extractor without
    threading a test-only argument through eight signatures. See MUTATIONS below."""

    which: str | None = None


SHELL_LANGS = {"sh", "bash", "shell", "console", "zsh", "dockerfile", "docker", ""}
PYTHON_LANGS = {"python", "py", "python3"}

SHELL_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "while", "do", "done", "case", "esac",
    "function", "return", "break", "continue", "local", "declare", "readonly", "in",
    "select", "until", "shift", "alias", "trap", "and", "or", "not", "the", "run",
    "from", "copy", "add", "workdir", "entrypoint", "cmd", "expose", "label", "arg",
    "as", "volume", "user", "shell", "onbuild", "healthcheck", "stopsignal",
}

FENCE = re.compile(r"^\s*```+\s*([A-Za-z0-9_+-]*)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

FLAG = re.compile(r"(?<![\w-])--([A-Za-z][A-Za-z0-9._-]*)")
ENV_ASSIGN = re.compile(r"(?<![\w])([A-Z][A-Z0-9_]{2,})=")
ENV_REF = re.compile(r"\$\{?([A-Z][A-Z0-9_]{2,})\}?")
ENV_PY = re.compile(r"environ(?:\.get)?[\[(]\s*[\"']([A-Z][A-Z0-9_]{2,})[\"']")
PATHISH = re.compile(r"(/(?:dev|v1|sys|proc|opt/intel)(?:/[A-Za-z0-9._*-]+)*)")
PY_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.MULTILINE)
PY_DOTTED = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")
SHELL_FUNCTION = re.compile(r"^\s*(?:function\s+)?([a-z_][a-z0-9_]*)\s*\(\)\s*\{?", re.MULTILINE)
CMD_SPLIT = re.compile(r"[|;&]{1,2}|\$\(|`")
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]{3,}")


def flags(text: str) -> set[str]:
    if Broken.which == "M3":
        return set()
    return {f"flag:--{match.group(1).rstrip('.,')}" for match in FLAG.finditer(text)}


def shared_elements(text: str) -> set[str]:
    """Environment variables and endpoint paths, spelled the same in either language."""
    found = {f"env:{name}" for name in ENV_ASSIGN.findall(text)}
    found |= {f"env:{name}" for name in ENV_REF.findall(text)}
    found |= {f"env:{name}" for name in ENV_PY.findall(text)}
    found |= {f"path:{path.rstrip('/.,')}" for path in PATHISH.findall(text)}
    return found


def shell_elements(text: str) -> set[str]:
    """Commands and flags: argv-0 after a line start, a pipe, a `&&` or a `$(`.

    Two rules below are measured rather than assumed. A head must be lower case, or a
    console block's own output (`Detected 2 devices`), a heredoc marker (`EOF`) and a
    traceback name all arrive as commands — in sibling skills together, inflating exactly
    the pairs being judged. And a head the script defines itself is dropped, because
    `usage`, `record` and `die` are house style: two skills that both define `usage()`
    have nothing in common. Before that, the preflight pair shared thirteen "actions", ten
    of them their own function names and log words.
    """
    local = set(SHELL_FUNCTION.findall(text))
    found = flags(text) | shared_elements(text)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\$\s+|^>\s+|^#\s+", "", line)
        for segment in CMD_SPLIT.split(line):
            words = segment.strip().split()
            while words and ("=" in words[0] or words[0] in {"sudo", "time", "!", "then", "do"}):
                words.pop(0)
            if not words:
                continue
            # `xe)` and `i915)` are case labels: values matched, not commands run.
            if words[0].endswith(")") and "(" not in words[0]:
                continue
            head = words[0].strip("\\'\"()")
            if head in SHELL_KEYWORDS or head in local:
                continue
            if not re.fullmatch(r"[a-z_][a-z0-9_.+-]*", head):
                continue
            found.add(f"cmd:{head}")
    return found


def inline_elements(text: str) -> set[str]:
    """Flags, variables and endpoints in inline code spans — those three only.

    An inline span is usually a *value*, not a command (`bf16`, `awq`, `int4`, `model`),
    and reading those as commands put a dozen into every vLLM skill at once. A flag, an
    all-caps variable and a `/v1/...` path are unambiguous by shape; a bare word is not.
    """
    prose = FENCE.sub("", text)
    spans = "\n".join(INLINE_CODE.findall(prose))
    return flags(spans) | shared_elements(spans)


def python_elements(text: str) -> set[str]:
    """Imports and two-segment dotted paths — `torch.xpu`, `dpnp.asnumpy`.

    NOT_AN_API covers the two shapes the regex cannot tell apart from a call: a filename
    and a host. `config.json` and `huggingface.co` were being reported as shared actions.
    """
    found = shared_elements(text)
    for module in PY_IMPORT.findall(text):
        found.add(f"mod:{module.split('.')[0]}")
    for head, attribute in PY_DOTTED.findall(text):
        if head in GENERIC_PY_HEAD or head.isupper():
            continue
        if attribute in NOT_AN_API:
            continue
        found.add(f"api:{head}.{attribute}")
    return found


def code_blocks(body: str) -> list[tuple[str, str]]:
    """(language, text) for every fenced block."""
    blocks: list[tuple[str, str]] = []
    language, buffer, inside = "", [], False
    for line in body.splitlines():
        fence = FENCE.match(line)
        if fence:
            if inside:
                blocks.append((language, "\n".join(buffer)))
                buffer, inside = [], False
            else:
                language, inside = fence.group(1).lower(), True
            continue
        if inside:
            buffer.append(line)
    if inside:
        blocks.append((language, "\n".join(buffer)))
    return blocks


def document_elements(text: str) -> set[str]:
    """A markdown document's actions: every fenced block by its language, plus prose spans."""
    found = inline_elements(text)
    for language, block in code_blocks(text):
        python = language in PYTHON_LANGS and Broken.which != "M2"
        found |= python_elements(block) if python else shell_elements(block)
    return found


def signature(body: str, bundled: dict[str, str]) -> set[str]:
    """Every action this skill performs, minus the platform layer.

    Bundled `.md` counts: `vllm-xpu-bench` reaches `/v1/chat/completions` only in
    `references/sweep-and-compare.md`. The catalog's four `.c` files are not read, so
    `onetbb-quickstart` is under-measured — recorded in --self-test rather than hidden.
    """
    found = document_elements(body)
    for name, text in bundled.items():
        if name.endswith(".md"):
            found |= document_elements(text)
        elif name.endswith(".py") and Broken.which != "M2":
            found |= python_elements(text)
        elif name.endswith((".sh", ".bash")):
            found |= shell_elements(text)
    return found - (set() if Broken.which == "M1" else PLATFORM)


def load(directory: Path, report: Report) -> dict | None:
    main = directory / "SKILL.md"
    if not main.is_file():
        return None
    text = main.read_text(encoding="utf-8")
    front, body = split_frontmatter(text, f"skills/{directory.name}/SKILL.md", report)
    body = HTML_COMMENT.sub("", body)

    prose = [text]
    bundled: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "SKILL.md":
            continue
        if path.suffix in {".md", ".sh", ".bash", ".py"}:
            content = path.read_text(encoding="utf-8", errors="replace")
            bundled[path.relative_to(directory).as_posix()] = content
            if path.suffix == ".md":
                prose.append(content)

    reference_text = HTML_COMMENT.sub("", "\n".join(prose))
    if Broken.which == "M6" and directory.name == "torch-xpu-profile":
        reference_text += "\n\nTo run inference rather than profile it, see torch-xpu-run.\n"
    if Broken.which == "M7" and directory.name == "vllm-xpu-run":
        reference_text = re.sub(r"(?<![\w-])vllm-xpu-profile(?![\w-])", "", reference_text)

    return {
        "name": directory.name,
        "description": str(front.get("description", "")),
        "signature": signature(body, bundled),
        "reference_text": reference_text,
        "imported": (directory / ".source.json").is_file(),
    }


def load_all(report: Report) -> list[dict]:
    skills = []
    for directory in sorted(SKILLS_DIR.iterdir()):
        if directory.is_dir():
            loaded = load(directory, report)
            if loaded is not None:
                skills.append(loaded)
    return skills


def mentions(skill: dict, other: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9-]){re.escape(other)}(?![A-Za-z0-9-])"
    return re.search(pattern, skill["reference_text"]) is not None


def handoff_index(skills: list[dict]) -> dict[str, set[str]]:
    """Which catalog skills each body names, in one pass per body rather than one per pair.

    This is where the cost of a large catalog turned out to live: at 300 skills the
    per-pair form spent 22.0s of 22.0s here, against 0.03s for the action comparison
    itself. Same boundaries as mentions(), and --self-test asserts the two agree on every
    pair in the tree rather than trusting that they do.
    """
    names = sorted((skill["name"] for skill in skills), key=len, reverse=True)
    if not names:
        return {}
    alternation = "|".join(re.escape(name) for name in names)
    pattern = re.compile(rf"(?<![A-Za-z0-9-])({alternation})(?![A-Za-z0-9-])")
    return {
        skill["name"]: set(pattern.findall(
            skill["reference_text"][: len(skill["reference_text"]) // 2]
            if Broken.which == "M10" else skill["reference_text"]))
        for skill in skills
    }


def words(description: str) -> set[str]:
    """No stoplist, deliberately: a hand-written stoplist moves the number it produces, so
    this feeds ranking only, never a verdict."""
    return {token.lower() for token in TOKEN.findall(description)}


def name_tokens(name: str) -> set[str]:
    """vllm-xpu-run -> {vllm, xpu, run}. Single characters carry no family signal."""
    return {part for part in name.split("-") if len(part) > 1}


def vocabulary(skills: list[dict]) -> dict[str, tuple[set[str], set[str]]]:
    """Name parts and description words per skill, hoisted out of the pair loop.

    Recomputing words() inside the loop costs 1.10s at 300 skills against 0.04s hoisted.
    """
    return {
        skill["name"]: (name_tokens(skill["name"]), words(skill["description"]))
        for skill in skills
    }


def pairs(skills: list[dict], min_shared: int,
          index: dict[str, set[str]] | None = None) -> list[dict]:
    """Score every pair. Under five actions on the smaller side there is too little
    material to mean anything either way, so the pair is not scored at all.

    Name and description overlap ride along as columns because they are nearly free once
    hoisted out of the loop, and they order the queue in lexical(). They gate nothing: on
    this tree a shared-name-token precondition would drop 4 of the 14 judgeable pairs,
    linux-perf | performance-patterns among them.
    """
    if index is None:
        index = handoff_index(skills)
    vocab = vocabulary(skills)
    out = []
    for left, right in combinations(skills, 2):
        if Broken.which == "M9" and not (
            vocab[left["name"]][0] & vocab[right["name"]][0]
        ):
            continue
        shared = left["signature"] & right["signature"]
        smaller = min(len(left["signature"]), len(right["signature"]))
        if smaller < 5:
            continue
        left_words, right_words = vocab[left["name"]][1], vocab[right["name"]][1]
        union = left_words | right_words
        out.append({
            "left": left["name"],
            "right": right["name"],
            "containment": len(shared) / smaller,
            "shared": sorted(shared),
            "handoff": (
                ("left" if right["name"] in index.get(left["name"], ()) else "")
                + ("right" if left["name"] in index.get(right["name"], ()) else "")
            ),
            "authored": not (left["imported"] and right["imported"]),
            "jaccard": len(left_words & right_words) / len(union) if union else 0.0,
            "name_shared": len(vocab[left["name"]][0] & vocab[right["name"]][0]),
            "reach": len(shared) >= min_shared,
        })
    out.sort(key=lambda pair: (-pair["containment"], -len(pair["shared"])))
    return out


def lexical(skills: list[dict], min_shared: int,
            index: dict[str, set[str]] | None = None) -> list[dict]:
    """Rank the pairs the action axis cannot judge, by shared name parts then description.

    A skill with almost no code has no action signature, so pairs() either skips it or
    cannot reach min_shared - but every skill has a name and a description. Declared pairs
    are dropped for the same reason the action axis drops them.

    Ordered by how blind the action axis is to the pair first, and only then by prose:
    ranking on the words alone buries a new contribution behind the catalog's existing name
    families, which is the case that matters most - a prose-only candidate sat at rank 113
    of 466 under a name-first key and rank 7 under this one. It still does not name the
    right counterpart (0.11 against the wrong skill, measured), which is why this is a
    reading order and not a verdict.
    """
    if index is None:
        index = handoff_index(skills)
    vocab = vocabulary(skills)
    out = []
    for left, right in combinations(skills, 2):
        shared = left["signature"] & right["signature"]
        smaller = min(len(left["signature"]), len(right["signature"]))
        if smaller >= 5 and len(shared) >= min_shared:
            continue
        if right["name"] in index.get(left["name"], ()) or \
                left["name"] in index.get(right["name"], ()):
            continue
        left_names, left_words = vocab[left["name"]]
        right_names, right_words = vocab[right["name"]]
        union = left_words | right_words
        out.append({
            "left": left["name"],
            "right": right["name"],
            "name_shared": len(left_names & right_names),
            "jaccard": len(left_words & right_words) / len(union) if union else 0.0,
            "actions": smaller,
        })
    out.sort(key=lambda pair: (pair["actions"] >= min_shared,
                               -pair["jaccard"], -pair["name_shared"]))
    return out


def spread(ranked: list[dict], queue: int) -> list[dict]:
    """Take the top rows, at most one per skill before any skill gets a second.

    The skill with the least code wins every comparison it is in, so without this the
    whole queue is one name repeated - four of the five rows on this tree.
    """
    seen: set[str] = set()
    first, rest = [], []
    for pair in ranked:
        if pair["left"] in seen or pair["right"] in seen:
            rest.append(pair)
            continue
        seen.update((pair["left"], pair["right"]))
        first.append(pair)
    return (first + rest)[:queue]


def worst_no_edge(scored: list[dict], min_shared: int) -> dict | None:
    candidates = [p for p in scored if not p["handoff"] and len(p["shared"]) >= min_shared]
    return max(candidates, key=lambda p: p["containment"], default=None)


def verdict(pair: dict, max_overlap: float, min_shared: int) -> str:
    """ok, WARN, or REVIEW - the last meaning someone here can fix it, not that CI fails.

    A hand-off explains a division of labour. At containment 1.0 there is no division to
    explain, so it is reported even when declared: a file copied under a new name carries
    the original's name in its own text and would otherwise silence itself, which is how
    this rule was found.
    """
    if pair["containment"] <= max_overlap or len(pair["shared"]) < min_shared:
        return "ok"
    if pair["handoff"] and (pair["containment"] < 1.0 or Broken.which == "M8"):
        return "ok"
    return "REVIEW" if pair["authored"] else "WARN"


# Fixtures, inline rather than in a tools/fixtures/ directory: tools/ is flat .py files
# with no data, and --self-test needs them in memory anyway.
DUPLICATE = {
    "name": "_fixture-paraphrased-duplicate",
    "description": (
        "Stand up a local OpenAI-compatible text generation endpoint on Intel discrete "
        "graphics using the vLLM engine, then confirm it answers."
    ),
    "body": """
## Steps

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
docker run --rm -it --device /dev/dri --group-add video \\
  -e ZE_AFFINITY_MASK=0 -p 8000:8000 intel/vllm:latest \\
  vllm serve Qwen/Qwen2.5-7B-Instruct --dtype float16 --max-model-len 4096 \\
  --enforce-eager --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \\
  --kv-cache-dtype fp8 --block-size 64 --attention-backend TRITON_ATTN \\
  --trust-remote-code --quantization awq
```

Then check it replies:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hi"}]}'
```
""",
}

# The same duplicate with a hand-off line, which is what a copied file has for free: it
# carries the original's name in its own text. Found by emulating a contribution rather than
# by reasoning about one - a byte-for-byte copy of vllm-xpu-run silenced itself.
DECLARED_COPY = {
    "name": "_fixture-declared-copy",
    "description": DUPLICATE["description"],
    "body": DUPLICATE["body"] + "\nFor the maintained path use vllm-xpu-run.\n",
}

THIN = {
    "name": "_fixture-thin-restatement",
    "description": "Notes on serving models with vLLM on Intel GPUs.",
    "body": """
## Notes

```bash
vllm serve $MODEL --dtype float16 --max-model-len 4096
```
""",
}

NEGATIVE = {
    "name": "_fixture-openvino-model-server",
    "description": (
        "Serve an OpenVINO IR model with OpenVINO Model Server on Intel graphics and "
        "query it over gRPC."
    ),
    "body": """
## Steps

```bash
docker run --rm -d --device /dev/dri --group-add video \\
  -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest \\
  --model_path /models/resnet50 --model_name resnet --target_device GPU \\
  --grpc_port 9000 --rest_port 9001 --batch_size auto --nireq 4 \\
  --layout NHWC --plugin_config '{"PERFORMANCE_HINT": "THROUGHPUT"}'
```

```bash
xpu-smi discovery
curl http://localhost:9001/v1/config
benchmark_app -m /models/resnet50/1/model.xml -d GPU -hint throughput
```

```python
from ovmsclient import make_grpc_client

client = make_grpc_client("localhost:9000")
metadata = client.get_model_metadata(model_name="resnet")
```
""",
}


def as_skill(fixture: dict) -> dict:
    return {
        "name": fixture["name"],
        "description": fixture["description"],
        "signature": signature(fixture["body"], {}),
        "reference_text": fixture["body"],
        "imported": False,
    }


def against(fixture: dict, skills: list[dict], min_shared: int) -> list[dict]:
    """Score one candidate against the committed catalog — the NVIDIA --catalog shape."""
    return pairs(skills + [as_skill(fixture)], min_shared=min_shared)


def queued(fixture: dict, skills: list[dict], min_shared: int) -> list[dict]:
    """Where a candidate lands in the name/description queue, catalog included."""
    ranked = lexical(skills + [as_skill(fixture)], min_shared=min_shared)
    return [pair for pair in ranked if fixture["name"] in (pair["left"], pair["right"])]


def self_test(max_overlap: float | None, min_shared: int) -> int:
    """Assert the detector still detects, against skills/ as committed. Writes nothing.

    Every way this check breaks is silent: a fence regex that stops matching scores every
    pair 0, which reads exactly like a catalog with no overlap.
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name} - {detail}")
        if not ok:
            failures.append(name)

    report = Report()
    skills = load_all(report)
    if len(skills) < 10:
        sys.exit(f"FAIL --self-test found {len(skills)} skills under skills/")
    index = {skill["name"]: skill for skill in skills}

    expected = [
        ("vllm-xpu-run", "cmd:vllm"),
        ("vllm-xpu-run", "flag:--enforce-eager"),
        ("torch-xpu-run", "api:torch.xpu"),
        ("dpnp-io", "api:dpnp.asnumpy"),
        ("torch-xpu-bench", "mod:transformers"),
        ("vllm-xpu-bench", "path:/v1/chat/completions"),
    ]
    missing = [
        f"{name}:{element}"
        for name, element in expected
        if name in index and element not in index[name]["signature"]
    ]
    check("every element kind is still extracted", not missing,
          ", ".join(missing) or f"cmd/flag/api/mod/path all seen across {len(skills)} skills")

    def edge(left: str, right: str) -> bool:
        return left in index and right in index and (
            mentions(index[left], right) or mentions(index[right], left)
        )

    check("a mutually declared pair is seen as declared", edge("vllm-xpu-bench", "vllm-xpu-run"),
          "vllm-xpu-bench and vllm-xpu-run name each other")
    check("a one-directional hand-off counts", edge("vllm-xpu-run", "vllm-xpu-profile"),
          "vllm-xpu-run names vllm-xpu-profile; the reverse is not required")
    check("the torch pair is undeclared in both directions",
          not edge("torch-xpu-run", "torch-xpu-profile"),
          "neither skills/torch-xpu-run/SKILL.md nor skills/torch-xpu-profile/SKILL.md "
          "names the other - independently confirmed by grep")

    handoffs = handoff_index(skills)
    disagree = [
        f"{left['name']}|{right['name']}"
        for left, right in combinations(skills, 2)
        for indexed, scanned in [(
            (right["name"] in handoffs[left["name"]],
             left["name"] in handoffs[right["name"]]),
            (mentions(left, right["name"]), mentions(right, left["name"])),
        )]
        if indexed != scanned
    ]
    check("the hand-off index agrees with scanning every pair", not disagree,
          ", ".join(disagree[:5]) or
          f"identical on all {len(skills) * (len(skills) - 1) // 2} pairs - the index is "
          f"what makes this affordable at a few hundred skills (22.0s of the 22.0s spent "
          f"at 300 was the per-pair scan; the action comparison itself was 0.03s)")

    frequency: dict[str, int] = {}
    for skill in skills:
        for element in skill["signature"]:
            frequency[element] = frequency.get(element, 0) + 1
    unclassified = sorted(
        element for element, count in frequency.items()
        if count >= 10 and element not in PLATFORM and element not in IDENTITY
    )
    check("no unclassified high-frequency element", not unclassified,
          ", ".join(unclassified) or
          f"every element at document frequency >= 10 is in PLATFORM or IDENTITY")

    scored = pairs(skills, min_shared=min_shared)
    worst = worst_no_edge(scored, min_shared)
    if worst is None:
        sys.exit("FAIL --self-test found no undeclared pair to calibrate against")

    # A band, not equality: the worst undeclared pair goes *down* when someone writes a
    # hand-off line, so pinning the threshold to it would force a workflow edit per fix.
    if max_overlap is None:
        found = {float(value) for value in GATE_THRESHOLD.findall(
            WORKFLOW.read_text(encoding="utf-8") if WORKFLOW.is_file() else "")}
        check("the workflow names the threshold exactly once", len(found) == 1,
              f"validate.yml runs --max-overlap {sorted(found) or 'nothing'}")
        budget = min(found) if found else 0.0
    else:
        budget = max_overlap
    # One band, both bounds read off the tree: above every pair already merged, so nothing
    # in the catalog is reported and a legitimate sibling arriving next is not either; and
    # no more than half again above it, or the gate carries slack nobody chose. Calibrating
    # against the highest *judgeable* pair rather than the highest undeclared one is
    # deliberate - the undeclared ceiling falls whenever someone writes a hand-off line,
    # while this one only moves when the catalog does.
    judgeable = [pair for pair in scored if len(pair["shared"]) >= min_shared]
    ceiling = max(pair["containment"] for pair in judgeable)
    top_pair = max(judgeable, key=lambda pair: pair["containment"])
    check("the threshold sits just above the whole tree",
          ceiling < budget <= 1.5 * ceiling,
          f"highest of the {len(judgeable)} judgeable pair(s) is {top_pair['left']}|"
          f"{top_pair['right']} at {ceiling:.4f}, threshold {budget:.2f}, margin "
          f"{budget / ceiling:.2f}x (want 1.0-1.5x); worst undeclared pair {worst['left']}|"
          f"{worst['right']} at {worst['containment']:.4f}")

    thin = sorted(
        skill["name"] for skill in skills
        if len(skill["signature"]) < min_shared
    )
    check("skills below the min-shared reach are named", True,
          ", ".join(thin) or "none - every skill carries at least "
          f"{min_shared} actions")

    # The cheap precondition is a ranking, never a filter. Without this, the obvious
    # optimisation - only compare skills whose names share a part - looks free and is not.
    blind = [
        f"{pair['left']}|{pair['right']}"
        for pair in judgeable if pair["name_shared"] == 0
    ]
    check("a pair whose names share nothing is still scored",
          any(pair["left"] == "linux-perf" and pair["right"] == "performance-patterns"
              or pair["left"] == "performance-patterns" and pair["right"] == "linux-perf"
              for pair in scored),
          f"{len(blind)} of the {len(judgeable)} judgeable pairs share no name part "
          f"({', '.join(blind) or 'none'}), so a shared-name precondition would drop them; "
          f"name and description order the queue and decide nothing")

    queue = spread(lexical(skills, min_shared=min_shared), 3)
    unreachable = set(thin)
    check("the queue reaches pairs the action axis cannot judge", bool(queue),
          "; ".join(f"{pair['left']} | {pair['right']} names {pair['name_shared']} "
                    f"desc {pair['jaccard']:.2f} ({pair['actions']} actions)"
                    for pair in queue)
          + (f" - includes {', '.join(sorted(unreachable))}, which carry too few actions "
             f"to be judged at all" if any(pair["left"] in unreachable
                                           or pair["right"] in unreachable
                                           for pair in queue) else ""))

    duplicate = [
        pair for pair in against(DUPLICATE, skills, min_shared)
        if DUPLICATE["name"] in (pair["left"], pair["right"])
    ]
    top = max(duplicate, key=lambda pair: pair["containment"])
    check("a paraphrased duplicate is caught and put in front of a reviewer",
          verdict(top, budget, min_shared) == "REVIEW",
          f"scores {top['containment']:.4f} with {len(top['shared'])} shared actions "
          f"against {top['right'] if top['left'] == DUPLICATE['name'] else top['left']}, "
          f"while its description shares {top['jaccard']:.2f} Jaccard - the prose is not "
          f"what caught it")

    copied = [
        pair for pair in against(DECLARED_COPY, skills, min_shared)
        if DECLARED_COPY["name"] in (pair["left"], pair["right"])
    ]
    top_copy = max(copied, key=lambda pair: pair["containment"])
    check("a copy cannot excuse itself with the hand-off it inherited",
          verdict(top_copy, budget, min_shared) == "REVIEW" and bool(top_copy["handoff"]),
          f"scores {top_copy['containment']:.4f} against vllm-xpu-run and names it "
          f"(hand-off {top_copy['handoff'] or 'none'}), reported anyway because at "
          f"containment 1.0 there is no division of labour for a hand-off to describe")

    thin_pairs = [
        pair for pair in against(THIN, skills, min_shared)
        if THIN["name"] in (pair["left"], pair["right"])
    ]
    flagged = [pair for pair in thin_pairs if verdict(pair, budget, min_shared) == "REVIEW"]
    check("a thin restatement passes, and that is the documented limit",
          not flagged and not thin_pairs,
          f"its {len(as_skill(THIN)['signature'])} action(s) are under the 5-action floor, "
          f"so it is never scored at all: a 20-line skill that restates a real one in its "
          f"own words is out of this check's reach by construction, not by accident")

    # Coverage, not correctness: the queue is the only view that reaches a skill with too
    # little code to score, and that is all it is asserted to do. It does not rank this
    # fixture next to the skill it restates - measured, and the reason the number is a
    # reading order rather than a finding.
    thin_queue = queued(THIN, skills, min_shared)
    neighbours = [pair["right"] if pair["left"] == THIN["name"] else pair["left"]
                  for pair in thin_queue[:3]]
    check("the queue reaches it even though the action axis never scores it",
          len(thin_queue) == len(skills),
          f"queued against {len(thin_queue)} of {len(skills)} skills where the action axis "
          f"scored it 0 times; nearest by name and description are "
          f"{', '.join(neighbours)} and none of them is the vLLM skill it restates, so this "
          f"orders a reviewer's reading and settles nothing")

    negative_pairs = [
        pair for pair in against(NEGATIVE, skills, min_shared)
        if NEGATIVE["name"] in (pair["left"], pair["right"])
    ]
    loud = [
        pair for pair in negative_pairs
        if verdict(pair, budget, min_shared) != "ok"
    ]
    worst_negative = max(negative_pairs, key=lambda p: p["containment"], default=None)
    check("a genuinely different server is silent", not loud and worst_negative is not None,
          ", ".join(f"{p['left']}|{p['right']} {p['containment']:.2f}" for p in loud)
          or (f"{len(as_skill(NEGATIVE)['signature'])} actions, scored against "
              f"{len(negative_pairs)} skill(s), worst {worst_negative['containment']:.4f} "
              f"({worst_negative['left']} | {worst_negative['right']}): everything it "
              f"shares with the vLLM skills is container plumbing, which is the whole "
              f"reason PLATFORM exists" if worst_negative else "it was never scored"))

    if failures:
        print(f"\nFAIL {len(failures)} self-test(s): " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nOK the detector detects. A regex that stopped matching, or a platform list")
    print("that swallowed a tool's identity, would fail here rather than report a clean")
    print("zero for every pair and leave the gate passing anything.")
    return 0


MUTATIONS = {
    "M1": "PLATFORM = set()",
    "M2": "read Python fences as shell",
    "M3": "delete the flag: regex",
    "M4": "--max-overlap 0.30",
    "M5": "--max-overlap 0.95",
    "M6": "inject a hand-off line into torch-xpu-profile",
    "M7": "delete the hand-off line from vllm-xpu-run",
    "M8": "let an inherited hand-off excuse a verbatim copy",
    "M9": "score only pairs whose names share a part",
    "M10": "index hand-offs from half of each body",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-overlap", type=float, default=None, metavar="F",
                        help="the line above which an undeclared pair is worth a reviewer's "
                             "attention. Omit to report only.")
    parser.add_argument("--min-shared", type=int, default=8, metavar="N",
                        help="a pair needs this many shared actions before it is judged")
    parser.add_argument("--advisory", action="store_true",
                        help="report and exit 0, annotating each finding for the pull "
                             "request. How CI runs it: which of two overlapping skills wins "
                             "is a judgement call, so it is put in front of a reviewer "
                             "rather than made by a threshold.")
    parser.add_argument("--show", metavar="SKILL", help="print one skill's signature")
    parser.add_argument("--top", type=int, default=15, metavar="N",
                        help="rows of the ranked table to print")
    parser.add_argument("--queue", type=int, default=5, metavar="N",
                        help="pairs to list by name and description overlap alone, for the "
                             "pairs the action axis cannot judge. A reading order, not a "
                             "verdict - and the dial for anything downstream that pays per "
                             "pair, since all-pairs is 44850 comparisons at 300 skills.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit nonzero on warnings too")
    parser.add_argument("--self-test", action="store_true",
                        help="assert the detector still detects, against skills/ as "
                             "committed. Writes nothing. Run it before the gate.")
    parser.add_argument("--mutate", choices=sorted(MUTATIONS),
                        help="break the detector on purpose and prove --self-test notices")
    args = parser.parse_args()

    if not SKILLS_DIR.is_dir():
        sys.exit(f"FAIL no {SKILLS_DIR.relative_to(REPO_ROOT).as_posix()}")

    Broken.which = args.mutate
    max_overlap = args.max_overlap
    if Broken.which == "M4":
        max_overlap = 0.30
    elif Broken.which == "M5":
        max_overlap = 0.95
    if Broken.which:
        print(f"# mutation {Broken.which}: {MUTATIONS[Broken.which]}")

    if args.self_test:
        return self_test(max_overlap, args.min_shared)

    report = Report()
    skills = load_all(report)
    for error in report.errors:
        print(f"ERR  {error}", file=sys.stderr)

    if args.show:
        match = next((s for s in skills if s["name"] == args.show), None)
        if match is None:
            sys.exit(f"FAIL no skills/{args.show}")
        print(f"{match['name']} - {len(match['signature'])} action(s), "
              f"{'imported' if match['imported'] else 'authored here'}")
        for element in sorted(match["signature"]):
            print(f"    {element}")
        return 0

    handoffs = handoff_index(skills)
    scored = pairs(skills, min_shared=args.min_shared, index=handoffs)
    queue = spread(lexical(skills, min_shared=args.min_shared, index=handoffs), args.queue)
    if args.json:
        print(json.dumps({"scored": scored, "queue": queue}, indent=2))
        return 0

    budget = max_overlap if max_overlap is not None else 1.01
    needs_review = [p for p in scored if verdict(p, budget, args.min_shared) == "REVIEW"]

    print(f"{'containment':>11} {'shared':>6} {'name':>4} {'jaccard':>7} {'hand-off':>8}  pair")
    for pair in scored[: args.top]:
        edge = pair["handoff"] or "-"
        print(f"{pair['containment']:11.4f} {len(pair['shared']):6} {pair['name_shared']:4} "
              f"{pair['jaccard']:7.2f} {edge:>8}  {pair['left']} | {pair['right']}")

    # What the action axis cannot see: a pair whose smaller side carries too little code to
    # compare still shares a name and a description. Printed as a reading order, because
    # neither number is strong enough to be a finding - the near-verbatim fixture scores
    # 0.05 on description while its actions score 1.0000.
    if queue:
        print(f"\nwhere actions cannot judge, closest by description then name "
              f"(top {len(queue)}, least code first):")
        for pair in queue:
            print(f"       names {pair['name_shared']} desc {pair['jaccard']:.2f} "
                  f"({pair['actions']} action(s) on the smaller side)  "
                  f"{pair['left']} | {pair['right']}")

    # No second threshold: the report is every pair with enough shared material to judge and
    # nothing written down to separate them - no hand-off, or a hand-off that cannot be true
    # because one skill does everything the other does. REVIEW is that finding with a skill
    # authored here on one side, ranked first because someone here can fix it.
    undeclared = [
        pair for pair in scored
        if len(pair["shared"]) >= args.min_shared
        and (not pair["handoff"] or pair["containment"] >= 1.0)
    ]
    for pair in undeclared:
        over = verdict(pair, budget, args.min_shared) == "REVIEW"
        stream = sys.stderr if over and not args.advisory else sys.stdout
        subsumed = pair["containment"] >= 1.0
        because = ("everything the smaller one does, the larger one already does, so the "
                   "hand-off cannot be what separates them"
                   if subsumed else "no hand-off in either direction")
        fix = ("fix: drop one, or narrow one so it stops being contained in the other"
               if subsumed else "fix: name the other skill in one of the two descriptions")
        sys.stdout.flush()  # or the buffered table lands after the unbuffered failures
        print(f"\n{'REVIEW' if over else 'WARN  '} overlap {pair['containment']:.4f} "
              f"{pair['left']} | {pair['right']}: {because}.", file=stream)
        print(f"       shared ({len(pair['shared'])}): " + ", ".join(pair["shared"]),
              file=stream)
        print(f"       {fix}"
              + ("" if pair["authored"] else " (upstream - both are imported)"), file=stream)
        if args.advisory and os.environ.get("GITHUB_ACTIONS"):
            sys.stdout.flush()
            print(f"::warning file=skills/{pair['left']}/SKILL.md::{pair['left']} and "
                  f"{pair['right']} drive {len(pair['shared'])} of the same actions "
                  f"(containment {pair['containment']:.2f}). {because.capitalize()}. "
                  f"Say in this pull request which one a request should route to, and why "
                  f"both belong.")

    print()
    sys.stdout.flush()
    if needs_review and not args.advisory:
        print(f"FAIL {len(needs_review)} undeclared overlap(s) above {budget:.2f} touching a "
              "skill authored here: "
              + ", ".join(f"{p['left']}|{p['right']}" for p in needs_review), file=sys.stderr)
        return 1
    worst = worst_no_edge(scored, args.min_shared)
    summary = (f"{len(skills)} skill(s), {len(scored)} scored pair(s), {len(undeclared)} "
               f"undeclared overlap(s) reported")
    if worst:
        summary += f", worst {worst['left']} | {worst['right']} at {worst['containment']:.4f}"
    if needs_review:
        print(f"REVIEW {summary}. {len(needs_review)} of them are over {budget:.2f} with a skill "
              f"authored here: "
              + ", ".join(f"{p['left']}|{p['right']}" for p in needs_review)
              + ". Advisory - which skill wins is a reviewer's call, not this check's, so "
                "the merge is not blocked.")
    else:
        print(f"OK {summary}, none over {budget:.2f}.")
    return 1 if args.strict and undeclared else 0


if __name__ == "__main__":
    sys.exit(main())
