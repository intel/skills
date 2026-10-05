#!/usr/bin/env python3
"""Report skill pairs that do the same things without saying which one to use.

    python3 tools/lint_skill_overlap.py                      # every pair, ranked
    python3 tools/lint_skill_overlap.py --show vllm-xpu-run  # one skill's signature
    python3 tools/lint_skill_overlap.py --max-overlap 0.75 --min-shared 8 --advisory

Two skills driving the same tool compete for the same request. Fine as long as one says
which is which: `vllm-xpu-bench` and `vllm-xpu-run` name each other, `torch-xpu-run` and
`torch-xpu-profile` never have. So this compares what each skill *does* - commands, flags,
variables, API calls, endpoint paths, from fences, inline spans and bundled files - and
subtracts every pair with a hand-off written down. Name and description only order a queue
of the pairs with too little code to compare; they decide nothing.

Duplicated *text* is the blind spot: `torch-xpu-bench` and `vllm-xpu-bench` share an
`## Env vars` table and score 0.0833. The OK line says so, being the line at risk of reading
as a verdict on duplication.

CI runs `--advisory`: which of two overlapping skills wins is a reviewer's call. Two things
block - `--self-test`, and containment 1.0 on a pair with a skill authored here. Rationale
and calibration: MAINTAINERS.md.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
# The two floors are different kinds of thing and only one of them is a flag. --min-shared
# is a property of a pair: how much they have in common. This is a property of one skill's
# signature, and decides whether the pair is scored at all - under five actions a
# containment ratio is arithmetic on noise. linux-perf and onetbb-quickstart share
# cmd:double, cmd:i and cmd:int, three loop variables out of a C snippet, for a containment
# of 1.0 against onetbb-quickstart's three actions: the only 1.0 pair in this tree, and the
# floor is the whole reason the gate does not block on it.
MIN_SIGNATURE = 5
# --self-test reads the threshold out of the workflow rather than keeping a second copy, and
# asserts that every other file documenting the gate names the same number.
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validate.yml"
GATE_THRESHOLD = re.compile(r"--max-overlap\s+([0-9.]+)")
THRESHOLD_TEXT = {".md", ".yml", ".yaml", ".py", ".sh", ".json", ".txt"}

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
    "mamba", "bash", "sh", "zsh", "npm", "npx", "node", "pytest", "seq",
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
    "flag:--extra-index-url", "flag:--pre", "flag:--editable",
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


# Which deliberate break --mutate is running, readable from every extractor without
# threading a test-only argument through eight signatures. See MUTATIONS below.
BROKEN = SimpleNamespace(which=None)

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
    if BROKEN.which == "M3":
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

    Lower case only, or console output and heredoc markers arrive as commands; a head the
    script defines itself is house style (`usage`, `die`), not a shared action.
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
    """Flags, variables and endpoints in inline code spans - those three only.

    An inline span is usually a value (`bf16`, `awq`), not a command; those three are
    unambiguous by shape, a bare word is not.
    """
    prose = FENCE.sub("", text)
    spans = "\n".join(INLINE_CODE.findall(prose))
    return flags(spans) | shared_elements(spans)


def python_elements(text: str) -> set[str]:
    """Imports and two-segment dotted paths - `torch.xpu`, `dpnp.asnumpy`.

    NOT_AN_API drops the two shapes a dotted regex cannot tell from a call: a filename and
    a host (`config.json`, `huggingface.co`).
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
        python = language in PYTHON_LANGS and BROKEN.which != "M2"
        found |= python_elements(block) if python else shell_elements(block)
    return found


def signature(body: str, bundled: dict[str, str]) -> set[str]:
    """Every action this skill performs, minus the platform layer.

    Bundled `.md` counts - `vllm-xpu-bench` reaches `/v1/chat/completions` only there. `.c`
    files are not read, so `onetbb-quickstart` is under-measured.
    """
    found = document_elements(body)
    for name, text in bundled.items():
        if name.endswith(".md"):
            found |= document_elements(text)
        elif name.endswith(".py") and BROKEN.which != "M2":
            found |= python_elements(text)
        elif name.endswith((".sh", ".bash")):
            found |= shell_elements(text)
    return found - (set() if BROKEN.which == "M1" else PLATFORM)


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
    if BROKEN.which == "M4":
        return False
    pattern = rf"(?<![A-Za-z0-9-]){re.escape(other)}(?![A-Za-z0-9-])"
    return re.search(pattern, skill["reference_text"]) is not None


def handoff_index(skills: list[dict]) -> dict[str, set[str]]:
    """Which catalog skills each body names, in one pass per body rather than one per pair.

    Where the cost lives: at 300 skills the per-pair form spent 22.0s of 22.0s here. Same
    boundaries as mentions(), and --self-test asserts the two agree on every pair.
    """
    names = sorted((skill["name"] for skill in skills), key=len, reverse=True)
    if not names or BROKEN.which == "M4":
        return {skill["name"]: set() for skill in skills}
    alternation = "|".join(re.escape(name) for name in names)
    pattern = re.compile(rf"(?<![A-Za-z0-9-])({alternation})(?![A-Za-z0-9-])")
    return {
        skill["name"]: set(pattern.findall(
            skill["reference_text"][: len(skill["reference_text"]) // 2]
            if BROKEN.which == "M7" else skill["reference_text"]))
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


def below_floor(skills: list[dict]) -> int:
    """How many pairs MIN_SIGNATURE removes from the budget, without enumerating them.

    Every pair touching a skill under the floor, counted once: each thin skill against
    every thick one, plus the thin ones among themselves.
    """
    thin = sum(1 for skill in skills if len(skill["signature"]) < MIN_SIGNATURE)
    if BROKEN.which == "M10":
        return 0
    return thin * (len(skills) - thin) + thin * (thin - 1) // 2


def pairs(skills: list[dict], min_shared: int,
          index: dict[str, set[str]] | None = None) -> list[dict]:
    """Score every pair. Under MIN_SIGNATURE actions on the smaller side there is too little
    material to mean anything either way, so the pair is not scored at all.

    Name and description overlap ride along as nearly-free columns for lexical() to rank by.
    They gate nothing: a shared-name precondition would drop 4 of the 14 judgeable pairs.
    """
    if index is None:
        index = handoff_index(skills)
    vocab = vocabulary(skills)
    out = []
    for left, right in combinations(skills, 2):
        if BROKEN.which == "M6" and not (
            vocab[left["name"]][0] & vocab[right["name"]][0]
        ):
            continue
        shared = left["signature"] & right["signature"]
        smaller = min(len(left["signature"]), len(right["signature"]))
        if smaller < MIN_SIGNATURE and BROKEN.which != "M9":
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

    A skill with almost no code has no action signature, but every skill has a name and a
    description. Blindness of the action axis is the first sort key and prose only the
    second: ranking on words alone buried a prose-only candidate at rank 113 of 466, against
    rank 7 here. It still names the wrong counterpart, so this is a reading order, not a
    verdict.
    """
    if index is None:
        index = handoff_index(skills)
    vocab = vocabulary(skills)
    out = []
    for left, right in combinations(skills, 2):
        shared = left["signature"] & right["signature"]
        smaller = min(len(left["signature"]), len(right["signature"]))
        if smaller >= MIN_SIGNATURE and len(shared) >= min_shared:
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

    A hand-off explains a division of labour, and at containment 1.0 there is none to
    explain: a copy carries the original's name in its own text and would silence itself.
    """
    if pair["containment"] <= max_overlap or len(pair["shared"]) < min_shared:
        return "ok"
    if pair["handoff"] and (pair["containment"] < 1.0 or BROKEN.which == "M5"):
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


def strongest(scored: list[dict], fixture: dict) -> dict:
    """The fixture's highest-scoring pair, or a named failure when it has none.

    Not inlined as max(): M3 and M6 leave nothing for the fixture to be scored against, and
    a bare max() would end those mutations in a traceback rather than in the assertion that
    is supposed to catch them.
    """
    ranked = [pair for pair in scored if fixture["name"] in (pair["left"], pair["right"])]
    if not ranked:
        sys.exit(f"FAIL --self-test scored {fixture['name']} against nothing: it shares no "
                 f"action with any skill, so the assertion about it could not run")
    return max(ranked, key=lambda pair: pair["containment"])


def queued(fixture: dict, skills: list[dict], min_shared: int) -> list[dict]:
    """Where a candidate lands in the name/description queue, catalog included."""
    ranked = lexical(skills + [as_skill(fixture)], min_shared=min_shared)
    return [pair for pair in ranked if fixture["name"] in (pair["left"], pair["right"])]


def threshold_copies() -> dict[float, list[str]]:
    """Every file that documents the gate threshold, keyed by the value it names.

    A walk, not a hand-list: the drift worth catching is a fifth copy nobody thought of. The
    rule it implies - any --max-overlap outside skills/ is a copy of the gate's number, so
    document another run in prose rather than at another value. skills/ is content.
    """
    found: dict[float, list[str]] = {}
    for base, directories, names in os.walk(REPO_ROOT):
        # Dotted directories other than .github are not the repository: a worktree under
        # .worktrees/ holds an older copy of the workflow, and that is not drift.
        directories[:] = sorted(d for d in directories if d != "skills"
                                and (d == ".github" or not d.startswith(".")))
        for name in sorted(names):
            path = Path(base) / name
            if path.suffix not in THRESHOLD_TEXT:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for value in GATE_THRESHOLD.findall(text):
                found.setdefault(float(value), []).append(
                    path.relative_to(REPO_ROOT).as_posix())
    return found


def refuses(value: str) -> str:
    """argparse's message for a rejected --min-shared, or "" if the parser accepted it.

    Through the parser, not shared_actions(): the wiring is what regresses, and with
    `type=int` back in place the flag takes 3 and reads as stricter rather than narrower.
    """
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            build_parser().parse_args(["--min-shared", value])
    except SystemExit:
        return " ".join(stderr.getvalue().split()).partition("error: ")[2]
    return ""


def suggested_threshold(ceiling: float) -> float | None:
    """The 0.05 step to put in the workflow for a tree with this ceiling, or None.

    1.15x first, then the lowest step the band allows: at the band's edge the next skill
    moves the ceiling again. Above 0.90 nothing is offered - two skills that close are near
    subsumption, and the answer is to narrow one, not to raise a number past them.
    """
    room = [step / 20 for step in range(2, 19) if ceiling < step / 20 <= 1.5 * ceiling]
    return min([v for v in room if v >= 1.15 * ceiling] or room) if room else None


def self_test(max_overlap: float | None, min_shared: int) -> int:
    """Assert the detector still detects, against skills/ as committed. Writes nothing.

    Every way this check breaks is silent: a fence regex that stops matching scores every
    pair 0, which reads exactly like a catalog with no overlap.

    check() fails the build and is restricted to facts a contribution cannot move: a
    detector that stopped detecting, an exact subset, a number written twice and disagreeing.
    note() prints calibration against today's tree - the threshold band, the hand-off edges
    that exist - and never fails, because adding a skill changes all of it.
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name} - {detail}")
        if not ok:
            failures.append(name)

    def note(name: str, ok: bool, detail: str) -> None:
        """Calibration against the catalog as it stands today, printed and never failed.

        A contribution that only adds a skill can move every one of these, so failing them
        would charge every contributor for the shape of the tree they arrived at.
        """
        print(f"{'ok  ' if ok else 'note'} {name} - {detail}")

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

    router = as_skill({"name": "_fixture-router", "description": "",
                       "body": "For the maintained path use _fixture-target.\n"})
    target = as_skill({"name": "_fixture-target", "description": "", "body": "\n"})
    check("a body that names another skill declares a hand-off",
          mentions(router, target["name"]) and not mentions(target, router["name"])
          and handoff_index([router, target])[router["name"]] == {target["name"]},
          "one direction is enough, and the index agrees with the scan on the fixture pair")

    def edge(left: str, right: str) -> bool:
        return left in index and right in index and (
            mentions(index[left], right) or mentions(index[right], left)
        )

    note("the catalog's hand-off edges are where they were",
         edge("vllm-xpu-bench", "vllm-xpu-run") and edge("vllm-xpu-run", "vllm-xpu-profile")
         and not edge("torch-xpu-run", "torch-xpu-profile"),
         "the two vllm pairs name each other, the torch pair names neither - a note, because "
         "writing the missing hand-off is the fix this check asks for and must not fail a "
         "build")

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
    note("no unclassified high-frequency element", not unclassified,
          ", ".join(unclassified) or
          "every element at document frequency >= 10 is in PLATFORM or IDENTITY")

    scored = pairs(skills, min_shared=min_shared)
    worst = worst_no_edge(scored, min_shared)
    if worst is None:
        sys.exit("FAIL --self-test found no undeclared pair to calibrate against")

    # The workflow is the one copy CI reads; the rest are documentation that would otherwise
    # tell a contributor to run a gate this repository does not run.
    copies = threshold_copies()
    workflow_name = WORKFLOW.relative_to(REPO_ROOT).as_posix()
    check("every copy of the threshold names the same number",
          len(copies) == 1 and any(workflow_name in paths for paths in copies.values()),
          "; ".join(f"{value:.2f} in {', '.join(sorted(set(paths)))}"
                    for value, paths in sorted(copies.items()))
          or "nothing names --max-overlap, so CI runs no gate")
    if max_overlap is None:
        found = {value for value, paths in copies.items() if workflow_name in paths}
        budget = min(found) if found else 0.0
    else:
        budget = max_overlap
    # Both bounds read off the tree: above every pair already merged, and no more than half
    # again above it. Calibrated on the highest *judgeable* pair, not the highest undeclared
    # one, which falls whenever someone writes a hand-off line.
    judgeable = [pair for pair in scored if len(pair["shared"]) >= min_shared]
    ceiling = max(pair["containment"] for pair in judgeable)
    top_pair = max(judgeable, key=lambda pair: pair["containment"])
    in_band = ceiling < budget <= 1.5 * ceiling
    measured = (f"highest of the {len(judgeable)} judgeable pair(s) is {top_pair['left']}|"
                f"{top_pair['right']} at {ceiling:.4f}, threshold {budget:.2f}, margin "
                f"{budget / ceiling:.2f}x (want 1.0-1.5x)")
    if in_band:
        detail = (f"{measured}; worst undeclared pair {worst['left']}|{worst['right']} at "
                  f"{worst['containment']:.4f}")
    else:
        # A note, not a failure: the pull request that moves the ceiling is usually one that
        # only added a skill, and it does not own this number. Name the file and the value so
        # a maintainer can move it in one edit when they decide to.
        suggestion = suggested_threshold(ceiling)
        where = ", ".join(sorted({path for paths in copies.values() for path in paths})) \
            or workflow_name
        cause = ("the catalog grew into the threshold" if budget <= ceiling
                 else "the threshold carries slack nobody chose")
        detail = (f"{measured}. To fix: {cause}, legal range is above {ceiling:.4f} and up "
                  f"to {1.5 * ceiling:.4f}, so "
                  + (f"set --max-overlap to {suggestion:.2f} in {where}. "
                     f"{top_pair['left']} | {top_pair['right']} is not itself a finding: it "
                     f"is reported only while the number is left below it."
                     if suggestion else
                     f"no step up to 0.90 leaves room above it: narrow {top_pair['left']} "
                     f"or {top_pair['right']} instead, since at {ceiling:.4f} one of the "
                     f"two nearly does everything the other does."))
    note("the threshold sits just above the whole tree", in_band, detail)

    thin = sorted(
        skill["name"] for skill in skills
        if len(skill["signature"]) < min_shared
    )
    check("skills below the min-shared reach are named", True,
          ", ".join(thin) or "none - every skill carries at least "
          f"{min_shared} actions")

    # The other floor, and the one that is not a flag. Asserted through the pair budget
    # because that is where it is visible: the summary line adds up only while every pair
    # the floor removes is a pair pairs() declined to score.
    hidden = sorted(
        ((left["signature"] & right["signature"], smaller, left, right)
         for left, right in combinations(skills, 2)
         if (smaller := min(len(left["signature"]), len(right["signature"])))
         < MIN_SIGNATURE),
        key=lambda row: len(row[0]) / row[1])
    # An empty floor is a pass, not a break: a catalog where every skill carries
    # MIN_SIGNATURE actions removes no pair, and there is then no strongest removed pair.
    noise = hidden[-1] if hidden else None
    check("the action floor is a property of a skill, not of a pair",
          len(hidden) + len(scored) == len(skills) * (len(skills) - 1) // 2
          and below_floor(skills) == len(hidden),
          f"{len(hidden)} removed + {len(scored)} scored of "
          f"{len(skills) * (len(skills) - 1) // 2} pair(s), {below_floor(skills)} of them "
          f"reported as removed"
          + (f"; the strongest removed pair is {noise[2]['name']} | {noise[3]['name']} at "
             f"{len(noise[0]) / noise[1]:.4f} on {', '.join(sorted(noise[0]))}, which is "
             f"what a containment ratio over {noise[1]} action(s) is worth"
             if noise else " - no skill is under the floor"))

    check("--min-shared under the floor is refused, not clipped",
          bool(refuses(str(MIN_SIGNATURE - 1))) and not refuses(str(MIN_SIGNATURE)),
          refuses(str(MIN_SIGNATURE - 1))
          or f"--min-shared {MIN_SIGNATURE - 1} was accepted, and the pairs it asks for are "
             f"the ones the {MIN_SIGNATURE}-action floor already removed")

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

    top = strongest(against(DUPLICATE, skills, min_shared), DUPLICATE)
    check("a paraphrased duplicate is caught and put in front of a reviewer",
          verdict(top, budget, min_shared) == "REVIEW",
          f"scores {top['containment']:.4f} with {len(top['shared'])} shared actions "
          f"against {top['right'] if top['left'] == DUPLICATE['name'] else top['left']}, "
          f"while its description shares {top['jaccard']:.2f} Jaccard - the prose is not "
          f"what caught it")

    top_copy = strongest(against(DECLARED_COPY, skills, min_shared), DECLARED_COPY)
    check("a copy cannot excuse itself with the hand-off it inherited",
          verdict(top_copy, budget, min_shared) == "REVIEW" and bool(top_copy["handoff"]),
          f"scores {top_copy['containment']:.4f} against vllm-xpu-run and names it "
          f"(hand-off {top_copy['handoff'] or 'none'}), reported anyway because at "
          f"containment 1.0 there is no division of labour for a hand-off to describe")

    copy_scored = against(DECLARED_COPY, skills, min_shared)
    blocked = subsumed_pairs(copy_scored)
    committed = subsumed_pairs(scored)
    check("a copy fails the gate even under --advisory",
          bool(blocked) and not committed,
          f"{len(blocked)} pair(s) block with the copy injected and {len(committed)} on "
          f"skills/ as committed, whose highest judgeable pair is {ceiling:.4f} - a cliff "
          f"with no traffic on it, not a line the catalog is drifting toward"
          + (f". The gate reports the same pair(s) with the same fix, and that report is the "
             f"one to read: " + ", ".join(f"{p['left']}|{p['right']}" for p in committed)
             if committed else ""))

    # The reach of --min-shared is not the reach of the copy rule, and inheriting it was a
    # real hole: 4 of the 33 skills carry too few actions for a verbatim copy of them to
    # share 8 of anything. Built from the thinnest skill the catalog has rather than from an
    # inline fixture, so it keeps testing whatever the thin end of the catalog becomes.
    thinnest = min((s for s in skills if MIN_SIGNATURE <= len(s["signature"]) < min_shared),
                   key=lambda s: len(s["signature"]), default=None)
    if thinnest:
        clone = {**thinnest, "name": f"{thinnest['name']}-copy", "imported": False}
        thin_copy = subsumed_pairs(pairs(skills + [clone], min_shared=min_shared))
        check("a copy under the --min-shared reach still fails the gate",
              any(clone["name"] in (pair["left"], pair["right"]) for pair in thin_copy),
              f"a verbatim copy of {thinnest['name']} shares its "
              f"{len(thinnest['signature'])} action(s), under --min-shared {min_shared}, and "
              f"{len(thin_copy)} pair(s) block - scoring it ok because it is too small to be "
              f"a near-miss is how a copy merged green")

    upstream = pairs(skills + [as_skill(DECLARED_COPY) | {"imported": True}],
                     min_shared=min_shared)
    upstream_top = strongest(upstream, DECLARED_COPY)
    check("the same copy between two imports stays advisory",
          verdict(upstream_top, budget, min_shared) == "WARN"
          and not subsumed_pairs(upstream),
          f"scores {upstream_top['containment']:.4f} against vllm-xpu-run and is a "
          f"{verdict(upstream_top, budget, min_shared)}, so the merge is not blocked - that "
          f"repair lives in someone else's repository")

    thin_pairs = [
        pair for pair in against(THIN, skills, min_shared)
        if THIN["name"] in (pair["left"], pair["right"])
    ]
    flagged = [pair for pair in thin_pairs if verdict(pair, budget, min_shared) == "REVIEW"]
    check("a thin restatement passes, and that is the documented limit",
          not flagged and not thin_pairs,
          f"its {len(as_skill(THIN)['signature'])} action(s) are under the "
          f"{MIN_SIGNATURE}-action floor, "
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
    "M4": "stop detecting hand-offs, in both the scan and the index",
    "M5": "let an inherited hand-off excuse a verbatim copy",
    "M6": "score only pairs whose names share a part",
    "M7": "index hand-offs from half of each body",
    "M8": "let --advisory pass a copy",
    "M9": f"score pairs under the {MIN_SIGNATURE}-action floor",
    "M10": "report no pairs under the floor",
}


def shared_actions(value: str) -> int:
    """--min-shared, refused below the floor rather than silently clipped to it.

    Below MIN_SIGNATURE it reads as more sensitivity and delivers none: those pairs are
    already gone, and --queue is where they are listed.
    """
    count = int(value)
    if count < MIN_SIGNATURE:
        raise argparse.ArgumentTypeError(
            f"{count} is below the {MIN_SIGNATURE}-action floor a signature needs before any "
            f"pair it is in is scored, so nothing under {MIN_SIGNATURE} widens what is "
            f"judged; pass {MIN_SIGNATURE} or more, and see --queue N for the pairs under "
            "the floor")
    return count


def build_parser() -> argparse.ArgumentParser:
    """Every flag, kept out of main() so what main() does is legible in one screen."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-overlap", type=float, default=None, metavar="F",
                        help="the line above which an undeclared pair is worth a reviewer's "
                             "attention. Omit to report only.")
    parser.add_argument("--min-shared", type=shared_actions, default=8, metavar="N",
                        help=f"a pair needs this many shared actions before it is judged. "
                             f"Not the only floor: a skill with fewer than {MIN_SIGNATURE} "
                             f"actions of its own is never scored against anything, however "
                             f"low this goes")
    parser.add_argument("--advisory", action="store_true",
                        help="report and exit 0, annotating each finding for the pull "
                             "request, unless a pair with a skill authored here reaches "
                             "containment 1.0. How CI runs it: which of two overlapping "
                             "skills wins is a reviewer's call, but a skill that does "
                             "nothing another does leaves nothing to judge.")
    parser.add_argument("--show", metavar="SKILL", help="print one skill's signature")
    parser.add_argument("--top", type=int, default=15, metavar="N",
                        help="rows of the ranked table to print")
    parser.add_argument("--queue", type=int, default=5, metavar="N",
                        help="pairs to list by name and description overlap alone, for the "
                             "pairs the action axis cannot judge. A reading order, not a "
                             "verdict - and the dial for anything downstream that pays per "
                             "pair, since all-pairs is 44850 comparisons at 300 skills.")
    parser.add_argument("--json", action="store_true",
                        help="every scored pair and the lexical queue as one object, for "
                             "reproducing a number in this file's docstring")
    parser.add_argument("--self-test", action="store_true",
                        help="assert the detector still detects, against skills/ as "
                             "committed. Writes nothing. Run it before the gate.")
    parser.add_argument("--mutate", choices=sorted(MUTATIONS),
                        help="break the detector on purpose and prove --self-test notices")
    return parser


def show_signature(skills: list[dict], name: str) -> int:
    """--show: the actions extracted from one skill, for arguing with the extractor."""
    match = next((skill for skill in skills if skill["name"] == name), None)
    if match is None:
        sys.exit(f"FAIL no skills/{name}")
    print(f"{match['name']} - {len(match['signature'])} action(s), "
          f"{'imported' if match['imported'] else 'authored here'}")
    for element in sorted(match["signature"]):
        print(f"    {element}")
    return 0


def print_ranked(scored: list[dict], queue: list[dict], top: int) -> None:
    """The table, then the pairs the action axis cannot judge - a reading order, not a
    finding: the near-verbatim fixture scores 0.05 on description and 1.0000 on actions."""
    print(f"{'containment':>11} {'shared':>6} {'name':>4} {'jaccard':>7} {'hand-off':>8}  pair")
    for pair in scored[:top]:
        edge = pair["handoff"] or "-"
        print(f"{pair['containment']:11.4f} {len(pair['shared']):6} {pair['name_shared']:4} "
              f"{pair['jaccard']:7.2f} {edge:>8}  {pair['left']} | {pair['right']}")
    if queue:
        print(f"\nwhere actions cannot judge, closest by description then name "
              f"(top {len(queue)}, least code first):")
        for pair in queue:
            print(f"       names {pair['name_shared']} desc {pair['jaccard']:.2f} "
                  f"({pair['actions']} action(s) on the smaller side)  "
                  f"{pair['left']} | {pair['right']}")


def undeclared_pairs(scored: list[dict], min_shared: int) -> list[dict]:
    """Every pair with enough shared material to judge and nothing written down to separate
    them - no hand-off, or a hand-off that cannot be true because one skill does everything
    the other does. No second threshold: the threshold only decides WARN from REVIEW.
    """
    return [
        pair for pair in scored
        if (len(pair["shared"]) >= min_shared or pair["containment"] >= 1.0)
        and (not pair["handoff"] or pair["containment"] >= 1.0)
    ]


def subsumed_pairs(scored: list[dict]) -> list[dict]:
    """The REVIEW pairs `--advisory` has nothing to defer: one skill does everything the
    other does, so there is no division of labour to weigh.

    Deliberately not routed through verdict(), which would inherit --min-shared: a copy of a
    7-action skill shares 7 actions, and under `--min-shared 8` the gate called it ok and
    merged it green. MIN_SIGNATURE stays the only floor, `authored` the only scoping - an
    upstream copy is repaired upstream.
    """
    if BROKEN.which == "M8":
        return []
    return [pair for pair in scored
            if pair["authored"] and pair["containment"] >= 1.0]


def report_undeclared(undeclared: list[dict], budget: float, args: argparse.Namespace) -> None:
    """One block per finding, on stderr only when the gate is enforcing it as a failure."""
    for pair in undeclared:
        subsumed = pair["containment"] >= 1.0
        # A copy is reported on its own terms, not through --min-shared: a copy of a
        # 7-action skill shares 7 actions and would otherwise print nothing at all.
        blocks_as_copy = subsumed and pair["authored"] and args.max_overlap is not None
        over = verdict(pair, budget, args.min_shared) == "REVIEW" or blocks_as_copy
        blocking = blocks_as_copy or (over and not args.advisory)
        stream = sys.stderr if blocking else sys.stdout
        because = ("everything the smaller one does, the larger one already does, so the "
                   "hand-off cannot be what separates them"
                   if subsumed else "no hand-off in either direction")
        remedy = ("drop one, or narrow one so it stops being contained in the other"
                  if subsumed else "name the other skill in one of the two descriptions")
        sys.stdout.flush()  # or the buffered table lands after the unbuffered failures
        print(f"\n{'REVIEW' if over else 'WARN  '} overlap {pair['containment']:.4f} "
              f"{pair['left']} | {pair['right']}: {because}.", file=stream)
        print(f"       shared ({len(pair['shared'])}): " + ", ".join(pair["shared"]),
              file=stream)
        print(f"       fix: {remedy}"
              + ("" if pair["authored"] else " (upstream - both are imported)"), file=stream)
        if args.advisory and os.environ.get("GITHUB_ACTIONS"):
            sys.stdout.flush()
            # Keyed on subsumption rather than on blocking, so an upstream copy is not
            # asked for the hand-off the line above it has just ruled out.
            if subsumed:
                ask = f"Fix: {remedy}" + (
                    "." if pair["authored"] else ", upstream - both are imported.")
            else:
                ask = ("Say in this pull request which one a request should route to, and "
                       "why both belong.")
            # Both sides of a copy. One of the two names is alphabetically first and the
            # other is the file the pull request added; annotating only the first showed the
            # finding on an untouched file, where GitHub renders nothing inline.
            for on in ([pair["left"], pair["right"]] if subsumed else [pair["left"]]):
                print(f"::{'error' if blocking else 'warning'} "
                      f"file=skills/{on}/SKILL.md::{pair['left']} and "
                      f"{pair['right']} drive {len(pair['shared'])} of the same actions "
                      f"(containment {pair['containment']:.2f}). {because.capitalize()}. "
                      f"{ask}")


def summarize(skills: list[dict], scored: list[dict], undeclared: list[dict],
              budget: float, args: argparse.Namespace) -> int:
    """The last line and the exit code. REVIEW names the pairs someone here can fix."""
    needs_review = [p for p in scored if verdict(p, budget, args.min_shared) == "REVIEW"]
    # Report-only is report-only: --max-overlap's help promises it, and a run somebody
    # started to look around the catalog is not the run that should decide a merge.
    subsumed = subsumed_pairs(scored) if args.max_overlap is not None else []
    print()
    sys.stdout.flush()
    if needs_review and not args.advisory:
        print(f"FAIL {len(needs_review)} undeclared overlap(s) above {budget:.2f} touching a "
              "skill authored here: "
              + ", ".join(f"{p['left']}|{p['right']}" for p in needs_review), file=sys.stderr)
        return 1
    worst = worst_no_edge(scored, args.min_shared)
    # Both floors in the one line, because a pair count on its own reads as coverage: most
    # of this tree is out of reach of the action axis, and only --queue looks at it.
    judged = len([p for p in scored if len(p["shared"]) >= args.min_shared])
    summary = (f"{len(skills)} skill(s), {len(skills) * (len(skills) - 1) // 2} pair(s) = "
               f"{judged} judged + {len(scored) - judged} under --min-shared "
               f"{args.min_shared} + {below_floor(skills)} under the {MIN_SIGNATURE}-action "
               f"floor, {len(undeclared)} undeclared overlap(s) reported")
    if worst:
        summary += f", worst {worst['left']} | {worst['right']} at {worst['containment']:.4f}"
    # A copy is reported whether or not it reaches --min-shared, so the last line counts it
    # too: an OK line printed above a FAIL line is worse than either of them alone.
    flagged = needs_review + [p for p in subsumed if p not in needs_review]
    if flagged:
        print(f"REVIEW {summary}. {len(flagged)} of them touch a skill authored here, over "
              f"{budget:.2f} or contained outright: "
              + ", ".join(f"{p['left']}|{p['right']}" for p in flagged)
              + (". Advisory - which skill wins is a reviewer's call, not this check's, so "
                 "the merge is not blocked." if not subsumed else "."))
    else:
        # Only on this line. A REVIEW line already has something to act on; an OK line is
        # the one a reader can mistake for "no duplication in this catalog".
        print(f"OK {summary}, none over {budget:.2f}. Scope: shared actions, not shared "
              "text - two skills can duplicate a whole section and score near zero here.")
    if subsumed:
        sys.stdout.flush()
        print(f"FAIL {len(subsumed)} pair(s) where a skill authored here does nothing the "
              "other does not: "
              + ", ".join(f"{p['left']}|{p['right']} at {p['containment']:.4f}"
                          for p in subsumed)
              + ". --advisory defers which of two overlapping skills a request should route "
                "to; it does not defer whether one of them is a copy. Drop one, or narrow "
                "one so it stops being contained in the other.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if not SKILLS_DIR.is_dir():
        sys.exit(f"FAIL no {SKILLS_DIR.relative_to(REPO_ROOT).as_posix()}")

    BROKEN.which = args.mutate
    max_overlap = args.max_overlap
    if BROKEN.which:
        print(f"# mutation {BROKEN.which}: {MUTATIONS[BROKEN.which]}")

    if args.self_test:
        return self_test(max_overlap, args.min_shared)

    report = Report()
    skills = load_all(report)
    for error in report.errors:
        print(f"ERR  {error}", file=sys.stderr)

    if args.show:
        return show_signature(skills, args.show)

    handoffs = handoff_index(skills)
    scored = pairs(skills, min_shared=args.min_shared, index=handoffs)
    queue = spread(lexical(skills, min_shared=args.min_shared, index=handoffs), args.queue)
    if args.json:
        print(json.dumps({"scored": scored, "queue": queue}, indent=2))
        return 0

    budget = max_overlap if max_overlap is not None else 1.01
    print_ranked(scored, queue, args.top)
    undeclared = undeclared_pairs(scored, args.min_shared)
    report_undeclared(undeclared, budget, args)
    return summarize(skills, scored, undeclared, budget, args)


if __name__ == "__main__":
    sys.exit(main())
