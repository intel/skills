# Harbor task example

A worked `evaluation/harbor/tasks/<task-name>/` for the Level 2 differential. **A skill
written here ships one**; a skill imported from a project repository does not. You need
no Harbor access to write it: CI runs the oracle arm on every task directory, so pushing
the five files is how you find out it works.

Runnable examples in this repository: the five `dpnp-*` directories under
[`evaluation/harbor/tasks/`](../evaluation/harbor/tasks). Layout and gate:
[`evaluation/harbor/README.md`](../evaluation/harbor/README.md).

## Files

```text
evaluation/harbor/tasks/<task-name>/
  task.toml                 metadata, limits, and the skill link
  instruction.md            the prompt the agent receives
  environment/Dockerfile    the container
  solution/solve.sh         the oracle solution — proves the task is solvable
  tests/test.sh             the verifier entry point
  tests/test_*.py           pytest assertions
  tests/rubric.json         instead of pytest, when the task scores prose
```

Shell scripts must have **LF** line endings. CRLF fails inside the container as
`/bin/bash^M: bad interpreter`, which surfaces as a missing reward with no hint
about the cause.

## `task.toml`

```toml
schema_version = "1.3"

[metadata]
# The task -> skill link is this field, not the directory path: `harbor run` needs
# a single discovery root, so tasks cannot nest under skills/.
skill = "your-skill-name"
# What the task exercises, named by you. Once a suite in ../../suites.json claims
# the task these must be that suite's capability ids and validate_skills.py fails if
# the two disagree; until then any non-empty list passes, and a maintainer reconciles
# the names when the suite is written. The capability class and the
# smoke/discriminating role live in suites.json only — one place to change when a
# task turns out not to separate the arms after all.
covers = ["some-capability-id"]
author_name = "<name>"
author_email = "<name>@intel.com"
difficulty = "easy"                 # easy | medium | hard
difficulty_explanation = "<one line: what the agent has to get right>"
category = "programming"
tags = ["python", "intel"]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0             # a ceiling on `environment.start()`, build included
# Network is for the image build. The agent and the verifier run without it, so a
# task cannot be solved by downloading someone's answer.
network_mode = "no-network"
cpus = 4
memory_mb = 4096
storage_mb = 8192
gpus = 0                            # keep at 0 unless the task truly needs a GPU
```

`gpus = 0` is worth defending: a CPU-only task runs on any maintainer's machine
and in CI. A task that needs an Intel GPU can only ever be run by whoever holds
that hardware, so it stops being a gate and becomes a favour someone does.

`build_timeout_sec` is easy to set too low, and it fails in a way that reads like a
broken task: harbor spends it on `environment.start()` as a whole, so an image that
compiles a library from source spends it on `docker build`, and running out raises
`EnvironmentStartTimeoutError` — an error whose message names no task, no step and
no command. Time your build cold (`docker build --no-cache`) and then multiply,
because CI shares a Docker daemon between concurrent builds and starts with no layer
cache; `onetbb-parallel-sort` builds oneTBB in 49 s on an idle 16-core host and
still exceeded 300 s there. It is a ceiling and not a budget: nothing waits for it,
so err high.

## `instruction.md`

The prompt the agent sees. Two rules that decide whether the task is worth
running at all:

- **Do not include the fix.** If the instruction contains the code, the API, or
  the pattern the skill teaches, the no-skill arm passes without knowing anything
  and the task measures nothing. This is the most common way a task ends up at a
  ceiling.
- **State the acceptance criteria, not the method.** Say which file to write and
  what must hold; let the agent choose how.

## `solution/solve.sh` and the oracle

The oracle agent applies `solve.sh` and nothing else. It uses no model and no API
key, which makes it the one arm runnable on a fork's pull request:

```bash
harbor run --path evaluation/harbor/tasks --agent oracle \
  --include-task-name '<task-name>' \
  --job-name smoke --jobs-dir harbor-jobs --yes

python3 tools/check_harbor_job.py harbor-jobs/smoke/result.json \
  --expected-trials 1 --reward-floor 1.0
```

An oracle below 1.0 means the task is broken, not that the agent is weak. Fix the
task before reading anything into a scored run.

## `tests/` — the verifier

The verifier writes the reward to `/logs/verifier/reward.txt`. No file there is an
*error*, not a zero — Harbor reports `RewardFileNotFoundError`, and
`check_harbor_job.py` keeps that separate from a genuine low score.

```bash
#!/bin/bash
# tests/test.sh
set -uo pipefail
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p /logs/verifier
set +e
python3 -m pytest /tests/test_solution.py -v 2>&1
EXIT_CODE=$?
set -e
if [ $EXIT_CODE -eq 0 ]; then echo 1 > "$REWARD_FILE"; else echo 0 > "$REWARD_FILE"; fi
exit $EXIT_CODE
```

Note the `set +e` around pytest: with `-e` in force a failing test would abort the
script before the `0` is written, and a legitimate failure would be reported as a
missing reward instead.

Assert on **behavior**, not on the shape of the code. A verifier that greps for an
import rewards the agent for typing a word; one that runs the result and compares
against a reference rewards it for being right.

For a task that scores prose instead, `tests/rubric.json` holds grouped regular
expressions plus `unsupported_claims`, where one forbidden claim zeros the reward.
That is the only place in this repository where patterns are evaluated as
patterns — `evals.json` checks are literal substrings.

## Registering it — maintainer-side

Your task directory is enough to open the pull request. `tools/validate_skills.py`
checks it and the oracle job runs it either way; it warns that no suite lists it yet,
and that warning is addressed to a maintainer, not to you.

Registering means adding the task to
[`evaluation/harbor/suites.json`](../evaluation/harbor/suites.json) with
`status: "implemented"`, the same `covers` list, and a capability declaring the class
and the smoke/discriminating role — which is what makes the reward count toward
something. A suite with an implemented task also needs
`evaluation/harbor/instructions/use-<skill>.md`, the neutral instruction the skill arms
receive; deciding both is a judgement about the whole portfolio, so it happens when a
maintainer takes the skill toward `validated`.
