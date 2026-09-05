---
name: your-skill-name
description: >-
  What this skill covers, and when an agent should load it. Write it in the words a user would type, not the canonical product name, and say what it is not for. Two or three sentences: the descriptions in this catalog run 450-600 characters, and 1024 is the hard limit.
license: Apache-2.0
---

# Your Skill Name

One or two sentences: what an agent can do with this that it could not do without it.

## Workflow

The steps, in the order someone actually does them. Code blocks that run as written.

```bash
# a command that runs as written, not a sketch of one
```

## When not to use this

The cases where this is the wrong tool, and what to reach for instead. An agent that
knows the limits gives better answers than one that only knows the happy path.

<!--
No required headings: the two above are a suggestion, not a schema. Checked by CI:
<= 500 lines, every file you ship mentioned by its path, every path you mention exists.
name and description are the only required keys. license is prefilled but optional: a
skill written here is Apache-2.0 whether the line is there or not. skills.yaml is where
the licence is recorded, and a license here may not contradict it.
Checked by a reviewer: no measured numbers here (those go in perf/).
Field reference: CONTRIBUTING.md.
-->
