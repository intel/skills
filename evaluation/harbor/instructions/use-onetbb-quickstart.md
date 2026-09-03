# Treatment instruction — onetbb-quickstart

A skill named `onetbb-quickstart` is installed in your skills directory. Read it
before you start, and follow its guidance where it applies to this task.

If the skill does not cover something the task needs, solve that part however you
normally would. If the skill contradicts what you observe in the environment,
trust the environment and say so in your final message.

<!--
Why this file exists, and what it costs.

Harbor copies `--skill <dir>` into the agent's own skills directory, so without
this file the agent would have to find the skill from its `description` alone.
This instruction removes that step deliberately: the Level 2 gate is a controlled
comparison, and description-reachability is a second variable that would confound
the reward delta. So this arm measures the skill's BODY, not its discoverability.

Discoverability is measured elsewhere and must not be assumed from a green Level 2:
  - the description test in CONTRIBUTING.md, at authoring time
  - Level 3, where the skill is offered the way an agent would really meet it

Rules for editing this file:
  - It must not name a task, a symptom, an API, or a fix. Anything task-specific
    here becomes part of the treatment and inflates the delta.
  - The `no_skill` arm runs with no `--extra-instruction-path` at all. That is the
    intended asymmetry: the arms differ by "a skill is present and used", which is
    the thing being tested, not by an unrelated hint.
  - Changing this text changes what past deltas mean. Re-measure the arms rather
    than comparing a new candidate against a number produced under other wording.
-->
