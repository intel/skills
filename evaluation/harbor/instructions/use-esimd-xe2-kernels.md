# Treatment instruction — esimd-xe2-kernels

Four skills are installed in your skills directory: `esimd-base`,
`esimd-lsc-2d-gather-scatter`, `esimd-lsc-slm`, and `xe2-dpas-patterns`. Read them
before you start, and follow their guidance where it applies to this task.

If the skills do not cover something the task needs, solve that part however you
normally would. If a skill contradicts what you observe in the environment, trust
the environment and say so in your final message.

<!--
Why this file exists, and what it costs.

Harbor copies each `--skill <dir>` into the agent's own skills directory, so without
this file the agent would have to find the skills from their `description` alone.
This instruction removes that step deliberately: the Level 2 gate is a controlled
comparison, and description-reachability is a second variable that would confound
the reward delta. So this arm measures the skills' BODIES, not their discoverability.

This is a bundle suite: the result measures the four skills together, not any one of
them. It is not individual evidence for promoting a member skill to `validated`.

Rules for editing this file:
  - It must not name a task, a symptom, an API, or a fix. Anything task-specific
    here becomes part of the treatment and inflates the delta.
  - The `no_skill` arm runs with no `--extra-instruction-path` at all. That is the
    intended asymmetry: the arms differ by "the skills are present and used", which
    is the thing being tested, not by an unrelated hint.
  - Changing this text changes what past deltas mean. Re-measure the arms rather
    than comparing a new candidate against a number produced under other wording.
-->
