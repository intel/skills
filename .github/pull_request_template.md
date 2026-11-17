## What this changes

<!-- One or two sentences. If you are adding a skill, say what problem it solves. -->

## Checklist

- [ ] I checked `description` against requests a user would really type — see
      CONTRIBUTING.md.
- [ ] `python3 tools/validate_skills.py` passes locally.

<!--
Everything else is checked by CI, so there is nothing to attest here.

If you are bringing a skill that already lives in another repository, put the pin in your
skills.yaml entry and run `python3 tools/sync_external.py --write` — that generates the
directory, its evals/ and its .source.json, so upstream needs evals/evals.json first. If you
wrote the skill here, add evals/evals.json (templates/evals.json) and one Harbor task
(templates/task_example.md).

Measurement (the three-arm differential and the discoverability run) is run by
maintainers; it needs credentials a fork cannot have, and no contributor is asked for it.
See MAINTAINERS.md.
-->
