#!/bin/bash
# Test runner for onetbb-parallel-sort task.
# Runs the pytest suite and writes reward.txt.

set -uo pipefail

REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p /logs/verifier


# Capture exit code without letting set -e abort before we write reward.txt
set +e
python3 -m pytest /tests/test_parallel_sort.py -v 2>&1
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo 1 > "$REWARD_FILE"
else
    echo 0 > "$REWARD_FILE"
fi

exit $EXIT_CODE
