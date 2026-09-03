#!/bin/bash
set -uo pipefail
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p /logs/verifier
set +e
python3 -m pytest /tests/test_onetbb_flow_graph.py -v 2>&1
EXIT_CODE=$?
set -e
if [ $EXIT_CODE -eq 0 ]; then echo 1 > "$REWARD_FILE"; else echo 0 > "$REWARD_FILE"; fi
exit $EXIT_CODE
