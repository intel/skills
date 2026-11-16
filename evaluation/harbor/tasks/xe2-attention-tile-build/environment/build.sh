#!/bin/bash
# Builds /app/check.cpp (which includes /app/attention.hpp) ahead of time for an
# Intel Arc B-series (Battlemage) GPU. No GPU is needed to build.
set -euo pipefail
src="${1:-/app/check.cpp}"
out="${2:-/app/check}"
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -I/app "$src" -o "$out"
echo "built $out"
