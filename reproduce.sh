#!/usr/bin/env bash
# Recompute every headline number from the released per-step outputs. No model calls, no network, standard library only.
set -euo pipefail
cd "$(dirname "$0")"
for m in claude-opus-5-5 claude-sonnet-5-5 claude-haiku-5-5; do
  echo "== routing to $m (run 1)"; MODEL=$m python3 -I bench/ac_router_score.py bench/ac_fresh1000.json
done
echo "== routing-score alternatives and small model alone"; python3 -I bench/ac_router_curves.py | tail -1
