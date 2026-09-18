#!/usr/bin/env bash
# Local CI: python suite, plugin suite, and the cross-language oracle checks.
# The parity tests read reports/slot-ood-baseline.json; the third step fails the
# build if that committed oracle no longer matches a fresh regeneration, which
# is how cross-language drift of the slot rule gets caught. The plugin also
# vendors a copy for the standalone layout, so the fourth step compares the two
# committed copies byte for byte.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== python suite =="
python -m pytest -q

echo "== dsh-igm-memory suite =="
(cd dsh-igm-memory && npm test)

echo "== slot oracle consistency (python vs committed baseline) =="
python -m memory_arch.run_slot_ood --output "${TMPDIR:-/tmp}/slot-ood-check.json" > /dev/null
python - "${TMPDIR:-/tmp}/slot-ood-check.json" <<'EOF'
import json
import sys

fresh = json.load(open(sys.argv[1], encoding="utf-8"))["oracle"]
committed = json.load(open("reports/slot-ood-baseline.json", encoding="utf-8"))["oracle"]
if fresh != committed:
    sys.exit("slot-ood oracle drifted from reports/slot-ood-baseline.json: "
             "run python -m memory_arch.run_slot_ood and review the diff")
print("oracle matches the committed baseline")
EOF

echo "== vendored oracle copy (plugin vs reports) =="
cmp reports/slot-ood-baseline.json dsh-igm-memory/test/fixtures/slot-ood-baseline.json
echo "vendored copy matches reports/slot-ood-baseline.json"
echo "all checks passed"
