#!/bin/bash
# validate-local.sh — pre-submission checklist runner
# Usage: bash validate-local.sh [HF_SPACE_URL]
set -e

SPACE_URL="${1:-http://localhost:8000}"
PASS=0; FAIL=0

ok()   { echo "✅ $1"; ((PASS++)); }
fail() { echo "❌ $1"; ((FAIL++)); }

echo "=== CloudOpsEnv pre-submission validator ==="
echo ""

# 1. Docker build
echo "--- Check 1: docker build ---"
if docker build -t cloudops-env . -q > /dev/null 2>&1; then
    ok "docker build succeeded"
else
    fail "docker build FAILED"
fi

# 2. Server liveness (assumes server already running or Docker started)
echo "--- Check 2: /health ping ---"
HTTP=$(curl -s -o /dev/null -w "%{http_code}" "$SPACE_URL/health" 2>/dev/null || echo "000")
if [ "$HTTP" = "200" ]; then
    ok "/health returned 200"
else
    fail "/health returned $HTTP (is the server running?)"
fi

# 3. /reset with empty body
echo "--- Check 3: POST /reset {} ---"
HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SPACE_URL/reset" \
    -H "Content-Type: application/json" -d '{}' 2>/dev/null || echo "000")
if [ "$HTTP" = "200" ]; then
    ok "POST /reset {} returned 200"
else
    fail "POST /reset {} returned $HTTP"
fi

# 4. openenv validate
echo "--- Check 4: openenv validate ---"
if command -v openenv &> /dev/null; then
    if openenv validate . > /dev/null 2>&1; then
        ok "openenv validate passed"
    else
        fail "openenv validate FAILED"
    fi
else
    echo "⚠️  openenv not installed — skipping (pip install openenv-core)"
    ((PASS++))
fi

# 5. inference smoke test (requires HF_TOKEN)
echo "--- Check 5: inference.py smoke test ---"
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set — skipping inference smoke test"
    ((PASS++))
else
    if timeout 60 python inference.py > /dev/null 2>&1; then
        ok "inference.py completed"
    else
        fail "inference.py FAILED or timed out"
    fi
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "🚀 READY TO SUBMIT" || echo "🔧 Fix failures before submitting"
exit $FAIL
