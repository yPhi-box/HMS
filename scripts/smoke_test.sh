#!/bin/bash
# HMS Smoke Test — verifies installation is working
# Usage: bash smoke_test.sh [port]

set -euo pipefail

PORT="${1:-8765}"
URL="http://localhost:$PORT"
PASS=0
FAIL=0
TOTAL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

check() {
    TOTAL=$((TOTAL + 1))
    local name="$1"
    local result="$2"
    
    if [[ "$result" == "true" ]]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}✓${NC} $name"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}✗${NC} $name"
    fi
}

echo "HMS Smoke Test"
echo "=============="
echo "Server: $URL"
echo ""

# 1. Health check
HEALTH=$(curl -s "$URL/health" 2>/dev/null)
check "Health endpoint responds" "$(echo "$HEALTH" | grep -q 'ok' && echo true || echo false)"

# 2. Version endpoint
VERSION=$(curl -s "$URL/version" 2>/dev/null)
check "Version endpoint responds" "$(echo "$VERSION" | grep -q 'version' && echo true || echo false)"

# 3. Stats endpoint
STATS=$(curl -s "$URL/stats" 2>/dev/null)
check "Stats endpoint responds" "$(echo "$STATS" | grep -q 'total_chunks' && echo true || echo false)"

# 4. Create test data
TEST_DIR=$(mktemp -d)
cat > "$TEST_DIR/test.md" << 'EOF'
# Test Document

## Configuration
- Server port: 9999
- Admin email: test@example.com
- API key: sk-test-12345

## Notes
The memory system uses hybrid search combining semantic vectors and keyword matching.
It runs entirely locally with zero API costs.
EOF

# 5. Index test data
INDEX=$(curl -s -X POST "$URL/index" \
    -H "Content-Type: application/json" \
    -d "{\"directory\": \"$TEST_DIR\", \"pattern\": \"**/*.md\", \"force\": true}" 2>/dev/null)
check "Index endpoint works" "$(echo "$INDEX" | grep -q 'Indexing complete' && echo true || echo false)"

CHUNKS=$(echo "$INDEX" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('stats',{}).get('total_chunks',0))" 2>/dev/null || echo 0)
check "Chunks created ($CHUNKS)" "$([ "$CHUNKS" -gt 0 ] && echo true || echo false)"

# 6. Search tests
sleep 2  # Let index settle

search_test() {
    local query="$1"
    local expect="$2"
    local result=$(curl -s -X POST "$URL/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\", \"max_results\": 3}" 2>/dev/null)
    
    local found=$(echo "$result" | grep -qi "$expect" && echo true || echo false)
    check "Search: \"$query\" → found \"$expect\"" "$found"
}

search_test "What is the server port?" "9999"
search_test "admin email address" "test@example.com"
search_test "API key" "sk-test"
search_test "hybrid search" "semantic"
search_test "API costs" "zero"

# 7. Empty query handling
EMPTY=$(curl -s -X POST "$URL/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "", "max_results": 3}' 2>/dev/null)
check "Empty query handled gracefully" "$(echo "$EMPTY" | grep -q 'results' && echo true || echo false)"

# Cleanup
rm -rf "$TEST_DIR"

# Results
echo ""
echo "=============="
echo -e "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} / $TOTAL total"

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
