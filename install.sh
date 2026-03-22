#!/bin/bash
# ============================================================================
# HMS v2.2 — Hybrid Memory Server
# One-liner installer for OpenClaw
#
# Usage:
#   curl -sSL <url>/install.sh | bash
#   OR
#   bash install.sh [options]
#
# Options:
#   --dir PATH       Install directory (default: ~/hms)
#   --port PORT      Server port (default: 8765)
#   --watch PATH     Directory to watch for changes (default: ~/.openclaw/workspace)
#   --no-service     Don't install systemd service
#   --no-plugin      Don't install OpenClaw plugin
#   --help           Show this help
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[HMS]${NC} $1"; }
warn() { echo -e "${YELLOW}[HMS]${NC} $1"; }
err()  { echo -e "${RED}[HMS]${NC} $1"; exit 1; }

# Defaults
INSTALL_DIR="$HOME/hms"
PORT=8765
WATCH_DIR="$HOME/.openclaw/workspace"
INSTALL_SERVICE=true
INSTALL_PLUGIN=true

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)       INSTALL_DIR="$2"; shift 2;;
        --port)      PORT="$2"; shift 2;;
        --watch)     WATCH_DIR="$2"; shift 2;;
        --no-service) INSTALL_SERVICE=false; shift;;
        --no-plugin) INSTALL_PLUGIN=false; shift;;
        --help)
            head -18 "$0" | tail -12
            exit 0;;
        *) warn "Unknown option: $1"; shift;;
    esac
done

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  HMS v2.2 — Hybrid Memory Server         ║${NC}"
echo -e "${BLUE}║  Local embeddings · Zero API costs        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# ---- Pre-flight checks ----
log "Checking prerequisites..."

# Python 3.10+
if ! command -v python3 &>/dev/null; then
    err "Python 3 not found. Install it first: sudo apt install python3 python3-pip python3-venv"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    err "Python 3.10+ required (found $PY_VERSION)"
fi
log "  Python $PY_VERSION ✓"

# python3-venv (Ubuntu may need version-specific package like python3.12-venv)
VENV_TEST="/tmp/.hms-venv-test-$$"
if ! python3 -m venv "$VENV_TEST" &>/dev/null 2>&1; then
    rm -rf "$VENV_TEST"
    warn "python3-venv not working. Installing..."
    sudo apt install -y "python3.${PY_MINOR}-venv" 2>/dev/null || \
    sudo apt install -y python3-venv || err "Failed to install python3-venv"
fi
rm -rf "$VENV_TEST"
log "  python3-venv ✓"

# pip
if ! python3 -m pip --version &>/dev/null 2>&1; then
    warn "pip not found. Installing..."
    sudo apt install -y python3-pip || err "Failed to install pip"
fi
log "  pip ✓"

# build-essential (needed for hnswlib compilation)
if ! command -v gcc &>/dev/null; then
    warn "build-essential not found. Installing..."
    sudo apt install -y build-essential python3-dev || err "Failed to install build tools"
fi
log "  build tools ✓"

# ---- Install ----
log "Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

# Copy source files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/src" ]]; then
    cp "$SCRIPT_DIR/src"/*.py "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/src/requirements.txt" "$INSTALL_DIR/"
    log "  Source files copied ✓"
else
    err "Source directory not found at $SCRIPT_DIR/src"
fi

# Create virtual environment
log "Creating Python virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Install dependencies
log "Installing dependencies (this may take a minute)..."
pip install --quiet --upgrade pip
pip install --quiet -r "$INSTALL_DIR/requirements.txt"

# Also need hnswlib
pip install --quiet hnswlib tqdm

log "  Dependencies installed ✓"

# Download embedding model on first run
log "Downloading embedding model (one-time, ~90MB)..."
python3 -c "
import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f'Model loaded. Dimension: {model.get_sentence_embedding_dimension()}')
"
log "  Embedding model ready ✓"

deactivate

# ---- Systemd service ----
if [[ "$INSTALL_SERVICE" == "true" ]]; then
    log "Installing systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/hms.service"
    sudo tee "$SERVICE_FILE" > /dev/null << SVCEOF
[Unit]
Description=HMS - Hybrid Memory Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/server.py
Restart=always
RestartSec=5
Environment=TRANSFORMERS_VERBOSITY=error
Environment=HMS_WATCH_PATHS=$WATCH_DIR
Environment=HMS_PORT=$PORT

[Install]
WantedBy=multi-user.target
SVCEOF

    sudo systemctl daemon-reload
    sudo systemctl enable hms
    sudo systemctl start hms
    
    # Wait for it to come up
    sleep 8
    
    if curl -s "http://localhost:$PORT/health" | grep -q "ok"; then
        log "  Service running on port $PORT ✓"
    else
        warn "  Service may still be starting (model loading takes ~10s)"
        warn "  Check: sudo systemctl status hms"
    fi
fi

# ---- OpenClaw plugin ----
if [[ "$INSTALL_PLUGIN" == "true" ]]; then
    PLUGIN_DIR="$INSTALL_DIR/openclaw-plugin"
    
    if [[ -d "$SCRIPT_DIR/plugin" ]]; then
        log "Installing OpenClaw plugin..."
        mkdir -p "$PLUGIN_DIR"
        cp "$SCRIPT_DIR/plugin"/* "$PLUGIN_DIR/"
        
        # Update server URL in plugin config if non-default port
        if [[ "$PORT" != "8765" ]]; then
            sed -i "s/127.0.0.1:8765/127.0.0.1:$PORT/g" "$PLUGIN_DIR/openclaw.plugin.json"
        fi
        
        log "  Plugin files installed to $PLUGIN_DIR ✓"
        log ""
        log "  To activate the plugin, add to your OpenClaw config:"
        log "    plugins:"
        log "      entries:"
        log "        hms-memory:"
        log "          path: $PLUGIN_DIR"
    fi
fi

# ---- Initial index ----
if [[ -d "$WATCH_DIR" ]]; then
    log "Running initial index of $WATCH_DIR..."
    source "$INSTALL_DIR/venv/bin/activate"
    python3 -c "
from pathlib import Path
from indexer import MemoryIndexer
import os
os.chdir('$INSTALL_DIR')
indexer = MemoryIndexer(Path('memory.db'))
indexer.index_directory(Path('$WATCH_DIR'), pattern='**/*.md', force=True)
stats = indexer.get_stats()
print(f\"Indexed: {stats['total_chunks']} chunks from {stats['total_files']} files, {stats['total_entities']} entities\")
indexer.close()
" 2>&1 | tail -5
    deactivate
    deactivate
    
    # Restart service to pick up the new index
    if [[ "$INSTALL_SERVICE" == "true" ]]; then
        sudo systemctl restart hms
        sleep 8
    fi
    
    log "  Initial index complete ✓"
else
    warn "Watch directory $WATCH_DIR not found. Index will build when files appear."
fi

# ---- Drop HMS memory protocol into workspace ----
if [[ -d "$WATCH_DIR" ]]; then
    HMS_RULES="$WATCH_DIR/HMS-RULES.md"
    if [[ ! -f "$HMS_RULES" ]]; then
        log "Installing memory search protocol to workspace..."
        cat > "$HMS_RULES" << 'RULES_EOF'
# HMS Memory Search Protocol

You have HMS (Hybrid Memory Server) installed. It searches your entire memory in under 500ms.

## Rule: Search Before Speaking

Before answering ANY question involving facts, machines, people, credentials, IPs, configs, history, or prior decisions:

1. **Run `memory_search` FIRST.** Every time. No exceptions.
2. **Do NOT guess.** Do not say "I don't know." Do not ask the human for info you could search.
3. **Do NOT trust vibes.** If you "feel like" you know something, verify it with a search.
4. **The search takes 500ms.** There is no excuse to skip it.

This rule exists because skipping memory search leads to wrong answers, broken trust, and wasted time.
RULES_EOF
        log "  HMS-RULES.md installed to workspace ✓"
    else
        log "  HMS-RULES.md already exists in workspace, skipping"
    fi
fi

# ---- Done ----
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  HMS v2.2 installed successfully!         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
log "Server:  http://localhost:$PORT"
log "Health:  curl http://localhost:$PORT/health"
log "Search:  curl -X POST http://localhost:$PORT/search -H 'Content-Type: application/json' -d '{\"query\": \"test\"}'"
log "Stats:   curl http://localhost:$PORT/stats"
log "Logs:    sudo journalctl -u hms -f"
log "Restart: sudo systemctl restart hms"
echo ""
