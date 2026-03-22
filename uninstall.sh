#!/bin/bash
# HMS Uninstaller
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="${1:-$HOME/hms}"

echo -e "${YELLOW}Uninstalling HMS from $INSTALL_DIR${NC}"
echo ""
read -p "Are you sure? This will remove all HMS data including the index. (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Stop and disable service
if systemctl is-active --quiet hms 2>/dev/null; then
    echo -e "${GREEN}[HMS]${NC} Stopping service..."
    sudo systemctl stop hms
fi

if systemctl is-enabled --quiet hms 2>/dev/null; then
    echo -e "${GREEN}[HMS]${NC} Disabling service..."
    sudo systemctl disable hms
fi

if [[ -f /etc/systemd/system/hms.service ]]; then
    echo -e "${GREEN}[HMS]${NC} Removing service file..."
    sudo rm /etc/systemd/system/hms.service
    sudo systemctl daemon-reload
fi

# Remove install directory
if [[ -d "$INSTALL_DIR" ]]; then
    echo -e "${GREEN}[HMS]${NC} Removing $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"
fi

echo ""
echo -e "${GREEN}HMS uninstalled.${NC}"
echo -e "${YELLOW}Note: The OpenClaw plugin config entry (if any) must be removed manually from your OpenClaw config.${NC}"
