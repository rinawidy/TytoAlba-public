#!/bin/bash

# Simple cron setup - runs every 15 minutes, wrapper checks time window
# Time window: 10:30 - 16:00

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WRAPPER_SCRIPT="$SCRIPT_DIR/cron_wrapper.sh"
LOG_DIR="/tmp/tytoalba/logs"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "PLN Data Fetch - Cron Setup (Simple)"
echo "=========================================="
echo ""
echo "Schedule: Every 15 minutes (wrapper checks 10:30-16:00)"
echo "Script: $WRAPPER_SCRIPT"
echo "Log file: $LOG_DIR/pln-fetch.log"
echo ""

# Remove existing cron entries for PLN fetch
crontab -l 2>/dev/null | grep -v "fetch_pln_data\|cron_wrapper" | crontab -

# Add new cron entry (every 15 minutes)
CRON_CMD="$WRAPPER_SCRIPT >> $LOG_DIR/pln-fetch.log 2>&1"

(crontab -l 2>/dev/null; echo "# PLN Data Fetch - Every 15 minutes (10:30-16:00 checked by wrapper)") | crontab -
(crontab -l 2>/dev/null; echo "*/15 * * * * $CRON_CMD") | crontab -

echo "✅ Cron job installed successfully!"
echo ""
echo "Current crontab:"
echo "----------------------------------------"
crontab -l | grep -A 1 "PLN Data Fetch"
echo "----------------------------------------"
echo ""
echo "📋 Next steps:"
echo "  1. Cron will run every 15 minutes"
echo "  2. Wrapper checks if time is 10:30-16:00"
echo "  3. Only fetches data within time window"
echo ""
echo "📝 Useful commands:"
echo "  View logs:     tail -f $LOG_DIR/pln-fetch.log"
echo "  Test manually: $WRAPPER_SCRIPT"
echo "  Remove cron:   crontab -e (delete PLN lines)"
echo "  List cron:     crontab -l"
echo ""
