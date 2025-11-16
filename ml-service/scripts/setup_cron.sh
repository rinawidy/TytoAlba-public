#!/bin/bash
# Setup cron job for automated data collection every 15 minutes
# Runs 24/7 continuously

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH=$(which python3)
COLLECTION_SCRIPT="$SCRIPT_DIR/automated_data_collection.py"
LOG_DIR="$SCRIPT_DIR/../logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Cron job command
CRON_CMD="*/15 * * * * cd $SCRIPT_DIR && $PYTHON_PATH $COLLECTION_SCRIPT >> $LOG_DIR/data_collection.log 2>&1"

echo "========================================="
echo "  PLN Data Collection - Cron Setup"
echo "========================================="
echo ""
echo "Script: $COLLECTION_SCRIPT"
echo "Python: $PYTHON_PATH"
echo "Logs: $LOG_DIR/data_collection.log"
echo ""
echo "Schedule: Every 15 minutes, 24/7 continuous"
echo "Cron expression: */15 * * * *"
echo ""
echo "========================================="
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "automated_data_collection.py"; then
    echo "⚠️  Cron job already exists!"
    echo ""
    echo "Current cron jobs:"
    crontab -l | grep "automated_data_collection.py"
    echo ""
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove old cron job
        crontab -l | grep -v "automated_data_collection.py" | crontab -
        echo "✓ Removed old cron job"
    else
        echo "❌ Cancelled. No changes made."
        exit 0
    fi
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo "✅ Cron job added successfully!"
echo ""
echo "To verify, run: crontab -l"
echo "To remove, run: crontab -e (then delete the line)"
echo ""
echo "Testing the script now..."
echo ""

# Test run
$PYTHON_PATH "$COLLECTION_SCRIPT"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Logs will be written to: $LOG_DIR/data_collection.log"
echo "To monitor logs in real-time: tail -f $LOG_DIR/data_collection.log"
