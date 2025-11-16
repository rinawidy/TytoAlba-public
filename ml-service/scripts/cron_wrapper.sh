#!/bin/bash

# Wrapper script for cron - runs data collection pipeline 24/7
# Removed time window restriction - now runs continuously every 15 minutes

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/../venv"
FETCH_SCRIPT="$SCRIPT_DIR/fetch_pln_data.py"
SYNTHESIS_SCRIPT="$SCRIPT_DIR/data_synthesizer.py"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Running data collection pipeline..."
cd "$SCRIPT_DIR/.."

# Step 1: Fetch real data from PLN API
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📡 Fetching PLN API data..."
"$VENV_DIR/bin/python" "$FETCH_SCRIPT"
FETCH_EXIT_CODE=$?

if [ $FETCH_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Data fetch successful"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Data fetch failed (exit code: $FETCH_EXIT_CODE)"
fi

# Step 2: Synthesize missing data (fill gaps)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔬 Synthesizing missing data..."
"$VENV_DIR/bin/python" "$SYNTHESIS_SCRIPT"
SYNTH_EXIT_CODE=$?

if [ $SYNTH_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Data synthesis complete"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Data synthesis failed (exit code: $SYNTH_EXIT_CODE)"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Pipeline complete"
