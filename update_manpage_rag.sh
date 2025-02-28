#!/bin/bash

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create data directory if it doesn't exist
mkdir -p data

# Log file for the update process
LOG_FILE="data/update_$(date +%Y%m%d_%H%M%S).log"

echo "Starting man page RAG update at $(date)" | tee -a "$LOG_FILE"

# Check for new man pages
echo "Checking for new man pages..." | tee -a "$LOG_FILE"
python extract_manpages.py 2>&1 | tee -a "$LOG_FILE"

# Process new man pages if found
if [ -f "data/last_update.txt" ]; then
    LAST_UPDATE=$(cat data/last_update.txt)
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_UPDATE))

    # If last update was less than an hour ago, process new pages
    if [ $TIME_DIFF -lt 3600 ]; then
        echo "Processing new man pages..." | tee -a "$LOG_FILE"
        python process_and_load.py 2>&1 | tee -a "$LOG_FILE"
    else
        echo "No new man pages found in this update." | tee -a "$LOG_FILE"
    fi
else
    echo "No update timestamp found. Skipping processing." | tee -a "$LOG_FILE"
fi

echo "Update process complete at $(date)" | tee -a "$LOG_FILE"
