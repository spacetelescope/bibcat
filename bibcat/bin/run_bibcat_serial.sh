#!/bin/bash

# Each bibcat job sleeps 5 mins after completion to be safe and not to run into RateLimitError because one job of 1000 API calls takes ~4500 seconds. To run this script on the terminal, run `chmod +x run_bibcat_serial.sh` and `./run_bibcat_serial.sh /path/to/batch_dir /path/to/logs`. If you want to dry-run, `./run_bibcat_serial.sh /path/to/batch_dir /path/to/logs --dry-run`.

# Initialize total compute time
TOTAL_COMPUTE_SECONDS=0
SCRIPT_START=$(date +%s)
JOB_END=$SCRIPT_START


# Validate Batch Directory
BATCH_DIR="$1"
if [[ -z "$BATCH_DIR" || "$BATCH_DIR" == --* || ! -d "$BATCH_DIR" ]]; then
    echo "Usage: $0 /path/to/batch_dir [/path/to/log_dir] [--dry-run]" >&2
    exit 1
fi
# Find batch files
mapfile -t BATCH_FILES < <(find "$BATCH_DIR" -type f -name "*.txt" | sort)

# Check if any .txt files are found
if [[ ${#BATCH_FILES[@]} -eq 0 ]]; then
    echo "No batch files found in $BATCH_DIR" | tee -a "$LOG"
    exit 0
fi


# Directory for logs
LOG_DIR="$2"

# Validate Log Directory
if [[ -z "$LOG_DIR" || ""$LOG_DIR == --* ]]; then
    echo "Usage: $0 /path/to/batch_dir [/path/to/log_dir] [--dry-run]" >&2
    exit 1
fi

# Create Log directory
mkdir -p "$LOG_DIR" || {
    echo "ERROR: Failed to create or access log directory: $LOG_DIR" >&2
    exit 1
}

# Setting Log file and Logging start
LOG="$LOG_DIR/$(date '+%F_%H-%M-%S')_serial_job_log.txt"
> "$LOG"

DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

# Run bibcat one at a time and wait 5 mins before the next run for batch files.
for idx in "${!BATCH_FILES[@]}"; do # for all batch files found in BATCH_DIR
    i_file="${BATCH_FILES[$idx]}"

    echo "$(date '+%F %T') - Starting bibcat on $i_file" | tee -a "$LOG"

    JOB_START=$(date +%s)

    if $DRY_RUN; then
        echo "[$(date '+%F %T')]" DRY-RUN: bibcat llm batch run -p $i_file | tee -a "$LOG"
        JOB_END=$(date +%s)
        JOB_DURATION=$((JOB_END - JOB_START))
        TOTAL_COMPUTE_SECONDS=$((TOTAL_COMPUTE_SECONDS + JOB_DURATION))
    else
        # run each file
        if bibcat llm batch run -p "$i_file"; then
            JOB_END=$(date +%s)
            JOB_DURATION=$((JOB_END - JOB_START))
            TOTAL_COMPUTE_SECONDS=$((TOTAL_COMPUTE_SECONDS + JOB_DURATION))

            # recording the time for current job
            MINUTES=$((JOB_DURATION / 60))
            SECONDS_REMAIN=$((JOB_DURATION % 60))
            echo "[$(date '+%F %T')] Success running: $i_file — Duration: ${MINUTES}m ${SECONDS_REMAIN}s" | tee -a "$LOG"

        else
            echo "[$(date '+%F %T')] ERROR: Failed: $i_file (exit code $?)" | tee -a "$LOG"
        fi
    fi
    # Only sleep if this is not the last file
    if [ "$idx" -lt "$(( ${#BATCH_FILES[@]} - 1 ))" ]; then
        if $DRY_RUN; then
            echo "DRY_RUN: would sleep for 5 mins" | tee -a "$LOG"
        else
            echo "Sleeping for 5 mins before the next run" | tee -a "$LOG"
            sleep 5m
        fi
    fi
done

# Logging time spent for all batch jobs
T_MINUTES=$((TOTAL_COMPUTE_SECONDS / 60))
T_SECONDS_REMAIN=$((TOTAL_COMPUTE_SECONDS % 60))

# Logging time spent for the whole operation time
TOTAL_OPERATION_SECONDS=$((JOB_END - SCRIPT_START))
T_OPS_MINUTES=$((TOTAL_OPERATION_SECONDS / 60))
T_OPS_SECONDS_REMAIN=$((TOTAL_OPERATION_SECONDS % 60))

echo "[$(date '+%F %T')] All jobs complete." | tee -a "$LOG"
echo "Total compute time (excluding sleeps): ${T_MINUTES} min ${T_SECONDS_REMAIN} sec" | tee -a "$LOG"
echo "Total compute time (including sleeps): ${T_OPS_MINUTES} min ${T_OPS_SECONDS_REMAIN} sec" | tee -a "$LOG"
