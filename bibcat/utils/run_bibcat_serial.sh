#!/bin/bash

# Each bibcat job sleeps 2 hours after completion to be safe and not to run into RateLimitError because one job of 1000 API calls takes ~4500 seconds. To run this script, update `LOG_DIR` and `i_file` paths according to your directory paths and the batch filenames. In the terminal, run `./run_bibcat_serial.sh`.

# Initialize total compute time
TOTAL_COMPUTE_SECONDS=0
SCRIPT_START=$(date +%s)

# List of input files
START_INDEX=0
END_INDEX=43


# Directory for logs
LOG_DIR="/your/log/dir/path"
mkdir -p "$LOG_DIR" # check if  log directory exists

# Error log for failed jobs
LOG="$LOG_DIR/$(date '+%F_%H-%M-%S')_serial_job_log.txt"
> "$LOG"  # Clear file at start

# function to run bibcat one at a time and wait 2 hrs before the next run.
for i in $(seq $START_INDEX $END_INDEX); do # For sequence from start_index to end_index
# for i in 9 10 18 25 26 29 37; do # For a list of specific batch files
    i_file="/dataset/to/batch/files/batch_${i}.txt"

    if [[ ! -f "$i_file" ]]; then
    echo "[$(date '+%F %T')] WARNING: File not found: $i_file — skipping." | tee -a "$LOG"
    continue
    fi

    echo "$(date '+%F %T') - Starting bibcat on $i_file" | tee -a "$LOG"

    JOB_START=$(date +%s)

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

    echo "Sleeping for 2 hours before the next run" | tee -a "$LOG"
    sleep 2h
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
