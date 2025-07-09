#!/bin/bash

# run bibcat serially every 2 hours to be safe and not to run into RateLimitError because one job of 1000 API calls takes ~4500 seconds.

# Initialize total compute time
TOTAL_COMPUTE_SECONDS=0

# List of input files
START_INDEX=0
END_INDEX=43


# Directory for logs
LOG_DIR="/home/jyoon/bibcat/logs"
mkdir -p "$LOG_DIR" # check if  log directory exists

# Error log for failed jobs
LOG="$LOG_DIR/$(date '+%F_%H-%M-%S')_serial_job_log.txt"
> "$LOG"  # Clear file at start

# function to run bibcat one at a time and wait 2 hrs before the next run.
for i in $(seq $START_INDEX $END_INDEX); do
    i_file="/home/jyoon/bibcat_project/dataset/historic_dataset/batch_bibcodes_bibcode_2018_2023_${i}.txt"

    echo "$(date '+%F %T') - Starting bibcat on $i_file" | tee -a "$LOG"

    JOB_START=$(date +%s)

    if bibcat run-gpt-batch -p "$i_file"; then
        JOB_END=$(date +%s)
        JOB_DURATION=$((JOB_END - JOB_START))
        TOTAL_COMPUTE_SECONDS=$((TOTAL_COMPUTE_SECONDS + JOB_DURATION))

        # recording the time for current job
        MINUTES=$((JOB_DURATION / 60))
        SECONDS_REMAIN=$((JOB_DURATION % 60))
        echo "[$(date '+%F %T')] Success running: $i_file — Duration: ${MINUTES}m ${SECONDS_REMAIN}s" | tee -a "$LOG"

    else
        echo "[$(date '+%F %T')] ERROR: Failed: $i_file" | tee -a "$LOG"
    fi

    echo "Sleeping for 2 hours before the next run" | tee -a "$LOG"
    sleep 2h
done

# Logging time spent for all jobs
T_MINUTES=$((TOTAL_COMPUTE_SECONDS / 60))
T_SECONDS_REMAIN=$((TOTAL_COMPUTE_SECONDS % 60))

echo "[$(date '+%F %T')] All jobs complete." | tee -a "$LOG"
echo "Total compute time (excluding sleeps): ${T_MINUTES} min ${T_SECONDS_REMAIN} sec" | tee -a "$LOG"
