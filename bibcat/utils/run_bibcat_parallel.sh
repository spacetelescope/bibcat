#!/bin/bash

### THIS script didn't work the way I wanted because of the openai rate limit error. We need to change our codebase to accomodate this, but Brian Cherinka will implement this so I should just wait. ####
SECONDS=0 # start timer

# Directory for logs
LOG_DIR="/home/jyoon/bibcat/logs"
mkdir -p "$LOG_DIR" # check if  log directory exists

# Error log for failed jobs
ERROR_LOG="$LOG_DIR/$(date '+%F_%H-%M-%S')_failed_parallel_jobs.txt"
> "$ERROR_LOG"  # Clear file at start

# maximum number of processing
MAX_JOBS=8

# function to wait if too many background jobs are running
wait_for_jobs() {
    while (( $(jobs -r | wc -l) >= MAX_JOBS)); do
        sleep 1
    done
}

# Classification wrapper with failure logging

run_bibcat_job() {

    # format the command with current index
    local i=$1
    local i_file="/home/jyoon/bibcat_project/dataset/historic_dataset/batch_bibcodes_bibcode_2018_2023_${i}.txt"
    local o_file="historic_2018-2023_llm_output_${i}.json"

    echo "$(date '+%F %T') - Starting Bibcat on $i_file"

    if ! bibcat run-gpt-batch -p "$i_file" -c "$o_file"; then
        echo "$(date '+%F %T') - Job failed for historic_2018-2023_llm_output_${i}.json" >> "$ERROR_LOG"
    fi
}

# Loop through the batch files (5 through 43)

for i in $(seq 5 43); do
    # control parallel job count
    wait_for_jobs
    run_bibcat_job "$i" &
done

wait

# Logging time spent for all jobs
MINUTES=$((SECONDS / 60))
SECONDS_REMAIN=$((SECONDS % 60))

echo "All jobs complete. Any failures are recorded in $ERROR_LOG."
echo "Total time spent: ${MINUTES} min ${SECONDS_REMAIN} sec."
