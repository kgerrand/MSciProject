# This script is used to monitor the jobs submitted to BC4. 
# It checks the status of the job every 30 minutes and resubmits the job if the job is not completed within 6 hours i.e. the data collection is incomplete.


#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <YEAR>"
    exit 1
fi

year=$1
resubmission_count=0

while [ $resubmission_count -lt 5 ]; do
  # submit job
  JOB_ID=$(sbatch submit.sh $year| awk '{print $4}')

  # waiting for the job to appear in the job queue
  while [ -z "$(squeue -h -j $JOB_ID)" ]; do
    sleep 1
  done

  # monitor job
  while true; do
      JOB_STATUS=$(squeue -h -j $JOB_ID -o "%T")
      
      # check if job is running or pending
      if [ "$JOB_STATUS" == "RUNNING" ]; then
          # leave for 30 minutes if running
          sleep 1800
      elif [ "$JOB_STATUS" == "PENDING" ]; then
          # job is pending, so leave for 5 minutes
          sleep 300
      else
          # job is not running or pending, break out of the monitoring loop
          break
      fi
  done

  # check if the job completed successfully or encountered an error
  JOB_EXIT_CODE=$(sacct -n -o ExitCode -j $JOB_ID | tr -dc '0-9')

  # Check if JOB_EXIT_CODE is not empty and is equal to 0
  if [ -n "$JOB_EXIT_CODE" ] && [ "$JOB_EXIT_CODE" -eq 0 ]; then
      # job completed successfully, so finish as all data collected
      break
  fi 

  CURRENT_TIME=$(squeue -h -j $JOB_ID -o "%L")
  TIME_LIMIT="6:00:00"

  if [[ "$CURRENT_TIME" > "$TIME_LIMIT" ]]; then
    # time limit exceeded so resubmit the job
    ((resubmission_count++))
    echo "Data collection incomplete. Resubmitting job. Attempt $resubmission_count"
  fi
done

