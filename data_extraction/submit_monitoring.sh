# This script submits the monitoring job to BC4. It is the only script ran by the user.

#!/bin/bash

#SBATCH --job-name=ECMWF
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:0 
#SBATCH --mem-per-cpu=100M
#SBATCH --account=PHYS030544
#SBATCH --mem=490000M

# Change to working directory, where job was submitted from
cd "${SLURM_SUBMIT_DIR}"

# Submitting and timing code runs
# Recording start time
start_time=$(date +%s)

# File run
year = $1
./monitoring_jobs.sh $year

# End recording the end time
end_time=$(date +%s)

# Calculate and print the runtime
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"