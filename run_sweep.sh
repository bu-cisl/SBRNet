#!/bin/bash

# Initialize a new sweep, and extract the agent command
agent_command=$(wandb sweep --project rjbona_sbrnet --entity cisl-bu configs_deprecated/sweep_config.yaml 2>&1 | grep 'wandb agent' | awk -F': ' '{print $3}')

# Append '--count 1' in the correct position within the agent command
agent_command_modified="${agent_command% *} --count 1 ${agent_command##* }"

# Define a unique job name
timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds
random_suffix=$(($RANDOM % 1000))  # Generates a random number between 0 and 999
jobname="sweep_sbrnet_${timestamp}_${random_suffix}"

# Create a new .qsub file with dynamic agent command and job details
cat <<EOF > "/projectnb/tianlabdl/rjbona/sbrnet/config_qsubs/sweep.qsub"
#!/bin/bash -l

# Set SCC project
#$ -P tianlabdl

# Specify hard time limit for the job.
#$ -l h_rt=12:00:00

# Combine output and error files into a single file
#$ -j y

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : \$(date)"
echo "Job name : \$JOB_NAME"
echo "Job ID : \$JOB_ID  \$SGE_TASK_ID"
echo "=========================================================="

# Specify number of cores 16: a whole node with at least 128gb ram
#$ -pe omp 2
#$ -t 1-18

#Specify the number of GPUs (1 is recommended!)
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l gpu_memory=16G

module load miniconda
source /projectnb/tianlabdl/rjbona/sbrnet/.direnv/bin/activate
$agent_command_modified
EOF

# Submit the job
qsub -N "${jobname}" -o "/projectnb/tianlabdl/rjbona/sbrnet/.log/qsubs/${jobname}.qlog" "/projectnb/tianlabdl/rjbona/sbrnet/config_qsubs/sweep.qsub"