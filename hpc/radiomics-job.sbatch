#!/bin/bash
#SBATCH --job-name=radiomics
#SBATCH --partition=p_nsm_compute
#SBATCH --output=radiomics-logs/rad-%a.out
#SBATCH --qos=q_nsm_student_w
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

mkdir -p ~/radiomics-logs

# --- CRITICAL: Activate your Python Environment ---
source ~/radiomics_env/bin/activate

# --- Select Subject ---
# We use the master list of 1231 subjects you already have
SUBJECT_ID=$(ls -1 ~/freesurfer_subjects | grep -v 'fsaverage' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -n "$SUBJECT_ID" ]; then
    python3 ~/run_radiomics.py "$SUBJECT_ID"
else
    echo "No subject found for Task ID $SLURM_ARRAY_TASK_ID"
fi

