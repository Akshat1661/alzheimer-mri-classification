#!/bin/bash
#SBATCH --job-name=segmentHA
#SBATCH --partition=p_nsm_compute_short
#SBATCH --output=segmentHA-logs/segmentHA-%a.out
#SBATCH --qos=q_nsm_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

echo "--- Starting Hippocampal Segmentation for Task ID: $SLURM_ARRAY_TASK_ID ---"

# --- Setup ---
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
# IMPORTANT: This points to your existing results folder
export SUBJECTS_DIR=$HOME/freesurfer_subjects
mkdir -p ~/segmentHA-logs

# --- Data Selection ---
# This finds a completed subject folder and gives it to the job
SUBJECT_ID=$(ls $SUBJECTS_DIR | grep -v 'fsaverage' | sed -n "${SLURM_ARRAY_TASK_ID}p")

# --- Execution ---
if [ -n "$SUBJECT_ID" ]; then
    echo "--- Processing Subject: $SUBJECT_ID ---"
    # This is the new command Aditi suggested
    segmentHA_T1.sh "$SUBJECT_ID"
    echo "--- Finished Subject: $SUBJECT_ID ---"
else
    echo "--- No subject found for this Task ID. Exiting. ---"
fi
