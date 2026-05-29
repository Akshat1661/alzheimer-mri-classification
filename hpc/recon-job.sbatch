#!/bin/bash
#SBATCH --job-name=recon-all
#SBATCH --partition=p_nsm_compute
#SBATCH --output=recon-logs/recon-%a.out
#SBATCH --qos=q_nsm_student_w
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00

mkdir -p ~/recon-logs

export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$HOME/freesurfer_subjects
mkdir -p $SUBJECTS_DIR

INPUT_FILE=$(find $HOME/nifti_output -name "*.nii.gz" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
SUBJECT_ID=$(basename "$INPUT_FILE" .nii.gz)

echo "--- Processing Subject: $SUBJECT_ID from file: $INPUT_FILE ---"
recon-all -s $SUBJECT_ID -i $INPUT_FILE -all -openmp $SLURM_CPUS_PER_TASK
echo "--- Finished Subject: $SUBJECT_ID ---"
