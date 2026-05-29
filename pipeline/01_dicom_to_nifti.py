import os
import subprocess
import pandas as pd
import shutil

# ==============================================================================
#                 ADNI DICOM to BIDS NIfTI Conversion Script
# ==============================================================================
# This script prepares your raw ADNI MPRAGE DICOM data for the FreeSurfer
# pipeline. It performs two critical steps:
#   1. Converts each subject's DICOM series to a single NIfTI file using dcm2niix.
#   2. Renames the output NIfTI file into BIDS format, embedding the subject's
#      diagnosis (AD, MCI, CN) into the filename.
# ==============================================================================

# --- 1. USER CONFIGURATION: UPDATE THESE PATHS ---

# Path to the dcm2niix.exe application you downloaded.
# IMPORTANT: Use forward slashes '/' or double backslashes '\\' in paths.
DCM2NIIX_PATH = r"C:/Tools/dcm2niix.exe"

# Path to the top-level folder containing all your raw MPRAGE subject folders.
# This is the folder that contains sub-folders like '002_S_10814', '002_S_4262', etc.
RAW_MPRAGE_DICOM_DIR = r"C:/D drive/alz_final_zip/newapproach/AD_mprage_spgr/MPRAGE"

# Path to your AD_mprage_spgr_9_03_2025.csv file. This is needed to get the diagnosis for each subject.
ADNIMERGE_PATH = r"C:/D drive/alz_final_zip/newapproach/AD_mprage_spgr_9_03_2025.csv"

# Path to the new, clean directory where the final NIfTI files will be saved.
# This script will create this folder for you.
BIDS_OUTPUT_DIR = r"C:/D drive/alz_final_zip/newapproach/MPRAGE_BIDS_for_FreeSurfer"


# --- 2. HELPER FUNCTION TO GET DIAGNOSIS ---

def get_diagnosis_map(adnimerge_path):
    """
    Reads the AD_mprage_spgr_9_03_2025.csv file and creates a dictionary mapping
    Subject ID (e.g., "002_S_1081") to their baseline diagnosis (AD, MCI, CN).
    """
    print("Reading AD_mprage_spgr_9_03_2025.csv to get subject diagnoses...")
    try:
        # Specify dtype for columns that might have mixed types to avoid warnings
        dtype_spec = {'PTGENDER': str, 'PTEDUCAT': str, 'PTETHCAT': str, 'PTRACCAT': str, 'PTMARRY': str, 'APOE4': str}
        df = pd.read_csv(adnimerge_path, low_memory=False, dtype=dtype_spec)
        
        # We need the diagnosis at the time of the scan (baseline)
        # Use 'DX.bl' for ADNI1 and 'DX' for later phases with VISCODE 'bl'
        baseline_df = df[df['VISCODE'] == 'bl'].copy()
        
        # Simplify the diagnosis column (grouping EMCI and LMCI into MCI)
        baseline_df['Group'] = baseline_df['DX'].replace({'EMCI': 'MCI', 'LMCI': 'MCI', 'Dementia': 'AD', 'CN': 'CN'})
        
        # Keep only the groups we need
        baseline_df = baseline_df[baseline_df['Group'].isin(['AD', 'MCI', 'CN'])]

        # Create the dictionary {SubjectID: Diagnosis}
        diagnosis_map = pd.Series(baseline_df.Group.values, index=baseline_df.PTID).to_dict()
        
        print(f"Successfully loaded baseline diagnoses for {len(diagnosis_map)} subjects.")
        return diagnosis_map
    except FileNotFoundError:
        print(f"ERROR: AD_mprage_spgr_9_03_2025.csv not found at {adnimerge_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

# --- 3. MAIN CONVERSION LOGIC ---

def main():
    # Load the diagnosis data first
    diagnosis_map = get_diagnosis_map(ADNIMERGE_PATH)
    if diagnosis_map is None:
        print("Halting execution due to error reading diagnosis file.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(BIDS_OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {BIDS_OUTPUT_DIR}")

    # Get a list of all subject folders in the raw data directory
    subject_folders = [f for f in os.listdir(RAW_MPRAGE_DICOM_DIR) if os.path.isdir(os.path.join(RAW_MPRAGE_DICOM_DIR, f))]
    
    total_subjects = len(subject_folders)
    converted_count = 0

    print(f"\nFound {total_subjects} subject folders. Starting conversion...")

    for i, subject_id in enumerate(subject_folders):
        print(f"\n--- Processing Subject {i+1}/{total_subjects}: {subject_id} ---")
        
        # Get the diagnosis for this subject
        diagnosis = diagnosis_map.get(subject_id)
        if not diagnosis:
            print(f"Warning: No baseline diagnosis found for {subject_id}. Skipping.")
            continue

        # Define the BIDS-compliant subject name (e.g., sub-AD002S1081)
        bids_subject_name = f"sub-{diagnosis}_{subject_id}"
        
        # Define the final output filename (e.g., sub-AD_002_S_1081_T1w)
        # dcm2niix will add the .nii.gz extension automatically
        output_filename = f"{bids_subject_name}_T1w"
        
        # Get the full path to the subject's raw DICOM folder
        input_dicom_path = os.path.join(RAW_MPRAGE_DICOM_DIR, subject_id)

        # Construct the full dcm2niix command
        command = [
            DCM2NIIX_PATH,
            "-f", output_filename,    # Set the output filename format
            "-o", BIDS_OUTPUT_DIR,    # Set the output directory
            "-z", "y",                # Enable gzip compression (.nii.gz)
            "-b", "y",                # Create BIDS sidecar .json file
            input_dicom_path          # The input folder with DICOM files
        ]

        try:
            print(f"Running command: {' '.join(command)}")
            # Run the command
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully converted {subject_id} to {output_filename}.nii.gz")
            converted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"ERROR: dcm2niix failed for subject {subject_id}.")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"ERROR: dcm2niix not found at '{DCM2NIIX_PATH}'. Please check the path.")
            break # Stop the script if the tool can't be found
        except Exception as e:
            print(f"An unexpected error occurred for subject {subject_id}: {e}")

    print("\n==================================================================")
    print("Conversion process finished.")
    print(f"Successfully converted {converted_count} out of {total_subjects} subjects.")
    print(f"Your BIDS-formatted NIfTI files are ready in: {BIDS_OUTPUT_DIR}")
    print("==================================================================")


if __name__ == "__main__":
    main()

