# Alzheimer's Disease Classification via Hippocampal Radiomics and Clinical Biomarkers

A multi-stage research pipeline that classifies Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects using 3D T1-weighted MRI scans from the ADNI cohort. The pipeline combines FreeSurfer hippocampal/amygdala subfield segmentation, PyRadiomics feature extraction, and an XGBoost classifier with LASSO feature selection.

---

## Results

| Experiment | Feature Set | Test Subjects | Accuracy |
|---|---|---|---|
| A | Clinical biomarkers only (CDRSB, MMSE, ADAS, APOE4, etc.) | 55 | **85.45%** |
| B | Radiomics only (LASSO-selected from ~4000 features) | 246 | **73.17%** |
| C | Combined (Clinical + Radiomics) | 55 | **92.73%** |

Full confusion matrices, ROC curves, and feature importance plots are in [`results/`](results/).

---

## Pipeline Overview

```
Raw ADNI DICOM
      │
      ▼
[Stage 1] DICOM → NIfTI (BIDS format)
      pipeline/01_dicom_to_nifti.py  +  dcm2niix
      │
      ▼
[Stage 2] FreeSurfer recon-all (full brain reconstruction)
      hpc/recon-job.sbatch   (~1231 subjects, run on HPC)
      │
      ▼
[Stage 3] Hippocampal & Amygdala Subfield Segmentation
      hpc/segmentHA-job.sbatch   (FreeSurfer segmentHA_T1.sh)
      │
      ▼
[Stage 4] PyRadiomics Feature Extraction
      hpc/radiomics-job.sbatch   (per-subject, parallelized)
      │
      ▼
[Stage 5] ML Classification (LASSO + XGBoost)
      pipeline/02_final_ml_pipeline.py
      │
      ▼
results/  (confusion matrices, ROC curves, feature importance)
```

---

## Dataset — ADNI

### What is ADNI?

The **Alzheimer's Disease Neuroimaging Initiative (ADNI)** is a longitudinal multi-site study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of Alzheimer's disease. Access to data requires a free registration at [adni.loni.usc.edu](https://adni.loni.usc.edu).

> **Note:** ADNI data is governed by a Data Use Agreement. You must apply for access through the LONI Image & Data Archive (IDA) before downloading any files. Raw MRI data and clinical spreadsheets are therefore **not** included in this repository.

### Cohort Used in This Study

| Group | Description | Count (approx.) |
|---|---|---|
| AD | Alzheimer's Disease | ~320 |
| MCI | Mild Cognitive Impairment | ~560 |
| CN | Cognitively Normal | ~350 |
| **Total** | | **~1,229 subjects** |

### What to Download from ADNI

After logging in to the LONI IDA portal, you need **three components**:

#### 1. MRI Scans (two acquisition types)

Navigate to: **Download → Image Collections → Advanced Search**

Search filters used in this study:
- **Modality:** MRI
- **Image Description:** `MPRAGE` (3D T1-weighted) — primary acquisition
- **Image Description:** `SPGR` (Spoiled Gradient Echo T1) — secondary acquisition
- **Study:** ADNI1, ADNI2, ADNI3
- **Format:** DCM (DICOM)

Download all results and organize them by subject ID. You will get folders structured as `SubjectID/Visit/SeriesDate/`.

#### 2. Clinical Spreadsheet (ADNIMERGE)

Navigate to: **Download → Study Data → Study Info → ADNIMERGE**

Download `ADNIMERGE.csv`. This single file contains diagnosis labels (DX), cognitive scores (CDRSB, MMSE, ADAS11, ADAS13, MOCA, FAQ), and APOE4 genotype for all subjects and visits.

#### 3. Scan Metadata CSV

From the same Advanced Search used in step 1, after selecting your images, click **"1-Click Download"** and also download the accompanying **metadata CSV** (contains Image Data ID, Subject, Group, Sex, Age, Visit columns). This file is used to link MRI scans to their diagnosis labels.

---

## Repository Structure

```
.
├── pipeline/
│   ├── 01_dicom_to_nifti.py          # Converts raw ADNI DICOMs to BIDS NIfTI
│   └── 02_final_ml_pipeline.py       # Final LASSO + XGBoost classification (3 experiments)
│
├── supplementary/
│   ├── run_multimodal_ml.py          # Earlier multimodal XGBoost experiment
│   ├── run_radiomics_refinery.py     # Standalone LASSO biomarker discovery script
│   └── run_grand_master_v4.py        # Most complete data-merge pipeline (robust ID handling)
│
├── hpc/
│   ├── recon-job.sbatch              # SLURM: FreeSurfer recon-all array job
│   ├── segmentHA-job.sbatch          # SLURM: Hippocampal subfield segmentation array job
│   └── radiomics-job.sbatch          # SLURM: PyRadiomics feature extraction array job
│
└── results/
    ├── Fixed_Feature_Importance.png
    ├── Biomarkers_Translated_For_Paper.csv
    ├── analysis_plots_all/               # 64 statistical boxplots (all subfields by group)
    ├── Experiment_A_Clinical_Only/       # confusion matrix, ROC, feature importance, report
    ├── Experiment_B_Radiomics_Only/      # same + Significant_LASSO_Biomarkers_List.csv
    └── Experiment_C_Combined/            # same
```

---

## Environment Setup

**Python 3.9+ recommended.**

```bash
pip install -r requirements.txt
```

FreeSurfer (v7.4.1) must be installed separately for Stages 2–3:
```bash
# Download from https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

---

## Running the Pipeline

### Stage 1 — DICOM to NIfTI

Requires [dcm2niix](https://github.com/rordenlab/dcm2niix) installed and the ADNI DICOM folder.

```bash
# Edit paths at the top of the script before running
python pipeline/01_dicom_to_nifti.py
```

The script:
- Reads each subject's DICOM series
- Converts to a single `.nii.gz` file via dcm2niix
- Renames output into BIDS format: `SubjectID_Group.nii.gz`
- Output goes to `nifti_output/`

### Stage 2 — FreeSurfer Reconstruction (HPC)

Run as a SLURM array job — one job per subject:

```bash
# N = total number of NIfTI files in nifti_output/
sbatch --array=1-N hpc/recon-job.sbatch
```

Resources per job: 8 CPUs, 24 GB RAM, 10-hour wall time.  
Output: `~/freesurfer_subjects/<SubjectID>/`

### Stage 3 — Hippocampal Subfield Segmentation (HPC)

Run after recon-all completes for all subjects:

```bash
# N = number of completed subjects in freesurfer_subjects/
sbatch --array=1-N hpc/segmentHA-job.sbatch
```

Resources per job: 1 CPU, 4 GB RAM, 2-hour wall time.  
This runs FreeSurfer's `segmentHA_T1.sh` which adds `hipposubfields.*.stats` and `amygdalar-nuclei.*.stats` files to each subject.

### Stage 4 — Radiomics Feature Extraction (HPC)

```bash
sbatch --array=1-N hpc/radiomics-job.sbatch
```

Resources per job: 1 CPU, 16 GB RAM, 30-minute wall time.  
Requires a Python environment (`radiomics_env`) with PyRadiomics installed on the cluster.  
Output: one CSV per subject in `radiomics_results/`

After all jobs complete, merge into master dataset:
```bash
python pipeline/02_final_ml_pipeline.py  # reads MASTER_Radiomics_Dataset.csv
```

> To regenerate `MASTER_Radiomics_Dataset.csv` from individual per-subject CSVs, use `supplementary/run_radiomics_refinery.py`.

### Stage 5 — ML Classification

```bash
python pipeline/02_final_ml_pipeline.py
```

**Input files required** (not in repo — see Dataset section):
- `MASTER_Radiomics_Dataset.csv` — merged radiomics features (~1229 rows)
- `AD_mprage_spgr_9_03_2025.csv` — scan metadata with Group labels
- `ADNIMERGE_03Jun2025.csv` — clinical scores

**Output** (written to `results_clinical/`, `results_radiomics/`, `results_combined/`):
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`
- `report.txt`
- `biomarkers.csv` (LASSO-selected features)

---

## Key Findings

- **LASSO** selected a compact subset of radiomics features from ~4,000 candidates extracted from hippocampal and amygdala subfield segmentations.
- The most discriminative features came from **left hippocampal subfields** (CA1, subiculum, molecular layer) and **basolateral amygdala** nuclei, consistent with known neurodegeneration patterns in AD.
- Combining radiomics with clinical scores (MMSE, CDRSB, APOE4) pushed accuracy from 85% → **92.7%**, showing that structural imaging and cognitive assessments are complementary.
- MCI classification remains the hardest class across all experiments — a known challenge in the field due to its heterogeneity.

---

## Citation / Acknowledgements

Data used in the preparation of this work were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database ([adni.loni.usc.edu](https://adni.loni.usc.edu)). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD.

---

## License

Code in this repository is released under the MIT License.  
ADNI data is subject to the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData) and may not be redistributed.
