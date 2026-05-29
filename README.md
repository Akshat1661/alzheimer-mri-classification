# Alzheimer's Disease Classification via Hippocampal & Amygdala Subfield Radiomics

A multi-stage research pipeline for classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects using T1-weighted MRI from the ADNI cohort. The pipeline integrates FreeSurfer subfield segmentation, high-throughput PyRadiomics feature extraction, a novel "Subject Squashing" data engineering algorithm, and a two-stage LASSO + XGBoost classifier.

---

## Results

| Experiment | Feature Set | Cohort (n) | Accuracy |
|---|---|---|---|
| A — Clinical Baseline | Age, Sex, APOE4, MMSE, CDRSB, ADAS11, ADAS13, FAQ, MOCA (d=9) | 275 (validated subset) | **85.45%** |
| B — Radiomics Only | 77 LASSO-selected subfield texture & shape features | 1,229 (full cohort) | **73.17%** |
| C — Combined Multimodal | Clinical (d=9) + Radiomics (d=77) → d=86 | 275 (validated subset) | **92.73%** |

- Radiomics-only model outperforms the random baseline (33%) by **more than 2×** using only brain micro-structure — no cognitive tests, no demographics.
- Combined ROC-AUC approached **0.95+**, indicating strong robustness across decision thresholds.
- Full confusion matrices, ROC curves, and feature importance plots: [`results/`](results/)

---

## Key Biological Findings

LASSO reduced the feature space from **5,980 → 77 biomarkers** (98.7% noise reduction). The top predictors were:

| Rank | Region | Feature | Importance | Interpretation |
|---|---|---|---|---|
| 1 | Left Basal Amygdala | FirstOrder: Skewness | 0.066 | Asymmetric intensity distribution indicating early tissue degradation — neuronal loss and amyloid deposition create "holes" before gross atrophy is visible |
| 2 | Right Lateral Amygdala | FirstOrder: Maximum | 0.041 | Hyper-intense signal change, potentially reflecting gliosis |
| 3 | Left CA3 Head | Shape: Surface-to-Volume Ratio | 0.037 | Geometric hallmark of atrophic neurodegeneration — subfield "shriveling" |
| 4 | Left HATA | GLCM: Correlation | — | Texture degradation in the hippocampus–amygdala transition zone |

> **Note on feature names:** The LASSO biomarkers CSV (`results/Experiment_B_Radiomics_Only/Significant_LASSO_Biomarkers_List.csv`) uses FreeSurfer numeric label IDs (e.g. `L_Hippo_Label7003_firstorder_Skewness`). Labels in the 7000+ range are amygdala nuclei — they appear under the `L_Hippo_` prefix because the extraction pipeline grouped all hippoAmygLabels outputs together. The anatomical names in the table above follow the FreeSurfer v7.1 probabilistic atlas.

**Critical finding:** Texture features (GLCM, GLRLM) and distribution statistics (Skewness, Kurtosis) consistently outperformed volumetric features. Volume is a *late-stage* marker; texture is an *early-stage* marker — micro-structural heterogeneity is a more sensitive predictor of early AD than macroscopic shrinkage.

---

## Pipeline Overview

```
Raw ADNI DICOM (MPRAGE)
        │
        ▼
[Stage 1]  DICOM → NIfTI  (BIDS format, dcm2niix + nibabel)
           pipeline/01_dicom_to_nifti.py
        │
        ▼
[Stage 2]  FreeSurfer recon-all  (cortical reconstruction, v7.3.2)
           hpc/recon-job.sbatch  ·  8 CPUs, 24 GB RAM, 10 h/subject
        │
        ▼
[Stage 3]  Hippocampal & Amygdala Subfield Segmentation  (segmentHA_T1.sh, v7.1 atlas)
           hpc/segmentHA-job.sbatch  ·  1 CPU, 4 GB RAM, ~30–45 min/subject
           → 56 sub-regions (28 per hemisphere): CA1/CA3/CA4 head+body,
             Subiculum, GC-ML-DG, Presubiculum, HATA, Fimbria, Molecular Layer,
             Lateral/Basal/AAA/Central/Medial/Cortical/Paralaminar Amygdala …
        │
        ▼
[Stage 4]  PyRadiomics Feature Extraction  (v3.0.1, parallelized on HPC)
           hpc/radiomics-job.sbatch  ·  1 CPU, 16 GB RAM
           → 5,980 features/subject  (Shape ×14, FirstOrder ×18, GLCM/GLRLM/GLSZM/NGTDM ×75)
        │
        ▼
[Stage 5]  "Subject Squashing"  (custom data engineering — collapses long-format
           per-subfield rows into one 1×6,010 vector per subject)
           → recovers cohort from 275 fragmented rows → 1,229 unique subjects
        │
        ▼
[Stage 6]  LASSO + XGBoost Classification
           pipeline/02_final_ml_pipeline.py
           → 5,980 → 77 features (LASSO) → 3-class XGBoost
        │
        ▼
results/  (confusion matrices, ROC curves, feature importance, biomarker list)
```

---

## Dataset — ADNI

### What is ADNI?

The **Alzheimer's Disease Neuroimaging Initiative (ADNI)** is a longitudinal multi-site study for developing clinical, imaging, genetic, and biochemical biomarkers for early detection of Alzheimer's disease. Access requires a free registration at [adni.loni.usc.edu](https://adni.loni.usc.edu).

> **Note:** ADNI data is governed by a Data Use Agreement. You must apply for access through the LONI Image & Data Archive (IDA). Raw MRI data and clinical spreadsheets are **not** included in this repository.

### Cohort

Data were aggregated across **ADNI-GO, ADNI-2, and ADNI-3** study phases.

| Group | N | Description |
|---|---|---|
| CN | 672 | Cognitively Normal controls |
| MCI | 282 | Mild Cognitive Impairment (transitional stage) |
| AD | 275 | Alzheimer's Disease dementia |
| **Total** | **1,229** | Large-scale validation cohort |

### Inclusion Criteria

- **Imaging:** High-resolution T1-weighted 3D MPRAGE (sagittal acquisition), 3T scanners preferred (1.5T included, harmonized via FreeSurfer preprocessing)
- **Clinical:** Confirmed baseline diagnosis (CN/MCI/AD) with complete MMSE and CDRSB scores in the ADNIMERGE registry
- **QC exclusions:** Scans with segmentation failures, geometric distortions, or incomplete subfield masks

### What to Download from ADNI

After logging in to the LONI IDA portal:

#### 1. MRI Scans

Navigate to: **Download → Image Collections → Advanced Search**

Filters used in this study:
- Modality: `MRI`
- Image Description: `MPRAGE` (3D T1-weighted, primary acquisition)
- Study: `ADNI-GO`, `ADNI-2`, `ADNI-3`
- Format: `DCM` (DICOM)

Download all results. Subject folders will be structured as `SubjectID/Visit/SeriesDate/`.

#### 2. Clinical Spreadsheet (ADNIMERGE)

Navigate to: **Download → Study Data → Study Info → ADNIMERGE**

Download `ADNIMERGE.csv`. This contains diagnosis labels, cognitive scores (CDRSB, MMSE, ADAS11, ADAS13, MOCA, FAQ), and APOE4 genotype for all subjects and visits.

#### 3. Scan Metadata CSV

From the same Advanced Search results page, after selecting your images, also download the accompanying **metadata CSV** (contains Image Data ID, Subject, Group, Sex, Age, Visit). This is used to link MRI image IDs to diagnosis labels.

---

## Repository Structure

```
.
├── pipeline/
│   ├── 01_dicom_to_nifti.py       # Converts ADNI DICOMs → BIDS NIfTI via dcm2niix
│   └── 02_final_ml_pipeline.py    # Full LASSO + XGBoost pipeline (all 3 experiments)
│
├── supplementary/
│   ├── run_multimodal_ml.py       # Earlier multimodal XGBoost experiment
│   ├── run_radiomics_refinery.py  # Standalone LASSO biomarker discovery
│   └── run_grand_master_v4.py     # Most robust 3-table merge with ID collision handling
│
├── hpc/
│   ├── recon-job.sbatch           # SLURM array: FreeSurfer recon-all (8 CPU, 24 GB, 10 h)
│   ├── segmentHA-job.sbatch       # SLURM array: segmentHA_T1.sh (1 CPU, 4 GB, 2 h)
│   └── radiomics-job.sbatch       # SLURM array: PyRadiomics extraction (1 CPU, 16 GB, 30 min)
│
└── results/
    ├── Fixed_Feature_Importance.png
    ├── Biomarkers_Translated_For_Paper.csv
    ├── analysis_plots_all/                  # 64 statistical boxplots by group (all subfields)
    ├── Experiment_A_Clinical_Only/          # confusion matrix, ROC, feature importance, report
    ├── Experiment_B_Radiomics_Only/         # same + Significant_LASSO_Biomarkers_List.csv
    └── Experiment_C_Combined/               # same
```

---

## Environment Setup

**Python 3.9+ recommended.**

```bash
pip install -r requirements.txt
```

FreeSurfer v7.3.2 must be installed separately for Stages 2–3:

```bash
# https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$HOME/freesurfer_subjects
```

---

## Running the Pipeline

### Stage 1 — DICOM to NIfTI

Requires [dcm2niix](https://github.com/rordenlab/dcm2niix). Update the **four path variables** at the top of the script before running:

```python
DCM2NIIX_PATH    = "/path/to/dcm2niix"
RAW_MPRAGE_DICOM_DIR = "/path/to/ADNI/MPRAGE"
ADNIMERGE_PATH   = "/path/to/AD_mprage_spgr_metadata.csv"
BIDS_OUTPUT_DIR  = "/path/to/nifti_output"   # ← must match what recon-job.sbatch expects
```

```bash
python pipeline/01_dicom_to_nifti.py
```

Outputs files named `sub-{Group}_{SubjectID}_T1w.nii.gz` (e.g. `sub-AD_002_S_1081_T1w.nii.gz`) into `BIDS_OUTPUT_DIR`.

> **Important:** `hpc/recon-job.sbatch` scans `$HOME/nifti_output` for input files. Set `BIDS_OUTPUT_DIR` in the script to `~/nifti_output` (or update the sbatch path) so both steps point to the same folder.

### Stage 2 — FreeSurfer Reconstruction (HPC)

```bash
# N = number of NIfTI files in nifti_output/
sbatch --array=1-N hpc/recon-job.sbatch
```

8 CPUs · 24 GB RAM · 10 h wall time per subject.  
Output: `~/freesurfer_subjects/<SubjectID>/`

### Stage 3 — Hippocampal Subfield Segmentation (HPC)

```bash
# N = number of completed subjects in freesurfer_subjects/
sbatch --array=1-N hpc/segmentHA-job.sbatch
```

Runs FreeSurfer's `segmentHA_T1.sh`, adding `hipposubfields.*.stats` and `amygdalar-nuclei.*.stats` to each subject directory. 1 CPU · 4 GB RAM · ~30–45 min per subject.

### Stage 4 — Radiomics Feature Extraction (HPC)

```bash
sbatch --array=1-N hpc/radiomics-job.sbatch
```

Requires a Python environment (`radiomics_env`) with PyRadiomics on the cluster.  
Output: one CSV per subject (5,980 features) in `radiomics_results/`.

### Stage 5 — ML Classification

```bash
python pipeline/02_final_ml_pipeline.py
```

**Required input files** (not in repo — obtain via ADNI):
- `MASTER_Radiomics_Dataset.csv` — merged radiomics features (1,229 rows × 5,980+ cols)
- `AD_mprage_spgr_9_03_2025.csv` — scan metadata with Group labels
- `ADNIMERGE_03Jun2025.csv` — clinical scores from ADNI

**Outputs** (written to three subfolders):
- `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `report.txt`
- `biomarkers.csv` — the 77 LASSO-selected features (Experiment B)

---

## Limitations

- **Cross-sectional only:** Single baseline time-point; cannot predict MCI-to-AD conversion without longitudinal follow-up.
- **Scanner heterogeneity:** Multi-site ADNI data (Siemens/GE/Philips, 1.5T/3T) may introduce texture variability despite FreeSurfer normalization.
- **MCI sensitivity:** Recall ~0.21 for MCI reflects the biological reality that MCI is a heterogeneous transition state, not a distinct pathological entity.
- **No external validation:** Model was trained and tested within ADNI; validation on independent cohorts (AIBL, OASIS) is needed.

---

## Citation / Acknowledgements

Data used in the preparation of this work were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database ([adni.loni.usc.edu](https://adni.loni.usc.edu)). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD.

**References:**
1. Mueller et al. (2007). Measurement of hippocampal subfields with high resolution MRI at 4T. *Neurobiology of Aging, 28*(5), 719–726.
2. Gillies et al. (2016). Radiomics: images are more than pictures, they are data. *Radiology, 278*(2), 563–577.
3. Iglesias et al. (2015). A computational atlas of the hippocampal formation using ex vivo ultra-high resolution MRI. *NeuroImage, 115*, 117–137.
4. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B, 58*(1), 267–288.
5. Chen, T. (2016). XGBoost: A Scalable Tree Boosting System. *arXiv:1603.02754.*
6. Weiner et al. (2010). The Alzheimer's disease neuroimaging initiative: progress report and future plans. *Alzheimer's & Dementia, 6*(3), 202–211.

---

## License

Code in this repository is released under the MIT License.  
ADNI data is subject to the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/adni-data/#AccessData) and may not be redistributed.
