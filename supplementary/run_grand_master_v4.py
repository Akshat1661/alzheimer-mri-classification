import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# --- FILES ---
FILE_RADIOMICS = 'MASTER_Radiomics_Dataset.csv'
FILE_SCANS = 'AD_mprage_spgr_9_03_2025.csv'
FILE_CLINICAL = 'ADNIMERGE_03Jun2025.csv'

def clean_id(val):
    return str(val).replace('I', '').strip()

print(f"--- 1. LOADING DATA ---")
df_rad = pd.read_csv(FILE_RADIOMICS, low_memory=False)
df_scan = pd.read_csv(FILE_SCANS, low_memory=False)
df_clin = pd.read_csv(FILE_CLINICAL, low_memory=False)

# Fix Duplicate Columns in Clinical File
df_clin = df_clin.loc[:, ~df_clin.columns.duplicated()]

# --- 2. SQUASHING LOGIC ---
print("--- Applying 'Squashing' Logic ---")

rad_id_col = 'Image Data ID'
if rad_id_col not in df_rad.columns:
    candidates = [c for c in df_rad.columns if 'Image Data ID' in c]
    if candidates: rad_id_col = candidates[0]

if 'Subject' not in df_rad.columns and 'Subject_x' in df_rad.columns:
    df_rad.rename(columns={'Subject_x': 'Subject'}, inplace=True)

if df_rad['Subject'].duplicated().any():
    print(f"Detected {len(df_rad)} rows. Collapsing duplicates...")
    df_rad = df_rad.groupby('Subject').first().reset_index()
    print(f"Collapsed to {len(df_rad)} unique subjects.")

# --- 3. MERGING ---
print("--- Merging Data ---")

df_rad['Join_Key'] = df_rad[rad_id_col].apply(clean_id)
df_scan['Join_Key'] = df_scan['Image Data ID'].apply(clean_id)
df_scan['Subject'] = df_scan['Subject'].astype(str).str.strip()

# MERGE 1: Radiomics + Scan List
df_mri_full = pd.merge(df_rad, df_scan[['Join_Key', 'Group', 'Subject', 'Age', 'Sex']], on='Join_Key', how='inner')

# --- FIX: COLUMN RESCUE (Group & Subject) ---
# Fix Group
if 'Group' not in df_mri_full.columns:
    if 'Group_x' in df_mri_full.columns: df_mri_full.rename(columns={'Group_x': 'Group'}, inplace=True)
    elif 'Group_y' in df_mri_full.columns: df_mri_full.rename(columns={'Group_y': 'Group'}, inplace=True)

# Fix Subject (The crash fix!)
if 'Subject' not in df_mri_full.columns:
    if 'Subject_x' in df_mri_full.columns: df_mri_full.rename(columns={'Subject_x': 'Subject'}, inplace=True)
    elif 'Subject_y' in df_mri_full.columns: df_mri_full.rename(columns={'Subject_y': 'Subject'}, inplace=True)

print(f"Full MRI Cohort: {len(df_mri_full)} subjects")

# PREPARE CLINICAL
score_cols = ['CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'FAQ', 'MOCA', 'APOE4']
score_cols = [c for c in score_cols if c in df_clin.columns]

cols_to_fetch = ['PTID'] + score_cols
df_clin_grouped = df_clin[cols_to_fetch].groupby('PTID').first().reset_index()

# MERGE 2: Gold Subset
df_subset = pd.merge(df_mri_full, df_clin_grouped, left_on='Subject', right_on='PTID', how='inner')
print(f"Gold Subset: {len(df_subset)} subjects")

# --- 4. EXP A: CLINICAL (Subset) ---
print(f"\nEXP A: CLINICAL ONLY (n={len(df_subset)})")
df_subset = df_subset.dropna(subset=['Group'])
y_sub = LabelEncoder().fit_transform(df_subset['Group'])

for c in score_cols:
    if c in df_subset.columns:
        df_subset[c] = pd.to_numeric(df_subset[c], errors='coerce')
        df_subset[c] = df_subset[c].fillna(df_subset[c].median())

clinical_feats = [c for c in score_cols if c in df_subset.columns]
if 'Age' in df_subset.columns: clinical_feats.append('Age')
if 'Sex' in df_subset.columns:
    le_sex = LabelEncoder()
    df_subset['Sex'] = df_subset['Sex'].fillna('Unknown').astype(str)
    df_subset['Sex_Code'] = le_sex.fit_transform(df_subset['Sex'])
    clinical_feats.append('Sex_Code')

X_clin = df_subset[clinical_feats]

X_train, X_test, y_train, y_test = train_test_split(X_clin, y_sub, test_size=0.2, random_state=42, stratify=y_sub)
model_clin = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', use_label_encoder=False)
model_clin.fit(X_train, y_train)
acc_clin = accuracy_score(y_test, model_clin.predict(X_test))
print(f"--> Clinical Accuracy: {acc_clin:.2%}")

# --- 5. EXP B: RADIOMICS (Full Dataset) ---
print(f"\nEXP B: RADIOMICS ONLY (n={len(df_mri_full)})")
df_mri_full = df_mri_full.dropna(subset=['Group'])
y_full = LabelEncoder().fit_transform(df_mri_full['Group'])

metadata = ['Join_Key', 'Group', 'Subject', 'Age', 'Sex', 'Image Data ID', 
            'Subject_x', 'Subject_y', 'PTID', 'Visit', 'Description', 'Acq Date', 'Format', 'Downloaded', 'Type', 'Modality']
X_rad = df_mri_full.drop(columns=metadata, errors='ignore').select_dtypes(include=[np.number])

# CLEANUP: Fill NaNs with 0
X_rad = X_rad.fillna(0)

# Feature Selection (Top 50)
selector = SelectKBest(f_classif, k=50)
X_rad_selected = selector.fit_transform(X_rad, y_full)

X_train, X_test, y_train, y_test = train_test_split(X_rad_selected, y_full, test_size=0.2, random_state=42, stratify=y_full)

model_rad = xgb.XGBClassifier(
    objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False,
    n_estimators=200, max_depth=6, learning_rate=0.05, colsample_bytree=0.3
)
model_rad.fit(X_train, y_train)
acc_rad = accuracy_score(y_test, model_rad.predict(X_test))
print(f"--> Radiomics Accuracy: {acc_rad:.2%}")

# --- 6. FINAL NUMBERS ---
print(f"\n{'='*30}")
print(f"FINAL NUMBERS FOR ADITI")
print(f"{'='*30}")
print(f"Clinical Model (Subset):  {acc_clin:.2%}")
print(f"Radiomics Model (Full):   {acc_rad:.2%}")
print(f"{'='*30}")
