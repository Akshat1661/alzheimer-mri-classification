import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from itertools import cycle

# --- CONFIGURATION ---
FILE_RADIOMICS = 'MASTER_Radiomics_Dataset.csv'
FILE_SCANS = 'AD_mprage_spgr_9_03_2025.csv'
FILE_CLINICAL = 'ADNIMERGE_03Jun2025.csv'

def clean_id(val):
    return str(val).replace('I', '').strip()

def save_plots(model, X_test, y_test, classes, folder_name, feature_names=None):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 1. Report
    report = classification_report(y_test, y_pred, target_names=classes)
    with open(f"{folder_name}/report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Acc: {acc:.1%})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{folder_name}/confusion_matrix.png")
    plt.close()
    
    # 3. ROC Curve
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(len(classes)), colors):
        if i < y_test_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{folder_name}/roc_curve.png")
    plt.close()

    # 4. Feature Importance (if applicable)
    if feature_names is not None:
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(10,8))
        plt.barh(feature_names[sorted_idx], imp[sorted_idx])
        plt.gca().invert_yaxis()
        plt.title("Top 20 Features")
        plt.tight_layout()
        plt.savefig(f"{folder_name}/feature_importance.png")
        plt.close()
        
    print(f"Saved results to {folder_name}/ (Acc: {acc:.2%})")
    return acc

# --- LOADING & PREP ---
print("--- Loading & Squashing Data ---")
df_rad = pd.read_csv(FILE_RADIOMICS, low_memory=False)
df_scan = pd.read_csv(FILE_SCANS, low_memory=False)
df_clin = pd.read_csv(FILE_CLINICAL, low_memory=False)
df_clin = df_clin.loc[:, ~df_clin.columns.duplicated()]

rad_id_col = 'Image Data ID'
if rad_id_col not in df_rad.columns:
    candidates = [c for c in df_rad.columns if 'Image Data ID' in c]
    if candidates: rad_id_col = candidates[0]

if 'Subject' not in df_rad.columns and 'Subject_x' in df_rad.columns:
    df_rad.rename(columns={'Subject_x': 'Subject'}, inplace=True)

if df_rad['Subject'].duplicated().any():
    df_rad = df_rad.groupby('Subject').first().reset_index()

df_rad['Join_Key'] = df_rad[rad_id_col].apply(clean_id)
df_scan['Join_Key'] = df_scan['Image Data ID'].apply(clean_id)
df_scan['Subject'] = df_scan['Subject'].astype(str).str.strip()

# Full MRI Merge
df_mri_full = pd.merge(df_rad, df_scan[['Join_Key', 'Group', 'Subject', 'Age', 'Sex']], on='Join_Key', how='inner')
if 'Group' not in df_mri_full.columns:
    if 'Group_x' in df_mri_full.columns: df_mri_full.rename(columns={'Group_x': 'Group'}, inplace=True)
    elif 'Group_y' in df_mri_full.columns: df_mri_full.rename(columns={'Group_y': 'Group'}, inplace=True)
if 'Subject' not in df_mri_full.columns:
    if 'Subject_x' in df_mri_full.columns: df_mri_full.rename(columns={'Subject_x': 'Subject'}, inplace=True)

# Subset (Clinical) Merge
score_cols = ['CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'FAQ', 'MOCA', 'APOE4']
score_cols = [c for c in score_cols if c in df_clin.columns]
cols_to_fetch = ['PTID'] + score_cols
df_clin_grouped = df_clin[cols_to_fetch].groupby('PTID').first().reset_index()
df_subset = pd.merge(df_mri_full, df_clin_grouped, left_on='Subject', right_on='PTID', how='inner')

print(f"Full Cohort: {len(df_mri_full)}")
print(f"Gold Subset: {len(df_subset)}")

# --- EXP A: CLINICAL RESULTS ---
print("\nRunning Clinical Model...")
df_subset = df_subset.dropna(subset=['Group'])
le = LabelEncoder()
y_sub = le.fit_transform(df_subset['Group'])
classes = le.classes_

for c in score_cols:
    if c in df_subset.columns:
        df_subset[c] = pd.to_numeric(df_subset[c], errors='coerce').fillna(df_subset[c].median())

clinical_feats = [c for c in score_cols if c in df_subset.columns]
if 'Age' in df_subset.columns: clinical_feats.append('Age')
# Sex
if 'Sex' in df_subset.columns:
    df_subset['Sex_Code'] = LabelEncoder().fit_transform(df_subset['Sex'].fillna('Unknown').astype(str))
    clinical_feats.append('Sex_Code')

X_clin = df_subset[clinical_feats]
X_train, X_test, y_train, y_test = train_test_split(X_clin, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

model_clin = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_estimators=100)
model_clin.fit(X_train, y_train)
save_plots(model_clin, X_test, y_test, classes, 'results_clinical', X_clin.columns)

# --- EXP B: RADIOMICS RESULTS ---
print("\nRunning Radiomics Model...")
df_mri_full = df_mri_full.dropna(subset=['Group'])
y_full = le.fit_transform(df_mri_full['Group'])

metadata = ['Join_Key', 'Group', 'Subject', 'Age', 'Sex', 'Image Data ID', 
            'Subject_x', 'Subject_y', 'PTID', 'Visit', 'Description', 'Acq Date', 'Format', 'Downloaded', 'Type', 'Modality']
X_rad = df_mri_full.drop(columns=metadata, errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Cleanup
selector = VarianceThreshold(threshold=0)
X_var = selector.fit_transform(X_rad)
feature_names = X_rad.columns[selector.get_support()]

# Scale & LASSO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var)
lasso = LassoCV(cv=5, random_state=42, max_iter=100000, n_jobs=-1).fit(X_scaled, y_full)
mask = np.abs(lasso.coef_) > 0
X_lasso = X_scaled[:, mask]
selected_names = feature_names[mask]
print(f"LASSO selected {len(selected_names)} features.")

X_train, X_test, y_train, y_test = train_test_split(X_lasso, y_full, test_size=0.2, random_state=42, stratify=y_full)
model_rad = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, 
                              n_estimators=300, max_depth=5, learning_rate=0.05, colsample_bytree=0.5)
model_rad.fit(X_train, y_train)
save_plots(model_rad, X_test, y_test, classes, 'results_radiomics', selected_names)

# Save Biomarkers
pd.DataFrame({'Feature': selected_names}).to_csv('results_radiomics/biomarkers.csv', index=False)

# --- EXP C: COMBINED RESULTS ---
print("\nRunning Combined Model...")
# Apply LASSO mask to subset radiomics
X_rad_sub_raw = df_subset.drop(columns=metadata, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
# Note: We must ensure columns match X_rad. Re-align columns.
X_rad_sub_aligned = X_rad_sub_raw.reindex(columns=X_rad.columns, fill_value=0)
X_sub_var = selector.transform(X_rad_sub_aligned)
X_sub_scaled = scaler.transform(X_sub_var)
X_sub_lasso = X_sub_scaled[:, mask]

# Concatenate
X_comb = np.hstack([X_clin.values, X_sub_lasso])
feat_comb = np.concatenate([clinical_feats, selected_names])

X_train, X_test, y_train, y_test = train_test_split(X_comb, y_sub, test_size=0.2, random_state=42, stratify=y_sub)
model_comb = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, n_estimators=100)
model_comb.fit(X_train, y_train)
save_plots(model_comb, X_test, y_test, classes, 'results_combined', feat_comb)

print("\n--- DONE! CHECK ALL 3 FOLDERS ---")
