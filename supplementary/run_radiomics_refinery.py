import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV

# --- CONFIGURATION ---
FILE_RADIOMICS = 'MASTER_Radiomics_Dataset.csv'
FILE_SCANS = 'AD_mprage_spgr_9_03_2025.csv'

def clean_id(val):
    return str(val).replace('I', '').strip()

print(f"--- 1. LOADING GOLD MINE ---")
df_rad = pd.read_csv(FILE_RADIOMICS, low_memory=False)
df_scan = pd.read_csv(FILE_SCANS, low_memory=False)

# --- 2. SQUASHING & PREP (Recovering 1229 Subjects) ---
rad_id_col = 'Image Data ID'
if rad_id_col not in df_rad.columns:
    candidates = [c for c in df_rad.columns if 'Image Data ID' in c]
    if candidates: rad_id_col = candidates[0]

if 'Subject' not in df_rad.columns and 'Subject_x' in df_rad.columns:
    df_rad.rename(columns={'Subject_x': 'Subject'}, inplace=True)

# Collapse rows to fix sparsity
if df_rad['Subject'].duplicated().any():
    print(f"Collapsing {len(df_rad)} rows...")
    df_rad = df_rad.groupby('Subject').first().reset_index()

# Clean Keys
df_rad['Join_Key'] = df_rad[rad_id_col].apply(clean_id)
df_scan['Join_Key'] = df_scan['Image Data ID'].apply(clean_id)

# Merge to get Labels
df_mri = pd.merge(df_rad, df_scan[['Join_Key', 'Group']], on='Join_Key', how='inner')

# Fix Group
if 'Group' not in df_mri.columns:
    if 'Group_x' in df_mri.columns: df_mri.rename(columns={'Group_x': 'Group'}, inplace=True)
    elif 'Group_y' in df_mri.columns: df_mri.rename(columns={'Group_y': 'Group'}, inplace=True)

print(f"Dataset Size: {len(df_mri)} subjects")

# --- 3. REFINING PIPELINE ---
print("\n--- STARTING REFINERY PROCESS ---")

df_mri = df_mri.dropna(subset=['Group'])
y = LabelEncoder().fit_transform(df_mri['Group'])

# Get pure numeric features
metadata = ['Join_Key', 'Group', 'Subject', 'Image Data ID', 'Age', 'Sex', 'Visit', 'Description', 'Acq Date']
X_raw = df_mri.drop(columns=metadata, errors='ignore').select_dtypes(include=[np.number])

# A. Impute & Remove Constants
X_filled = X_raw.fillna(0)
selector = VarianceThreshold(threshold=0)
X_var = selector.fit_transform(X_filled)
print(f"1. Original Features: {X_raw.shape[1]}")
print(f"2. Non-Constant Features: {X_var.shape[1]}")

# B. Scale (Crucial for LASSO)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var)
feature_names = X_raw.columns[selector.get_support()]

# C. LASSO SELECTION (The Gold Standard)
# We use Lasso to find the features that actually matter
print("3. Running LASSO to find Biomarkers (this takes a moment)...")
# We treat this as a regression problem just to find important features
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# Get the non-zero coefficients (the "Gold" features)
importance = np.abs(lasso.coef_)
selected_mask = importance > 0
X_lasso = X_scaled[:, selected_mask]
selected_names = feature_names[selected_mask]

print(f"4. LASSO selected {X_lasso.shape[1]} important biomarkers.")

# --- 4. FINAL XGBOOST ON REFINED DATA ---
print("\n--- Training Model on Refined Biomarkers ---")

X_train, X_test, y_train, y_test = train_test_split(X_lasso, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=3, 
    eval_metric='mlogloss', 
    use_label_encoder=False,
    n_estimators=300, 
    max_depth=5, 
    learning_rate=0.05
)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

print(f"\n{'='*40}")
print(f"FINAL REFINED ACCURACY: {acc:.2%}")
print(f"{'='*40}")

# --- 5. EXPORT BIOMARKERS FOR PAPER ---
print("\nTop 10 Biomarkers Identified:")
top_indices = np.argsort(model.feature_importances_)[::-1][:10]
for i in top_indices:
    print(f"{i+1}. {selected_names[i]}")

# Save them to CSV for Aditi
biomarkers_df = pd.DataFrame({'Feature': selected_names, 'Importance': model.feature_importances_})
biomarkers_df = biomarkers_df.sort_values(by='Importance', ascending=False)
biomarkers_df.head(50).to_csv('top_50_biomarkers.csv', index=False)
print("\nSaved 'top_50_biomarkers.csv'. Send this list to Aditi!")
