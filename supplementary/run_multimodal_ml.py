import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
RADIOMICS_CSV = 'MASTER_Radiomics_Dataset.csv'
CLINICAL_CSV = 'ADNIMERGE_03Jun2025.csv'

print("--- 1. Loading Data ---")
# Load Radiomics
print(f"Loading Radiomics from {RADIOMICS_CSV}...")
df_rad = pd.read_csv(RADIOMICS_CSV, low_memory=False)

# Clean Subject ID
if 'Subject' not in df_rad.columns and 'Subject_x' in df_rad.columns:
    df_rad.rename(columns={'Subject_x': 'Subject'}, inplace=True)

# Deduplicate Radiomics (Just in case)
df_rad = df_rad.drop_duplicates(subset=['Subject'], keep='first')
print(f"Radiomics Subjects: {len(df_rad)}")

# Load Clinical
print(f"Loading Clinical Data from {CLINICAL_CSV}...")
df_clin = pd.read_csv(CLINICAL_CSV, low_memory=False)

# --- 2. STRICT CLEANING (The Fix) ---
print("Cleaning Clinical Data...")

if 'PTID' in df_clin.columns:
    # 1. Sort by Date (Earliest first)
    # Note: We try to find 'EXAMDATE', if not, we rely on the order
    if 'EXAMDATE' in df_clin.columns:
        df_clin = df_clin.sort_values('EXAMDATE')
    
    # 2. FORCE ONE ROW PER PATIENT
    # We keep only the FIRST entry for each PTID (Baseline)
    df_clin_subset = df_clin.drop_duplicates(subset=['PTID'], keep='first')
    
    print(f"Clinical Data reduced from {len(df_clin)} rows to {len(df_clin_subset)} unique patients.")
    
    # 3. Select Columns
    clinical_cols = ['PTID', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE', 'ADAS13']
    clinical_cols = [c for c in clinical_cols if c in df_clin_subset.columns]
    df_clin_subset = df_clin_subset[clinical_cols]
    
    # 4. MERGE
    print("Merging...")
    df_final = pd.merge(df_rad, df_clin_subset, left_on='Subject', right_on='PTID', how='inner')
    
    print(f"Final Dataset Size: {len(df_final)} subjects")
    
    # SAFETY CHECK
    if len(df_final) > len(df_rad):
        print("!!! ERROR: Dataset Exploded! Stopping.")
        exit()
else:
    print("ERROR: 'PTID' not found in Clinical CSV.")
    exit()

# --- 3. Prepare Features ---
if 'PTGENDER' in df_final.columns:
    df_final['Male'] = df_final['PTGENDER'].apply(lambda x: 1 if x == 'Male' else 0)

clin_feats = ['AGE', 'PTEDUCAT', 'APOE4', 'MMSE', 'ADAS13', 'Male']
clin_feats = [c for c in clin_feats if c in df_final.columns]

rad_feats = [c for c in df_final.columns if ('Hippo' in c or 'Amyg' in c) and 'Whole' not in c]
rad_feats = [c for c in rad_feats if pd.api.types.is_numeric_dtype(df_final[c])]

# Fill Missing Values
for col in clin_feats + rad_feats:
    df_final[col] = df_final[col].fillna(df_final[col].median())

# Target
df_final = df_final.dropna(subset=['Group'])
le = LabelEncoder()
Y = le.fit_transform(df_final['Group'])
X = df_final[clin_feats + rad_feats]

print(f"\nTraining on {len(df_final)} unique subjects.")

# --- 4. Train Model ---
print("\n--- Training Model ---")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

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
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\n" + "="*40)
print(f"FINAL VALIDATED ACCURACY: {acc:.2%}")
print("="*40)
print(classification_report(y_test, preds, target_names=le.classes_))

# Plot
plt.figure(figsize=(10, 8))
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1][:20]
top_features = X.columns[sorted_idx]
top_scores = importance[sorted_idx]
colors = ['red' if f in clin_feats else 'blue' for f in top_features]
sns.barplot(x=top_scores, y=top_features, palette=colors)
plt.title(f'Feature Importance (Acc: {acc:.2%})\nRed=Clinical, Blue=Radiomics')
plt.tight_layout()
plt.savefig('Fixed_Feature_Importance.png')
print("Saved plot to 'Fixed_Feature_Importance.png'")