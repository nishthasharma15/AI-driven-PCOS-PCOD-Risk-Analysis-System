import pandas as pd
import numpy as np

# ================================
# STEP 1: Load Dataset
# ================================

df = pd.read_csv("PCOS_data.csv")

print("Initial Shape:", df.shape)
print(df.head())

# ================================
# STEP 2: Clean Column Names
# ================================

df.columns = df.columns.str.strip()

# ================================
# STEP 2.1: Remove Unnamed Columns (CSV Artifacts)
# ================================

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ================================
# STEP 3: Remove Duplicate Rows
# ================================

initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows - df.shape[0]} duplicate rows")

# ================================
# STEP 4: Drop Irrelevant / Identifier Columns
# ================================

drop_cols = [
    'Sl. No',
    'Patient File No.',
    'Blood Group',
    'I_beta-HCG(mIU/mL)',
    'II_beta-HCG(mIU/mL)'
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')
print("After dropping unnecessary columns:", df.shape)

# ================================
# STEP 5: Drop Completely Empty Columns
# ================================

empty_cols = df.columns[df.isnull().all()]
df.drop(columns=empty_cols, inplace=True)
print(f"Dropped empty columns: {list(empty_cols)}")

# ================================
# STEP 6: Ensure Target Column is Valid
# ================================

df['PCOS (Y/N)'] = pd.to_numeric(df['PCOS (Y/N)'], errors='coerce')

# Drop rows with missing target
df = df.dropna(subset=['PCOS (Y/N)'])

# Ensure binary integer target
df['PCOS (Y/N)'] = df['PCOS (Y/N)'].astype(int)

# ================================
# STEP 7: Encode Yes/No Categorical Features
# ================================

yes_no_columns = [
    'Pregnant(Y/N)',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Fast food (Y/N)',
    'Reg.Exercise(Y/N)'
]

for col in yes_no_columns:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({'Y': 1, 'N': 0})
        )

# ================================
# STEP 8: Encode Cycle Regularity
# ================================

if 'Cycle(R/I)' in df.columns:
    df['Cycle(R/I)'] = (
        df['Cycle(R/I)']
        .astype(str)
        .str.strip()
        .str.upper()
        .map({'R': 0, 'I': 1})
    )

# ================================
# STEP 9: Convert All Columns to Numeric
# ================================

df = df.apply(pd.to_numeric, errors='coerce')

# ================================
# STEP 10: Handle Missing Values (Safe & Robust)
# ================================

for col in df.columns:
    if df[col].isnull().sum() > 0:

        # Binary / categorical columns
        if df[col].nunique(dropna=True) <= 2:
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])
            else:
                df[col] = df[col].fillna(0)

        # Continuous numerical columns
        else:
            df[col] = df[col].fillna(df[col].median())

# ================================
# STEP 11: Final Validation
# ================================

print("\nFinal Dataset Shape:", df.shape)
print("\nRemaining Missing Values Per Column:\n", df.isnull().sum())

# ================================
# STEP 12: Save Cleaned Dataset
# ================================

df.to_csv("PCOS_cleaned.csv", index=False)

print("\nData preprocessing completed successfully.")
