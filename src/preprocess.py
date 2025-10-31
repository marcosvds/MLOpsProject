# --- Disable all plotting and display output (for terminal execution) ---
import matplotlib
matplotlib.use("Agg")
import builtins
builtins.display = lambda *args, **kwargs: None

import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

# --- Core libraries ---
import pandas as pd
import numpy as np
import time
import re
from pathlib import Path

# --- Scientific and statistical tools ---
import scipy.stats as stats

# --- Geospatial utilities ---
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- Text processing ---
from unidecode import unidecode


# Load datasets - Update the paths to reflect the new mountpoint
DATA_DIR = Path(r"C:\Users\utaka\MLOPS\Final Assignment\MLOpsProject\data")

train_raw = pd.read_csv(DATA_DIR / "train_raw.csv")
test_raw  = pd.read_csv(DATA_DIR / "test_raw.csv")

train = train_raw.copy()
test  = test_raw.copy()

# Initial exploration
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Drop weight column since all house weights are equal 
train.drop(columns=['w'], inplace=True)

# Overview
print(train.info())
print(test.info())

# Check for missing values
print("\nMissing in Train:\n", train.isnull().sum())
print("\nMissing in Test:\n", test.isnull().sum())

# Peek at data
display(train.head())
display(test.head())


# In[176]:


# Calculate percentage of missing values
missing_train_percent = (train.isnull().sum() / len(train)) * 100
missing_test_percent = (test.isnull().sum() / len(test)) * 100

# Combine and filter out columns with no missing values
missing_data = pd.concat([missing_train_percent[missing_train_percent > 0], missing_test_percent[missing_test_percent > 0]], axis=1, keys=['Train', 'Test'])

# Plotting the histogram
if not missing_data.empty:
    missing_data.plot(kind='bar', figsize=(12, 6))
    plt.title('Percentage of Missing Values per Column')
    plt.ylabel('Percentage')
    plt.xlabel('Columns')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found in either dataset.")


# ### Missing Data Imputation Strategy
# 
# Handling missing data is a critical step before model training.  
# We applied a structured approach based on the amount and nature of missingness in each variable.
# 
# | Scenario | Description | Action Taken |
# |-----------|--------------|---------------|
# | **≤ 5% missing (MCAR)** | Values missing completely at random — no systematic pattern. | Dropped or simply imputed using median/mode. |
# | **5–20% missing (MAR)** | Missingness depends on other observed features (e.g., zone, condition). | Group-based imputation using median, mode, or assigning `"unknown"`. |
# | **> 20% missing (MNAR)** | Missingness itself may carry information or strong bias. | Modeled missingness with an indicator variable (e.g., `feature_missing = 1`) or dropped the feature if unreliable. |
# | **Missing values in test data** | Missing values appear only in the unseen dataset. | Always imputed using the same logic as in training — never dropped. |
# 
# > **Rule of Thumb:**  
# > - Small random gaps → drop or simple impute  
# > - Moderate gaps → contextual impute  
# > - Large or biased gaps → model or remove  
# > - Test data → always impute  
# 
# This framework ensured consistent, explainable imputations and avoided information loss that could distort model learning.
# 

# ## 1 - Availability

# In[ ]:



# In[178]:


train['availability'].unique()


# In[179]:


# Separate rows with missing and non-missing 'availability' values
available_not_null = train[train['availability'].notnull()]
available_is_null = train[train['availability'].isnull()]

# Check if 'availability' contains missing values
if available_is_null.empty:
    print("No missing values in the 'availability' column.")
else:
    print(f"Number of rows with missing availability: {len(available_is_null)}")
    print(f"Number of rows with non-missing availability: {len(available_not_null)}")

    # Compare a numerical variable (e.g., 'y') between groups with and without missing 'availability'
    # This helps evaluate whether the missingness pattern might be random or systematic (MCAR vs. MAR/MNAR)

    if 'y' in train.columns and np.issubdtype(train['y'].dtype, np.number):
        # Proceed only if both groups contain more than one valid observation
        if len(available_not_null) > 1 and len(available_is_null) > 1:
            # Perform an independent samples t-test to compare group means
            ttest_result = stats.ttest_ind(available_not_null['y'].dropna(), available_is_null['y'].dropna())
            print(f"\nT-test comparison of 'price' for rows with/without missing 'availability':")
            print(f"  T-statistic: {ttest_result.statistic:.4f}")
            print(f"  P-value: {ttest_result.pvalue:.4f}")

            # Interpret the result based on the significance level
            if ttest_result.pvalue < 0.05:
                print("\nInterpretation: The difference in 'price' is statistically significant.")
                print("Missingness in 'availability' may not be random (MAR or MNAR).")
            else:
                print("\nInterpretation: No statistically significant difference in 'price'.")
                print("Missingness in 'availability' may be closer to MCAR with respect to 'price'.")
        else:
            print("\nCannot perform T-test: Insufficient data points in one or both groups.")
    else:
        print("\n'y' column not found or not numerical. Unable to perform T-test.")
        print("Consider comparing missingness with another relevant numerical feature.")

    # Visual comparison of 'y' distributions for both groups
    if 'y' in train.columns and np.issubdtype(train['y'].dtype, np.number):
        plt.figure(figsize=(10, 6))
        plt.hist(available_not_null['y'].dropna(), bins=30, alpha=0.5, label='Availability Not Missing')
        plt.hist(available_is_null['y'].dropna(), bins=30, alpha=0.5, label='Availability Missing')
        plt.title('Distribution of Price Based on Availability Missingness')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


# In[180]:


# Identify rows with and without missing 'availability' values in the TEST dataset
available_not_null_test = test[test['availability'].notnull()]
available_is_null_test = test[test['availability'].isnull()]

# Check for missing values in 'availability' within the TEST data
if available_is_null_test.empty:
    print("No missing values in the 'availability' column in the TEST data.")
else:
    print(f"Number of rows with missing availability in TEST: {len(available_is_null_test)}")
    print(f"Number of rows with non-missing availability in TEST: {len(available_not_null_test)}")

    print("\nExploring relationships of missing 'availability' with other columns in TEST data:")

    # --- Categorical Columns Analysis ---
    # Evaluate the proportion of missing 'availability' within each category of categorical variables
    categorical_columns = test.select_dtypes(include='object').columns.tolist()
    if categorical_columns:
        print("\nAnalyzing missing 'availability' vs. Categorical Columns:")
        for col in categorical_columns:
            missing_proportion_by_category = test.groupby(col)['availability'].apply(lambda x: x.isnull().sum() / len(x) * 100)
            if not missing_proportion_by_category.empty and (missing_proportion_by_category > 0).any():
                print(f"\nMissing 'availability' percentage by '{col}' category:")
                display(missing_proportion_by_category[missing_proportion_by_category > 0])
                # Optional: visualize with a bar chart if desired

    # --- Numerical Columns Analysis ---
    # Compare numerical feature distributions between rows with and without missing 'availability'
    numerical_columns = test.select_dtypes(include=np.number).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'availability' and not col.lower().endswith('_id')]

    if numerical_columns and len(available_not_null_test) > 1 and len(available_is_null_test) > 1:
        print("\nAnalyzing missing 'availability' vs. Numerical Columns (comparing means with T-test):")
        for col in numerical_columns:
            # Perform t-tests only if both groups contain sufficient valid observations
            if available_not_null_test[col].dropna().shape[0] > 1 and available_is_null_test[col].dropna().shape[0] > 1:
                ttest_result_col = stats.ttest_ind(available_not_null_test[col].dropna(), available_is_null_test[col].dropna())
                print(f"\nT-test comparison of '{col}' for rows with/without missing 'availability' in TEST:")
                print(f"  T-statistic: {ttest_result_col.statistic:.4f}")
                print(f"  P-value: {ttest_result_col.pvalue:.4f}")

                if ttest_result_col.pvalue < 0.05:
                    print(f"  Interpretation: Statistically significant difference in '{col}'. Missingness in 'availability' may be related to '{col}'.")
                else:
                    print(f"  Interpretation: No statistically significant difference in '{col}'. Missingness in 'availability' may be MCAR with respect to '{col}'.")
            else:
                print(f"\nNot enough data points to compare '{col}' for rows with/without missing 'availability' in TEST.")
    elif numerical_columns:
        print("\nNot enough data points in one or both groups with/without missing 'availability' to perform numerical column comparisons.")

    # --- Summary of Findings ---
    print("\nSummary of Missingness Analysis in TEST:")
    print("- If missingness in 'availability' is related to other variables, it may indicate MAR.")
    print("- If missingness appears unrelated to any observed variables, it may indicate MCAR.")
    print("- If missingness depends on the unobserved value of 'availability' itself, it would be MNAR.")
    print("\nBased on this analysis and standard handling of test data, missing 'availability' values will be imputed accordingly.")


# In[181]:


train['availability'] = train['availability'].fillna('unknown')
test['availability'] = test['availability'].fillna('unknown')


# In[182]:


train.columns


# In[183]:


train['availability']


# In[184]:


test['availability'].unique()


# In[185]:


train['availability'].isnull().sum()
test['availability'].isnull().sum()


# In[186]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to extract date from 'available from DD/MM/YYYY'
def extract_availability_date(val):
    try:
        if isinstance(val, str) and "available from" in val.lower():
            date_str = val.lower().replace("available from", "").strip()
            return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        return pd.NaT
    except:
        return pd.NaT

# Apply to both datasets
train['availability_date'] = train['availability'].apply(extract_availability_date)
test['availability_date'] = test['availability'].apply(extract_availability_date)

# Combine dates
all_dates = pd.concat([train['availability_date'], test['availability_date']]).dropna()

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(all_dates, bins=50, color='steelblue', edgecolor='black')
plt.title("Availability Date Distribution")
plt.xlabel("Availability Date")
plt.ylabel("Number of Listings")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Detect outliers (e.g., before 2022 or after 2025)
outlier_low = all_dates[all_dates < pd.to_datetime("2022-01-01")]
outlier_high = all_dates[all_dates > pd.to_datetime("2025-01-01")]

print(f"Outliers before 2022: {len(outlier_low)} rows")
print(outlier_low.sort_values().unique())

print(f"\nOutliers after 2025: {len(outlier_high)} rows")
print(outlier_high.sort_values().unique())


# In[187]:


# Group the combined dates by year and count the observations
yearly_counts = all_dates.dt.year.value_counts().sort_index()

# Print the counts for each year
print("Number of observations with availability date per year:")
print(yearly_counts)


# In[188]:


# Define valid range
min_valid = pd.to_datetime("2022-01-01")
max_valid = pd.to_datetime("2024-12-31")

train = train[(train['availability_date'].isna()) |
              ((train['availability_date'] >= min_valid) &
               (train['availability_date'] <= max_valid))].copy()

# Create binary flag
train['is_available'] = train['availability'].apply(
    lambda x: 1 if isinstance(x, str) and x.strip().lower() == 'available' else 0
)
test['is_available'] = test['availability'].apply(
    lambda x: 1 if isinstance(x, str) and x.strip().lower() == 'available' else 0
)


# In[189]:


train[['availability_date','availability']]


# In[190]:


print("\nMissing values summary after handling 'availability':")
print("\nMissing in Train:\n", train.isnull().sum())
print("\nMissing in Test:\n", test.isnull().sum())


# In[191]:


avail_map = {'available': 2, 'unknown': 1, 'not available': 0}
train['availability_encoded'] = train['availability'].fillna('unknown').map(avail_map)
test['availability_encoded'] = test['availability'].fillna('unknown').map(avail_map)


# In[192]:


# Replace NaN values in 'availability_encoded' with 0
# Ensures consistent encoding for entries originally labeled as 'available from date' or missing values
train['availability_encoded'] = train['availability_encoded'].fillna(0)
test['availability_encoded'] = test['availability_encoded'].fillna(0)


# In[193]:


train[['is_available','availability', 'availability_encoded']].head(20)


# In[194]:


# Step 1: Show rows in train with missing 'other_features'
print("📋 Missing 'other_features' in train:")
display(train[train['other_features'].isna()])

# Step 2: Show rows in test with missing 'other_features'
print("\n📋 Missing 'other_features' in test:")
display(test[test['other_features'].isna()])


# In[195]:


train.isnull().sum()


# In[196]:


test.isnull().sum()


# ## 2 - Energy Efficiency Class

# In[197]:


# Rows in train with missing 'energy_efficiency_class'
print("📋 Rows in train with missing 'energy_efficiency_class':")
display(train[train['energy_efficiency_class'].isna()])

# Rows in test with missing 'energy_efficiency_class'
print("\n📋 Rows in test with missing 'energy_efficiency_class':")
display(test[test['energy_efficiency_class'].isna()])


# In[198]:


import pandas as pd

# Step 1: Encode energy class (a=7 best, g=1 worst, unknown=None)
energy_map = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1, 'unknown': None}
train['energy_class_num'] = train['energy_efficiency_class'].map(energy_map)

# Step 2: Select numeric variables
corr_df = train[['energy_class_num', 'square_meters', 'condominium_fees']].copy()

# Step 3: One-hot encode categorical predictors
cond_dummies = pd.get_dummies(train['conditions'], prefix='cond')
zone_dummies = pd.get_dummies(train['zone'], prefix='zone')

# Step 4: Combine into correlation dataset
corr_df = pd.concat([corr_df, cond_dummies, zone_dummies], axis=1)

# Step 5: Compute correlation with energy_class_num
correlations = corr_df.corr()['energy_class_num'].sort_values(ascending=False)

# Step 6: Print top 20 correlated variables
print("🔍 Top 20 correlations with energy_class_num:\n")
print(correlations.head(20))

# Step 7: Show correlation with condominium_fees directly
print("\n📌 Correlation with 'condominium_fees':")
print(correlations['condominium_fees'])


# In[199]:


# Step 2: Average energy score by 'conditions'
condition_avg_map = train.groupby('conditions')['energy_class_num'].mean().sort_values(ascending=False)

# Step 3: Average energy score by 'zone'
zone_avg_map = train.groupby('zone')['energy_class_num'].mean().sort_values(ascending=False)

# Display
print("📋 Mean Energy Class by 'conditions':\n")
print(condition_avg_map)

print("\n📋 Mean Energy Class by 'zone':\n")
print(zone_avg_map)


# In[200]:


# Step 1: Define energy score mapping
energy_map = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1, 'unknown': None}

# Step 2: Map values to energy_class_num
train['energy_class_num'] = train['energy_efficiency_class'].map(energy_map)
test['energy_class_num'] = test['energy_efficiency_class'].map(energy_map)

# Step 3: Create group means from available (non-null) rows in train
cond_avg = train.loc[train['energy_class_num'].notna()].groupby('conditions')['energy_class_num'].mean()
zone_avg = train.loc[train['energy_class_num'].notna()].groupby('zone')['energy_class_num'].mean()

# Step 4: Define imputation function
def impute_energy(row):
    if pd.notna(row['energy_class_num']):
        return row['energy_class_num']
    cond_val = cond_avg.get(row['conditions'])
    if pd.notna(cond_val):
        return cond_val
    zone_val = zone_avg.get(row['zone'])
    if pd.notna(zone_val):
        return zone_val
    return 4  # fallback average if nothing else available

# Step 5: Apply to train and test
train['energy_class_num'] = train.apply(impute_energy, axis=1)
test['energy_class_num'] = test.apply(impute_energy, axis=1)

# Step 6: Final formatting
train['energy_efficiency_class_encoded'] = train['energy_class_num'].round().clip(1, 7).astype(int)
test['energy_efficiency_class_encoded'] = test['energy_class_num'].round().clip(1, 7).astype(int)


# In[201]:


test.isnull().sum()


# In[202]:


train.isnull().sum()


# ## 3 - Conditions

# In[203]:


# Group by 'conditions' and compute summary stats
condition_price_stats = (
    train.groupby('conditions')['y']
    .agg(['median', 'mean', 'count'])
    .sort_values(by='median', ascending=False)
    .reset_index()
)

# Rename for clarity
condition_price_stats.columns = ['conditions', 'median_rent', 'mean_rent', 'num_listings']

# Display
print(" Rent Price Stats by Condition:\n")
print(condition_price_stats)


# Inference:
# 
# "New" and "Excellent" conditions correspond to higher rent levels.
# 
# "Good condition" appears for cheaper properties.
# 
# This confirms: conditions is strongly associated with price, and we can impute it based on the rent y value of missing rows.

# In[204]:


# Reference: median price by zone + energy + condition
condition_ref = (
    train[train['conditions'].notna()]
    .groupby(['zone', 'energy_efficiency_class_encoded', 'conditions'])['y']
    .median()
    .reset_index()
    .rename(columns={'y': 'median_rent'})
)

# Also create fallback: most frequent condition by zone + energy
fallback_ref = (
    train[train['conditions'].notna()]
    .groupby(['zone', 'energy_efficiency_class_encoded'])['conditions']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)


# In[205]:


import seaborn as sns

# Recompute top zone-energy pairs
top_pairs = (
    train.groupby(['zone', 'energy_efficiency_class_encoded'])
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()[['zone', 'energy_efficiency_class_encoded']]
)

# Filter condition_ref for visualization
viz_data = condition_ref.merge(top_pairs, on=['zone', 'energy_efficiency_class_encoded'])

# Plot median rent by condition across top zone-energy combinations
plt.figure(figsize=(14, 6))
sns.barplot(
    data=viz_data,
    x='zone',
    y='median_rent',
    hue='conditions'
)
plt.title("Median Rent by Condition for Top 10 Zone + Energy Class Combinations")
plt.ylabel("Median Rent (€)")
plt.xticks(rotation=45)
plt.legend(title="Condition")
plt.tight_layout()
plt.show()


# In[206]:


# For train: use price proximity to impute
def refined_impute_condition_train(row):
    if pd.notna(row['conditions']):
        return row['conditions']

    candidates = condition_ref[
        (condition_ref['zone'] == row['zone']) &
        (condition_ref['energy_efficiency_class_encoded'] == row['energy_efficiency_class_encoded'])
    ]

    if not candidates.empty:
        closest = candidates.iloc[(candidates['median_rent'] - row['y']).abs().argsort()[:1]]
        return closest['conditions'].values[0]

    # fallback: most frequent in train
    return train['conditions'].mode().iloc[0]

train['conditions'] = train.apply(refined_impute_condition_train, axis=1)

# For test: no rent available, fallback to most frequent
def refined_impute_condition_test(row):
    if pd.notna(row['conditions']):
        return row['conditions']

    match = fallback_ref[
        (fallback_ref['zone'] == row['zone']) &
        (fallback_ref['energy_efficiency_class_encoded'] == row['energy_efficiency_class_encoded'])
    ]

    if not match.empty:
        return match['conditions'].values[0]

    return train['conditions'].mode().iloc[0]  # fallback

test['conditions'] = test.apply(refined_impute_condition_test, axis=1)


# ### 📝 Imputation Strategy for `conditions`
# 
# To handle missing values in the `conditions` variable, we adopted a **two-tiered imputation strategy** applied separately to the **training** and **test** datasets.  
# This approach preserved meaningful data patterns while avoiding information leakage.
# 
# ---
# 
# #### ✅ Training Set Imputation
# 
# In the training data (where the true rent price `y` is available), we used a **price-guided imputation** method:
# 
# 1. Grouped the data by `zone`, `energy_efficiency_class_encoded`, and `conditions`, then computed the **median rent price** for each combination.  
# 2. For each row with missing `conditions`, selected the group (matching `zone` and energy class) whose median rent was **closest to that row’s actual rent**.  
# 3. If no suitable group was found, defaulted to the **most frequent condition** overall in the training set.
# 
# This approach ensured that imputed `conditions` values aligned with **economic characteristics** (via price similarity) and **categorical context** (zone and efficiency).
# 
# ---
# 
# #### ✅ Test Set Imputation
# 
# Since the rent price `y` is not available in the test set, we used a **frequency-based approach**:
# 
# 1. Precomputed the most common `conditions` for each `(zone, energy_efficiency_class_encoded)` group in the training data.  
# 2. For each missing value in the test set, assigned the **most common condition** from its matching group.  
# 3. If no match existed, defaulted to the **overall most frequent condition** in the training set.
# 
# This maintained **consistency with the training distribution** and leveraged both **spatial** and **efficiency-based patterns** for realistic estimation.
# 
# ---
# 
# Overall, this two-step method balanced interpretability, consistency, and predictive relevance without introducing target leakage.
# 

# In[207]:


test.isnull().sum()


# In[208]:


train.isnull().sum()


# In[209]:


# Define ordinal mapping (lowest = worst, highest = best)
condition_map = {
    'good condition': 0,
    'excellent': 1,
    'new': 2
}

# Apply mapping
train['conditions_encoded'] = train['conditions'].map(condition_map)
test['conditions_encoded'] = test['conditions'].map(condition_map)


# In[210]:


train.head()


# ## 4- Floor

# In[211]:


missing_floor_train = train[train['floor'].isna()]
missing_floor_test = test[test['floor'].isna()]

print("Missing 'floor' in Train Data (all attributes):")
if not missing_floor_train.empty:
    display(missing_floor_train)
else:
    print("No missing 'floor' values found in the training dataset.")

print("\nMissing 'floor' in Test Data (all attributes):")
if not missing_floor_test.empty:
    display(missing_floor_test)
else:
    print("No missing 'floor' values found in the test dataset.")


# In[212]:


train['floor'].unique()


# In[213]:


# Grouped mode lookup
floor_mode_by_group = (
    train.groupby(['zone', 'elevator'])['floor']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .reset_index()
)

# Define function
def impute_floor(row, df_lookup):
    if pd.notna(row['floor']):
        return row['floor']
    match = df_lookup[
        (df_lookup['zone'] == row['zone']) &
        (df_lookup['elevator'] == row['elevator'])
    ]
    if not match.empty:
        return match['floor'].values[0]
    return train['floor'].mode().iloc[0]  # fallback

# Apply to both datasets
train['floor'] = train.apply(lambda row: impute_floor(row, floor_mode_by_group), axis=1)
test['floor'] = test.apply(lambda row: impute_floor(row, floor_mode_by_group), axis=1)


# ###  Imputation Strategy for `floor`
# 
# To address the small number of missing values in the `floor` variable, we applied a **grouped mode imputation** based on **zone** and **elevator availability**:
# 
# 1. Grouped the training data by `zone` and `elevator` status, then calculated the most frequent (**mode**) `floor` within each group.  
# 2. For each row with a missing `floor`, imputed the value using the **mode of its matching (zone, elevator)** group.  
# 3. If no matching group was found (e.g., rare combinations), defaulted to the **overall most frequent floor** in the training set.
# 
# This approach leverages **real estate intuition** — the relationship between floor level and rental desirability often depends on whether an elevator is present, especially for higher floors.
# 

# In[214]:


train['floor'].unique()


# In[215]:


# Define mapping
floor_map = {
    'Semi-basement': -1,
    'Ground floor': 0,
    'Mezzanine': 0.5,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9
}

# Apply to both datasets
train['floor_num'] = train['floor'].map(floor_map)
test['floor_num'] = test['floor'].map(floor_map)


# In[216]:


train['floor_num'].unique()


# In[217]:


train['elevator']


# In[218]:


# Step 1: Encode elevator as 1 (yes) and -1 (no)
train['elevator_sign'] = train['elevator'].map({'yes': 1, 'no': -1})
test['elevator_sign'] = test['elevator'].map({'yes': 1, 'no': -1})

# Step 2: Create interaction feature: floor × elevator_sign
# (Assumes 'floor' has already been mapped to numeric values)
train['floor_effect'] = train['floor_num'] * train['elevator_sign']
test['floor_effect'] = test['floor_num'] * test['elevator_sign']


# In[219]:


test.isnull().sum()


# ## 5 - Description

# In[220]:


train['description'].unique()


# In[221]:


test['description'].unique()


# In[222]:


import re

def parse_description(text):
    text = str(text).lower()
    features = {
        'total_rooms': None,
        'num_bedrooms': 0,
        'num_other_rooms': 0,
        'num_bathrooms': 0,
        'kitchen_open': 0,
        'kitchen_diner': 0,
        'kitchen_nook': 0,
        'kitchen_semi': 0,
        'kitchenette': 0,
        'kitchen_none': 1,
        'suitable_disabled': int('suitable for disabled' in text)
    }

    # Total rooms
    match_total = re.search(r'(\d+)\s*(?:rooms|\()', text)
    if match_total:
        features['total_rooms'] = int(match_total.group(1))

    # Bedrooms and other rooms
    match_bed = re.search(r'(\d+)\s*bedrooms?', text)
    match_other = re.search(r'(\d+)\s*others?', text)
    if match_bed:
        features['num_bedrooms'] = int(match_bed.group(1))
    if match_other:
        features['num_other_rooms'] = int(match_other.group(1))

    # Bathrooms
    match_bath = re.search(r'(\d+)[\+]*\s*bathrooms?', text)
    if match_bath:
        features['num_bathrooms'] = int(match_bath.group(1))

    # Kitchen type
    kitchen_patterns = {
    'kitchen_open': r'\bopen kitchen\b',
    'kitchen_diner': r'\bkitchen diner\b',
    'kitchen_nook': r'\bkitchen nook\b',
    'kitchen_semi': r'\bsemi[-\s]?habitable kitchen\b',
    'kitchenette': r'\bkitchenette\b'}

    # Check each pattern
    for key, pattern in kitchen_patterns.items():
        if re.search(pattern, text):
            features[key] = 1
            features['kitchen_none'] = 0

    return pd.Series(features)

# Apply to both train and test sets
train = train.join(train['description'].apply(parse_description))
test = test.join(test['description'].apply(parse_description))


# In[223]:


# Check counts for each kitchen type
kitchen_counts_train = train[['kitchen_open', 'kitchen_diner', 'kitchen_nook', 'kitchen_semi', 'kitchenette', 'kitchen_none']].sum()
kitchen_counts_test = test[['kitchen_open', 'kitchen_diner', 'kitchen_nook', 'kitchen_semi', 'kitchenette', 'kitchen_none']].sum()

print("\nCounts of Kitchen Features (Train):")
print(kitchen_counts_train)

print("\nCounts of Kitchen Features (Test):")
kitchen_counts_test

# Note: If a description mentions multiple kitchen types (e.g., "open kitchen and kitchenette"),
# the current parsing will count it in multiple categories. This is expected based on
# how the binary flags are set in the `parse_description` function.
# 'kitchen_none' indicates that *none* of the specific types ('open', 'diner', 'nook', 'semi', 'kitchenette') were found.


# In[224]:


train.columns


# In[225]:


test.columns


# In[226]:


sample = train[['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms', 'num_bathrooms',
                'kitchen_open', 'kitchen_diner', 'kitchen_nook', 'kitchen_semi',
                'kitchenette', 'kitchen_none', 'suitable_disabled']].sample(10, random_state=42)

for i, row in sample.iterrows():
    print(f"📦 Description: {row['description']}")
    print(f"🔢 Parsed: Rooms={row['total_rooms']}, Bedrooms={row['num_bedrooms']}, Others={row['num_other_rooms']}, Bathrooms={row['num_bathrooms']}")
    print(f"🍳 Kitchens: Open={row['kitchen_open']}, Diner={row['kitchen_diner']}, Nook={row['kitchen_nook']}, Semi={row['kitchen_semi']}, Kitchenette={row['kitchenette']}, None={row['kitchen_none']}")
    print(f"♿ Suitable for Disabled: {row['suitable_disabled']}")
    print('-'*80)


# In[227]:


# Null total_rooms but valid description
fail_total = train[train['total_rooms'].isnull() & train['description'].notnull()]
print("Descriptions where total_rooms could not be parsed:")
display(fail_total[['description']].head(10))

# No kitchen type detected
fail_kitchen = train[
    (train[['kitchen_open', 'kitchen_diner', 'kitchen_nook', 'kitchen_semi', 'kitchenette']].sum(axis=1) == 0) &
    (train['kitchen_none'] == 0)
]
print("Descriptions where kitchen type parsing might have failed:")
display(fail_kitchen[['description']].head(10))


# In[228]:


mismatch = train[(train['total_rooms'].notnull()) &
                 (train['num_bedrooms'] + train['num_other_rooms'] != train['total_rooms'])]

print("Cases where sum(bedrooms + other) ≠ total_rooms:")
display(mismatch[['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].head(10))


# In[229]:


mismatch.count()


# In[230]:


mismatch['total_rooms'].mean()


# In[231]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distributions of total_rooms, num_bedrooms, and num_other_rooms
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(train['total_rooms'].dropna(), bins=15, kde=False, ax=axs[0], color='skyblue')
axs[0].set_title('Distribution of Total Rooms')
axs[0].set_xlabel('Total Rooms')

sns.histplot(train['num_bedrooms'].dropna(), bins=15, kde=False, ax=axs[1], color='lightgreen')
axs[1].set_title('Distribution of Number of Bedrooms')
axs[1].set_xlabel('Bedrooms')

sns.histplot(train['num_other_rooms'].dropna(), bins=15, kde=False, ax=axs[2], color='salmon')
axs[2].set_title('Distribution of Other Rooms')
axs[2].set_xlabel('Other Rooms')

plt.tight_layout()
plt.show()


# In[232]:


import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Identify missing room breakdowns
missing_test_mask = (
    (test['total_rooms'].notnull()) &
    (test['num_bedrooms'] == 0) &
    (test['num_other_rooms'] == 0)
)
num_missing_rows = missing_test_mask.sum()

# STEP 2: Plot total_rooms distribution before imputation
plt.figure(figsize=(7, 4))
sns.histplot(test.loc[missing_test_mask, 'total_rooms'], bins=10, kde=False, color='orange')
plt.title(f'Distribution of total_rooms for {num_missing_rows} rows\nmissing room breakdown (before imputation)')
plt.xlabel('Total Rooms')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# STEP 3: Apply imputation (1 other room, rest as bedrooms)
test.loc[missing_test_mask, 'num_other_rooms'] = 1
test.loc[missing_test_mask, 'num_bedrooms'] = test.loc[missing_test_mask, 'total_rooms'] - 1

# STEP 4: Plot imputed distributions
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(test.loc[missing_test_mask, 'num_bedrooms'], bins=10, ax=axs[0], color='green')
axs[0].set_title('Imputed Number of Bedrooms')
axs[0].set_xlabel('Bedrooms')

sns.histplot(test.loc[missing_test_mask, 'num_other_rooms'], bins=10, ax=axs[1], color='blue')
axs[1].set_title('Imputed Number of Other Rooms')
axs[1].set_xlabel('Other Rooms')

plt.tight_layout()
plt.show()


# In[233]:


train['num_bedrooms'].unique()


# In[234]:


train['num_other_rooms'].unique()


# In[235]:


test['num_other_rooms'].unique()


# In[236]:


test[(test['num_other_rooms'] == 8) | (test['num_other_rooms'] == 9)][['description', 'num_other_rooms']]


# In[237]:


test['num_bedrooms'].unique()


# In[238]:


# Find rows where the sum of bedrooms and other rooms exceeds the parsed total_rooms
room_sum_exceeds_total = (
    (test['num_bedrooms'] + test['num_other_rooms'] > test['total_rooms']) &
    (test['total_rooms'].notnull())
)

# Display those rows for inspection
invalid_totals_df = test[room_sum_exceeds_total][['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']]
print(invalid_totals_df)


# In[239]:


train.columns


# In[240]:


# Filter train and test separately where num_bedrooms + num_other_rooms > 5
train_5plus = train[(train['num_bedrooms'] + train['num_other_rooms']) > 5][
    ['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']
].copy()
train_5plus['source'] = 'train'

test_5plus = test[(test['num_bedrooms'] + test['num_other_rooms']) > 5][
    ['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']
].copy()
test_5plus['source'] = 'test'

# Display them separately
print("🔹 Train Set (Bedrooms + Other Rooms >= 5):")
print(train_5plus.reset_index(drop=True).to_string(index=False))

print("\n🔸 Test Set (Bedrooms + Other Rooms >= 5):")
print(test_5plus.reset_index(drop=True).to_string(index=False))


# In[241]:


# Impute total_rooms where it's NaN using num_bedrooms + num_other_rooms
train.loc[
    (train['total_rooms'].isna()) & ((train['num_bedrooms'] + train['num_other_rooms']) > 5),
    'total_rooms'
] = train['num_bedrooms'] + train['num_other_rooms']

test.loc[
    (test['total_rooms'].isna()) & ((test['num_bedrooms'] + test['num_other_rooms']) > 5),
    'total_rooms'
] = test['num_bedrooms'] + test['num_other_rooms']


# In[242]:


# Filter train and test separately where num_bedrooms + num_other_rooms > 5
train_5plus = train[(train['num_bedrooms'] + train['num_other_rooms']) > 5][
    ['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']
].copy()
train_5plus['source'] = 'train'

test_5plus = test[(test['num_bedrooms'] + test['num_other_rooms']) > 5][
    ['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']
].copy()
test_5plus['source'] = 'test'

# Display them separately
print("🔹 Train Set (Bedrooms + Other Rooms >= 5):")
print(train_5plus.reset_index(drop=True).to_string(index=False))

print("\n🔸 Test Set (Bedrooms + Other Rooms >= 5):")
print(test_5plus.reset_index(drop=True).to_string(index=False))


# In[243]:


# Impute total_rooms where it's NaN using num_bedrooms + num_other_rooms
train.loc[
    (train['total_rooms'].isna()) & ((train['num_bedrooms'] + train['num_other_rooms']) > 5),
    'total_rooms'
] = train['num_bedrooms'] + train['num_other_rooms']

test.loc[
    (test['total_rooms'].isna()) & ((test['num_bedrooms'] + test['num_other_rooms']) > 5),
    'total_rooms'
] = test['num_bedrooms'] + test['num_other_rooms']


# In[244]:


# Filter train and test separately where num_bedrooms + num_other_rooms > 5
train_5plus = train[(train['num_bedrooms'] + train['num_other_rooms']) > 5][
    ['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']
].copy()
train_5plus['source'] = 'train'

test_5plus = test[(test['num_bedrooms'] + test['num_other_rooms']) > 5][
    ['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']
].copy()
test_5plus['source'] = 'test'

# Display them separately
print("🔹 Train Set (Bedrooms + Other Rooms >= 5):")
print(train_5plus.reset_index(drop=True).to_string(index=False))

print("\n🔸 Test Set (Bedrooms + Other Rooms >= 5):")
print(test_5plus.reset_index(drop=True).to_string(index=False))


# In[245]:


print("Missing 'total_rooms' in Train:", train['total_rooms'].isnull().sum())
print("Missing 'total_rooms' in Test:", test['total_rooms'].isnull().sum())


# In[246]:


print("Missing 'num_bedrooms' in Train:", train['num_bedrooms'].isnull().sum())
print("Missing 'num_bedrooms' in Test:", test['num_bedrooms'].isnull().sum())


# In[247]:


print("Missing 'num_other_rooms' in Train:", train['num_other_rooms'].isnull().sum())
print("Missing 'num_other_rooms' in Test:", test['num_other_rooms'].isnull().sum())


# In[248]:


# Get rows where 'total_rooms' is missing in train
missing_total_rooms_train = train[train['total_rooms'].isnull()]
print("Rows in Train with Missing 'total_rooms':")
print(missing_total_rooms_train[['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].to_string())

print("\nDescription of Missing 'total_rooms' in Train:")
print(missing_total_rooms_train['total_rooms'].describe(include='all'))


# Get rows where 'total_rooms' is missing in test
missing_total_rooms_test = test[test['total_rooms'].isnull()]
print("\nRows in Test with Missing 'total_rooms':")
print(missing_total_rooms_test[['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].to_string())

print("\nDescription of Missing 'total_rooms' in Test:")
print(missing_total_rooms_test['total_rooms'].describe(include='all'))


# In[249]:


# Fix 1: If "1 room" is in description and both bedroom/other are 0 → assign 1 bedroom
for df in [train, test]:
    mask_room_only = (
        df['description'].str.lower().str.contains(r'\b1 room\b', na=False) &
        (df['num_bedrooms'] == 0) &
        (df['num_other_rooms'] == 0)
    )
    df.loc[mask_room_only, 'num_bedrooms'] = 1

# Fix 2: Impute missing total_rooms as the sum of bedrooms + other rooms
for df in [train, test]:
    mask_missing_total = df['total_rooms'].isna()
    df.loc[mask_missing_total, 'total_rooms'] = (
        df.loc[mask_missing_total, 'num_bedrooms'] + df.loc[mask_missing_total, 'num_other_rooms']
    )


# In[250]:


# Get rows where 'total_rooms' is missing in train
missing_total_rooms_train = train[train['total_rooms'].isnull()]
print("Rows in Train with Missing 'total_rooms':")
print(missing_total_rooms_train[['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].to_string())

print("\nDescription of Missing 'total_rooms' in Train:")
print(missing_total_rooms_train['total_rooms'].describe(include='all'))


# Get rows where 'total_rooms' is missing in test
missing_total_rooms_test = test[test['total_rooms'].isnull()]
print("\nRows in Test with Missing 'total_rooms':")
print(missing_total_rooms_test[['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].to_string())

print("\nDescription of Missing 'total_rooms' in Test:")
print(missing_total_rooms_test['total_rooms'].describe(include='all'))


# In[251]:


print("Missing 'total_rooms' in Train:", train['total_rooms'].isnull().sum())
print("Missing 'total_rooms' in Test:", test['total_rooms'].isnull().sum())


# In[252]:


# Check for inconsistencies: total_rooms not equal to num_bedrooms + num_other_rooms
train_inconsistent = train[
    train['total_rooms'] != (train['num_bedrooms'] + train['num_other_rooms'])
][['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']]

test_inconsistent = test[
    test['total_rooms'] != (test['num_bedrooms'] + test['num_other_rooms'])
][['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']]

# Display result
print("Inconsistent rows in TRAIN:")
print(train_inconsistent)

print("\nInconsistent rows in TEST:")
print(test_inconsistent)


# In[253]:


# Define the mask to select the rows that will be targeted for imputation
# This mask represents the rows *before* the imputation that have the issue
mask_to_impute_train = (
    train['total_rooms'].notnull() &  # total_rooms is known
    (train['total_rooms'] > 0) &      # And total_rooms is more than zero
    (train['num_bedrooms'] == 0) &    # Bedrooms is 0
    (train['num_other_rooms'] == 0)   # Other rooms is 0
)

mask_to_impute_test = (
    test['total_rooms'].notnull() &   # total_rooms is known
    (test['total_rooms'] > 0) &       # And total_rooms is more than zero
    (test['num_bedrooms'] == 0) &     # Bedrooms is 0
    (test['num_other_rooms'] == 0)    # Other rooms is 0
)

print("Rows in Train that will be targeted for imputation (before fix):")
# Display the rows that fit this pattern
display(train[mask_to_impute_train][['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].head())


print("\nRows in Test that will be targeted for imputation (before fix):")
# Display the rows that fit this pattern
display(test[mask_to_impute_test][['description', 'total_rooms', 'num_bedrooms', 'num_other_rooms']].head())


print("\nCounts of rows that will be targeted for imputation (before fix):")
print(f"Train: {mask_to_impute_train.sum()}")
print(f"Test: {mask_to_impute_test.sum()}")


# In[254]:


# Define the mask to select the rows that will be targeted for imputation
# This mask represents the rows *before* the imputation that have the issue
mask_to_impute_train = (
    train['total_rooms'].notnull() &  # total_rooms is known
    (train['total_rooms'] > 0) &      # And total_rooms is more than zero
    (train['num_bedrooms'] == 0) &    # Bedrooms is 0
    (train['num_other_rooms'] == 0)   # Other rooms is 0
)

mask_to_impute_test = (
    test['total_rooms'].notnull() &   # total_rooms is known
    (test['total_rooms'] > 0) &       # And total_rooms is more than zero
    (test['num_bedrooms'] == 0) &     # Bedrooms is 0
    (test['num_other_rooms'] == 0)    # Other rooms is 0
)

print("Applying imputation to the following number of rows:")
print(f"Train: {mask_to_impute_train.sum()}")
print(f"Test: {mask_to_impute_test.sum()}")

# Apply the imputation ONLY to the rows identified by the masks
# Imputation strategy: 1 other room, rest are bedrooms

# For Train:
train.loc[mask_to_impute_train, 'num_other_rooms'] = 1
train.loc[mask_to_impute_train, 'num_bedrooms'] = train.loc[mask_to_impute_train, 'total_rooms'] - 1

# For Test: (This won't change anything since the count is 0, but it's good practice)
test.loc[mask_to_impute_test, 'num_other_rooms'] = 1
test.loc[mask_to_impute_test, 'num_bedrooms'] = test.loc[mask_to_impute_test, 'total_rooms'] - 1

print("\nImputation applied.")


# ## 6 - Other Features

# In[255]:


train['other_features'].unique()


# In[256]:


train['other_features'].nunique()


# In[257]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Combine for analysis
combined_features = pd.concat([train['other_features'], test['other_features']], axis=0).fillna('')

# Tokenize using '|' separator
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'), binary=True)
X = vectorizer.fit_transform(combined_features)

# Extract feature names and frequencies
feature_names = [f.strip() for f in vectorizer.get_feature_names_out()]
feature_counts = X.sum(axis=0).A1  # convert to 1D array
features_df = pd.DataFrame({'feature': feature_names, 'count': feature_counts})
features_df.sort_values(by='count', ascending=False, inplace=True)

# Print number of unique features
print(f"Total unique features: {features_df.shape[0]}")

# Show top 20 most frequent features
print("\nTop 20 features:")
print(features_df.head(20))

# Plot
plt.figure(figsize=(12, 6))
features_df.head(20).plot.bar(x='feature', y='count', legend=False, color='skyblue')
plt.title('Top 20 Most Frequent Features in `other_features`')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[258]:


# How many features are mentioned per listing?
num_features_per_row = (X > 0).sum(axis=1).A1
plt.figure(figsize=(8, 4))
plt.hist(num_features_per_row, bins=range(1, 15), color='orange', edgecolor='black')
plt.title('Number of Features per Listing')
plt.xlabel('Features per Row')
plt.ylabel('Count')
plt.show()


# In[259]:


# Show all features, not just top 20
print("\nAll unique features and their counts:")
# Use to_string() to print the entire DataFrame without truncation
print(features_df.to_string())


# In[260]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

# Step 1: Cleaning function
def clean_token(token):
    token = token.lower().strip()
    token = re.sub(r'\s+', ' ', token)
    return token

# Step 2: Improved tokenizer
def smart_tokenizer(text):
    if pd.isna(text): return []

    tokens = []
    for raw in text.split('|'):
        raw = clean_token(raw)

        # Handle broken "window frames + exposure" tokens
        if 'window frames' in raw and 'exposure' in raw:
            match = re.search(r'(.*?)(exposure[\w\s,/-]*)', raw)
            if match:
                base, exposure = match.groups()
                if base.strip(): tokens.append(base.strip())
                if exposure.strip(): tokens.append(exposure.strip())
            else:
                tokens.append(raw)
        else:
            tokens.append(raw)

    return list(set(tokens))  # remove duplicates

# Step 3: Combine features for inspection only
combined = pd.concat([train['other_features'], test['other_features']], axis=0).fillna('')
vectorizer = CountVectorizer(tokenizer=smart_tokenizer, binary=True)
X_combined = vectorizer.fit_transform(combined)

# Step 4: Inspect the cleaned feature names and frequencies
feature_names = [clean_token(f) for f in vectorizer.get_feature_names_out()]
token_counts = X_combined.sum(axis=0).A1
features_df = pd.DataFrame({'feature': feature_names, 'count': token_counts})
features_df = features_df.sort_values(by='count', ascending=False).reset_index(drop=True)

# Step 5: Print summary
print(f"✅ Extracted {len(feature_names)} cleaned binary features.\n")
print("🔝 Features:")
print(features_df.to_string(index=False))


# In[261]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

# Step 1: Cleaning function
def clean_token(token):
    token = token.lower().strip()
    token = re.sub(r'\s+', ' ', token)
    return token

# Step 2: Smart tokenizer with directional exposure parsing
def smart_tokenizer(text):
    if pd.isna(text): return []

    tokens = []
    for raw in text.split('|'):
        raw = clean_token(raw)

        # Case 1: Malformed "window frames in ... exposure"
        if 'window frames' in raw and 'exposure' in raw:
            match = re.search(r'(.*?)(exposure[\w\s,/-]*)', raw)
            if match:
                base, exposure = match.groups()
                if base.strip():
                    tokens.append(clean_token(base))
                # Extract exposure directions from suffix
                directions = re.findall(r'north|south|east|west', exposure)
                for dir in directions:
                    tokens.append(f"exposure_{dir}")
            else:
                tokens.append(raw)

        # Case 2: Normal exposure strings like "exposure south, east"
        elif 'exposure' in raw:
            directions = re.findall(r'north|south|east|west', raw)
            for dir in directions:
                tokens.append(f"exposure_{dir}")

        else:
            tokens.append(raw)

    return list(set(tokens))  # remove duplicates

# Step 3: Combine features for inspection only
combined = pd.concat([train['other_features'], test['other_features']], axis=0).fillna('')
vectorizer = CountVectorizer(tokenizer=smart_tokenizer, binary=True)
X_combined = vectorizer.fit_transform(combined)

# Step 4: Inspect the cleaned feature names and frequencies
feature_names = [clean_token(f) for f in vectorizer.get_feature_names_out()]
token_counts = X_combined.sum(axis=0).A1
features_df = pd.DataFrame({'feature': feature_names, 'count': token_counts})
features_df = features_df.sort_values(by='count', ascending=False).reset_index(drop=True)

# Step 5: Print summary
print(f"✅ Extracted {len(feature_names)} cleaned binary features.\n")
print("🔝 Features:")
print(features_df.to_string(index=False))


# In[262]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

# Step 1: Cleaning function
def clean_token(token):
    token = token.lower().strip()
    token = re.sub(r'\s+', ' ', token)
    return token

# Step 2: Final tokenizer with aggressive window + exposure grouping
def smart_tokenizer(text):
    if pd.isna(text): return []

    tokens = []
    for raw in text.split('|'):
        raw = clean_token(raw)

        # 🌞 Exposure directions (independent or inside other strings)
        if 'exposure' in raw:
            directions = re.findall(r'north|south|east|west', raw)
            for dir in directions:
                tokens.append(f"exposure_{dir}")

        # 🪟 Window frame grouping
        if 'window frames' in raw:
            if 'double glass' in raw:
                tokens.append('window_double_glass')
            if 'triple glass' in raw:
                tokens.append('window_triple_glass')
            if 'pvc' in raw:
                tokens.append('window_pvc')
            if 'wood' in raw:
                tokens.append('window_wood')
            if 'metal' in raw:
                tokens.append('window_metal')

        # 🎯 Add general cleaned tokens (excluding raw window frame/exposure strings)
        if 'window frames' not in raw and 'exposure' not in raw:
            tokens.append(raw)

    return list(set(tokens))  # remove duplicates

# Step 3: Apply tokenizer to train + test combined
combined = pd.concat([train['other_features'], test['other_features']], axis=0).fillna('')
vectorizer = CountVectorizer(tokenizer=smart_tokenizer, binary=True)
X_combined = vectorizer.fit_transform(combined)

# Step 4: Summarize and display feature frequencies
feature_names = [clean_token(f) for f in vectorizer.get_feature_names_out()]
token_counts = X_combined.sum(axis=0).A1
features_df = pd.DataFrame({'feature': feature_names, 'count': token_counts})
features_df = features_df.sort_values(by='count', ascending=False).reset_index(drop=True)

# Step 5: Print summary
print(f"✅ Extracted {len(feature_names)} compact binary features.\n")
print("🔝 Features:")
print(features_df.to_string(index=False))


# In[263]:


# Step 1: Filter features that appear in at least 15 listings
min_count = 15
final_features = features_df[features_df['count'] >= min_count]['feature'].tolist()
print(f"✅ Retained {len(final_features)} features with count ≥ {min_count}")

# Step 2: Recreate DataFrames from X_combined using final features
# Rebuild full feature matrix DataFrame
X_full_df = pd.DataFrame(X_combined.toarray(), columns=feature_names)

# Step 3: Split back into aligned train/test
X_train_df = X_full_df.iloc[:len(train)][final_features].set_index(train.index)
X_test_df = X_full_df.iloc[len(train):][final_features].set_index(test.index)

# Step 4: Safely assign to original datasets
train = pd.concat([train, X_train_df], axis=1)
test = pd.concat([test, X_test_df], axis=1)

print("✅ Final filtered binary features successfully and safely added to train and test.")


# In[264]:


train.head()


# In[265]:


test.head()


# ## 7 - Condominimum

# In[266]:


train['condominium_fees'].isnull().sum()


# In[267]:


test['condominium_fees'].isnull().sum()


# In[271]:


# Group medians by elevator_sign, full day concierge, centralized tv system, and shared garden
group_medians = train.groupby(['elevator_sign', 'full day concierge', 'centralized tv system', 'shared garden'])['condominium_fees'].median().reset_index()

# Group counts for significance
group_counts = train.groupby(['elevator_sign', 'full day concierge', 'centralized tv system', 'shared garden']).size().reset_index(name='count')

# Merge
group_summary = pd.merge(group_medians, group_counts, on=['elevator_sign', 'full day concierge', 'centralized tv system', 'shared garden'])

# Correlation (numeric only)
# Include the new binary features and the encoded zone for a broader view
correlations = train[['condominium_fees', 'elevator_sign', 'full day concierge', 'centralized tv system', 'shared garden']].copy()
cor_matrix = correlations.corr(numeric_only=True)['condominium_fees'][1:] # Exclude self-correlation

print("📊 Group Median Summary by Features")
display(group_summary)

print("\n📈 Correlation with Condominium Fees:")
print(cor_matrix.round(4))


# In[272]:


# Group medians by elevator_sign, full day concierge, and shared garden
group_medians = train.groupby(['elevator_sign', 'full day concierge', 'shared garden'])['condominium_fees'].median().reset_index()

# Group counts for significance
group_counts = train.groupby(['elevator_sign', 'full day concierge', 'shared garden']).size().reset_index(name='count')

# Merge
group_summary = pd.merge(group_medians, group_counts, on=['elevator_sign', 'full day concierge', 'shared garden'])

# Correlation (numeric only)
# Include the remaining binary features and the encoded zone for a broader view
correlations = train[['condominium_fees', 'elevator_sign', 'full day concierge', 'shared garden']].copy()
cor_matrix = correlations.corr(numeric_only=True)['condominium_fees'][1:] # Exclude self-correlation

print("📊 Group Median Summary by Features")
display(group_summary)

print("\n📈 Correlation with Condominium Fees:")
print(cor_matrix.round(4))


# In[273]:


# Step 3: Define imputation function
def impute_condo_fees(row):
    if pd.isna(row['condominium_fees']):
        key = (row['elevator_sign'], row['full day concierge'], row['shared garden'])
        return group_medians.get(key, np.nan)
    return row['condominium_fees']

# Step 4: Apply imputation
train['condominium_fees'] = train.apply(impute_condo_fees, axis=1)
test['condominium_fees'] = test.apply(impute_condo_fees, axis=1)

# Step 5: Fallback to global median (no chained assignment)
overall_median = train['condominium_fees'].median()
train['condominium_fees'] = train['condominium_fees'].fillna(overall_median)
test['condominium_fees'] = test['condominium_fees'].fillna(overall_median)


# In[274]:


train['condominium_fees'].isnull().sum()


# In[275]:


test['condominium_fees'].isnull().sum()


# ## 8 - Zone

# In[276]:


train['zone'].nunique()


# In[277]:


# 1. Compute average rent (y) per zone from train
zone_target_mean = train.groupby('zone')['y'].mean()

# 2. Map it back to the train and test sets
train['zone_encoded'] = train['zone'].map(zone_target_mean)
test['zone_encoded'] = test['zone'].map(zone_target_mean)


# In[278]:


train[['zone_encoded', 'y']].corr()


# In[279]:


# Get unique zones and calculate median rent (y) for each zone in the training data
zone_median_rent = train.groupby('zone')['y'].median().reset_index()
zone_median_rent.columns = ['zone', 'median_rent']

# Get counts (observations) per zone in the training data
zone_counts = train.groupby('zone').size().reset_index(name='observations')

# Calculate mean rent (y) per zone in the training data
zone_mean_rent = train.groupby('zone')['y'].mean().reset_index()
zone_mean_rent.columns = ['zone', 'mean_rent']

# Combine the information
zone_summary = pd.merge(zone_median_rent, zone_mean_rent, on='zone')
zone_summary = pd.merge(zone_summary, zone_counts, on='zone')

# Sort by zone for consistency
zone_summary = zone_summary.sort_values(by='zone').reset_index(drop=True)

# Display the combined information for all 132 zones
print("📊 Zone Summary: Median Rent, Mean Rent, and Observations per Zone\n")
# Use to_string() to ensure all rows are displayed
print(zone_summary.to_string())

# Display the total number of zones
print(f"\nTotal unique zones: {len(zone_summary)}")


# In[280]:


# Step 1: Count observations
zone_counts = train['zone'].value_counts()
threshold = 5

# Step 2: Define frequent zones
frequent_zones = zone_counts[zone_counts >= threshold].index.tolist()

# Step 3: Group rare zones
train['zone_grouped'] = train['zone'].apply(lambda z: z if z in frequent_zones else 'zone_other')
test['zone_grouped'] = test['zone'].apply(lambda z: z if z in frequent_zones else 'zone_other')

# Step 4: Create zone_encoded (average rent by zone_grouped)
zone_avg = train.groupby('zone_grouped')['y'].mean()
train['zone_encoded'] = train['zone_grouped'].map(zone_avg)
test['zone_encoded'] = test['zone_grouped'].map(zone_avg)
test['zone_encoded'] = test['zone_encoded'].fillna(train['y'].mean())

# ✅ Step 5: Create zone_rent_per_sqm = avg(y / sqm) per grouped zone
train['rent_per_sqm'] = train['y'] / train['square_meters']
zone_rpsqm = train.groupby('zone_grouped')['rent_per_sqm'].mean()

train['zone_rent_per_sqm'] = train['zone_grouped'].map(zone_rpsqm)
test['zone_rent_per_sqm'] = test['zone_grouped'].map(zone_rpsqm)
test['zone_rent_per_sqm'] = test['zone_rent_per_sqm'].fillna(train['y'].mean() / train['square_meters'].mean())

# ✅ Step 6: Create estimated_zone_price = rent_per_sqm * sqm
train['estimated_zone_price'] = train['zone_rent_per_sqm'] * train['square_meters']
test['estimated_zone_price'] = test['zone_rent_per_sqm'] * test['square_meters']


# In[281]:


train.drop(columns=['rent_per_sqm', 'estimated_zone_price', 'zone_grouped'], inplace=True, errors='ignore')
test.drop(columns=['rent_per_sqm', 'estimated_zone_price', 'zone_grouped'], inplace=True, errors='ignore')


# ## 9 - Contract Type

# In[282]:


train['contract_type'].unique()


# In[283]:


# Aggregate both mean 'y' and count for each 'contract_type'
contract_type_stats = train.groupby('contract_type')['y'].agg(['mean', 'count']).reset_index()

# Rename the columns to be more descriptive
contract_type_stats.columns = ['contract_type', 'average_rent', 'num_listings']

print("Average Rent Price and Number of Listings by Contract Type:")
display(contract_type_stats)


# In[284]:


# Aggregate the count for each 'contract_type' by getting the size of each group
# This results in a Series, so we convert it back to a DataFrame and reset the index
contract_type_stats = test.groupby('contract_type').size().reset_index(name='num_listings')


# The columns are already correctly named if using reset_index(name=...)
# contract_type_stats.columns = ['contract_type', 'num_listings'] # This line is no longer needed

print("Number of Listings by Contract Type in Test Data:")
display(contract_type_stats)


# In[285]:


# Filter the train DataFrame to create the required DataFrames
student_contract = train[train['contract_type'] == 'rent | students (6 - 36 months)']
six_plus_six_contract = train[train['contract_type'] == 'rent | 6+6']
three_plus_two_contract = train[train['contract_type'] == 'rent | 3+2'] # Filter for 'rent | 3+2'

# Create a single figure for the plots
plt.figure(figsize=(18, 6))

# Plot histogram for 'rent | students (6 - 36 months)'
plt.subplot(1, 3, 1) # Changed to 1 row, 3 columns
plt.hist(student_contract['y'], bins=10, edgecolor='black')
plt.title("Rent Distribution: Students (6 - 36 months)")
plt.xlabel("Rent (€)")
plt.ylabel("Frequency")

# Plot histogram for 'rent | 6+6'
plt.subplot(1, 3, 2) # Changed to 1 row, 3 columns
plt.hist(six_plus_six_contract['y'], bins=10, edgecolor='black')
plt.title("Rent Distribution: 6+6 Contracts")
plt.xlabel("Rent (€)")
plt.ylabel("Frequency")

# Plot histogram for 'rent | 3+2'
plt.subplot(1, 3, 3) # Added a new subplot for the third distribution
plt.hist(three_plus_two_contract['y'], bins=10, edgecolor='black', color='purple') # Changed to histogram
plt.title("Rent Distribution: 3+2 Contracts")
plt.xlabel("Rent (€)")
plt.ylabel("Frequency") # Label changed back to Frequency

plt.tight_layout()
plt.show()


# In[286]:


# Step 1: Combine train and test for consistent dummy encoding
combined = pd.concat([train, test], axis=0, ignore_index=True)

# Step 2: One-hot encode the original contract_type (no grouping)
contract_dummies = pd.get_dummies(combined['contract_type'], prefix='contract')

# Step 3: Re-split dummies back into train and test sets
train_dummies = contract_dummies.iloc[:len(train)].reset_index(drop=True)
test_dummies = contract_dummies.iloc[len(train):].reset_index(drop=True)

# Step 4: Add dummies back to main train and test sets (modifies them in-place)
train = pd.concat([train.reset_index(drop=True), train_dummies], axis=1)
test = pd.concat([test.reset_index(drop=True), test_dummies], axis=1)


# In[287]:


train.columns


# ## 10 - Distance from center

# In[290]:

# Duomo coordinates
duomo_coords = (45.4642, 9.1900)
geolocator = Nominatim(user_agent="duomo_distance_calc")

# Combine zones to geocode only once
combined_zones = pd.concat([train['zone'], test['zone']]).dropna().unique()

# Geocoding helper function
def geocode_zone(zone):
    try:
        location = geolocator.geocode(f"{zone}, Milan, Italy")
        if location:
            return (location.latitude, location.longitude)
    except:
        pass

    cleaned = re.sub(r'[^\w\s]', '', unidecode(zone.lower()))
    try:
        location = geolocator.geocode(f"{cleaned}, Milan, Italy")
        if location:
            return (location.latitude, location.longitude)
    except:
        pass

    if '-' in zone:
        parts = [p.strip() for p in zone.split('-')]
        for part in parts:
            try:
                location = geolocator.geocode(f"{part}, Milan, Italy")
                if location:
                    return (location.latitude, location.longitude)
            except:
                pass
    return None

# Geocode all zones with rate limit
zone_coords = {}
for zone in combined_zones:
    zone_coords[zone] = geocode_zone(zone)
    time.sleep(1)

# Compute distance to Duomo
def compute_distance(zone):
    coords = zone_coords.get(zone)
    return geodesic(coords, duomo_coords).km if coords else None

# Add distance column directly to train and test
train['distance_to_duomo_km'] = train['zone'].apply(compute_distance)
test['distance_to_duomo_km'] = test['zone'].apply(compute_distance)

# Check remaining missing zones
print("🚨 Still missing in train:", train[train['distance_to_duomo_km'].isna()]['zone'].unique())
print("🚨 Still missing in test:", test[test['distance_to_duomo_km'].isna()]['zone'].unique())


# In[291]:


# Get unique zones from train and test dataframes
train_zones = train['zone'].unique()
test_zones = test['zone'].unique()

# Count unique zones in each dataset
unique_train_count = len(train_zones)
unique_test_count = len(test_zones)

print(f"Number of unique zones in train: {unique_train_count}")
print(f"Number of unique zones in test: {unique_test_count}")


# In[292]:


test['distance_to_duomo_km'].unique()


# In[293]:


# Get unique distances and their counts in the training dataset
unique_distances_train = train['distance_to_duomo_km'].value_counts().sort_index()

# Get unique distances and their counts in the test dataset
unique_distances_test = test['distance_to_duomo_km'].value_counts().sort_index()

# Filter for distances where the count (observations) is less than 5
low_observation_distances_train = unique_distances_train[unique_distances_train < 5]
low_observation_distances_test = unique_distances_test[unique_distances_test < 5]

print("Unique Distances to Duomo (km) with < 5 Observations in TRAIN:")
if not low_observation_distances_train.empty:
    print(low_observation_distances_train.to_string())
else:
    print("No distances found with less than 5 observations in TRAIN.")

print("\nUnique Distances to Duomo (km) with < 5 Observations in TEST:")
if not low_observation_distances_test.empty:
    print(low_observation_distances_test.to_string())
else:
    print("No distances found with less than 5 observations in TEST.")


# ## 11 - Save Preprocessed Data

# Keep only the meaningful interactions (based on prior importance analysis)

train['bathrooms_x_bedrooms'] = train['num_bathrooms'] * train['num_bedrooms']
train['zone_x_sqm'] = train['zone_encoded'] * train['square_meters']

test['bathrooms_x_bedrooms'] = test['num_bathrooms'] * test['num_bedrooms']
test['zone_x_sqm'] = test['zone_encoded'] * test['square_meters']

# Drop columns that are not needed for training
drop_cols = [
    "contract_type", "availability_encoded", "availability", "availability_date",
    "description", "other_features", "energy_efficiency_class",
    "energy_class_num", "total_rooms", "elevator", "floor",
    "conditions", "zone", "zone_grouped"
]

train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")


# Save cleaned and imputed datasets directly in the data folder
train.to_csv(DATA_DIR / "train_preprocessed.csv", index=False)
test.to_csv(DATA_DIR / "test_preprocessed.csv", index=False)

print("✅ Preprocessed datasets saved successfully:")
print(f"  • {DATA_DIR / 'train_preprocessed.csv'}")
print(f"  • {DATA_DIR / 'test_preprocessed.csv'}")
