import pandas as pd
import numpy as np

# ── Load cleaned dataset ──────────────────────────────────────────────
sold = pd.read_csv('sold_final.csv', encoding='latin-1')
print(f"Loaded: {len(sold)} rows, {len(sold.columns)} columns")

# Convert dates back to datetime (lost when saving to CSV)
date_cols = ['CloseDate', 'PurchaseContractDate', 
             'ListingContractDate', 'ContractStatusChangeDate']
for col in date_cols:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')

# ── WEEK 6: Feature Engineering ───────────────────────────────────────

# Price metrics
sold['price_ratio']          = sold['ClosePrice'] / sold['OriginalListPrice']
sold['price_per_sqft']       = sold['ClosePrice'] / sold['LivingArea']
sold['close_to_orig_ratio']  = sold['ClosePrice'] / sold['OriginalListPrice']

# Time series variables
sold['close_year']  = sold['CloseDate'].dt.year
sold['close_month'] = sold['CloseDate'].dt.month
sold['close_yrmo']  = sold['CloseDate'].dt.to_period('M').astype(str)

# Timeline metrics (in days)
sold['listing_to_contract_days'] = (
    sold['PurchaseContractDate'] - sold['ListingContractDate']
).dt.days

sold['contract_to_close_days'] = (
    sold['CloseDate'] - sold['PurchaseContractDate']
).dt.days

# ── Sample output ─────────────────────────────────────────────────────
print("\n=== ENGINEERED FEATURES (sample) ===")
sample_cols = ['ClosePrice', 'OriginalListPrice', 'LivingArea',
               'price_ratio', 'price_per_sqft', 'close_yrmo',
               'listing_to_contract_days', 'contract_to_close_days']
print(sold[sample_cols].head(10).to_string())

# ── Segment summary by PropertySubType ───────────────────────────────
print("\n=== SEGMENT SUMMARY BY PropertySubType ===")
seg = sold.groupby('PropertySubType').agg(
    count           = ('ClosePrice', 'count'),
    median_price    = ('ClosePrice', 'median'),
    avg_ppsf        = ('price_per_sqft', 'mean'),
    avg_price_ratio = ('price_ratio', 'mean'),
    avg_dom         = ('DaysOnMarket', 'mean')
).round(2)
print(seg.to_string())

# ── Segment summary by CountyOrParish ────────────────────────────────
print("\n=== SEGMENT SUMMARY BY CountyOrParish ===")
county = sold.groupby('CountyOrParish').agg(
    count           = ('ClosePrice', 'count'),
    median_price    = ('ClosePrice', 'median'),
    avg_ppsf        = ('price_per_sqft', 'mean'),
    avg_dom         = ('DaysOnMarket', 'mean')
).round(2).sort_values('median_price', ascending=False)
print(county.to_string())

# ── WEEK 7: Outlier Detection (IQR) ──────────────────────────────────
print("\n=== WEEK 7: OUTLIER DETECTION ===")

def flag_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    flag_col = f'outlier_{col}'
    df[flag_col] = (df[col] < lower) | (df[col] > upper)
    print(f"  {col}: lower={lower:,.0f}, upper={upper:,.0f}, "
          f"outliers={df[flag_col].sum()}")
    return df

for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
    sold = flag_outliers_iqr(sold, col)

# Before/after comparison
sold_filtered = sold[
    ~sold['outlier_ClosePrice'] & 
    ~sold['outlier_LivingArea'] & 
    ~sold['outlier_DaysOnMarket']
].copy()

print(f"\nBefore outlier removal: {len(sold)} rows")
print(f"After outlier removal:  {len(sold_filtered)} rows")
print(f"Removed: {len(sold) - len(sold_filtered)} records")

print("\n=== MEDIAN VALUES BEFORE vs AFTER OUTLIER REMOVAL ===")
for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
    before = sold[col].median()
    after  = sold_filtered[col].median()
    print(f"  {col}: {before:,.1f} → {after:,.1f}")

# ── Save both datasets ────────────────────────────────────────────────
sold.to_csv('sold_features_flagged.csv', index=False)
sold_filtered.to_csv('sold_features_clean.csv', index=False)
print("\nSaved: sold_features_flagged.csv (all records with outlier flags)")
print("Saved: sold_features_clean.csv (outliers removed, Tableau-ready)")