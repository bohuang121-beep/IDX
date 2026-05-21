import pandas as pd
import glob

# ── Load ALL monthly sold files automatically ─────────────────────────
sold_files    = glob.glob('CRMLSSold*.csv')
listing_files = glob.glob('CRMLSListing*.csv')

print(f"Found {len(sold_files)} sold files:")
for f in sorted(sold_files): print(f"  {f}")

print(f"\nFound {len(listing_files)} listing files:")
for f in sorted(listing_files): print(f"  {f}")

# ── Concatenate all months ────────────────────────────────────────────
sold_dfs     = []
listing_dfs  = []

for f in sorted(sold_files):
    df = pd.read_csv(f, encoding='latin-1')
    df['source_file'] = f
    sold_dfs.append(df)
    print(f"Loaded {f}: {len(df)} rows")

for f in sorted(listing_files):
    df = pd.read_csv(f, encoding='latin-1')
    df['source_file'] = f
    listing_dfs.append(df)
    print(f"Loaded {f}: {len(df)} rows")

sold_all     = pd.concat(sold_dfs,    ignore_index=True)
listings_all = pd.concat(listing_dfs, ignore_index=True)

print(f"\nCombined sold:     {len(sold_all)} rows")
print(f"Combined listings: {len(listings_all)} rows")

# ── Filter to Residential only ────────────────────────────────────────
sold_res     = sold_all[sold_all['PropertyType'] == 'Residential'].copy()
listings_res = listings_all[listings_all['PropertyType'] == 'Residential'].copy()

print(f"\nAfter Residential filter:")
print(f"Sold: {len(sold_res)}, Listings: {len(listings_res)}")

# ── FRED Mortgage Rate Merge ──────────────────────────────────────────
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url)
mortgage.columns = ['date', 'rate_30yr_fixed']
mortgage['date'] = pd.to_datetime(mortgage['date'])
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean().reset_index()
)
print(f"\nMortgage data loaded: {len(mortgage_monthly)} months")

# Create join keys
sold_res['year_month']     = pd.to_datetime(sold_res['CloseDate'],              errors='coerce').dt.to_period('M')
listings_res['year_month'] = pd.to_datetime(listings_res['ListingContractDate'], errors='coerce').dt.to_period('M')

# Merge
sold_res     = sold_res.merge(mortgage_monthly,     on='year_month', how='left')
listings_res = listings_res.merge(mortgage_monthly, on='year_month', how='left')

print(f"Null rates in sold:     {sold_res['rate_30yr_fixed'].isnull().sum()}")
print(f"Null rates in listings: {listings_res['rate_30yr_fixed'].isnull().sum()}")

# ── Data Cleaning ─────────────────────────────────────────────────────
# Convert date columns
date_cols = ['CloseDate', 'PurchaseContractDate',
             'ListingContractDate', 'ContractStatusChangeDate']
for col in date_cols:
    if col in sold_res.columns:
        sold_res[col] = pd.to_datetime(sold_res[col], errors='coerce')

# Drop >90% missing columns
missing_pct = sold_res.isnull().mean()
drop_cols   = missing_pct[missing_pct >= 0.90].index.tolist()
sold_clean  = sold_res.drop(columns=drop_cols)
print(f"\nDropped {len(drop_cols)} columns above 90% missing")
print(f"Columns remaining: {len(sold_clean.columns)}")

# Ensure numeric types
numeric_cols = ['ClosePrice', 'ListPrice', 'OriginalListPrice',
                'LivingArea', 'LotSizeAcres', 'BedroomsTotal',
                'BathroomsTotalInteger', 'DaysOnMarket', 'GarageSpaces']
for col in numeric_cols:
    if col in sold_clean.columns:
        sold_clean[col] = pd.to_numeric(sold_clean[col], errors='coerce')

# Remove invalid records
before = len(sold_clean)
sold_clean = sold_clean[sold_clean['ClosePrice']   > 0]
sold_clean = sold_clean[sold_clean['LivingArea']   > 0]
sold_clean = sold_clean[sold_clean['DaysOnMarket'] >= 0]
print(f"Removed {before - len(sold_clean)} invalid records")

# Geographic flags
sold_clean['flag_missing_coords'] = (
    sold_clean['Latitude'].isnull() | sold_clean['Longitude'].isnull()
)
sold_clean['flag_bad_longitude']  = sold_clean['Longitude'] > 0
sold_clean['flag_out_of_state']   = (
    (sold_clean['Latitude']  < 32)   | (sold_clean['Latitude']  > 42) |
    (sold_clean['Longitude'] < -124) | (sold_clean['Longitude'] > -114)
)
sold_clean.loc[sold_clean['Latitude'].isnull(), 'flag_out_of_state'] = False

# Date consistency flags
sold_clean['listing_after_close_flag']  = sold_clean['ListingContractDate']  > sold_clean['CloseDate']
sold_clean['purchase_after_close_flag'] = sold_clean['PurchaseContractDate'] > sold_clean['CloseDate']

print(f"Missing coords:       {sold_clean['flag_missing_coords'].sum()}")
print(f"Bad longitude:        {sold_clean['flag_bad_longitude'].sum()}")
print(f"Out of state:         {sold_clean['flag_out_of_state'].sum()}")
print(f"Listing after close:  {sold_clean['listing_after_close_flag'].sum()}")
print(f"Purchase after close: {sold_clean['purchase_after_close_flag'].sum()}")

# ── Feature Engineering ───────────────────────────────────────────────
sold_clean['price_ratio']              = sold_clean['ClosePrice'] / sold_clean['OriginalListPrice']
sold_clean['price_per_sqft']           = sold_clean['ClosePrice'] / sold_clean['LivingArea']
sold_clean['close_year']               = sold_clean['CloseDate'].dt.year
sold_clean['close_month']              = sold_clean['CloseDate'].dt.month
sold_clean['close_yrmo']               = sold_clean['CloseDate'].dt.to_period('M').astype(str)
sold_clean['listing_to_contract_days'] = (sold_clean['PurchaseContractDate'] - sold_clean['ListingContractDate']).dt.days
sold_clean['contract_to_close_days']   = (sold_clean['CloseDate'] - sold_clean['PurchaseContractDate']).dt.days

# ── Outlier Detection IQR ─────────────────────────────────────────────
def flag_outliers_iqr(df, col):
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[f'outlier_{col}'] = (df[col] < lower) | (df[col] > upper)
    print(f"  {col}: lower={lower:,.0f}, upper={upper:,.0f}, outliers={df[f'outlier_{col}'].sum()}")
    return df

print("\n=== OUTLIER FLAGS ===")
for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
    sold_clean = flag_outliers_iqr(sold_clean, col)

sold_filtered = sold_clean[
    ~sold_clean['outlier_ClosePrice']    &
    ~sold_clean['outlier_LivingArea']    &
    ~sold_clean['outlier_DaysOnMarket']
].copy()

print(f"\nFull dataset:     {len(sold_clean)} rows")
print(f"Filtered dataset: {len(sold_filtered)} rows")
print(f"Removed:          {len(sold_clean) - len(sold_filtered)} outliers")

print("\n=== MEDIAN BEFORE vs AFTER OUTLIER REMOVAL ===")
for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
    before = sold_clean[col].median()
    after  = sold_filtered[col].median()
    print(f"  {col}: {before:,.1f} → {after:,.1f}")

# ── Save outputs ──────────────────────────────────────────────────────
sold_clean.to_csv('sold_all_flagged.csv',    index=False)
sold_filtered.to_csv('sold_all_clean.csv',   index=False)
listings_res.to_csv('listings_all_clean.csv', index=False)

print("\n=== FINAL SUMMARY ===")
print(f"Date range: {sold_filtered['close_yrmo'].min()} → {sold_filtered['close_yrmo'].max()}")
print(f"Rows in Tableau-ready sold dataset: {len(sold_filtered)}")
print(f"\nSaved: sold_all_clean.csv      ← use this in Tableau")
print(f"Saved: sold_all_flagged.csv    ← all records with flags")
print(f"Saved: listings_all_clean.csv  ← listings for Tableau")
