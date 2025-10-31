import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HDB RESALE PRICE PREDICTION - IMPROVED VERSION")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/8] Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"[OK] Train: {train_df.shape[0]:,} records")
print(f"[OK] Test: {test_df.shape[0]:,} records")

# Load auxiliary data with geographic information
mrt_df = pd.read_csv('auxiliary-data/sg-mrt-stations.csv')
mall_df = pd.read_csv('auxiliary-data/sg-shopping-malls.csv')
primary_df = pd.read_csv('auxiliary-data/sg-primary-schools.csv')
secondary_df = pd.read_csv('auxiliary-data/sg-secondary-schools.csv')
hawker_df = pd.read_csv('auxiliary-data/sg-gov-hawkers.csv')

# Filter only operational MRT stations
mrt_df = mrt_df[mrt_df['STATUS'] == 'open'].reset_index(drop=True)

print(f"[OK] MRT stations (open): {len(mrt_df)}")
print(f"[OK] Shopping malls: {len(mall_df)}")
print(f"[OK] Primary schools: {len(primary_df)}")
print(f"[OK] Secondary schools: {len(secondary_df)}")
print(f"[OK] Hawker centers: {len(hawker_df)}")

# Separate target
y_train = train_df['RESALE_PRICE'].copy()
train_df = train_df.drop('RESALE_PRICE', axis=1)

# ============================================================================
# 2. GEOGRAPHIC FEATURE ENGINEERING (MEANINGFUL & INTERPRETABLE)
# ============================================================================
print("\n[2/8] Creating geographic features from auxiliary data...")
print("      (Using town-level amenity density and accessibility metrics)")

def calculate_town_geographic_features(df, mrt_df, mall_df, primary_df, secondary_df, hawker_df):
    """
    Create meaningful geographic features based on town-level amenity density.
    
    Rationale:
    - Properties near MRT stations are more valuable (better connectivity)
    - Shopping malls indicate commercial development (convenience)
    - Schools are important for families (education accessibility)
    - Hawkers provide affordable dining options (lifestyle)
    """
    df = df.copy()
    df['TOWN'] = df['TOWN'].str.lower().str.strip()
    
    def get_amenity_counts(amenity_df, col_name='PLANNING_AREA'):
        """Count amenities by planning area (town)"""
        if col_name not in amenity_df.columns:
            return {}
        amenity_df = amenity_df.copy()
        amenity_df[col_name] = amenity_df[col_name].str.lower().str.strip()
        return amenity_df[col_name].value_counts().to_dict()
    
    # Count amenities by town
    mrt_count = get_amenity_counts(mrt_df, 'PLANNING_AREA')
    mall_count = get_amenity_counts(mall_df, 'PLANNING_AREA')
    primary_count = get_amenity_counts(primary_df, 'PLANNING_AREA')
    secondary_count = get_amenity_counts(secondary_df, 'PLANNING_AREA')
    
    # MRT accessibility (most important for property value)
    df['MRT_COUNT'] = df['TOWN'].map(mrt_count).fillna(0)
    df['HAS_MRT'] = (df['MRT_COUNT'] > 0).astype(int)
    df['MRT_DENSITY_CATEGORY'] = pd.cut(df['MRT_COUNT'], 
                                        bins=[-1, 0, 2, 5, 100],
                                        labels=['none', 'low', 'medium', 'high']).astype(str)
    
    # Shopping mall accessibility (commercial development indicator)
    df['MALL_COUNT'] = df['TOWN'].map(mall_count).fillna(0)
    df['HAS_MALL'] = (df['MALL_COUNT'] > 0).astype(int)
    df['IS_COMMERCIAL_HUB'] = (df['MALL_COUNT'] >= 3).astype(int)  # 3+ malls = commercial hub
    
    # School accessibility (important for families)
    df['PRIMARY_SCHOOL_COUNT'] = df['TOWN'].map(primary_count).fillna(0)
    df['SECONDARY_SCHOOL_COUNT'] = df['TOWN'].map(secondary_count).fillna(0)
    df['TOTAL_SCHOOL_COUNT'] = df['PRIMARY_SCHOOL_COUNT'] + df['SECONDARY_SCHOOL_COUNT']
    df['SCHOOL_DENSITY_CATEGORY'] = pd.cut(df['TOTAL_SCHOOL_COUNT'],
                                           bins=[-1, 3, 6, 10, 100],
                                           labels=['low', 'medium', 'high', 'very_high']).astype(str)
    
    # Education hub indicator (universities, polytechnics attract students/investors)
    education_hubs = ['bukit timah', 'queenstown', 'clementi', 'pasir ris', 'tampines']
    df['IS_EDUCATION_HUB'] = df['TOWN'].isin(education_hubs).astype(int)
    
    # Overall amenity score (weighted by importance)
    # Weights based on research: MRT > Malls > Schools
    df['AMENITY_SCORE'] = (
        df['MRT_COUNT'] * 4.0 +           # MRT most important
        df['MALL_COUNT'] * 2.5 +          # Malls second
        df['PRIMARY_SCHOOL_COUNT'] * 1.0 + # Schools moderate importance
        df['SECONDARY_SCHOOL_COUNT'] * 1.0
    )
    
    # Normalize amenity score (0-100 scale for interpretability)
    if df['AMENITY_SCORE'].max() > 0:
        df['AMENITY_SCORE_NORMALIZED'] = (df['AMENITY_SCORE'] / df['AMENITY_SCORE'].max() * 100).round(2)
    else:
        df['AMENITY_SCORE_NORMALIZED'] = 0
    
    # Connectivity index (combination of transport and commercial)
    df['CONNECTIVITY_INDEX'] = (
        df['HAS_MRT'] * 50 + 
        df['MRT_COUNT'] * 10 + 
        df['HAS_MALL'] * 30 + 
        df['IS_COMMERCIAL_HUB'] * 20
    )
    
    # Family-friendly score (schools + parks/amenities)
    df['FAMILY_FRIENDLY_SCORE'] = (
        df['PRIMARY_SCHOOL_COUNT'] * 3 + 
        df['SECONDARY_SCHOOL_COUNT'] * 2 +
        df['HAS_MALL'] * 5  # Malls often have family facilities
    )
    
    # Livability score (comprehensive quality of life indicator)
    df['LIVABILITY_SCORE'] = (
        df['AMENITY_SCORE_NORMALIZED'] * 0.5 +
        df['CONNECTIVITY_INDEX'] * 0.3 +
        df['FAMILY_FRIENDLY_SCORE'] * 0.2
    )
    
    return df

train_df = calculate_town_geographic_features(train_df, mrt_df, mall_df, primary_df, secondary_df, hawker_df)
test_df = calculate_town_geographic_features(test_df, mrt_df, mall_df, primary_df, secondary_df, hawker_df)

print("[OK] Created 15+ geographic features based on amenity accessibility")

# ============================================================================
# 3. TEMPORAL FEATURES (MARKET TRENDS)
# ============================================================================
print("\n[3/8] Creating temporal features...")
print("      (Capturing market trends and seasonal patterns)")

def create_temporal_features(df):
    """
    Time-based features capturing market conditions and trends.
    
    Rationale:
    - Singapore property market has clear trends (bull/bear markets)
    - Seasonal patterns exist (peak buying seasons)
    - Recent transactions reflect current market sentiment
    """
    df = df.copy()
    
    df['MONTH'] = pd.to_datetime(df['MONTH'])
    df['YEAR'] = df['MONTH'].dt.year
    df['MONTH_NUM'] = df['MONTH'].dt.month
    df['QUARTER'] = df['MONTH'].dt.quarter
    df['YEAR_MONTH'] = df['YEAR'] * 12 + df['MONTH_NUM']
    
    # Market period classification (based on Singapore housing market history)
    # 2017-2019: Steady
    # 2020-2021: COVID dip then recovery
    # 2022-2025: Strong growth
    df['MARKET_PERIOD'] = 'stable'
    df.loc[df['YEAR'].isin([2020, 2021]), 'MARKET_PERIOD'] = 'covid'
    df.loc[df['YEAR'] >= 2022, 'MARKET_PERIOD'] = 'growth'
    
    # Cyclical features (seasonal patterns)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)
    
    # Peak buying seasons (Chinese New Year, mid-year, year-end)
    df['IS_PEAK_SEASON'] = df['MONTH_NUM'].isin([1, 2, 6, 7, 11, 12]).astype(int)
    
    # Years since 2017 (linear time trend)
    df['YEARS_SINCE_2017'] = df['YEAR'] - 2017
    
    return df

train_df = create_temporal_features(train_df)
test_df = create_temporal_features(test_df)

print("[OK] Created temporal features capturing market dynamics")

# ============================================================================
# 4. PROPERTY CHARACTERISTICS FEATURES
# ============================================================================
print("\n[4/8] Creating property characteristic features...")
print("      (Physical attributes affecting property value)")

def create_property_features(df):
    """
    Property-specific features based on physical characteristics.
    
    Rationale:
    - Larger flats command premium per sqm
    - Remaining lease crucial (99-year leases depreciate)
    - Floor level affects views, privacy, and price
    - Flat model indicates quality and amenities
    """
    df = df.copy()
    
    # ========== Lease Features ==========
    df['LEASE_COMMENCE_DATA'] = pd.to_numeric(df['LEASE_COMMENCE_DATA'], errors='coerce')
    df['LEASE_AGE'] = df['YEAR'] - df['LEASE_COMMENCE_DATA']
    df['REMAINING_LEASE'] = 99 - df['LEASE_AGE']
    
    # Lease depreciation stages (non-linear relationship)
    df['LEASE_STAGE'] = 'prime'  # 80-99 years: Full value
    df.loc[df['REMAINING_LEASE'] < 80, 'LEASE_STAGE'] = 'mature'  # 60-80: Slight discount
    df.loc[df['REMAINING_LEASE'] < 60, 'LEASE_STAGE'] = 'aging'   # 40-60: Moderate discount
    df.loc[df['REMAINING_LEASE'] < 40, 'LEASE_STAGE'] = 'short'   # <40: Heavy discount
    
    # Lease ratios for modeling
    df['REMAINING_LEASE_RATIO'] = df['REMAINING_LEASE'] / 99
    df['LEASE_AGE_SQUARED'] = df['LEASE_AGE'] ** 2
    df['REMAINING_LEASE_SQUARED'] = df['REMAINING_LEASE'] ** 2
    
    # Critical lease threshold (below 60 years affects resale significantly)
    df['BELOW_60_YEARS'] = (df['REMAINING_LEASE'] < 60).astype(int)
    df['BELOW_40_YEARS'] = (df['REMAINING_LEASE'] < 40).astype(int)
    
    # ========== Floor Features ==========
    floor_range_map = {}
    for fr in df['FLOOR_RANGE'].unique():
        if pd.isna(fr):
            floor_range_map[fr] = (np.nan, np.nan, np.nan)
        else:
            parts = str(fr).lower().replace('to', ' ').split()
            if len(parts) >= 2:
                try:
                    low = int(parts[0])
                    high = int(parts[1])
                    mid = (low + high) / 2
                    floor_range_map[fr] = (low, high, mid)
                except:
                    floor_range_map[fr] = (np.nan, np.nan, np.nan)
            else:
                floor_range_map[fr] = (np.nan, np.nan, np.nan)
    
    df['FLOOR_LOW'] = df['FLOOR_RANGE'].map(lambda x: floor_range_map[x][0])
    df['FLOOR_HIGH'] = df['FLOOR_RANGE'].map(lambda x: floor_range_map[x][1])
    df['FLOOR_MID'] = df['FLOOR_RANGE'].map(lambda x: floor_range_map[x][2])
    
    # Floor categories (different price dynamics)
    df['FLOOR_CATEGORY'] = 'mid'
    df.loc[df['FLOOR_LOW'] <= 3, 'FLOOR_CATEGORY'] = 'low'        # Ground: Less privacy
    df.loc[df['FLOOR_LOW'] >= 10, 'FLOOR_CATEGORY'] = 'high'      # High: Better views
    df.loc[df['FLOOR_LOW'] >= 20, 'FLOOR_CATEGORY'] = 'very_high' # Very high: Premium views
    
    df['IS_HIGH_FLOOR'] = (df['FLOOR_LOW'] >= 10).astype(int)
    df['IS_VERY_HIGH_FLOOR'] = (df['FLOOR_LOW'] >= 20).astype(int)
    df['IS_LOW_FLOOR'] = (df['FLOOR_LOW'] <= 3).astype(int)
    df['FLOOR_SQUARED'] = df['FLOOR_MID'] ** 2
    
    # ========== Area Features ==========
    df['FLOOR_AREA_SQM'] = pd.to_numeric(df['FLOOR_AREA_SQM'], errors='coerce')
    
    # Room count from flat type
    room_map = {
        '1 room': 1, '2 room': 2, '3 room': 3, '4 room': 4, '5 room': 5,
        '5-room': 5, 'executive': 6, 'multi-generation': 7,
        '3-room': 3, '4-room': 4
    }
    df['NUM_ROOMS'] = df['FLAT_TYPE'].str.lower().map(room_map).fillna(4)
    
    # Area metrics (spaciousness indicators)
    df['AREA_PER_ROOM'] = df['FLOOR_AREA_SQM'] / df['NUM_ROOMS']
    df['IS_SPACIOUS'] = (df['AREA_PER_ROOM'] >= 30).astype(int)  # 30+ sqm per room is spacious
    
    # Area transformations for non-linear relationships
    df['AREA_SQUARED'] = df['FLOOR_AREA_SQM'] ** 2
    df['AREA_SQRT'] = np.sqrt(df['FLOOR_AREA_SQM'])
    df['AREA_LOG'] = np.log1p(df['FLOOR_AREA_SQM'])
    
    # Size categories
    df['SIZE_CATEGORY'] = 'medium'
    df.loc[df['FLOOR_AREA_SQM'] < 70, 'SIZE_CATEGORY'] = 'small'
    df.loc[df['FLOOR_AREA_SQM'] >= 100, 'SIZE_CATEGORY'] = 'large'
    df.loc[df['FLOOR_AREA_SQM'] >= 130, 'SIZE_CATEGORY'] = 'extra_large'
    
    # ========== Flat Model Features ==========
    df['FLAT_MODEL'] = df['FLAT_MODEL'].str.lower().str.strip()
    
    # Premium models (DBSS, premium apartments have better finishes)
    premium_models = ['dbss', 'premium apartment', 'premium maisonette', 
                      'premium apartment loft', 'type s1', 'type s2']
    df['IS_PREMIUM_MODEL'] = df['FLAT_MODEL'].isin(premium_models).astype(int)
    
    # Executive/Maisonette (larger units with special layouts)
    executive_models = ['executive', 'maisonette', 'multi-generation']
    df['IS_EXECUTIVE'] = df['FLAT_MODEL'].str.contains('|'.join(executive_models), na=False).astype(int)
    
    # Modern models (newer designs, better layouts)
    modern_models = ['model a', 'premium apartment', 'dbss', 'improved', 'type s1', 'type s2']
    df['IS_MODERN_MODEL'] = df['FLAT_MODEL'].isin(modern_models).astype(int)
    
    return df

train_df = create_property_features(train_df)
test_df = create_property_features(test_df)

print("[OK] Created comprehensive property features")

# ============================================================================
# 5. LOCATION FEATURES
# ============================================================================
print("\n[5/8] Creating location features...")
print("      (Regional characteristics and town attributes)")

def create_location_features(df):
    """
    Location-based features capturing regional characteristics.
    
    Rationale:
    - Central locations command significant premium (proximity to CBD)
    - Mature estates have established infrastructure
    - Some towns are more prestigious than others
    """
    df = df.copy()
    df['TOWN'] = df['TOWN'].str.lower().str.strip()
    
    # Central locations (proximity to CBD, Orchard, Marina Bay)
    central_towns = ['bishan', 'bukit merah', 'bukit timah', 'central area', 
                     'geylang', 'kallang/whampoa', 'marine parade', 'queenstown', 
                     'toa payoh', 'downtown core']
    df['IS_CENTRAL'] = df['TOWN'].isin(central_towns).astype(int)
    
    # Mature estates (established since 1970s-1980s, better infrastructure)
    mature_estates = ['ang mo kio', 'bedok', 'bishan', 'bukit merah', 'bukit timah',
                      'central area', 'clementi', 'geylang', 'kallang/whampoa',
                      'marine parade', 'pasir ris', 'queenstown', 'serangoon',
                      'tampines', 'toa payoh']
    df['IS_MATURE'] = df['TOWN'].isin(mature_estates).astype(int)
    
    # Waterfront/Coastal towns (sea views, recreational areas)
    waterfront_towns = ['bedok', 'marine parade', 'pasir ris', 'punggol', 
                        'sembawang', 'tampines']
    df['IS_WATERFRONT'] = df['TOWN'].isin(waterfront_towns).astype(int)
    
    # Northern region (generally more affordable)
    northern_towns = ['sembawang', 'woodlands', 'yishun', 'admiralty']
    df['IS_NORTHERN'] = df['TOWN'].isin(northern_towns).astype(int)
    
    # Eastern region (balanced pricing, near airport/Changi)
    eastern_towns = ['bedok', 'pasir ris', 'tampines']
    df['IS_EASTERN'] = df['TOWN'].isin(eastern_towns).astype(int)
    
    # Western region (developing areas)
    western_towns = ['bukit batok', 'bukit panjang', 'choa chu kang', 'clementi',
                     'jurong east', 'jurong west']
    df['IS_WESTERN'] = df['TOWN'].isin(western_towns).astype(int)
    
    # Prestigious towns (historically higher prices)
    prestigious_towns = ['bishan', 'bukit timah', 'marine parade', 'queenstown', 'central area']
    df['IS_PRESTIGIOUS'] = df['TOWN'].isin(prestigious_towns).astype(int)
    
    # Location desirability score
    df['LOCATION_SCORE'] = (
        df['IS_CENTRAL'] * 50 +
        df['IS_MATURE'] * 30 +
        df['IS_PRESTIGIOUS'] * 40 +
        df['IS_WATERFRONT'] * 20
    )
    
    # Block and street features
    df['STREET'] = df['STREET'].str.lower().str.strip()
    df['BLOCK'] = df['BLOCK'].astype(str).str.strip()
    df['BLOCK_NUM'] = df['BLOCK'].str.extract(r'(\d+)')[0].astype(float)
    df['HAS_BLOCK_SUFFIX'] = df['BLOCK'].str.contains('[A-Za-z]', na=False).astype(int)
    
    return df

train_df = create_location_features(train_df)
test_df = create_location_features(test_df)

print("[OK] Created location-based features")

# ============================================================================
# 6. MEANINGFUL INTERACTION FEATURES
# ============================================================================
print("\n[6/8] Creating interaction features...")
print("      (Capturing synergistic effects between features)")

def create_interaction_features(df):
    """
    Interaction features capturing combined effects.
    
    Rationale:
    - Large flats in central locations have exponential premiums
    - New flats with good connectivity are highly valuable
    - Floor premium varies by property size
    """
    df = df.copy()
    
    # Size × Location (large flats in good locations = premium)
    df['AREA_X_CENTRAL'] = df['FLOOR_AREA_SQM'] * df['IS_CENTRAL']
    df['AREA_X_MATURE'] = df['FLOOR_AREA_SQM'] * df['IS_MATURE']
    df['AREA_X_PRESTIGIOUS'] = df['FLOOR_AREA_SQM'] * df['IS_PRESTIGIOUS']
    df['AREA_X_LOCATION_SCORE'] = df['FLOOR_AREA_SQM'] * df['LOCATION_SCORE']
    
    # Size × Amenities (spacious flats near amenities)
    df['AREA_X_MRT'] = df['FLOOR_AREA_SQM'] * df['MRT_COUNT']
    df['AREA_X_MALL'] = df['FLOOR_AREA_SQM'] * df['MALL_COUNT']
    df['AREA_X_AMENITY_SCORE'] = df['FLOOR_AREA_SQM'] * df['AMENITY_SCORE_NORMALIZED']
    df['AREA_X_CONNECTIVITY'] = df['FLOOR_AREA_SQM'] * df['CONNECTIVITY_INDEX']
    
    # Lease × Location (newer flats in prime locations)
    df['LEASE_X_CENTRAL'] = df['REMAINING_LEASE'] * df['IS_CENTRAL']
    df['LEASE_X_MATURE'] = df['REMAINING_LEASE'] * df['IS_MATURE']
    df['LEASE_X_MRT'] = df['REMAINING_LEASE'] * df['MRT_COUNT']
    df['LEASE_X_AMENITY'] = df['REMAINING_LEASE'] * df['AMENITY_SCORE_NORMALIZED']
    
    # Lease × Size (larger flats with longer leases)
    df['LEASE_X_AREA'] = df['REMAINING_LEASE'] * df['FLOOR_AREA_SQM']
    df['LEASE_RATIO_X_AREA'] = df['REMAINING_LEASE_RATIO'] * df['FLOOR_AREA_SQM']
    
    # Floor × Size (high floor large flats)
    df['FLOOR_X_AREA'] = df['FLOOR_MID'] * df['FLOOR_AREA_SQM']
    df['HIGH_FLOOR_X_AREA'] = df['IS_HIGH_FLOOR'] * df['FLOOR_AREA_SQM']
    
    # Premium model × Size
    df['PREMIUM_X_AREA'] = df['IS_PREMIUM_MODEL'] * df['FLOOR_AREA_SQM']
    df['PREMIUM_X_CENTRAL'] = df['IS_PREMIUM_MODEL'] * df['IS_CENTRAL']
    
    # Rooms × Area (spaciousness)
    df['AREA_X_ROOMS'] = df['FLOOR_AREA_SQM'] * df['NUM_ROOMS']
    df['ROOMS_X_CENTRAL'] = df['NUM_ROOMS'] * df['IS_CENTRAL']
    
    # Complex interactions (triple)
    df['AREA_X_LEASE_X_CENTRAL'] = df['FLOOR_AREA_SQM'] * df['REMAINING_LEASE'] * df['IS_CENTRAL']
    df['AREA_X_LEASE_X_MRT'] = df['FLOOR_AREA_SQM'] * df['REMAINING_LEASE'] * df['MRT_COUNT']
    
    # Value density (composite score)
    df['VALUE_COMPOSITE'] = (
        df['FLOOR_AREA_SQM'] * 
        df['REMAINING_LEASE_RATIO'] * 
        (df['LOCATION_SCORE'] + 1) *
        (df['AMENITY_SCORE_NORMALIZED'] + 1)
    )
    
    return df

train_df = create_interaction_features(train_df)
test_df = create_interaction_features(test_df)

print("[OK] Created meaningful interaction features")

# ============================================================================
# 7. STATISTICAL AGGREGATION FEATURES
# ============================================================================
print("\n[7/8] Creating statistical aggregation features...")
print("      (Town and street-level price patterns)")

def create_statistical_features(train, test, y):
    """
    Statistical features capturing local market trends.
    
    Rationale:
    - Specific streets/towns have consistent price patterns
    - Historical prices in area indicate current value
    - Cross-validation prevents overfitting
    """
    train = train.copy()
    test = test.copy()
    
    # Prepare for target encoding
    y_series = pd.Series(y.values, index=train.index, name='target')
    
    # High-cardinality features for target encoding
    target_encode_cols = ['TOWN', 'STREET', 'FLAT_MODEL', 'BLOCK', 
                         'FLOOR_CATEGORY', 'LEASE_STAGE', 'MARKET_PERIOD']
    
    print(f"   - Applying target encoding to {len(target_encode_cols)} categorical features...")
    
    for col in target_encode_cols:
        if col + '_ENCODED' not in train.columns:
            continue
            
        encoded_col = col + '_ENCODED'
        train[col + '_TARGET_MEAN'] = 0.0
        
        # K-Fold target encoding (prevents overfitting)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(train):
            target_fold = y_series.iloc[train_idx]
            means = train.iloc[train_idx].groupby(encoded_col).apply(
                lambda x: target_fold.loc[x.index].mean()
            )
            global_mean = target_fold.mean()
            train.loc[train.index[val_idx], col + '_TARGET_MEAN'] = \
                train.iloc[val_idx][encoded_col].map(means).fillna(global_mean)
        
        # For test, use all training data
        means = train.groupby(encoded_col).apply(lambda x: y_series.loc[x.index].mean())
        global_mean = y_series.mean()
        test[col + '_TARGET_MEAN'] = test[encoded_col].map(means).fillna(global_mean)
        
        # Also calculate std dev (price volatility in area)
        stds = train.groupby(encoded_col).apply(lambda x: y_series.loc[x.index].std())
        train[col + '_TARGET_STD'] = train[encoded_col].map(stds).fillna(y_series.std())
        test[col + '_TARGET_STD'] = test[encoded_col].map(stds).fillna(y_series.std())
    
    print(f"   [OK] Created {len(target_encode_cols) * 2} target-encoded features")
    
    return train, test

# ============================================================================
# 8. ENCODING AND PREPARATION
# ============================================================================
print("\n[8/8] Encoding categorical features...")

# Label encode categorical columns
categorical_cols = ['TOWN', 'FLAT_TYPE', 'FLAT_MODEL', 'ECO_CATEGORY', 'FLOOR_RANGE', 
                   'STREET', 'BLOCK', 'FLOOR_CATEGORY', 'LEASE_STAGE', 'SIZE_CATEGORY',
                   'MRT_DENSITY_CATEGORY', 'SCHOOL_DENSITY_CATEGORY', 'MARKET_PERIOD']

label_encoders = {}
for col in categorical_cols:
    if col in train_df.columns:
        le = LabelEncoder()
        combined = pd.concat([
            train_df[col].fillna('missing').astype(str),
            test_df[col].fillna('missing').astype(str)
        ])
        le.fit(combined)
        
        train_df[col + '_ENCODED'] = le.transform(train_df[col].fillna('missing').astype(str))
        test_df[col + '_ENCODED'] = le.transform(test_df[col].fillna('missing').astype(str))
        label_encoders[col] = le

# Apply statistical features
train_df, test_df = create_statistical_features(train_df, test_df, y_train)

# Drop original categorical columns and MONTH
drop_cols = ['MONTH'] + categorical_cols
train_df = train_df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)

# Final preparation
train_df = train_df.fillna(-999)
test_df = test_df.fillna(-999)
train_df.columns = train_df.columns.astype(str)
test_df.columns = test_df.columns.astype(str)

print(f"[OK] Final feature count: {train_df.shape[1]}")
print(f"[OK] Ready for modeling")

# ============================================================================
# 9. TRAIN OPTIMIZED ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("TRAINING LIGHTGBM ENSEMBLE")
print("="*80)

# Optimized parameters based on feature characteristics
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 127,
    'max_depth': 10,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'min_split_gain': 0.01,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'force_col_wise': True
}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

models = []
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
    print(f"\nFold {fold}/{n_folds}")
    print("-" * 40)
    
    X_train = train_df.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val = train_df.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train_fold)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=4000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=500)
        ]
    )
    
    models.append(model)
    
    oof_predictions[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_predictions += model.predict(test_df, num_iteration=model.best_iteration) / n_folds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
    fold_scores.append(fold_rmse)
    print(f"Fold {fold} RMSE: ${fold_rmse:,.2f}")

cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: ${score:,.2f}")
print(f"\nMean CV RMSE: ${np.mean(fold_scores):,.2f}")
print(f"Std CV RMSE:  ${np.std(fold_scores):,.2f}")
print(f"\nOverall CV RMSE: ${cv_rmse:,.2f}")

# ============================================================================
# 10. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("TOP 30 MOST IMPORTANT FEATURES")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': train_df.columns,
    'importance': np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

for idx, row in feature_importance.head(30).iterrows():
    print(f"{row['feature']:.<50} {row['importance']:.2e}")

# ============================================================================
# 11. GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'Predicted': test_predictions
})

submission.to_csv('submission_improved.csv', index=False)

print(f"[OK] Submission saved: submission_improved.csv")
print(f"[OK] Number of predictions: {len(submission):,}")
print(f"\nPrediction statistics:")
print(f"  Mean:   ${submission['Predicted'].mean():,.2f}")
print(f"  Median: ${submission['Predicted'].median():,.2f}")
print(f"  Std:    ${submission['Predicted'].std():,.2f}")
print(f"  Min:    ${submission['Predicted'].min():,.2f}")
print(f"  Max:    ${submission['Predicted'].max():,.2f}")

print("\n" + "="*80)
print("[SUCCESS] COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"Cross-Validation RMSE: ${cv_rmse:,.2f}")
print(f"Submission File: submission_improved.csv")
print("="*80)

