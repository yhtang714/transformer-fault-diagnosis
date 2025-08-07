import pandas as pd
import numpy as np
from config import gas_columns

# Load pure data to compute correlations (used only for imputation)
df1 = pd.read_csv(r'/Users/yinghuatang/PycharmProjects/pythonProject/electricity/python/pure_data.csv')
df1 = df1.dropna(subset=gas_columns, how='all')

# Normalize to proportions
def normalize_proportions(df):
    total = df[gas_columns].sum(axis=1)
    total[total == 0] = 1e-10
    return df[gas_columns].div(total, axis=0)

# Correlation matrix from pure data
pure_props = normalize_proportions(df1)
correlation_matrix = pure_props.corr().fillna(0)

# Representative midpoints for 4 intervals (levels 0–3)
level_midpoints = [0.04, 0.135, 0.245, 0.65]

# Section thresholds from config logic
section_limits = {
    'H2': [0.488, 0.388, 0.288],
    'CH4': [0.388, 0.288, 0.188],
    'C2H2': [0.278, 0.178, 0.078],
    'C2H4': [0.288, 0.188, 0.088],
    'C2H6': [0.288, 0.188, 0.088]
}

def prop_to_level(gas, prop):
    limits = section_limits[gas]
    if prop >= limits[0]: return 3
    elif prop >= limits[1]: return 2
    elif prop >= limits[2]: return 1
    else: return 0


def impute_missing(df, gas_columns, models=None):
    df_imp = df.copy()
    for idx, row in df_imp.iterrows():
        known = row[gas_columns].dropna()
        missing = row[gas_columns].index[row[gas_columns].isnull()]
        if missing.any():
            # Normalize known gases to proportions
            tentative = row[gas_columns].copy()
            for gas in missing:
                best_score = -np.inf
                best_level = 0
                for level_idx, midpoint in enumerate(level_midpoints):
                    tentative[gas] = midpoint
                    total = tentative.sum()
                    proportions = tentative / (total if total != 0 else 1e-10)
                    levels = {g: prop_to_level(g, proportions[g]) for g in gas_columns if not pd.isna(proportions[g])}

                    score = 0.0
                    for known_gas in known.index:
                        if known_gas != gas:
                            corr = correlation_matrix.loc[gas, known_gas]
                            delta = abs(levels[gas] - levels[known_gas])
                            score -= corr * delta

                    if score > best_score:
                        best_score = score
                        best_level = level_idx

                # Assign only the predicted level index (0–3)
                df_imp.at[idx, gas] = best_level
    return df_imp

# No training needed for correlation-based method
def train_models(df, gas_columns):
    return None