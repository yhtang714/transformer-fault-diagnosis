import numpy as np
from config import gas_columns, section_limits

def discretize(df):
    total_gas = df[gas_columns].sum(axis=1)
    total_gas[total_gas == 0] = 1e-10
    proportions = df[gas_columns].div(total_gas, axis=0)

    for gas in gas_columns:
        levels = np.zeros(len(df), dtype=int)
        limits = section_limits[gas]
        if len(limits) == 3:
            if len(proportions[gas]) == len(levels):
                mask1 = proportions[gas] >= limits[2]
                mask2 = proportions[gas] >= limits[1]
                mask3 = proportions[gas] >= limits[0]
                levels[mask1] = 1
                levels[mask2] = 2
                levels[mask3] = 3
        df[gas] = levels
    return df
