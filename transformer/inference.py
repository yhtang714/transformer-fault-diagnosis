from config import gas_columns, section_limits
from imputer import impute_missing
import pandas as pd


def get_user_input():
    input_features = {}
    for gas in gas_columns:
        value = input(f"{gas}: ").strip().lower()
        if value in ("none", ""):
            input_features[gas] = None
        else:
            input_features[gas] = float(value)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_features])

    # Track which gases were missing
    missing_gases = [gas for gas, val in input_features.items() if val is None]

    # Impute missing gases using correlation-guided section prediction
    input_df_imputed = impute_missing(input_df, gas_columns)
    imputed_row = input_df_imputed.iloc[0][gas_columns].astype(int).to_dict()

    # Print imputed values
    if missing_gases:
        print("Imputed gas levels (section index):")
        for gas in missing_gases:
            print(f"  {gas}: {imputed_row[gas]} (level)")

    return imputed_row

def compute_evidence(input_features):
    sum_gas = sum(input_features.values())
    concentrations = {gas: val / sum_gas if sum_gas > 0 else 0 for gas, val in input_features.items()}
    evidence = {}
    for gas in gas_columns:
        conc = concentrations[gas]
        limits = section_limits[gas]
        if conc >= limits[0]:
            evidence[gas] = 3
        elif conc >= limits[1]:
            evidence[gas] = 2
        elif conc >= limits[2]:
            evidence[gas] = 1
        else:
            evidence[gas] = 0
    return evidence
