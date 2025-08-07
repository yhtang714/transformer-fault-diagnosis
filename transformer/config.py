gas_columns = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
fault_types = ['H_DIS', 'L_DIS', 'P_DIS', 'HO', 'MO', 'LO']
all_columns = gas_columns + fault_types

section_limits = {
    'H2': [0.488, 0.388, 0.288],
    'CH4': [0.388, 0.288, 0.188],
    'C2H2': [0.278, 0.178, 0.078],
    'C2H4': [0.288, 0.188, 0.088],
    'C2H6': [0.288, 0.188, 0.088]
}
