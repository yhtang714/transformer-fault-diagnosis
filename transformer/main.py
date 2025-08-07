from config import gas_columns, fault_types, all_columns
from data_loader import load_data
from imputer import train_models, impute_missing
from discretizer import discretize
from notears import notears_numpy
from causal_network import build_graph, visualize_graph, fit_causal_network
from inference import get_user_input, compute_evidence
from counterfactual import run_counterfactual_diagnosis
from pgmpy.inference import VariableElimination

# 1. Load and preprocess
df = load_data("pythonProject/electricity/python/training_less.csv")
models = train_models(df, gas_columns)
df = impute_missing(df, gas_columns, models)
df = discretize(df)

# 2. Learn DAG
X = df[all_columns].to_numpy()
W = notears_numpy(X)
G = build_graph(W, gas_columns, fault_types, all_columns)
visualize_graph(G)
model = fit_causal_network(G, df, all_columns)

# 3. Inference
input_features = get_user_input()
evidence = compute_evidence(input_features)
evidence = {k: v for k, v in evidence.items() if k in G.nodes()}
inference = VariableElimination(model)

# 4. Counterfactual Diagnosis
print("\nRunning counterfactual diagnosis with weighted scoring...\n")
ranked_faults = run_counterfactual_diagnosis(
    evidence=evidence,
    model=model,
    inference=inference,
    fault_types=fault_types
)
