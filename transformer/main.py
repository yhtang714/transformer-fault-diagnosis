from config import gas_columns, fault_types, all_columns
from data_loader import load_data
from imputer import train_models, impute_missing
from discretizer import discretize
from notears import notears_numpy
from causal_network import build_graph, visualize_graph, fit_causal_network
from inference import get_user_input, compute_evidence
from counterfactual import run_counterfactual_diagnosis
from pgmpy.inference import VariableElimination
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import numpy as np
from counterfactual import expected_disablement
from counterfactual import expected_sufficiency


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

# Optional: save top diagnosis to a variable for external use
if ranked_faults:
    top_fault, top_info = ranked_faults[0]
    print(f"Diagnosis: {top_fault} (Score: {top_info['Score']:.3f})")

ranked_faults = run_counterfactual_diagnosis(
    evidence=evidence,
    model=model,
    inference=inference,
    fault_types=fault_types,
    w1=0.4, w2=0.3, w3=0.3  # You can adjust weights here
)

if ranked_faults:
    top_fault, top_info = ranked_faults[0]
    print(f"Diagnosis: {top_fault} (Score: {top_info['Score']:.3f})")


class TempScalingRiskAssessor:
    def __init__(self, random_state: int = 42, gamma_grid=None):
        self.random_state = random_state
        self.gamma_grid = gamma_grid if gamma_grid is not None else np.linspace(0.0, 0.5, 26)
        self.T_ = 1.0
        self.gamma_ = None
        self.fault_types_ = None
        self.calib_indices_ = None

    def _softmax(self, scores):
        z = np.array(list(scores.values())) / max(self.T_, 1e-6)
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        return dict(zip(scores.keys(), p))

    def fit(self, df_discrete, fault_types, compute_scores_fn, calib_size: float = 0.2):
        """
        Fit temperature scaling parameter T and select gamma on calibration set.
        """
        self.fault_types_ = list(fault_types)

        # --- 1) Split calibration indices ---
        n = len(df_discrete)
        all_idx = np.arange(n)
        strat = (df_discrete[fault_types].sum(axis=1) > 0).astype(int).values
        _, calib_idx = train_test_split(
            all_idx, test_size=calib_size, stratify=strat, random_state=self.random_state
        )
        self.calib_indices_ = calib_idx

        # --- 2) Collect per-sample scores & correctness ---
        score_matrix = []
        y_top_is_correct = []
        for ridx in calib_idx:
            cf_res, y_true = compute_scores_fn(int(ridx))
            scores = {f: m['Score'] for f, m in cf_res.items()}
            if not scores:
                continue
            score_matrix.append(list(scores.values()))
            top_f = max(scores.items(), key=lambda kv: kv[1])[0]
            y_top_is_correct.append(int(y_true.get(top_f, 0)))

        score_matrix = np.array(score_matrix, dtype=float)
        y_top_is_correct = np.array(y_top_is_correct, dtype=int)

        # --- 3) Fit temperature T by minimizing NLL for top-vs-rest ---
        def nll(T):
            z = score_matrix / T
            z -= z.max(axis=1, keepdims=True)
            p_top = np.exp(z.max(axis=1)) / np.exp(z).sum(axis=1)
            eps = 1e-9
            return -np.mean(
                y_top_is_correct * np.log(p_top + eps) + (1 - y_top_is_correct) * np.log(1 - p_top + eps)
            )

        Ts = np.linspace(0.2, 5.0, 49)
        vals = [nll(T) for T in Ts]
        self.T_ = float(Ts[int(np.argmin(vals))])

        # --- 4) Cache uncertainties and hits with calibrated probs ---
        u_list, hit_list = [], []
        for ridx in calib_idx:
            cf_res, y_true = compute_scores_fn(int(ridx))
            scores = {f: m['Score'] for f, m in cf_res.items()}
            if not scores:
                continue
            probs = self._softmax(scores)
            top_f, top_p = max(probs.items(), key=lambda kv: kv[1])
            u = 1.0 - float(top_p)
            hit = 1 if int(y_true.get(top_f, 0)) == 1 else 0
            u_list.append(u)
            hit_list.append(hit)

        u_arr = np.asarray(u_list, dtype=float)
        hit_arr = np.asarray(hit_list, dtype=int)

        # --- 5) Choose gamma (≤ 0.5) maximizing coverage × accuracy ---
        def cov_acc_at(g):
            covered = (u_arr <= g)
            cov = covered.mean() if covered.size else 0.0
            acc = hit_arr[covered].mean() if covered.any() else 0.0
            return cov * acc, cov, acc

        best_score, best_gamma, best_cov, best_acc = -1.0, 0.5, 0.0, 0.0
        for g in self.gamma_grid:
            s, cov, acc = cov_acc_at(float(g))
            if s > best_score:
                best_score, best_gamma, best_cov, best_acc = s, float(g), cov, acc

        self.gamma_ = min(best_gamma, 0.5)
        print(f"[Calibration] T={self.T_:.3f}, gamma={self.gamma_:.3f} (coverage={best_cov:.3f}, acc={best_acc:.3f})")
        return self

    def decide(self, counterfactual_results, gamma=None):
        """
        Gate-only decision:
          - compute uncertainty u from temperature-scaled softmax probs;
          - if u > gamma -> ABSTAIN;
          - else AUTO and keep ORIGINAL top-by-Score label.
        """
        if gamma is None:
            gamma = self.gamma_
        if gamma is None:
            raise RuntimeError("gamma is None. Call fit() first or pass gamma explicitly.")

        scores = {f: m['Score'] for f, m in counterfactual_results.items()}
        if not scores:
            return {'action': 'ABSTAIN', 'top_fault': None, 'uncertainty': 1.0, 'calibrated_probs': {}}

        probs = self._softmax(scores)
        top_p = max(probs.values())
        u = 1.0 - float(top_p)

        if u > float(gamma):
            return {'action': 'ABSTAIN', 'top_fault': None, 'uncertainty': u, 'calibrated_probs': probs}

        # Keep original top-by-Score label
        top_fault = max(counterfactual_results.items(), key=lambda kv: kv[1]['Score'])[0]
        return {'action': 'AUTO', 'top_fault': top_fault, 'uncertainty': u, 'calibrated_probs': probs}


def _build_compute_scores_fn(df, fault_types, gas_columns, model, inference,
                             w1=0.4, w2=0.3, w3=0.3):
    """
    Callback that computes {Posterior, Ed, Es, Score} for a given row index,
    and returns (counterfactual_results, y_true). Uses your existing logic.
    """
    def compute_scores_fn(row_idx: int):
        evidence = {}
        for gas in gas_columns:
            val = df.loc[row_idx, gas]
            try:
                evidence[gas] = int(val)
            except Exception:
                evidence[gas] = val
        y_true = {f: int(df.loc[row_idx, f]) for f in fault_types}
        res = {}
        for fault in fault_types:
            if hasattr(model, 'nodes') and fault not in model.nodes():
                continue
            try:
                posterior = float(inference.query(variables=[fault], evidence=evidence).values[1])
            except Exception:
                posterior = 0.0
            try:
                ed = expected_disablement(fault, evidence, inference)
                es = expected_sufficiency(fault, evidence, inference, fault_types)
            except Exception:
                ed, es = 0.0, 0.0
            score = w1 * posterior + w2 * ed + w3 * es
            res[fault] = {'Posterior': posterior, 'Ed': ed, 'Es': es, 'Score': score}
        return res, y_true
    return compute_scores_fn


# === Train assessor once ===
try:
    assessor = TempScalingRiskAssessor(random_state=42)
    compute_scores_fn = _build_compute_scores_fn(df, fault_types, gas_columns, model, inference,
                                                 w1=0.4, w2=0.3, w3=0.3)
    assessor.fit(df, fault_types, compute_scores_fn, calib_size=0.2)
except Exception as _e:
    assessor = None
    print(f"[Risk Assessment] Skipped calibration: {_e}")

# === After your existing ranked_faults, print decision ===
try:
    if 'ranked_faults' in globals() and ranked_faults:
        cf_res_for_current = {f: m for f, m in ranked_faults}
        if assessor is not None and assessor.gamma_ is not None:
            decision = assessor.decide(cf_res_for_current)
            if decision['action'] == 'ABSTAIN':
                print(f"\n[Risk Assessment] High Uncertainty! Manual Review Required (uncertainty={decision['uncertainty']:.3f} > gamma={assessor.gamma_:.3f})")
            else:
                print(f"\n[Risk Assessment] AUTO -> {decision['top_fault']} "
                      f"(uncertainty={decision['uncertainty']:.3f} <= gamma={assessor.gamma_:.3f})")
            probs_str = ", ".join([f"{k}:{v:.3f}" for k, v in decision['calibrated_probs'].items()])
            print(f"[Risk Assessment] Calibrated probs: {probs_str}")
except Exception as _e:
    print(f"[Risk Assessment] Decision skipped: {_e}")
