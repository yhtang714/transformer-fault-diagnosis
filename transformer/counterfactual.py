def expected_disablement(fault, evidence, inference):
    ed = 0.0
    for gas, level in evidence.items():
        try:
            p_true = inference.query(variables=[gas], evidence={fault: 1}).values[level]
            p_false = inference.query(variables=[gas], evidence={fault: 0}).values[level]
            ed += max(0.0, p_true - p_false)
        except Exception:
            continue
    return ed

def expected_sufficiency(fault, evidence, inference, all_faults):
    other_faults = [f for f in all_faults if f != fault]
    fault_only = {f: 0 for f in other_faults}
    fault_only[fault] = 1

    es = 0.0
    for gas, level in evidence.items():
        try:
            p = inference.query(variables=[gas], evidence=fault_only).values[level]
            es += max(0.0, p)
        except Exception:
            continue
    return es

def run_counterfactual_diagnosis(evidence, model, inference, fault_types,
                                 w1=0.4, w2=0.3, w3=0.3):
    counterfactual_results = {}
    for fault in fault_types:
        if fault not in model.nodes():
            continue
        try:
            posterior = inference.query(variables=[fault], evidence=evidence).values[1]
        except Exception:
            posterior = 0.0
        ed = expected_disablement(fault, evidence, inference)
        es = expected_sufficiency(fault, evidence, inference, fault_types)
        score = w1 * posterior + w2 * ed + w3 * es
        counterfactual_results[fault] = {
            'Posterior': posterior,
            'Ed': ed,
            'Es': es,
            'Score': score
        }

    ranked_faults = sorted(counterfactual_results.items(), key=lambda x: -x[1]['Score'])

    print("Counterfactual Diagnostic Result (Weighted):")
    for fault, metrics in ranked_faults:
        print(f"  {fault}: Posterior={metrics['Posterior']:.3f}, Ed={metrics['Ed']:.3f}, Es={metrics['Es']:.3f}, Score={metrics['Score']:.3f}")

    if ranked_faults:
        top_fault = ranked_faults[0][0]
        top_score = ranked_faults[0][1]['Score']
        print(f"Most likely root cause: {top_fault} (score: {top_score:.3f})")
    else:
        print("No valid counterfactual result could be computed.")

    return ranked_faults
