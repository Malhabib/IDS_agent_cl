# simple_run_v6_triple_llm_v24.py
"""
EV-IDS-Agent VERSION 6 - TRIPLE LLM V24 Runner

Same interactive flow as V23 (scenario select, train/load, sample select,
manual per-LLM `ollama run` switching, comparison tables + heatmaps), PLUS:
  - honest per-model calibrated confidence surfaced in the comparison
  - SHAP<->LIME agreement + cross-model consensus summary
  - full 6-step reasoning-trace persisted in the JSON output

Install dependencies first:
    pip install shap lime
"""

import os, sys, json, pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix
from ev_ids_agent_v6_triple_llm_v24 import (
    SimpleOllamaClient, EVIDSAgentV6TripleLLMV24,
    auto_train_models_v6, get_column_mapping, classify_difficulty_zone,
    parse_datetime_to_timestamp, SHAP_AVAILABLE, LIME_AVAILABLE
)


def check_available_models(ollama_url="http://localhost:11434"):
    try:
        import ollama
        models_info = ollama.list()
        models = [m['name'] for m in models_info['models']]
        print(f"\nAVAILABLE MODELS:")
        for m in models:
            print(f"  {m}")
        return models
    except Exception as e:
        print(f"Error checking models: {e}")
        return []


def test_llm_connectivity(model_name, ollama_url="http://localhost:11434"):
    print(f"\nTesting {model_name}...")
    try:
        from ollama import chat
        resp = chat(model=model_name, messages=[{'role': 'user', 'content': 'Say OK'}])
        content = resp.message.content.strip()
        if content:
            print(f"  [OK] {model_name}: {content[:50]}")
            return True
        return False
    except Exception as e:
        print(f"  [FAIL] {model_name}: {e}")
        return False


def make_json_serializable(obj):
    if isinstance(obj, dict):            return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):          return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)):return float(obj)
    elif isinstance(obj, np.bool_):      return bool(obj)
    elif isinstance(obj, np.ndarray):    return obj.tolist()
    elif isinstance(obj, pd.Series):     return obj.tolist()
    elif hasattr(obj, 'item'):           return obj.item()
    return obj


def calculate_metrics(y_true, y_pred):
    # Standard IDS metrics over ALL samples — every session is classified
    # (no ABSTAIN), so the denominator is always N and models are comparable.
    y_t = [1 if l == 'Malicious' else 0 for l in y_true]
    y_p = [1 if l == 'Malicious' else 0 for l in y_pred]
    correct  = sum(a == b for a, b in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        far  = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {'accuracy': accuracy, 'precision': prec, 'recall': rec, 'f1': f1,
                'far': far, 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    return {'accuracy': accuracy, 'precision': 0, 'recall': 0, 'f1': 0, 'far': 0,
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}


def load_or_train_models(data_path, models_dir):
    split_path   = os.path.join(models_dir, 'split_info_v6.pkl')
    models_exist = os.path.exists(os.path.join(models_dir, 'ev_rf_model.pkl'))
    if models_exist and os.path.exists(split_path):
        with open(split_path, 'rb') as f:
            split_info = pickle.load(f)
        if 'feature_cols' in split_info and len(split_info.get('feature_cols', [])) > 4:
            print("\n*** MODELS TRAINED WITH DELIVERY_RATIO — RETRAINING ***")
            import glob
            for pkl in glob.glob(os.path.join(models_dir, '*.pkl')):
                os.remove(pkl)
        else:
            print(f"\nEXISTING MODELS: Train={len(split_info['train_indices'])}, "
                  f"Test={len(split_info['test_indices'])}")
            rf_path = os.path.join(models_dir, 'ev_rf_model.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    rf = pickle.load(f)
                df       = pd.read_csv(data_path)
                cmap     = get_column_mapping(df)
                sample   = df.loc[split_info['train_indices'][:20]]
                a_cols   = [cmap[c] for c in ['connectionTime', 'disconnectTime',
                                               'RequestedDemand', 'kWhDelivered']]
                X_s      = sample[a_cols].copy()
                X_s.columns = ['connectionTime', 'disconnectTime', 'RequestedDemand', 'kWhDelivered']
                for col in ['connectionTime', 'disconnectTime']:
                    X_s[col] = X_s[col].apply(parse_datetime_to_timestamp)
                X_s = X_s.fillna(0)
                sc_path = os.path.join(models_dir, 'ev_scaler.pkl')
                if os.path.exists(sc_path):
                    with open(sc_path, 'rb') as f:
                        sc = pickle.load(f)
                    try:
                        preds = rf.predict(sc.transform(X_s))
                        print(f"  RF predicts: {set(preds)}")
                    except Exception as e:
                        print(f"  RF prediction failed: {e} — retraining")
                        import glob
                        for pkl in glob.glob(os.path.join(models_dir, '*.pkl')):
                            os.remove(pkl)
                        models_exist = False
            if models_exist:
                resp = input("\nUse existing models? (yes/no): ").lower().strip()
                if resp in ['yes', 'y']:
                    return split_info['train_indices'], split_info['test_indices']

    print("\nTRAINING NEW MODELS")
    while True:
        try:
            raw = input("\nTrain ratio (default 0.40): ").strip()
            tr  = 0.40 if raw == "" else float(raw)
            if 0.1 <= tr <= 0.9:
                break
            print("Must be 0.1–0.9")
        except ValueError:
            print("Invalid")
    resp = input(f"\nTrain on {tr*100:.0f}%? (yes/no): ").lower().strip()
    if resp not in ['yes', 'y']:
        sys.exit(0)
    return auto_train_models_v6(data_path, models_dir, tr, 42)


def select_samples(df, test_indices):
    test_df = df.loc[test_indices].copy()
    print(f"\nTest set: {len(test_df)} samples")
    min_samples = 50
    print(f"  Minimum {min_samples} samples required.")
    while True:
        try:
            n = int(input(f"How many to test ({min_samples}–{len(test_df)}): "))
            if n < min_samples:
                print(f"  Must be at least {min_samples}.")
                continue
            if n <= len(test_df):
                break
        except ValueError:
            pass
    normal_df = test_df[test_df['label'] == 0]
    mal_df    = test_df[test_df['label'] == 1]
    nr  = len(normal_df) / len(test_df)
    nn  = int(n * nr)
    nm  = n - nn
    ns  = normal_df.sample(n=min(nn, len(normal_df)), random_state=42)
    ms  = mal_df.sample(n=min(nm, len(mal_df)), random_state=42)
    sel = pd.concat([ns, ms]).sample(frac=1, random_state=42)
    print(f"Selected {len(sel)} samples ({nn}N, {nm}M)")
    return sel.index.tolist()


def run_classification(agent, emoji, label, selected, df, model_name, ollama_url):
    print(f"\n{'='*80}")
    print(f"{emoji} RUNNING {label}")
    print(f"{'='*80}")
    results = []
    input(f"\nPress Enter when {label} is ready...")
    if not test_llm_connectivity(model_name, ollama_url):
        print(f"[FAIL] {label} — skipping")
        return results
    for i, idx in enumerate(selected, 1):
        gt = "Malicious" if df.loc[idx, 'label'] == 1 else "Normal"
        print(f"\n{'#'*80}")
        print(f"# {emoji} {label} {i}/{len(selected)} — Index {idx} (Truth: {gt})")
        print(f"{'#'*80}")
        try:
            result = agent.detect(line_number=idx)
            if result['status'] == 'success':
                pred    = result['result']['predicted_label']
                correct = (pred == gt)
                results.append({
                    'index':             idx,
                    'ground_truth':      gt,
                    'predicted':         pred,
                    'correct':           correct,
                    'model_predictions': result['model_predictions'],
                    'evidence_bundle':   result.get('evidence_bundle', {}),
                    'result':            result['result'],
                    'majority_vote':     result['majority_vote']
                })
                sym = "OK" if correct else "WRONG"
                print(f"\n  [{sym}] Truth={gt}, Predicted={pred}, "
                      f"Override={result['result']['llm_overrode_ml']}")
                if correct:
                    agent.store_correct_decision_in_ltm(idx, True, result['result'])
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    cc = sum(1 for r in results if r['correct'])
    if results:
        print(f"\n{label} COMPLETE: {cc}/{len(results)} correct "
              f"({cc/len(results)*100:.1f}%)")
    return results


def generate_confusion_heatmaps(results_dict, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping heatmaps")
        return []

    os.makedirs(output_dir, exist_ok=True)
    first_llm     = list(results_dict.values())[0]
    ground_truths = [r['ground_truth'] for r in first_llm]
    ml_names      = ['Random Forest', 'K-Nearest Neighbors', 'Logistic Regression',
                     'MLP', 'Support Vector Classifier', 'Decision Tree', 'Gradient Boosting']
    all_models_data = {}
    for mn in ml_names:
        all_models_data[mn] = [r['model_predictions'].get(mn, {}).get('prediction', 'Normal')
                                for r in first_llm]
    all_models_data['Majority Vote'] = [r['majority_vote'] for r in first_llm]
    for llm_name, results in results_dict.items():
        preds = [r['predicted'] if r['predicted'] != 'ABSTAIN' else r['majority_vote']
                 for r in results]
        all_models_data[f"{llm_name} Agent (V24)"] = preds

    n_models = len(all_models_data)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle('Confusion Matrices — V24 (Confidence + SHAP + LIME)',
                 fontsize=14, fontweight='bold', y=1.02)
    axes_flat = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, preds) in enumerate(all_models_data.items()):
        ax  = axes_flat[idx]
        y_t = [1 if l == 'Malicious' else 0 for l in ground_truths]
        y_p = [1 if l == 'Malicious' else 0 for l in preds]
        cm  = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn = fp = fn = tp = 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / len(ground_truths) if ground_truths else 0
        ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=max(cm.max(), 1))
        ax.set_title(f'{name}\nAcc={acc:.3f}', fontsize=8, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attack'], fontsize=7)
        ax.set_yticklabels(['Normal', 'Attack'], fontsize=7)
        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_ylabel('Actual', fontsize=7)
        labels_cm = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val   = cm[i, j]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, f"{val}\n{labels_cm[i][j]}",
                        ha='center', va='center', fontsize=8, color=color)

    for idx in range(len(all_models_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrices_v24.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return [path]


def print_xai_summary(results_llama):
    """XAI section: which features consistently drove ML predictions + agreement."""
    print(f"\n{'='*120}")
    print(f"XAI FEATURE CONSISTENCY ANALYSIS (SHAP + LIME + AGREEMENT)")
    print(f"{'='*120}")
    print(f"  Which features the ML ensemble consistently flagged, whether SHAP and")
    print(f"  LIME agreed, and whether the LLM's reasoning reflected those signals.\n")

    ml_names = ['Random Forest', 'K-Nearest Neighbors', 'Logistic Regression',
                'MLP', 'Support Vector Classifier', 'Decision Tree', 'Gradient Boosting']

    from collections import defaultdict
    shap_top_counts = defaultdict(int)
    lime_top_counts = defaultdict(int)
    agreement_counts = defaultdict(int)
    total_samples   = len(results_llama)

    for r in results_llama:
        xai = r['result'].get('shap_consensus', {})
        # cross-model consensus feature (per sample)
        if xai.get('top_feature'):
            shap_top_counts[xai['top_feature']] += 1
        # strong/conflict agreement rollups
        for m in r['result'].get('strong_agreement_models', []):
            agreement_counts['strong'] += 1
        for m in r['result'].get('conflict_models', []):
            agreement_counts['conflict'] += 1

    if shap_top_counts:
        print(f"  Cross-model SHAP consensus driver across {total_samples} samples:")
        for feat, count in sorted(shap_top_counts.items(), key=lambda x: -x[1])[:6]:
            pct = count / total_samples * 100
            print(f"    {feat:<35} {count:>5} times ({pct:.1f}%)")

    print(f"\n  SHAP<->LIME agreement (per model-instance across all samples):")
    print(f"    strong agreement:  {agreement_counts['strong']}")
    print(f"    conflict:          {agreement_counts['conflict']}")

    xai_mentioned = sum(1 for r in results_llama
                        if r['result'].get('llm_xai_assessment', ''))
    print(f"\n  LLM explicitly referenced SHAP/LIME (STEP 3) in: "
          f"{xai_mentioned}/{total_samples} sessions ({xai_mentioned/total_samples*100:.1f}%)")

    # reasoning completeness (V24)
    avg_steps = np.mean([r['result'].get('steps_present', 0) for r in results_llama])
    full      = sum(1 for r in results_llama if r['result'].get('steps_present', 0) == 6)
    print(f"  Reasoning completeness: avg {avg_steps:.1f}/6 steps, "
          f"{full}/{total_samples} sessions had all 6 steps "
          f"({full/total_samples*100:.1f}%)")


def print_triple_comparison(results_llama, results_qwen, results_glm, output_dir):
    sample   = results_llama[0]['result'] if results_llama else {}
    rag_on   = sample.get('knowledge_used', False)
    mem_on   = sample.get('memory_used', False)
    sc_label = ("SCENARIO A" if rag_on and mem_on else
                "SCENARIO B" if rag_on else
                "SCENARIO C" if mem_on else "SCENARIO D")

    print(f"\n{'='*160}")
    print(f"TRIPLE LLM COMPARISON — VERSION 6 V24 (CONFIDENCE + SHAP + LIME) | {sc_label}")
    print(f"{'='*160}")
    print(f"  SHAP: {'ON' if SHAP_AVAILABLE else 'OFF'}  |  "
          f"LIME: {'ON' if LIME_AVAILABLE else 'OFF'}")

    model_map = {'Random Forest': 'RF', 'Logistic Regression': 'LR',
                 'K-Nearest Neighbors': 'KNN', 'MLP': 'MLP',
                 'Decision Tree': 'DT', 'Support Vector Classifier': 'SVC',
                 'Gradient Boosting': 'GB'}
    ml_names = list(model_map.keys())

    ground_truths = [r['ground_truth'] for r in results_llama]
    llama_preds   = {n: [r['model_predictions'].get(n, {}).get('prediction', 'Normal')
                          for r in results_llama] for n in ml_names}
    llama_preds['Majority_Vote'] = [r['majority_vote']  for r in results_llama]
    llama_preds['IDS_Agent']     = [r['predicted']       for r in results_llama]
    qwen_preds  = {'IDS_Agent': [r['predicted'] for r in results_qwen]}
    glm_preds   = {'IDS_Agent': [r['predicted'] for r in results_glm]}

    llama_m = {m: calculate_metrics(ground_truths, llama_preds[m])
               for m in ml_names + ['Majority_Vote', 'IDS_Agent']}
    qwen_m  = {'IDS_Agent': calculate_metrics(ground_truths, qwen_preds['IDS_Agent'])}
    glm_m   = {'IDS_Agent': calculate_metrics(ground_truths, glm_preds['IDS_Agent'])}

    # METRICS TABLE
    print(f"\n{'Metric':<15} |", end='')
    for mn in ml_names:
        print(f" {model_map[mn]:>6} |", end='')
    print(f" {'Majority':>8} | {'Llama':>8} | {'Qwen':>8} | {'GLM':>8} |")
    print("-" * 140)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'far']:
        lab = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
               'f1': 'F1-Score', 'far': 'FAR'}[metric]
        print(f"{lab:<15} |", end='')
        for mn in ml_names:
            print(f" {llama_m[mn][metric]:>6.4f} |", end='')
        print(f" {llama_m['Majority_Vote'][metric]:>8.4f} |"
              f" {llama_m['IDS_Agent'][metric]:>8.4f} |"
              f" {qwen_m['IDS_Agent'][metric]:>8.4f} |"
              f" {glm_m['IDS_Agent'][metric]:>8.4f} |")
    print("=" * 140)

    # CONFUSION MATRIX TABLE
    print(f"\n{'Model':<20} | {'TP':>4} | {'TN':>4} | {'FP':>4} | {'FN':>4} | "
          f"{'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
    print("-" * 90)
    for mn in ml_names:
        m = llama_m[mn]
        print(f"{mn:<20} | {m['tp']:>4} | {m['tn']:>4} | {m['fp']:>4} | {m['fn']:>4} | "
              f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
              f"{m['recall']:>6.4f} | {m['f1']:>6.4f}")
    m = llama_m['Majority_Vote']
    print(f"{'Majority Vote':<20} | {m['tp']:>4} | {m['tn']:>4} | {m['fp']:>4} | {m['fn']:>4} | "
          f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
          f"{m['recall']:>6.4f} | {m['f1']:>6.4f}")
    for llm_name, metrics in [
            ('Llama Agent V24', llama_m['IDS_Agent']),
            ('Qwen Agent V24',  qwen_m['IDS_Agent']),
            ('GLM Agent V24',   glm_m['IDS_Agent'])]:
        m = metrics
        print(f"{llm_name:<20} | {m['tp']:>4} | {m['tn']:>4} | {m['fp']:>4} | {m['fn']:>4} | "
              f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
              f"{m['recall']:>6.4f} | {m['f1']:>6.4f}")

    # HEATMAPS
    print(f"\n{'='*120}")
    print(f"GENERATING CONFUSION MATRIX HEATMAPS")
    print(f"{'='*120}")
    generate_confusion_heatmaps(
        {'Llama': results_llama, 'Qwen': results_qwen, 'GLM': results_glm}, output_dir)

    # XAI FEATURE ANALYSIS
    print_xai_summary(results_llama)

    # LLM OVERRIDE ANALYSIS
    print(f"\n{'='*120}")
    print(f"LLM OVERRIDE ANALYSIS")
    print(f"{'='*120}")
    for llm_name, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
        overrides  = [r for r in results if r['result'].get('llm_overrode_ml')]
        correct_ov = sum(1 for r in overrides if r['correct'])
        agreed     = [r for r in results if not r['result'].get('llm_overrode_ml')]
        agreed_cor = sum(1 for r in agreed if r['correct'])
        print(f"\n  {llm_name}:")
        if agreed:
            print(f"    Agreed with ML: {len(agreed)}/{len(results)} "
                  f"({len(agreed)/len(results)*100:.1f}%) — "
                  f"correct: {agreed_cor}/{len(agreed)} ({agreed_cor/len(agreed)*100:.1f}%)")
        print(f"    Overrode ML:    {len(overrides)}/{len(results)} ({len(overrides)/len(results)*100:.1f}%)")
        if overrides:
            wrong_ov = len(overrides) - correct_ov
            print(f"      Correct: {correct_ov}/{len(overrides)} ({correct_ov/len(overrides)*100:.1f}%)")
            print(f"      Wrong:   {wrong_ov}/{len(overrides)} ({wrong_ov/len(overrides)*100:.1f}%)")
            print(f"      Examples:")
            for r in overrides[:5]:
                res = r['result']
                sym = "RIGHT" if r['correct'] else "WRONG"
                print(f"        [{sym}] idx={r['index']}, truth={r['ground_truth']}, "
                      f"ML={res['ml_majority']}, LLM={res['predicted_label']}, "
                      f"ratio={res['delivery_ratio']:.3f}, zone={res['difficulty_zone']}")

    # LLM VALUE PROPOSITION
    print(f"\n{'='*120}")
    print(f"LLM VALUE PROPOSITION vs ML BASELINE")
    print(f"{'='*120}")
    maj_correct = sum(1 for i, r in enumerate(results_llama)
                      if r['majority_vote'] == ground_truths[i])
    maj_acc = maj_correct / len(results_llama)
    print(f"  ML Majority baseline: {maj_acc:.4f}\n")
    print(f"  {'LLM':<12} | {'Accuracy':>10} | {'vs ML Maj':>10} | "
          f"{'Overrides':>10} | {'Net Fix':>8}")
    print(f"  {'-'*58}")
    for llm_name, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
        llm_cor    = sum(1 for r in results if r['correct'])
        llm_acc    = llm_cor / len(results) if results else 0
        delta      = llm_acc - maj_acc
        overrides  = [r for r in results if r['result'].get('llm_overrode_ml')]
        correct_ov = sum(1 for r in overrides if r['correct'])
        wrong_ov   = len(overrides) - correct_ov
        net_fix    = correct_ov - wrong_ov
        nf_sign    = "+" if net_fix >= 0 else ""
        print(f"  {llm_name:<12} | {llm_acc:>10.4f} | {delta:>+10.4f} | "
              f"{len(overrides):>10} | {nf_sign}{net_fix:>7}")

    # DIFFICULTY ZONE ANALYSIS
    print(f"\n{'='*120}")
    print(f"DIFFICULTY ZONE ANALYSIS")
    print(f"{'='*120}")
    zones = {}
    for r in results_llama:
        z = r['result'].get('difficulty_zone', 'UNKNOWN')
        if z not in zones:
            zones[z] = {'total': 0, 'gt': []}
        zones[z]['total'] += 1
        zones[z]['gt'].append(r['ground_truth'])
    for z, data in sorted(zones.items()):
        n_mal = sum(1 for g in data['gt'] if g == 'Malicious')
        print(f"  {z:<20}: {data['total']} ({n_mal}M, {data['total']-n_mal}N)")
    print(f"\n  {'Zone':<20} | {'Llama':>8} | {'Qwen':>8} | {'GLM':>8} | {'Majority':>8} | {'Best ML':>8}")
    print(f"  {'-'*80}")
    for zone in sorted(zones.keys()):
        zone_idx = [i for i, r in enumerate(results_llama)
                    if r['result'].get('difficulty_zone') == zone]
        if not zone_idx:
            continue
        zone_gt  = [ground_truths[i] for i in zone_idx]
        zone_accs = {}
        for lname, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
            zp    = [results[i]['predicted'] for i in zone_idx]
            valid = [(t, p) for t, p in zip(zone_gt, zp) if p != 'ABSTAIN']
            zone_accs[lname] = (sum(a == b for a, b in valid) / len(valid)) if valid else 0.0
        maj_preds = [llama_preds['Majority_Vote'][i] for i in zone_idx]
        zone_accs['Majority'] = sum(a == b for a, b in zip(zone_gt, maj_preds)) / len(zone_gt)
        zone_accs['Best ML']  = max(
            sum(a == b for a, b in zip(zone_gt, [llama_preds[mn][i] for i in zone_idx])) / len(zone_gt)
            for mn in ml_names
        )
        print(f"  {zone:<20} | {zone_accs['Llama']:>8.4f} | {zone_accs['Qwen']:>8.4f} | "
              f"{zone_accs['GLM']:>8.4f} | {zone_accs['Majority']:>8.4f} | {zone_accs['Best ML']:>8.4f}")

    # COMPLEXITY
    print(f"\n{'='*120}")
    print(f"COMPLEXITY (V24: includes XAI computation time)")
    print(f"{'='*120}")
    print(f"\n  {'Metric':<28} | {'Llama':>10} | {'Qwen':>10} | {'GLM':>10}")
    print(f"  {'-'*65}")
    for mname, key in [('Avg LLM Latency (sec)', 'llm_latency_sec'),
                        ('Avg Stage1 (sec)', 'stage1_latency'),
                        ('Avg Stage2 (sec)', 'stage2_latency'),
                        ('Avg Prompt (chars)', 'prompt_chars'),
                        ('Avg Response (words)', 'response_words'),
                        ('Avg Throughput (w/s)', 'tokens_per_sec'),
                        ('Models XAI-explained', 'xai_models_explained'),
                        ('Total Time (sec)', 'session_total_sec')]:
        vals = []
        for results in [results_llama, results_qwen, results_glm]:
            cdata = [r['result'].get('complexity', {}).get(key, 0) for r in results]
            vals.append(np.sum(cdata) if key == 'session_total_sec'
                        else np.mean(cdata) if cdata else 0)
        print(f"  {mname:<28} | {vals[0]:>10.2f} | {vals[1]:>10.2f} | {vals[2]:>10.2f}")

    # FINAL RANKINGS
    print(f"\n{'='*120}")
    print(f"FINAL RANKINGS | {sc_label} | V24 (CONFIDENCE + SHAP + LIME)")
    print(f"{'='*120}")
    best_ml_name = max(ml_names, key=lambda m: llama_m[m]['accuracy'])
    all_ranked   = sorted([
        ('GLM Agent V24',      glm_m['IDS_Agent']['accuracy']),
        ('Qwen Agent V24',     qwen_m['IDS_Agent']['accuracy']),
        ('Llama Agent V24',    llama_m['IDS_Agent']['accuracy']),
        ('Majority Vote',      llama_m['Majority_Vote']['accuracy']),
        (best_ml_name,         llama_m[best_ml_name]['accuracy']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(all_ranked):
        print(f"  {i+1}. {name:22s} {acc:.4f} ({acc*100:.1f}%)")


def main():
    print(f"""
========================================================================
  EV-IDS-Agent VERSION 6 — TRIPLE LLM COMPARISON V24

  Llama3 | Qwen3.5 | GLM-4.7-Flash

  NEW in V24 (vs V23):
  - Honest per-model calibrated confidence (SVC via decision margin)
  - SHAP normalised to % share + SHAP<->LIME agreement flag
  - Cross-model SHAP consensus + confidence-weighted vote (evidence only)
  - Optimally engineered prompt: forced 6-step reasoning + few-shot anchors
  - Full 6-step reasoning trace persisted per session
  - Stage-1 completeness validation with one retry; deterministic decoding
  - ALWAYS classifies (no ABSTAIN): unparseable LLM output falls back to the
    ML majority vote, so every sample is scored over the same denominator N
  - GLM thinking-channel recovery (no more near-empty responses)
  - ML model TRAINING is unchanged from V23 (by design)

  Status: SHAP={'ON' if SHAP_AVAILABLE else 'OFF (pip install shap)'}  |  LIME={'ON' if LIME_AVAILABLE else 'OFF (pip install lime)'}
========================================================================
    """)

    if not SHAP_AVAILABLE or not LIME_AVAILABLE:
        print("  [WARN] Install missing XAI libraries for full V24 functionality:")
        if not SHAP_AVAILABLE: print("    pip install shap")
        if not LIME_AVAILABLE: print("    pip install lime")
        resp = input("\n  Continue anyway? (yes/no): ").lower().strip()
        if resp not in ['yes', 'y']:
            return

    print("=" * 60)
    print("SELECT SCENARIO")
    print("=" * 60)
    print("""
  A: Full Pipeline (ML + RAG + LTM + XAI -> LLM)
  B: RAG + XAI only (ML + RAG + XAI -> LLM, no LTM)
  C: Memory + XAI only (ML + LTM + XAI -> LLM, no RAG)
  D: XAI baseline (ML + XAI -> LLM, no RAG, no LTM)  *** RUN FIRST ***

  RECOMMENDED ORDER: D → B → C → A
    """)
    while True:
        choice = input("Scenario (A/B/C/D): ").strip().upper()
        if choice in ['A', 'B', 'C', 'D']:
            break
        print("Enter A, B, C, or D")

    scenario_map = {
        'A': {'use_knowledge': True,  'use_memory': True,  'tag': 'A_RAG_LTM_XAI', 'desc': 'Full Pipeline + XAI'},
        'B': {'use_knowledge': True,  'use_memory': False, 'tag': 'B_RAG_XAI',      'desc': 'RAG + XAI only'},
        'C': {'use_knowledge': False, 'use_memory': True,  'tag': 'C_LTM_XAI',      'desc': 'Memory + XAI only'},
        'D': {'use_knowledge': False, 'use_memory': False, 'tag': 'D_XAI_baseline',  'desc': 'XAI baseline (no RAG/LTM)'},
    }
    sc            = scenario_map[choice]
    use_knowledge = sc['use_knowledge']
    use_memory    = sc['use_memory']
    scenario_tag  = sc['tag']
    print(f"\n  Scenario {choice}: {sc['desc']}")
    print(f"  RAG={'ON' if use_knowledge else 'OFF'}, LTM={'ON' if use_memory else 'OFF'}, XAI=ON")

    DATA_PATH = (r"D:\OneDrive - Hamad bin Khalifa University\project 2"
                 r"\ev_mil_framework_corrected\dataset"
                 r"\Processed_ACNdata_CMA6_1_fu_4_test.csv")
    WORKSPACE_DIR = "./ev_ids_workspace_v6"
    OLLAMA_URL    = "http://localhost:11434"

    if not os.path.exists(DATA_PATH):
        print(f"\nDataset not found: {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset: {len(df)} samples, Distribution: {dict(df['label'].value_counts())}")

    cmap = get_column_mapping(df)
    df['_ratio'] = df[cmap['kWhDelivered']] / df[cmap['RequestedDemand']].replace(0, np.nan)
    df['_zone']  = df['_ratio'].apply(lambda r: classify_difficulty_zone(r) if pd.notna(r) else 'UNKNOWN')
    print(f"\nDifficulty zone distribution:")
    for zone, count in df['_zone'].value_counts().items():
        n_mal = len(df[(df['_zone'] == zone) & (df[cmap['label']] == 1)])
        print(f"  {zone:<20}: {count} ({n_mal}M, {count-n_mal}N)")
    df.drop(columns=['_ratio', '_zone'], inplace=True)

    try:
        import requests
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        print(f"\nOllama running at {OLLAMA_URL}")
        check_available_models(OLLAMA_URL)
    except Exception as e:
        print(f"\nOllama error: {e}")
        return

    models_dir = os.path.join(WORKSPACE_DIR, "models")
    try:
        train_indices, test_indices = load_or_train_models(DATA_PATH, models_dir)
    except Exception as e:
        print(f"\nModel error: {e}")
        return

    selected = select_samples(df, test_indices)
    config = {
        'data_path':           DATA_PATH,
        'models_dir':          models_dir,
        'knowledge_base_path': os.path.join(WORKSPACE_DIR, 'knowledge_base'),
        'workspace_dir':       WORKSPACE_DIR
    }
    results_dir = os.path.join(WORKSPACE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # RUN LLAMA
    print(f"\nMANUAL: Start ollama run llama3:latest")
    llm_llama   = SimpleOllamaClient("llama3:latest", OLLAMA_URL)
    agent_llama = EVIDSAgentV6TripleLLMV24(config, llm_llama,
                   use_knowledge=use_knowledge, use_memory=use_memory,
                   scenario_tag=f"{scenario_tag}_llama")
    if use_memory and agent_llama.ltm:
        agent_llama.ltm.clear()
        agent_llama.seed_ltm_from_training(train_indices, n_seed=50)
    results_llama = run_classification(agent_llama, "[LLAMA]", "LLAMA3",
                                        selected, df, "llama3:latest", OLLAMA_URL)

    # RUN GLM
    print(f"\nMANUAL: Switch to ollama run glm-4.7-flash:latest")
    llm_glm   = SimpleOllamaClient("glm-4.7-flash:latest", OLLAMA_URL)
    agent_glm = EVIDSAgentV6TripleLLMV24(config, llm_glm,
                 use_knowledge=use_knowledge, use_memory=use_memory,
                 scenario_tag=f"{scenario_tag}_glm")
    if use_memory and agent_glm.ltm:
        agent_glm.ltm.clear()
        agent_glm.seed_ltm_from_training(train_indices, n_seed=50)
    results_glm = run_classification(agent_glm, "[GLM]", "GLM-4.7-FLASH",
                                      selected, df, "glm-4.7-flash:latest", OLLAMA_URL)

    # RUN QWEN
    print(f"\nMANUAL: Switch to ollama run qwen3.5:latest")
    llm_qwen   = SimpleOllamaClient("qwen3.5:latest", OLLAMA_URL)
    agent_qwen = EVIDSAgentV6TripleLLMV24(config, llm_qwen,
                  use_knowledge=use_knowledge, use_memory=use_memory,
                  scenario_tag=f"{scenario_tag}_qwen")
    if use_memory and agent_qwen.ltm:
        agent_qwen.ltm.clear()
        agent_qwen.seed_ltm_from_training(train_indices, n_seed=50)
    results_qwen = run_classification(agent_qwen, "[QWEN]", "QWEN3.5",
                                       selected, df, "qwen3.5:latest", OLLAMA_URL)

    if results_llama and results_qwen and results_glm:
        print_triple_comparison(results_llama, results_qwen, results_glm, results_dir)
    else:
        print("\nIncomplete results")
        return

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"triple_llm_v6_v24_{scenario_tag}_{ts}.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(make_json_serializable({
                'version':       '6_triple_llm_v24',
                'scenario':      scenario_tag,
                'use_knowledge': use_knowledge,
                'use_memory':    use_memory,
                'shap_used':     SHAP_AVAILABLE,
                'lime_used':     LIME_AVAILABLE,
                'timestamp':     ts,
                'description':   f'Scenario {choice}: {sc["desc"]}',
                'llms':          {'llama': 'llama3', 'qwen': 'qwen3.5', 'glm': 'glm-4.7-flash'},
                'results_llama': results_llama,
                'results_qwen':  results_qwen,
                'results_glm':   results_glm
            }), f, indent=2)
        print(f"\nSaved: {results_file}")
    except Exception as e:
        print(f"\nSave error: {e}")
    print(f"\nTRIPLE LLM V24 SCENARIO {choice} COMPLETE!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n\nFatal: {e}")
        import traceback
        traceback.print_exc()
