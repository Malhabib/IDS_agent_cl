# simple_run_v6_triple_llm_v25.py
"""
EV-IDS-Agent VERSION 6 - TRIPLE LLM V25 Runner
Same interactive flow, scenarios and analysis sections as V23. V25 adds:
  - backend selection: ollama (default) or vllm (PagedAttention +
    continuous batching server-side; launch `vllm serve <model> --port 8000`)
  - optional parallel sample processing (thread pool). With vLLM this
    engages continuous batching; with Ollama set OLLAMA_NUM_PARALLEL.
    Parallelism is auto-disabled when LTM is on (memory must grow in order).
  - shared XAI store: SHAP+LIME computed once per sample for all 3 LLMs
  - standard IDS metrics over ALL N samples (always-classify, no ABSTAIN);
    a Fallback column reports how often the LLM output was unparseable
  - per-LLM elapsed time + ETA during the run

Install dependencies first:
    pip install shap lime ollama requests
"""

import os, sys, json, pickle, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix
from ev_ids_agent_v6_triple_llm_v25 import (
    make_llm_client, EVIDSAgentV6TripleLLMV25, build_xai_store,
    auto_train_models_v6, get_column_mapping, classify_difficulty_zone,
    parse_datetime_to_timestamp, SHAP_AVAILABLE, LIME_AVAILABLE
)

PRINT_LOCK = threading.Lock()


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


def test_llm_connectivity(client):
    print(f"\nTesting {client.model_name}...")
    try:
        resp = client.generate_with_system("You are a helpful assistant.",
                                           "Say OK", max_tokens=10)
        if resp and not resp.startswith("Error:"):
            print(f"  [OK] {client.model_name}: {resp[:50]}")
            return True
        print(f"  [FAIL] {client.model_name}: {resp[:100]}")
        return False
    except Exception as e:
        print(f"  [FAIL] {client.model_name}: {e}")
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
    """Standard IDS metrics over ALL samples (always-classify, no ABSTAIN)."""
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


def _classify_one(agent, idx, gt):
    """Worker: classify one sample; returns the result row or None on error."""
    try:
        result = agent.detect(line_number=idx)
        if result['status'] != 'success':
            return None
        pred = result['result']['predicted_label']
        return {
            'index':             idx,
            'ground_truth':      gt,
            'predicted':         pred,
            'correct':           pred == gt,
            'model_predictions': result['model_predictions'],
            'result':            result['result'],
            'majority_vote':     result['majority_vote']
        }
    except Exception as e:
        with PRINT_LOCK:
            print(f"\n  [ERROR] idx={idx}: {e}")
            import traceback
            traceback.print_exc()
        return None


def run_classification(agent, emoji, label, selected, df, workers=1):
    print(f"\n{'='*80}")
    print(f"{emoji} RUNNING {label}  (workers={workers})")
    print(f"{'='*80}")
    results = []
    input(f"\nPress Enter when {label} is ready...")
    if not test_llm_connectivity(agent.llm_client):
        print(f"[FAIL] {label} — skipping")
        return results

    truths  = {idx: ("Malicious" if df.loc[idx, 'label'] == 1 else "Normal")
               for idx in selected}
    t_start = time.time()
    done    = 0
    total   = len(selected)

    def _progress(row):
        nonlocal done
        done += 1
        elapsed = time.time() - t_start
        eta     = (elapsed / done) * (total - done)
        sym = ("OK" if row['correct'] else "WRONG")
        fb  = " FB" if row['result'].get('used_fallback') else ""
        with PRINT_LOCK:
            print(f"  [{done:>3}/{total}] idx={row['index']:<6} "
                  f"truth={row['ground_truth']:<9} pred={row['predicted']:<9} "
                  f"[{sym}]{fb}  elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")

    if workers <= 1:
        for idx in selected:
            row = _classify_one(agent, idx, truths[idx])
            if row is not None:
                results.append(row)
                _progress(row)
                if row['correct']:
                    agent.store_correct_decision_in_ltm(idx, True, row['result'])
    else:
        # Parallel: memory-off scenarios only (enforced by caller). With the
        # vLLM backend this engages continuous batching server-side.
        agent.verbose = False
        order = {idx: i for i, idx in enumerate(selected)}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_classify_one, agent, idx, truths[idx]): idx
                    for idx in selected}
            for fut in as_completed(futs):
                row = fut.result()
                if row is not None:
                    results.append(row)
                    _progress(row)
        results.sort(key=lambda r: order[r['index']])

    cc = sum(1 for r in results if r['correct'])
    fb = sum(1 for r in results if r['result'].get('used_fallback'))
    el = time.time() - t_start
    if results:
        print(f"\n{label} COMPLETE: {cc}/{len(results)} correct "
              f"({cc/len(results)*100:.1f}%), fallbacks={fb}, "
              f"time={el/60:.1f} min ({el/len(results):.1f}s/sample)")
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
        all_models_data[f"{llm_name} Agent (V25)"] = [r['predicted'] for r in results]

    n_models = len(all_models_data)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle('Confusion Matrices — V25 (SHAP+LIME, always-classify)',
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
    path = os.path.join(output_dir, 'confusion_matrices_v25.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return [path]


def print_xai_summary(results_llama):
    print(f"\n{'='*120}")
    print(f"XAI FEATURE CONSISTENCY ANALYSIS (SHAP + LIME)")
    print(f"{'='*120}")
    total_samples = len(results_llama)
    xai_mentioned = sum(1 for r in results_llama
                        if r['result'].get('llm_xai_assessment', ''))
    print(f"  LLM explicitly referenced SHAP/LIME in: "
          f"{xai_mentioned}/{total_samples} sessions "
          f"({xai_mentioned/total_samples*100:.1f}%)")


def print_triple_comparison(results_llama, results_qwen, results_glm, output_dir):
    sample   = results_llama[0]['result'] if results_llama else {}
    rag_on   = sample.get('knowledge_used', False)
    mem_on   = sample.get('memory_used', False)
    sc_label = ("SCENARIO A" if rag_on and mem_on else
                "SCENARIO B" if rag_on else
                "SCENARIO C" if mem_on else "SCENARIO D")

    print(f"\n{'='*160}")
    print(f"TRIPLE LLM COMPARISON — VERSION 6 V25 (SHAP+LIME, always-classify) | {sc_label}")
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

    # FALLBACK COUNTS (LLM output unparseable -> ML majority used)
    print(f"\nFALLBACK COUNTS (unparseable LLM output -> ML majority):")
    for name, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
        n_fb = sum(1 for r in results if r['result'].get('used_fallback'))
        print(f"  {name}: {n_fb}/{len(results)}")

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
          f"{'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Fallback':>8}")
    print("-" * 100)
    for mn in ml_names:
        m = llama_m[mn]
        print(f"{mn:<20} | {m['tp']:>4} | {m['tn']:>4} | {m['fp']:>4} | {m['fn']:>4} | "
              f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
              f"{m['recall']:>6.4f} | {m['f1']:>6.4f} | {'—':>8}")
    m = llama_m['Majority_Vote']
    print(f"{'Majority Vote':<20} | {m['tp']:>4} | {m['tn']:>4} | {m['fp']:>4} | {m['fn']:>4} | "
          f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
          f"{m['recall']:>6.4f} | {m['f1']:>6.4f} | {'—':>8}")
    for llm_name, metrics, results in [
            ('Llama Agent V25', llama_m['IDS_Agent'], results_llama),
            ('Qwen Agent V25',  qwen_m['IDS_Agent'],  results_qwen),
            ('GLM Agent V25',   glm_m['IDS_Agent'],   results_glm)]:
        m    = metrics
        n_fb = sum(1 for r in results if r['result'].get('used_fallback'))
        print(f"{llm_name:<20} | {m['tp']:>4} | {m['tn']:>4} | {m['fp']:>4} | {m['fn']:>4} | "
              f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
              f"{m['recall']:>6.4f} | {m['f1']:>6.4f} | {n_fb:>8}")

    # HEATMAPS
    print(f"\n{'='*120}")
    print(f"GENERATING CONFUSION MATRIX HEATMAPS")
    print(f"{'='*120}")
    generate_confusion_heatmaps(
        {'Llama': results_llama, 'Qwen': results_qwen, 'GLM': results_glm}, output_dir)

    # XAI
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
        print(f"    Overrode ML:    {len(overrides)}/{len(results)} "
              f"({len(overrides)/len(results)*100:.1f}%)")
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
          f"{'Fallback':>8} | {'Overrides':>10} | {'Net Fix':>8}")
    print(f"  {'-'*70}")
    for llm_name, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
        llm_cor    = sum(1 for r in results if r['correct'])
        llm_acc    = llm_cor / len(results) if results else 0
        delta      = llm_acc - maj_acc
        overrides  = [r for r in results if r['result'].get('llm_overrode_ml')]
        correct_ov = sum(1 for r in overrides if r['correct'])
        wrong_ov   = len(overrides) - correct_ov
        n_fb       = sum(1 for r in results if r['result'].get('used_fallback'))
        net_fix    = correct_ov - wrong_ov
        nf_sign    = "+" if net_fix >= 0 else ""
        print(f"  {llm_name:<12} | {llm_acc:>10.4f} | {delta:>+10.4f} | "
              f"{n_fb:>8} | {len(overrides):>10} | {nf_sign}{net_fix:>7}")

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
            zp = [results[i]['predicted'] for i in zone_idx]
            zone_accs[lname] = sum(a == b for a, b in zip(zone_gt, zp)) / len(zone_gt)
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
    print(f"COMPLEXITY (V25: bounded generation + shared XAI)")
    print(f"{'='*120}")
    print(f"\n  {'Metric':<28} | {'Llama':>10} | {'Qwen':>10} | {'GLM':>10}")
    print(f"  {'-'*65}")
    for mname, key in [('Avg LLM Latency (sec)', 'llm_latency_sec'),
                        ('Avg Stage1 (sec)', 'stage1_latency'),
                        ('Avg Stage2 (sec)', 'stage2_latency'),
                        ('Avg XAI (sec)', 'xai_sec'),
                        ('Avg Prompt (chars)', 'prompt_chars'),
                        ('Avg Response (words)', 'response_words'),
                        ('Avg Throughput (w/s)', 'tokens_per_sec'),
                        ('Total Time (sec)', 'session_total_sec')]:
        vals = []
        for results in [results_llama, results_qwen, results_glm]:
            cdata = [r['result'].get('complexity', {}).get(key, 0) for r in results]
            vals.append(np.sum(cdata) if key == 'session_total_sec'
                        else np.mean(cdata) if cdata else 0)
        print(f"  {mname:<28} | {vals[0]:>10.2f} | {vals[1]:>10.2f} | {vals[2]:>10.2f}")

    # FINAL RANKINGS
    print(f"\n{'='*120}")
    print(f"FINAL RANKINGS | {sc_label} | V25 (SHAP+LIME, always-classify)")
    print(f"{'='*120}")
    best_ml_name = max(ml_names, key=lambda m: llama_m[m]['accuracy'])
    all_ranked   = sorted([
        ('GLM Agent V25',      glm_m['IDS_Agent']['accuracy']),
        ('Qwen Agent V25',     qwen_m['IDS_Agent']['accuracy']),
        ('Llama Agent V25',    llama_m['IDS_Agent']['accuracy']),
        ('Majority Vote',      llama_m['Majority_Vote']['accuracy']),
        (best_ml_name,         llama_m[best_ml_name]['accuracy']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(all_ranked):
        print(f"  {i+1}. {name:22s} {acc:.4f} ({acc*100:.1f}%)")


def main():
    print(f"""
========================================================================
  EV-IDS-Agent VERSION 6 — TRIPLE LLM COMPARISON V25
  (built on V23 — same idea, objective, flow and prompts)

  Llama3 | Qwen3.5 | GLM-4.7-Flash

  V25 fixes (performance/robustness only):
  - Bounded, deterministic generation (V23 sent no limits; GLM averaged
    6,066 words/sample => ~2-day runs)
  - num_ctx=8192 (V23 could silently truncate the system prompt)
  - Thinking capture for Qwen AND GLM (V23 lost GLM's verdict -> 24/50 abstain)
  - Always-classify: unparseable output -> ML majority (no ABSTAIN);
    standard IDS metrics over ALL N samples
  - SHAP+LIME computed ONCE per sample, shared by all 3 LLMs, disk-cached
  - Optional vLLM backend (PagedAttention + continuous batching) + workers

  Status: SHAP={'ON' if SHAP_AVAILABLE else 'OFF (pip install shap)'}  |  LIME={'ON' if LIME_AVAILABLE else 'OFF (pip install lime)'}
========================================================================
    """)

    if not SHAP_AVAILABLE or not LIME_AVAILABLE:
        print("  [WARN] Install missing XAI libraries for full functionality:")
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

    # BACKEND + PARALLELISM
    print("\n" + "=" * 60)
    print("BACKEND")
    print("=" * 60)
    print("""
  ollama : default. NOTE: Ollama does NOT implement PagedAttention or
           continuous batching (inference-server features). Parallel
           workers only help if the server runs with OLLAMA_NUM_PARALLEL.
  vllm   : OpenAI-compatible vLLM server — implements PagedAttention +
           continuous batching. Launch first:  vllm serve <model> --port 8000
    """)
    backend = input("Backend (ollama/vllm) [ollama]: ").strip().lower() or "ollama"
    if backend not in ('ollama', 'vllm'):
        backend = 'ollama'
    default_url = "http://localhost:8000" if backend == 'vllm' else "http://localhost:11434"
    base_url = input(f"Server URL [{default_url}]: ").strip() or default_url

    if use_memory:
        workers = 1
        print("\n  LTM is ON — parallel disabled (memory must accumulate in order).")
    else:
        try:
            workers = int(input("Parallel workers (1=sequential, 2-8) [1]: ").strip() or "1")
        except ValueError:
            workers = 1
        workers = max(1, min(8, workers))
        if backend == 'ollama' and workers > 1:
            print(f"  NOTE: for real gains set OLLAMA_NUM_PARALLEL={workers} on the server.")

    DATA_PATH = (r"D:\OneDrive - Hamad bin Khalifa University\project 2"
                 r"\ev_mil_framework_corrected\dataset"
                 r"\Processed_ACNdata_CMA6_1_fu_4_test.csv")
    WORKSPACE_DIR = "./ev_ids_workspace_v6"

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

    if backend == 'ollama':
        try:
            import requests
            requests.get(f"{base_url}/api/tags", timeout=5).raise_for_status()
            print(f"\nOllama running at {base_url}")
            check_available_models(base_url)
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

    # SHARED XAI STORE — SHAP/LIME computed once per sample for all 3 LLMs.
    # Build it from a throwaway agent's models/scaler (same pickles).
    print(f"\nBuilding shared XAI store (compute-once, disk-cached)...")
    bootstrap = EVIDSAgentV6TripleLLMV25(config,
                    llm_client=type('Null', (), {'model_name': 'none'})(),
                    use_knowledge=False, use_memory=False,
                    scenario_tag='bootstrap', xai_store=None, verbose=False)
    shared_store = bootstrap.xai_store

    def make_agent(model_name):
        client = make_llm_client(backend, model_name, base_url)
        return EVIDSAgentV6TripleLLMV25(
            config, client,
            use_knowledge=use_knowledge, use_memory=use_memory,
            scenario_tag=f"{scenario_tag}_{model_name.split(':')[0]}",
            xai_store=shared_store,
            verbose=(workers == 1))

    # RUN LLAMA
    print(f"\nMANUAL: make sure llama3:latest is available on the backend")
    agent_llama = make_agent("llama3:latest")
    if use_memory and agent_llama.ltm:
        agent_llama.ltm.clear()
        agent_llama.seed_ltm_from_training(train_indices, n_seed=50)
    results_llama = run_classification(agent_llama, "[LLAMA]", "LLAMA3",
                                        selected, df, workers=workers)
    shared_store.save()

    # RUN GLM
    print(f"\nMANUAL: switch to glm-4.7-flash:latest")
    agent_glm = make_agent("glm-4.7-flash:latest")
    if use_memory and agent_glm.ltm:
        agent_glm.ltm.clear()
        agent_glm.seed_ltm_from_training(train_indices, n_seed=50)
    results_glm = run_classification(agent_glm, "[GLM]", "GLM-4.7-FLASH",
                                      selected, df, workers=workers)
    shared_store.save()

    # RUN QWEN
    print(f"\nMANUAL: switch to qwen3.5:latest")
    agent_qwen = make_agent("qwen3.5:latest")
    if use_memory and agent_qwen.ltm:
        agent_qwen.ltm.clear()
        agent_qwen.seed_ltm_from_training(train_indices, n_seed=50)
    results_qwen = run_classification(agent_qwen, "[QWEN]", "QWEN3.5",
                                       selected, df, workers=workers)
    shared_store.save()

    if results_llama and results_qwen and results_glm:
        print_triple_comparison(results_llama, results_qwen, results_glm, results_dir)
    else:
        print("\nIncomplete results")
        return

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"triple_llm_v6_v25_{scenario_tag}_{ts}.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(make_json_serializable({
                'version':       '6_triple_llm_v25',
                'scenario':      scenario_tag,
                'backend':       backend,
                'workers':       workers,
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
    print(f"\nTRIPLE LLM V25 SCENARIO {choice} COMPLETE!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n\nFatal: {e}")
        import traceback
        traceback.print_exc()
