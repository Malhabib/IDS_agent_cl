# simple_run_v6_triple_llm_v30.py
"""
EV-IDS-Agent VERSION 6 - TRIPLE LLM V30 Runner  (final consolidated version)

Pipeline, prompts, 7-step flow, scenarios (A/B/C/D) and the V23-style
explainability output are UNCHANGED. V30 is V23 plus accumulated fixes.

RUN REQUIREMENTS
    All files must sit in the SAME folder:
        ev_ids_agent_v6_triple_llm_v30.py
        simple_run_v6_triple_llm_v30.py
        llm_eval_metrics_v30.py
        selftest_v30.py  integration_test_v30.py  healthcheck_v30.py
    pip install shap lime ollama requests scikit-learn pandas numpy matplotlib
    Edit DATA_PATH in main() to point at your CSV.

RUN THE GATES FIRST — IN THIS ORDER
    python selftest_v30.py           seconds, no GPU
    python integration_test_v30.py   ~1 minute, no GPU
    python healthcheck_v30.py --n 50 minutes, needs Ollama
    python simple_run_v6_triple_llm_v30.py

  Each gate must exit 0 before the next is worth running. The first two need
  neither Ollama nor your dataset; the third measures the real per-session cost
  and projects the total. Every unusable run this project has paid for was
  detectable by one of these three within minutes of starting.

WHAT THIS RUNNER PRODUCES
  - Full V23 per-session trace: every step, the Stage 1 prompt containing each
    model's SHAP/LIME evidence, both LLM responses, and the final decision.
    Each session is also written to workspace/transcripts/<tag>/session_<i>.txt
  - Standard IDS metrics over ALL N samples (always-classify, no ABSTAIN),
    plus a V23-comparable column that scores only LLM-answered sessions.
  - Zone analysis, override analysis, complexity, confusion-matrix heatmaps.
  - LLM EVALUATION METRICS: Wilson CIs, McNemar, Cohen's kappa, ECE,
    format compliance, XAI faithfulness, and the ratio>1.3 oracle baseline.
  - Optional temperature sweep: effect of decoding temperature on accuracy.

SAMPLING DESIGNS (prompted at run time)
  1) balanced   — class-proportional random draw. On this dataset that is
                  almost perfectly separable, so strong models score ~96-100%.
  2) stratified — force-includes the hard counterexamples that exist in the
                  data (normal sessions inside the attack ratio band). This is
                  the discriminative, publishable benchmark.
  Report both: the gap between them is the headline finding.
"""

import os, sys, json, pickle, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix
from llm_eval_metrics_v30 import (
    wilson_ci, mcnemar_test, cohens_kappa, expected_calibration_error,
    format_compliance, xai_faithfulness, temperature_table,
    ratio_oracle_predictions, llm_error_rate, answered_only,
)
from ev_ids_agent_v6_triple_llm_v30 import (
    make_llm_client, EVIDSAgentV6TripleLLMV30, build_xai_store,
    unload_ollama_model,
    auto_train_models_v6, get_column_mapping, classify_difficulty_zone,
    parse_datetime_to_timestamp, SHAP_AVAILABLE, LIME_AVAILABLE,
    NUM_CTX, min_viable_num_ctx
)

PRINT_LOCK = threading.Lock()

# ── CIRCUIT BREAKER (V30) ────────────────────────────────────────────────────
# Thresholds for stopping a run that is already known to be unusable. See
# run_classification._breaker(). These are deliberately generous: they do not
# stop a slow run, they stop a broken one.
BREAKER_MIN_SESSIONS   = 5      # never judge on fewer than this
BREAKER_MAX_ERROR_RATE = 0.20   # >20% of sessions with no LLM answer
BREAKER_MAX_HOURS      = 8.0    # projected wall-clock for the whole model


def check_available_models(ollama_url="http://localhost:11434"):
    """List Ollama models. Handles both old (dict['name']) and new
    (ListResponse .model attribute) ollama-python APIs."""
    try:
        import ollama
        info = ollama.list()
        raw  = info.get('models', []) if isinstance(info, dict) else getattr(info, 'models', [])
        models = []
        for m in raw:
            if isinstance(m, dict):
                name = m.get('name') or m.get('model')
            else:
                name = getattr(m, 'model', None) or getattr(m, 'name', None)
            if name:
                models.append(name)
        print(f"\nAVAILABLE MODELS:")
        for m in models:
            print(f"  {m}")
        return models
    except Exception as e:
        print(f"Error checking models: {e}")
        return []


def verify_models_installed(model_ids, base_url="http://localhost:11434"):
    """
    Confirm every model the run needs actually exists on THIS machine.

    Without this, a missing model is discovered only when its turn comes. On a
    three-model study that means the first model runs to completion -- an hour
    -- and only then does the second fail with a 404 and get skipped, leaving
    an incomplete result set and an hour of wasted GPU time. Model names are
    also not portable between machines: a configuration validated on one box
    referenced glm4 and qwen2.5:7b, neither of which was installed on the next.

    Returns the list of missing (slot, name) pairs; prints what IS installed and
    the exact pull commands.
    """
    import requests
    try:
        tags = requests.get(f"{base_url}/api/tags", timeout=10).json()
    except Exception as e:
        print(f"  Could not list models: {e}")
        return []
    installed = {}
    for m in tags.get('models', []):
        name = m.get('name') or m.get('model')
        if name:
            installed[name] = float(m.get('size', 0) or 0)

    missing = [(slot, name) for slot, name in model_ids.items()
               if name not in installed]
    if not missing:
        print(f"\n  All {len(model_ids)} configured models are installed.")
        return []

    print(f"\n  MISSING MODELS — the run would fail partway through")
    print(f"  {'-'*66}")
    for slot, name in missing:
        print(f"    {slot:<6} {name:<26} NOT INSTALLED")
    print(f"\n  Installed on this machine:")
    for name, size in sorted(installed.items(), key=lambda kv: kv[1]):
        gb = size / 1024 ** 3
        cloud = " (cloud — runs remotely, uses no local VRAM)" if size == 0 else ""
        print(f"    {name:<26} {gb:5.1f} GB{cloud}")
    print(f"\n  Either install what the study expects:")
    for _, name in missing:
        print(f"    ollama pull {name.split(':')[0] if name.endswith(':latest') else name}")
    print(f"  or edit model_ids in main() to use models you already have.")
    print(f"\n  Installing the same models on every machine is strongly preferred:")
    print(f"  results from different models are not comparable, so substituting")
    print(f"  one silently would make the two machines' numbers incommensurable.")
    return missing


def check_gpu_residency(model_name, base_url="http://localhost:11434"):
    """Load one model and report how much of it Ollama placed in VRAM.

    A model that does not fit in VRAM is not merely slow: on the target
    machine glm-4.7-flash (19 GB on an 8 GB card) ran at 67%/33% CPU/GPU and
    0.08 words/sec, so calls timed out, timed-out calls fell back to the ML
    verdict, and two runs of identical code on identical data at temperature
    0 produced different accuracies. Partial residency therefore invalidates
    the experiment rather than lengthening it, which is why this runs before
    any sample is classified.

    Returns the fraction of the model resident in VRAM (1.0 = fully on GPU),
    or None if Ollama could not be queried.
    """
    import requests
    try:
        # A one-token generate is the cheapest way to force the load.
        requests.post(f"{base_url}/api/generate",
                      json={'model': model_name, 'prompt': 'hi',
                            'stream': False, 'options': {'num_predict': 1}},
                      timeout=600).raise_for_status()
        ps = requests.get(f"{base_url}/api/ps", timeout=10).json()
    except Exception as e:
        print(f"  [{model_name}] residency check failed: {e}")
        return None

    for m in ps.get('models', []):
        if m.get('model') != model_name and m.get('name') != model_name:
            continue
        total = float(m.get('size', 0) or 0)
        vram  = float(m.get('size_vram', 0) or 0)
        if total <= 0:
            return None
        frac = vram / total
        gb   = lambda b: b / (1024 ** 3)
        ctx  = m.get('context_length') or m.get('details', {}).get('context_length')
        note = "100% GPU" if frac >= 0.999 else f"{(1-frac)*100:.0f}%/{frac*100:.0f}% CPU/GPU"
        print(f"  {model_name:<24} {gb(total):5.1f} GB  {note:<18}"
              + (f"  ctx={ctx}" if ctx else ""))
        return frac
    print(f"  {model_name:<24} not reported by /api/ps")
    return None


def preflight_gpu_residency(model_ids, base_url="http://localhost:11434"):
    """Check every model the run will use, unloading each one afterwards.

    Prints a table and returns the list of slots that are not fully resident.
    """
    print("\nGPU RESIDENCY PREFLIGHT")
    print("  Each model is loaded once and released. A model that reports")
    print("  anything other than 100% GPU will produce unreproducible results.")
    degraded = []
    for slot, name in model_ids.items():
        frac = check_gpu_residency(name, base_url)
        if frac is None or frac < 0.999:
            degraded.append((slot, name, frac))
        unload_ollama_model(name, base_url)
    if degraded:
        print("\n  NOT FULLY ON GPU:")
        for slot, name, frac in degraded:
            where = "unknown" if frac is None else f"{frac*100:.0f}% GPU"
            print(f"    {slot:<6} {name:<24} {where}")
        print("  Free VRAM (close browsers, Teams, Copilot and other desktop")
        print("  applications) or choose a smaller variant before running.")
    else:
        print("\n  All models fully resident in VRAM.")
    return degraded


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


def select_samples(df, test_indices, cmap, seed=42):
    """
    Two sampling designs.

    'balanced'  — the original: class-proportional random draw. On this dataset
                  every attack has ratio > 1.3 and every ratio in 0.8-1.2 is
                  normal, so a random draw is almost perfectly separable and a
                  strong LLM scores ~100%. That number is not defensible in a
                  paper: it measures dataset separability, not reasoning.

    'stratified'— deliberately includes the HARD CASES that already exist in
                  the data: normal sessions whose delivery ratio falls inside
                  the attack band (this dataset holds 13 in EASY_ATTACK and 4
                  in BORDERLINE_HIGH). Random sampling picks them with
                  probability ~17/5000, so they are effectively never tested.
                  Including them is not making the task artificially hard — it
                  is no longer ignoring the hard real data, and it is what
                  makes the evaluation discriminative between models.
    """
    test_df = df.loc[test_indices].copy()
    ratio   = (test_df[cmap['kWhDelivered']] /
               test_df[cmap['RequestedDemand']].replace(0, np.nan))
    test_df['_ratio'] = ratio
    test_df['_zone']  = ratio.apply(
        lambda r: classify_difficulty_zone(r) if pd.notna(r) else 'UNKNOWN')

    # Hard cases = label contradicts the ratio band
    hard = test_df[((test_df['_ratio'] > 1.3) & (test_df['label'] == 0)) |
                   ((test_df['_ratio'].between(0.8, 1.2)) & (test_df['label'] == 1))]

    print(f"\nTest set: {len(test_df)} samples")
    print(f"  HARD counterexamples available (label contradicts ratio band): {len(hard)}")
    print(f"    -> normals inside the attack ratio band, and vice versa.")
    print(f"    -> a purely random draw almost never includes these, which is")
    print(f"       why a strong LLM can score 100% on this task.")

    min_samples = 50
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

    print("""
  SAMPLING DESIGN
    1) balanced   — class-proportional random draw (previous behaviour)
    2) stratified — same, but force-includes the hard counterexamples
                    (RECOMMENDED for publishable, discriminative results)
""")
    design = input("  Design (1/2) [2]: ").strip() or "2"

    # SAMPLING SEED. A single 50-session batch is a fragile basis for a headline
    # number: one perfect run invites the reviewer question "would it hold on a
    # different draw?". Running several disjoint batches with different seeds
    # and reporting mean +/- std answers that directly.
    try:
        seed = int(input("  Sampling seed (change it for a fresh batch) [42]: ").strip() or "42")
    except ValueError:
        seed = 42
    print(f"  Sampling seed: {seed}")

    if design == "2" and len(hard) > 0:
        n_hard = min(len(hard), max(1, int(round(n * 0.20))))   # ~20% hard
        hs     = hard.sample(n=n_hard, random_state=seed)
        rest   = test_df.drop(index=hs.index)
        n_rem  = n - n_hard
        r_norm = rest[rest['label'] == 0]
        r_mal  = rest[rest['label'] == 1]
        frac_n = len(r_norm) / max(1, len(rest))
        nn     = int(n_rem * frac_n)
        nm     = n_rem - nn
        sel = pd.concat([
            hs,
            r_norm.sample(n=min(nn, len(r_norm)), random_state=seed),
            r_mal.sample(n=min(nm, len(r_mal)), random_state=seed),
        ]).sample(frac=1, random_state=seed)
        n_m = int((sel['label'] == 1).sum())
        print(f"  Selected {len(sel)} samples "
              f"({len(sel)-n_m}N, {n_m}M) including {n_hard} HARD counterexamples")
    else:
        normal_df = test_df[test_df['label'] == 0]
        mal_df    = test_df[test_df['label'] == 1]
        nr  = len(normal_df) / len(test_df)
        nn  = int(n * nr)
        nm  = n - nn
        sel = pd.concat([
            normal_df.sample(n=min(nn, len(normal_df)), random_state=seed),
            mal_df.sample(n=min(nm, len(mal_df)), random_state=seed),
        ]).sample(frac=1, random_state=seed)
        print(f"  Selected {len(sel)} samples ({nn}N, {nm}M) [balanced]")
    return sel.index.tolist(), seed


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
    n_error = 0
    aborted = None

    def _progress(row):
        nonlocal done, n_error
        done += 1
        if row['result'].get('llm_error'):
            n_error += 1
        elapsed = time.time() - t_start
        eta     = (elapsed / done) * (total - done)
        sym = ("OK" if row['correct'] else "WRONG")
        fb  = " FB" if row['result'].get('used_fallback') else ""
        er  = " LLM-ERROR" if row['result'].get('llm_error') else ""
        with PRINT_LOCK:
            print(f"  [{done:>3}/{total}] idx={row['index']:<6} "
                  f"truth={row['ground_truth']:<9} pred={row['predicted']:<9} "
                  f"[{sym}]{fb}{er}  elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")

    def _breaker():
        """
        Stop a run that is already known to be unusable.

        Two things make a run worthless no matter how long it continues: calls
        that do not complete, and a projected duration nobody will wait for.
        Both are visible within the first handful of sessions. Detecting them at
        session 5 instead of session 50 is the difference between losing four
        minutes and losing two days -- which is precisely what happened to the
        V29.1 and V29.2 runs.
        """
        if done < BREAKER_MIN_SESSIONS:
            return None
        if n_error / done > BREAKER_MAX_ERROR_RATE:
            return (f"{n_error} of the first {done} sessions were LLM errors "
                    f"({n_error/done*100:.0f}%). The transport is failing; the "
                    f"remaining {total-done} sessions would fail the same way.")
        projected_h = (time.time() - t_start) / done * total / 3600.0
        if projected_h > BREAKER_MAX_HOURS:
            return (f"measured {(time.time()-t_start)/done:.0f}s per session, so "
                    f"{total} sessions project to {projected_h:.1f} h "
                    f"(limit {BREAKER_MAX_HOURS} h). Almost always this means the "
                    f"model is not fully resident in VRAM.")
        return None

    if workers <= 1:
        for idx in selected:
            row = _classify_one(agent, idx, truths[idx])
            if row is not None:
                results.append(row)
                _progress(row)
                if row['correct'] and not row['result'].get('llm_error'):
                    agent.store_correct_decision_in_ltm(idx, True, row['result'])
            aborted = _breaker()
            if aborted:
                print(f"\n  [CIRCUIT BREAKER] {label} stopped after {done} of "
                      f"{total} sessions.\n    {aborted}")
                print(f"    Run 'python healthcheck_v30.py' to confirm the cause "
                      f"before retrying.")
                break
    else:
        # Parallel: memory-off scenarios only (enforced by caller). The full
        # V23 per-session trace is still printed — each session is buffered by
        # the agent and flushed as one contiguous block under the print lock.
        #
        # The circuit breaker MUST run here too. It was originally written only
        # into the serial branch above, so a run at workers=8 with a 28% LLM
        # error rate went the full 50 sessions and cost an hour before reporting
        # that it was unpublishable. A guard that only guards one code path is
        # not a guard.
        order = {idx: i for i, idx in enumerate(selected)}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_classify_one, agent, idx, truths[idx]): idx
                    for idx in selected}
            try:
                for fut in as_completed(futs):
                    row = fut.result()
                    if row is not None:
                        results.append(row)
                        _progress(row)
                    aborted = _breaker()
                    if aborted:
                        print(f"\n  [CIRCUIT BREAKER] {label} stopping after "
                              f"{done} of {total} sessions.\n    {aborted}")
                        print(f"    Cancelling queued sessions. Run 'python "
                              f"healthcheck_v30.py' to confirm the cause.")
                        for f in futs:
                            f.cancel()
                        break
            finally:
                # Threads already running cannot be cancelled, so the pool is
                # drained rather than left to finish silently in the background.
                pool.shutdown(wait=True, cancel_futures=True)
        results.sort(key=lambda r: order[r['index']])

    cc = sum(1 for r in results if r['correct'])
    fb = sum(1 for r in results if r['result'].get('used_fallback'))
    er = sum(1 for r in results if r['result'].get('llm_error'))
    el = time.time() - t_start
    if results:
        print(f"\n{label} COMPLETE: {cc}/{len(results)} correct "
              f"({cc/len(results)*100:.1f}%), fallbacks={fb}, llm_errors={er}, "
              f"time={el/60:.1f} min ({el/len(results):.1f}s/sample)")
        health = (agent.llm_client.health()
                  if hasattr(agent.llm_client, 'health') else {})
        if health:
            print(f"  transport: {health['calls']} calls, "
                  f"{health['timeouts']} timeouts, {health['errors']} errors, "
                  f"{health['budget_clamps']} budget clamps, "
                  f"granted ctx {health['granted_ctx']}")
        if er:
            print(f"  [WARNING] {er} of {len(results)} sessions had no LLM answer. "
                  f"Their labels come from the ML majority, NOT from the model, "
                  f"so the accuracy below understates and misattributes "
                  f"{label}'s behaviour. This run is not publishable as it "
                  f"stands.")
    if aborted:
        print(f"  [INCOMPLETE] {len(results)}/{total} sessions — {aborted}")
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
        all_models_data[f"{llm_name} Agent (V30)"] = [r['predicted'] for r in results]

    n_models = len(all_models_data)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle('Confusion Matrices — V30 (SHAP+LIME, always-classify)',
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
    path = os.path.join(output_dir, 'confusion_matrices_v30.png')
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
    print(f"TRIPLE LLM COMPARISON — VERSION 6 V30 (SHAP+LIME, always-classify) | {sc_label}")
    print(f"{'='*160}")
    xai_on = results_llama[0]['result'].get('xai_enabled', True) if results_llama else True
    print(f"  SHAP: {'ON' if SHAP_AVAILABLE else 'OFF'}  |  "
          f"LIME: {'ON' if LIME_AVAILABLE else 'OFF'}  |  "
          f"XAI IN PROMPT: {'ON' if xai_on else 'OFF (ablation)'}")

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
    # DECISION PROVENANCE. Reporting only a fallback count hid the fact that a
    # large share of decisions were previously taken by a word-frequency
    # heuristic rather than a stated verdict. Every decision is now attributable.
    print(f"\nDECISION PROVENANCE (where each verdict actually came from):")
    print(f"  {'LLM':<8} | {'stated':>7} | {'repaired':>9} | {'ML fallback':>12} | "
          f"{'LLM error':>10} | {'n':>4}")
    print(f"  {'-'*64}")
    any_error = False
    for name, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
        src = [r['result'].get('verdict_source', 'stated') for r in results]
        n_err = src.count('llm_error')
        any_error = any_error or n_err > 0
        print(f"  {name:<8} | {src.count('stated'):>7} | {src.count('repaired'):>9} | "
              f"{src.count('ml_fallback'):>12} | {n_err:>10} | {len(results):>4}")
    print(f"  Only 'stated' and 'repaired' reflect the model's own judgement.")
    print(f"  'ML fallback' = the model answered but stated no verdict.")
    print(f"  'LLM error'   = the model never answered (timeout or transport fault).")
    if any_error:
        print(f"\n  [NOT PUBLISHABLE] Sessions in the 'LLM error' column were not")
        print(f"  decided by any model. Their labels come from the ML majority, so")
        print(f"  the accuracy tables below mix model behaviour with transport")
        print(f"  failure. Two runs with different error counts will disagree even")
        print(f"  at temperature 0. Fix the environment (healthcheck_v30.py) and")
        print(f"  re-run before drawing any conclusion from these numbers.")

    # V23-COMPARABLE METRIC ----------------------------------------------------
    # V23 reported accuracy EXCLUDING abstained sessions, so its headline
    # numbers were computed on a subset (e.g. Qwen 95.65% on 46/50, GLM on
    # 26/50). Reporting only the all-N number would understate V27 against
    # those figures. Both are printed so the comparison is like-for-like.
    print(f"\n{'='*120}")
    print(f"ACCURACY — V23-COMPARABLE (LLM-answered sessions only) vs ALL-N")
    print(f"{'='*120}")
    print(f"  {'LLM':<10} | {'V23-style Acc':>14} | {'n answered':>11} | "
          f"{'All-N Acc':>10} | {'n':>4}")
    print(f"  {'-'*62}")
    for name, results in [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]:
        answered = [r for r in results if r['result'].get('llm_parsed', True)]
        acc_v23  = (sum(1 for r in answered if r['correct']) / len(answered)) if answered else 0.0
        acc_all  = (sum(1 for r in results if r['correct']) / len(results)) if results else 0.0
        print(f"  {name:<10} | {acc_v23:>14.4f} | {len(answered):>11} | "
              f"{acc_all:>10.4f} | {len(results):>4}")
    print(f"\n  V23-style column = same convention as the V23 tables you compared against.")
    print(f"  All-N column     = every session classified (IDS-standard).")

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
            ('Llama Agent V30', llama_m['IDS_Agent'], results_llama),
            ('Qwen Agent V30',  qwen_m['IDS_Agent'],  results_qwen),
            ('GLM Agent V30',   glm_m['IDS_Agent'],   results_glm)]:
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
    print(f"COMPLEXITY (V30: bounded generation + shared XAI)")
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

    # ── LLM EVALUATION METRICS (research-paper measures) ──────────────────
    print(f"\n{'='*120}")
    print(f"LLM EVALUATION METRICS")
    print(f"{'='*120}")

    llm_sets = [('Llama', results_llama), ('Qwen', results_qwen), ('GLM', results_glm)]

    # Trivial single-rule reference baseline
    oracle = ratio_oracle_predictions(results_llama, threshold=1.3)
    oracle_acc = sum(1 for p, t in zip(oracle, ground_truths) if p == t) / len(ground_truths)
    print(f"\n  REFERENCE BASELINE  'delivery_ratio > 1.3 => Attack': {oracle_acc:.4f}")
    print(f"    Reviewers will compute this; if it is close to the LLM scores,")
    print(f"    the sampled task is largely solvable without any model.")

    # Accuracy with Wilson 95% CI
    print(f"\n  ACCURACY WITH 95% CONFIDENCE INTERVAL (Wilson)")
    print(f"  {'LLM':<10} | {'Acc':>7} | {'95% CI':>18} | {'n':>4}")
    print(f"  {'-'*48}")
    for name, res in llm_sets:
        k = sum(1 for r in res if r['correct'])
        lo, hi = wilson_ci(k, len(res))
        print(f"  {name:<10} | {k/len(res):>7.4f} | [{lo:>6.4f}, {hi:>6.4f}] | {len(res):>4}")
    print(f"    Overlapping intervals mean the difference is NOT significant.")

    # Paired significance + agreement
    print(f"\n  PAIRWISE COMPARISON (McNemar exact test / Cohen's kappa)")
    print(f"  {'Pair':<18} | {'b':>3} | {'c':>3} | {'p-value':>9} | {'signif':>7} | {'kappa':>6}")
    print(f"  {'-'*64}")
    for (na, ra), (nb, rb) in [(llm_sets[0], llm_sets[1]),
                               (llm_sets[0], llm_sets[2]),
                               (llm_sets[1], llm_sets[2])]:
        ca = [r['correct'] for r in ra]
        cb = [r['correct'] for r in rb]
        mc = mcnemar_test(ca, cb)
        kp = cohens_kappa([r['predicted'] for r in ra], [r['predicted'] for r in rb])
        print(f"  {na+' vs '+nb:<18} | {mc['b']:>3} | {mc['c']:>3} | "
              f"{mc['p_value']:>9.4f} | {str(mc['significant']):>7} | {kp:>6.3f}")

    # Transport integrity — a validity precondition, printed before anything
    # that could be mistaken for a property of the models.
    print(f"\n  TRANSPORT INTEGRITY (did the calls complete at all?)")
    print(f"  {'LLM':<10} | {'Errors':>7} | {'Rate':>7} | {'Usable':>7}")
    print(f"  {'-'*40}")
    for name, res in llm_sets:
        er = llm_error_rate(res)
        print(f"  {name:<10} | {er['errors']:>7} | {er['rate']:>7.4f} | "
              f"{('yes' if er['valid'] else 'NO'):>7}")
    print(f"    'NO' means some sessions were labelled by the ML majority "
          f"because\n    the model never answered. Those runs are not comparable "
          f"to each other.")

    # Format compliance / instruction following
    print(f"\n  FORMAT COMPLIANCE (verdict emitted in the requested format)")
    print(f"  {'LLM':<10} | {'Rate':>7} | {'Parsed':>7} | {'Fallback':>9} | {'n':>4}")
    print(f"  {'-'*48}")
    for name, res in llm_sets:
        fc = format_compliance(res)
        print(f"  {name:<10} | {fc['rate']:>7.4f} | {fc['parsed']:>7} | "
              f"{fc['fallback']:>9} | {fc['n']:>4}")
    print(f"    n excludes sessions the model never answered.")

    # XAI faithfulness — substantiates the explainability contribution
    def _pct(v):
        # None = no sessions in this split. Printing 0.0000 for an empty
        # denominator reads as a catastrophic result rather than as no data.
        return "     n/a" if v is None else f"{v:>8.4f}"

    print(f"\n  XAI FAITHFULNESS (LLM reasoning cites the dominant SHAP driver)")
    print(f"  {'LLM':<10} | {'Overall':>8} | {'When correct':>13} | {'When wrong':>11}")
    print(f"  {'-'*50}")
    for name, res in llm_sets:
        xf = xai_faithfulness(res)
        print(f"  {name:<10} | {_pct(xf['rate'])} | {_pct(xf['rate_correct']):>13} | "
              f"{_pct(xf['rate_wrong']):>11}")
    print(f"    Lower faithfulness on wrong decisions = errors coincide with")
    print(f"    ignoring the SHAP/LIME evidence. 'n/a' = no sessions in that split.")

    # Calibration
    print(f"\n  CALIBRATION (Expected Calibration Error, confidence vs accuracy)")
    for name, res in llm_sets:
        ece = expected_calibration_error(
            [r['result'].get('confidence', 0.5) for r in res],
            [r['correct'] for r in res])
        print(f"    {name:<8} ECE={ece['ece']:.4f}")
        for b in ece['bins']:
            print(f"      conf {b['range']}  n={b['n']:>3}  "
                  f"mean_conf={b['confidence']:.3f}  acc={b['accuracy']:.3f}")

    # FINAL RANKINGS
    print(f"\n{'='*120}")
    print(f"FINAL RANKINGS | {sc_label} | V30 (SHAP+LIME, always-classify)")
    print(f"{'='*120}")
    best_ml_name = max(ml_names, key=lambda m: llama_m[m]['accuracy'])
    all_ranked   = sorted([
        ('GLM Agent V30',      glm_m['IDS_Agent']['accuracy']),
        ('Qwen Agent V30',     qwen_m['IDS_Agent']['accuracy']),
        ('Llama Agent V30',    llama_m['IDS_Agent']['accuracy']),
        ('Majority Vote',      llama_m['Majority_Vote']['accuracy']),
        (best_ml_name,         llama_m[best_ml_name]['accuracy']),
    ], key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(all_ranked):
        print(f"  {i+1}. {name:22s} {acc:.4f} ({acc*100:.1f}%)")


def run_temperature_sweep(config, backend, base_url, scenario_tag,
                          use_knowledge, use_memory, shared_store,
                          selected, df, temps, model_ids, results_dir, sc_k,
                          use_xai=True):
    """
    Effect of decoding temperature on accuracy across LLM models.

    Standard LLM-evaluation figure: the same sessions are re-classified at
    each temperature for every model, so differences are attributable to
    decoding alone (samples, prompts, ML votes and SHAP/LIME are identical).
    """
    print(f"\n{'='*120}")
    print(f"TEMPERATURE SWEEP — {len(temps)} temperatures x {len(model_ids)} models "
          f"x {len(selected)} sessions")
    print(f"{'='*120}")
    sweep = {}
    for slot, model_name in model_ids.items():
        sweep[slot] = {}
        for t in temps:
            if backend == 'ollama':
                unload_ollama_model(model_name, base_url)
            print(f"\n--- {slot} @ temperature={t} ---")
            client = make_llm_client(backend, model_name, base_url, temperature=t)
            agent  = EVIDSAgentV6TripleLLMV30(
                config, client, use_knowledge=use_knowledge, use_memory=use_memory,
                scenario_tag=f"{scenario_tag}_{slot}_T{t}",
                xai_store=shared_store, verbose=False, print_lock=PRINT_LOCK,
                self_consistency_k=1, use_xai=use_xai)   # k=1: isolate the temperature effect
            rows = []
            for idx in selected:
                gt  = "Malicious" if df.loc[idx, 'label'] == 1 else "Normal"
                row = _classify_one(agent, idx, gt)
                if row:
                    rows.append(row)
            acc = sum(1 for r in rows if r['correct']) / len(rows) if rows else 0
            print(f"    {slot} T={t}: accuracy={acc:.4f} on {len(rows)} sessions")
            sweep[slot][t] = rows

    print(f"\n{'='*120}")
    print(f"EFFECT OF TEMPERATURE ON ACCURACY ACROSS LLM MODELS")
    print(f"{'='*120}")
    print(temperature_table(sweep))

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"temperature_sweep_{scenario_tag}_{ts}.json")
    try:
        with open(path, 'w') as f:
            json.dump(make_json_serializable(
                {'temperatures': temps,
                 'sweep': {k: {str(t): v for t, v in d.items()}
                           for k, d in sweep.items()}}), f, indent=2)
        print(f"\n  Saved sweep data: {path}")
    except Exception as e:
        print(f"  Save error: {e}")
    return sweep


def main():
    print(f"""
========================================================================
  EV-IDS-Agent VERSION 6 — TRIPLE LLM COMPARISON V30
  (built on V23 — same idea, objective, flow and prompts)

  Llama3 | Qwen3.5 | GLM-4.7-Flash

  V30 = V23 pipeline + accumulated fixes:
  - SHAP/LIME direction convention CORRECTED: attributions are taken against
    the Attack class, so "+ = toward Attack" is literally true. Previously
    every model predicting Normal had its direction inverted in the prompt,
    which drove the false-alarm rate to 0.42-0.45.
  - V23 generation semantics restored (uncapped output, default temperature);
    a per-model policy only guards against runaway loops.
  - Robust verdict extraction + repair; unparseable output falls back to the
    ML majority, so every session is scored (no ABSTAIN).
  - SHAP+LIME computed once per sample, shared by all 3 LLMs, disk-cached and
    invalidated automatically when the ML models are retrained.
  - Stratified hard-case sampling option for publishable results.
  - LLM evaluation metrics + optional temperature sweep.

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

    # ── XAI ABLATION ──────────────────────────────────────────────────────
    print("""
============================================================
XAI (SHAP + LIME) — ABLATION SWITCH
============================================================
  ON  : each ML model is shown with its SHAP contributions and LIME weights
  OFF : each ML model is shown with prediction + confidence only

  Everything else is identical -- same prompts, same 7-step flow, same
  samples, same display. Running the SAME samples once with ON and once
  with OFF isolates the contribution of SHAP/LIME exactly.

  NOTE: SHAP/LIME here are computed on deliberately underfit models, so
  their attributions can be uninformative or misleading. Measuring this
  is the point of the switch.
""")
    xai_in   = input("  XAI (on/off) [on]: ").strip().lower() or "on"
    use_xai  = xai_in not in ("off", "no", "n", "0")
    scenario_tag = f"{scenario_tag}_{'XAIon' if use_xai else 'XAIoff'}"
    print(f"  RAG={'ON' if use_knowledge else 'OFF'}, "
          f"LTM={'ON' if use_memory else 'OFF'}, "
          f"XAI={'ON' if use_xai else 'OFF'}")

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

    # Model names per LLM slot (vLLM serves HuggingFace ids, not Ollama tags)
    # Model choice is constrained by VRAM. Measured on the target machine
    # (Quadro M4000, 8 GB, with roughly 7.7 GB held by desktop applications):
    #   glm-4.7-flash  19.0 GB  -> 2.4x the whole card; Ollama reported
    #                             67%/33% CPU/GPU, 0.08 words/sec and eleven
    #                             hours per session, and requests timed out
    #   glm4            5.5 GB  -> same family, fits alongside a 4 KV cache
    #   llama3          4.7 GB  -> fits
    #   qwen3.5         6.6 GB  -> fits only with desktop applications closed
    # A model that does not fit is not slow, it is unmeasurable: timed-out
    # calls become ML fallbacks and identical inputs stop giving identical
    # outputs.
    # THESE ARE THE STUDY'S TARGET MODELS. They are not chosen for convenience
    # and must not be substituted to make a run fit a particular GPU: results
    # from different models are not comparable, so a silent substitution would
    # change what the study measures. If a target model does not fit, the fix
    # is to give it more VRAM or a smaller num_ctx -- not a different model.
    # A substitution, if one is genuinely necessary, is a documented deviation
    # to be reported in the write-up, not a default.
    #
    # Measured loaded footprints (see healthcheck_v30.py and vram_footprints.json):
    #                     @num_ctx 8192   @num_ctx 6144
    #   llama3                  5.5 GB          5.2 GB
    #   qwen3.5                 5.8 GB          5.8 GB
    #   glm-4.7-flash          18.3 GB         18.2 GB   (17.7 GB of weights)
    #
    # glm-4.7-flash cannot be made to fit a 6 GB or an 8 GB card by any window
    # setting, because the weights alone exceed both. It needs roughly a 24 GB
    # GPU, or the cloud variant, which runs full-size weights remotely.
    #
    # Set EV_IDS_NUM_CTX=6144 on a tight card: it is the pipeline floor and it
    # shrinks the KV cache without changing which models are used.
    model_ids = {'llama': 'llama3:latest',
                 'glm':   'glm-4.7-flash:latest',
                 'qwen':  'qwen3.5:latest'}   # used by the run and the sweep

    # A per-machine override, so a second machine does not need the file edited.
    #   set EV_IDS_MODELS=llama3:latest,glm-5:cloud,qwen3.5:latest
    _override = os.environ.get('EV_IDS_MODELS', '').strip()
    if _override:
        parts = [p.strip() for p in _override.split(',') if p.strip()]
        if len(parts) == 3:
            model_ids = dict(zip(('llama', 'glm', 'qwen'), parts))
            print(f"\n  EV_IDS_MODELS override in effect: {model_ids}")
            print(f"  Any substitution away from the study's target models is a")
            print(f"  documented deviation and must be reported with the results.")
        else:
            print(f"\n  [WARN] EV_IDS_MODELS needs exactly 3 comma-separated names "
                  f"(llama,glm,qwen); got {len(parts)}. Ignoring it.")

    if backend == 'vllm':
        # Pre-flight: vLLM is a separate inference server. It does NOT run
        # natively on Windows (WSL2 or Docker required) and it serves
        # HuggingFace model ids — Ollama tags do not exist on it.
        try:
            import requests
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            r.raise_for_status()
            served = [m.get('id') for m in r.json().get('data', [])]
            print(f"\n  vLLM server OK. Served models: {served}")
            print("  Enter the served model id for each LLM slot:")
            for k in model_ids:
                v = input(f"    {k} model id [{model_ids[k]}]: ").strip()
                if v:
                    model_ids[k] = v
        except Exception as e:
            print(f"\n  [ERROR] No vLLM server reachable at {base_url}")
            print(f"          ({e})")
            print("""
  vLLM was not found running. Notes:
    - vLLM is a SEPARATE inference server; this script is only its client.
    - vLLM does not run natively on Windows — it needs WSL2 or Docker:
        pip install vllm            (inside WSL2/Linux)
        vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000
    - vLLM serves HuggingFace model ids; Ollama tags (llama3:latest)
      do not exist on a vLLM server.
  On a Windows GPU machine the practical option is Ollama parallelism:
        setx OLLAMA_NUM_PARALLEL 4     (then restart the Ollama service)
""")
            resp = input("  Fall back to the Ollama backend? (yes/no): ").lower().strip()
            if resp in ('yes', 'y'):
                backend  = 'ollama'
                base_url = "http://localhost:11434"
            else:
                return

    if use_memory:
        workers = 1
        print("\n  LTM is ON — parallel disabled (memory must accumulate in order).")
    else:
        print("""
  Parallel workers speed the run up ONLY if the server actually serves
  concurrent requests. For Ollama you must set this BEFORE starting it:
      Windows:  setx OLLAMA_NUM_PARALLEL 4     (then restart Ollama)
  Otherwise requests queue and you see bursts of N finishing together.
  The full V23 per-session trace is printed either way.

  MEASURED CONSEQUENCE — a run at workers=8 on an 8 GB card produced
  15 timeouts and 14 of 50 sessions with no LLM answer at all. Those
  sessions were labelled by the ML majority, so the reported 96.0%
  described the fallback, not the model, and the run was unpublishable.
  Parallel slots share one KV cache: each worker gets a fraction of
  num_ctx, generation is starved, and calls run past the deadline.
  Use 1 unless you have verified the granted context at your worker
  count with healthcheck_v30.py.""")
        try:
            workers = int(input("Parallel workers (1=sequential, 2-8) [1]: ").strip() or "1")
        except ValueError:
            workers = 1
        workers = max(1, min(8, workers))
        if backend == 'ollama' and workers > 1:
            env_par = os.environ.get('OLLAMA_NUM_PARALLEL')
            if env_par:
                print(f"  OLLAMA_NUM_PARALLEL={env_par} detected in this shell.")
            else:
                print(f"  [WARN] OLLAMA_NUM_PARALLEL is not set — Ollama will most "
                      f"likely queue your {workers} workers (bursts of {workers}).")
            print("  [WARN] ACCURACY RISK: Ollama splits the KV cache across parallel\n"
                  "         slots, so each request may get a fraction of num_ctx and the\n"
                  "         SHAP/LIME evidence can be truncated. Use 1 for best accuracy.")
            print("  [WARN] On an 8 GB card this re-creates the exact bug V30 fixed.\n"
                  "         A granted window of 4096 instead of 8192 collapses the\n"
                  "         Stage 2 generation budget to its 128-token floor, which is\n"
                  "         what produced GLM's empty answers and made two runs of\n"
                  "         identical code at temperature 0 disagree. Check the\n"
                  "         'granted context' line in healthcheck_v30.py output at the\n"
                  "         worker count you intend to use before trusting the results.")
            if input(f"  Type 'yes' to run with {workers} workers anyway: ")\
                    .strip().lower() != 'yes':
                workers = 1
                print("  Using workers=1.")

    # SELF-CONSISTENCY (V27 accuracy enhancement; pipeline unchanged)
    print("""
============================================================
SELF-CONSISTENCY (accuracy enhancement)
============================================================
  For AMBIGUOUS/BORDERLINE sessions only, Stage 2 is asked k times and the
  majority verdict is taken. Easy sessions are decided once, exactly as V23.
  k=1 reproduces V23 behaviour. k=3 is the recommended accuracy setting.
""")
    try:
        sc_k = int(input("Self-consistency k for ambiguous zones (1=off, 3=recommended) [3]: ").strip() or "3")
    except ValueError:
        sc_k = 3
    sc_k = max(1, min(5, sc_k))

    # RUN MODE
    print("""
============================================================
RUN MODE
============================================================
  1) standard    — one full run (detection metrics + LLM evaluation metrics)
  2) temperature — sweep decoding temperature across all LLMs and report
                   "Effect of temperature on accuracy across LLM models"
  3) both        — standard run, then the temperature sweep
""")
    run_mode = input("  Mode (1/2/3) [1]: ").strip() or "1"
    temps = [0.0, 0.2, 0.5, 0.7, 1.0]
    if run_mode in ("2", "3"):
        raw = input(f"  Temperatures (comma separated) {temps}: ").strip()
        if raw:
            try:
                temps = [float(x) for x in raw.split(',') if x.strip()]
            except ValueError:
                pass
        print(f"  Sweep temperatures: {temps}")
        # COST GUARD: the V27 run swept 5 temps x 3 models x 50 sessions = 750
        # extra sessions, five times the main run, which is what turned this
        # into a multi-day job. Default the sweep to a subset.
        try:
            sweep_n = int(input(f"  Sessions per sweep cell "
                                f"(<= main run n) [20]: ").strip() or "20")
        except ValueError:
            sweep_n = 20
        sweep_n = max(10, sweep_n)
        print(f"  Sweep cost: {len(temps)} temps x 3 models x {sweep_n} sessions "
              f"= {len(temps)*3*sweep_n} sessions")

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
        # Cheapest check first: a name that does not exist cannot be measured.
        if verify_models_installed(model_ids, base_url):
            print(f"\n  Aborted before any GPU time was spent.")
            return
        floor = min_viable_num_ctx()
        print(f"\n  Context window: {NUM_CTX} "
              f"(minimum this pipeline can work in: {floor})")
        if NUM_CTX < floor:
            print(f"  [ABORT] EV_IDS_NUM_CTX={NUM_CTX} is below the floor. Stage 2 "
                  f"replays the\n          system prompt, the Stage 1 prompt, the "
                  f"Stage 1 response and the\n          question; below {floor} "
                  f"there is no room left to write a verdict and\n          the "
                  f"model returns reasoning with no answer.")
            return
        degraded = preflight_gpu_residency(model_ids, base_url)
        if degraded:
            print("\n  A partly-resident model does not produce slow results, it")
            print("  produces unusable ones: calls time out, timeouts are recorded")
            print("  as LLM errors, and the run cannot be scored. Continuing is")
            print("  only sensible if you are deliberately measuring that.")
            if input("\n  Run anyway? (y/N): ").strip().lower() != 'y':
                print("  Aborted. Free VRAM and re-run.")
                return
        print("\n  Before a long run, measure the cost first:")
        print("      python healthcheck_v30.py --n <sample size>")
        print("  It exercises the full two-stage pipeline on six unambiguous")
        print("  sessions per model and projects the total hours.")

    models_dir = os.path.join(WORKSPACE_DIR, "models")
    try:
        train_indices, test_indices = load_or_train_models(DATA_PATH, models_dir)
    except Exception as e:
        print(f"\nModel error: {e}")
        return

    selected, sample_seed = select_samples(df, test_indices, cmap)
    scenario_tag = f"{scenario_tag}_seed{sample_seed}"
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
    bootstrap = EVIDSAgentV6TripleLLMV30(config,
                    llm_client=type('Null', (), {'model_name': 'none'})(),
                    use_knowledge=False, use_memory=False,
                    scenario_tag='bootstrap', xai_store=None, verbose=False)
    shared_store = bootstrap.xai_store

    def make_agent(model_name):
        client = make_llm_client(backend, model_name, base_url)  # default temp
        return EVIDSAgentV6TripleLLMV30(
            config, client,
            use_knowledge=use_knowledge, use_memory=use_memory,
            scenario_tag=f"{scenario_tag}_{model_name.split(':')[0]}",
            xai_store=shared_store,
            verbose=True, print_lock=PRINT_LOCK,
            self_consistency_k=sc_k, use_xai=use_xai)

    # RUN LLAMA
    print(f"\nMANUAL: make sure {model_ids['llama']} is available on the backend")
    agent_llama = make_agent(model_ids["llama"])
    if use_memory and agent_llama.ltm:
        agent_llama.ltm.clear()
        agent_llama.seed_ltm_from_training(train_indices, n_seed=50)
    results_llama = run_classification(agent_llama, "[LLAMA]", "LLAMA3",
                                        selected, df, workers=workers)
    shared_store.save()

    # RUN GLM — free the previous model's VRAM first so GLM gets the whole GPU
    if backend == 'ollama':
        unload_ollama_model(model_ids["llama"], base_url)
    print(f"\nMANUAL: switch to {model_ids['glm']}")
    agent_glm = make_agent(model_ids["glm"])
    if use_memory and agent_glm.ltm:
        agent_glm.ltm.clear()
        agent_glm.seed_ltm_from_training(train_indices, n_seed=50)
    results_glm = run_classification(agent_glm, "[GLM]",
                                      model_ids['glm'].split(':')[0].upper(),
                                      selected, df, workers=workers)
    shared_store.save()

    # RUN QWEN — free the previous model's VRAM first
    if backend == 'ollama':
        unload_ollama_model(model_ids["glm"], base_url)
    print(f"\nMANUAL: switch to {model_ids['qwen']}")
    agent_qwen = make_agent(model_ids["qwen"])
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

    if run_mode in ("2", "3"):
        run_temperature_sweep(config, backend, base_url, scenario_tag,
                              use_knowledge, use_memory, shared_store,
                              selected[:sweep_n], df, temps, model_ids,
                              results_dir, sc_k, use_xai=use_xai)

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"triple_llm_v6_v30_{scenario_tag}_{ts}.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(make_json_serializable({
                'version':       '6_triple_llm_v30',
                'scenario':      scenario_tag,
                'backend':       backend,
                'workers':       workers,
                'use_knowledge': use_knowledge,
                'use_memory':    use_memory,
                'use_xai':       use_xai,
                'sample_seed':   sample_seed,
                'shap_used':     SHAP_AVAILABLE,
                'lime_used':     LIME_AVAILABLE,
                'timestamp':     ts,
                'description':   f'Scenario {choice}: {sc["desc"]}',
                'llms':          {k: v for k, v in model_ids.items()},
                'results_llama': results_llama,
                'results_qwen':  results_qwen,
                'results_glm':   results_glm
            }), f, indent=2)
        print(f"\nSaved: {results_file}")
    except Exception as e:
        print(f"\nSave error: {e}")
    print(f"\nTRIPLE LLM V30 SCENARIO {choice} COMPLETE!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n\nFatal: {e}")
        import traceback
        traceback.print_exc()
