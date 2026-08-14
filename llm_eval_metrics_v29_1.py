# llm_eval_metrics_v29_1.py
"""
LLM evaluation metrics for the EV-IDS-Agent study.

Standard measures expected in LLM evaluation sections of research papers,
kept separate from the detection pipeline so the pipeline itself is unchanged.

Provided:
  - wilson_ci                : binomial confidence interval for accuracy
  - mcnemar_test             : paired significance test between two LLMs
  - cohens_kappa             : inter-model agreement
  - expected_calibration_err : ECE + reliability bins (confidence vs accuracy)
  - format_compliance        : how often the LLM emitted a parseable verdict
  - xai_faithfulness         : does the LLM's reasoning cite the feature that
                               SHAP actually identified as the top driver?
  - stability_agreement      : repeat-run agreement (self-consistency)
  - temperature_table        : accuracy/F1 vs decoding temperature per model
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional


# ── ACCURACY UNCERTAINTY ─────────────────────────────────────────────────────
def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval — the correct binomial CI for small n and for
    proportions near 0 or 1 (where the normal approximation fails, e.g. 50/50).
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── PAIRED SIGNIFICANCE ──────────────────────────────────────────────────────
def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, Any]:
    """
    McNemar's test on paired predictions — the right test for "is model A
    better than model B on the SAME samples". Uses the exact binomial test,
    which is required when the discordant count is small (it usually is here).

    Returns b (A right / B wrong), c (A wrong / B right), p-value.
    """
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    if n == 0:
        return {'b': 0, 'c': 0, 'p_value': 1.0, 'significant': False}
    # exact two-sided binomial test at p=0.5
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {'b': b, 'c': c, 'p_value': p, 'significant': p < 0.05}


def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Chance-corrected agreement between two models' predictions."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(list(ca) + list(cb)))
    return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe)


# ── CALIBRATION ──────────────────────────────────────────────────────────────
def expected_calibration_error(confidences: List[float], correct: List[bool],
                               n_bins: int = 5) -> Dict[str, Any]:
    """
    ECE: mean |confidence - accuracy| across confidence bins. A well-calibrated
    model that says 0.9 should be right ~90% of the time. Also returns the bins
    so a reliability diagram can be plotted.
    """
    if not confidences:
        return {'ece': 0.0, 'bins': []}
    bins = []
    total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, c in enumerate(confidences)
               if (c > lo or (i == 0 and c >= lo)) and c <= hi]
        if not idx:
            continue
        acc = sum(1 for j in idx if correct[j]) / len(idx)
        conf = sum(confidences[j] for j in idx) / len(idx)
        ece += (len(idx) / total) * abs(conf - acc)
        bins.append({'range': f"{lo:.1f}-{hi:.1f}", 'n': len(idx),
                     'confidence': round(conf, 3), 'accuracy': round(acc, 3)})
    return {'ece': round(ece, 4), 'bins': bins}


# ── LLM-SPECIFIC BEHAVIOURAL METRICS ─────────────────────────────────────────
def format_compliance(results: List[Dict]) -> Dict[str, Any]:
    """
    Fraction of sessions where the LLM produced a verdict in the requested
    format (no repair, no fallback). A standard robustness measure for
    instruction-following in LLM evaluation.
    """
    n = len(results)
    if n == 0:
        return {'rate': 0.0, 'parsed': 0, 'fallback': 0, 'n': 0}
    parsed   = sum(1 for r in results if r['result'].get('llm_parsed', True))
    fallback = sum(1 for r in results if r['result'].get('used_fallback'))
    return {'rate': round(parsed / n, 4), 'parsed': parsed,
            'fallback': fallback, 'n': n}


def xai_faithfulness(results: List[Dict]) -> Dict[str, Any]:
    """
    Explanation faithfulness: did the LLM's own reasoning cite the feature that
    SHAP identified as the dominant driver for this session?

    This is the metric that substantiates the XAI contribution — it measures
    whether SHAP/LIME actually guided the decision rather than being ignored.
    Reported overall and split by correct/incorrect decisions, since a drop in
    faithfulness on errors is evidence the LLM erred by ignoring the evidence.
    """
    cited = cited_correct = cited_wrong = 0
    n_correct = n_wrong = n = 0
    for r in results:
        res  = r['result']
        top  = (res.get('xai_top_feature') or '').lower()
        text = " ".join([
            res.get('llm_xai_assessment', '') or '',
            res.get('llm_reasoning', '') or '',
            res.get('llm_analysis', '') or '',
        ]).lower()
        if not top:
            continue
        n += 1
        # match on the distinctive word of the feature label
        key = 'delivered' if 'delivered' in top else \
              'requested' if 'requested' in top else \
              'connection' if 'connection' in top else \
              'disconnect' if 'disconnect' in top else top
        hit = key in text
        cited += hit
        if r['correct']:
            n_correct += 1
            cited_correct += hit
        else:
            n_wrong += 1
            cited_wrong += hit
    return {
        'rate':          round(cited / n, 4) if n else 0.0,
        'rate_correct':  round(cited_correct / n_correct, 4) if n_correct else 0.0,
        'rate_wrong':    round(cited_wrong / n_wrong, 4) if n_wrong else 0.0,
        'n': n, 'n_correct': n_correct, 'n_wrong': n_wrong,
    }


def stability_agreement(repeat_predictions: Dict[int, List[str]]) -> Dict[str, Any]:
    """
    Self-consistency / determinism: for samples classified more than once,
    how often did the model give the same answer?

    repeat_predictions: {sample_index: [pred_run1, pred_run2, ...]}
    """
    total = unanimous = 0
    for _, preds in repeat_predictions.items():
        if len(preds) < 2:
            continue
        total += 1
        unanimous += (len(set(preds)) == 1)
    return {'rate': round(unanimous / total, 4) if total else 1.0,
            'unanimous': unanimous, 'n_repeated': total}


# ── TEMPERATURE SENSITIVITY ──────────────────────────────────────────────────
def temperature_table(sweep: Dict[str, Dict[float, List[Dict]]]) -> str:
    """
    Effect of decoding temperature on accuracy across LLMs — a standard figure
    in LLM evaluation papers.

    sweep: {llm_name: {temperature: [result rows]}}
    Returns a printable table; per-cell values are accuracy (F1 underneath).
    """
    if not sweep:
        return "  (no temperature sweep data)"
    temps = sorted({t for per_t in sweep.values() for t in per_t})
    lines = []
    lines.append(f"  {'LLM':<10} | " + " | ".join(f"T={t:<5.1f}" for t in temps))
    lines.append(f"  {'-' * (12 + 10 * len(temps))}")
    for llm, per_t in sweep.items():
        cells = []
        for t in temps:
            rows = per_t.get(t) or []
            if not rows:
                cells.append(f"{'—':>7}")
                continue
            acc = sum(1 for r in rows if r['correct']) / len(rows)
            cells.append(f"{acc:>7.4f}")
        lines.append(f"  {llm:<10} | " + " | ".join(cells))
    # stability row: spread across temperatures
    lines.append(f"  {'-' * (12 + 10 * len(temps))}")
    for llm, per_t in sweep.items():
        accs = []
        for t in temps:
            rows = per_t.get(t) or []
            if rows:
                accs.append(sum(1 for r in rows if r['correct']) / len(rows))
        if len(accs) > 1:
            spread = max(accs) - min(accs)
            mean   = sum(accs) / len(accs)
            lines.append(f"  {llm:<10} | mean={mean:.4f}  spread={spread:.4f} "
                         f"(temperature sensitivity)")
    return "\n".join(lines)


# ── REFERENCE BASELINE ───────────────────────────────────────────────────────
def ratio_oracle_predictions(results: List[Dict], threshold: float = 1.3) -> List[str]:
    """
    Trivial single-rule baseline: delivery_ratio > threshold => Attack.
    Reviewers will compute this themselves; reporting it up front shows how
    much of the task is solvable without any model at all.
    """
    return ['Malicious' if r['result'].get('delivery_ratio', 0) > threshold
            else 'Normal' for r in results]


# ── MULTI-BATCH AGGREGATION ──────────────────────────────────────────────────
def aggregate_runs(result_files: List[str]) -> str:
    """
    Combine several result JSONs (different sampling seeds) into a mean +/- std
    table.

    A single 50-session batch is a weak basis for a headline number: a perfect
    score on one draw invites the reviewer question "would it hold on another?".
    Reporting the mean and standard deviation across independent batches, with a
    pooled confidence interval, answers that directly and is what reviewers
    expect. It also removes the awkwardness of publishing an unqualified 1.00.

    Usage:
        python llm_eval_metrics_v29_1.py results/run_seed42.json results/run_seed7.json ...
    """
    import json, statistics
    per_llm = {'llama': [], 'qwen': [], 'glm': []}
    pooled  = {'llama': [0, 0], 'qwen': [0, 0], 'glm': [0, 0]}   # [correct, n]
    seeds   = []
    for path in result_files:
        with open(path) as f:
            blob = json.load(f)
        seeds.append(blob.get('sample_seed', '?'))
        for k in per_llm:
            rows = blob.get(f'results_{k}') or []
            if not rows:
                continue
            corr = sum(1 for r in rows if r.get('correct'))
            per_llm[k].append(corr / len(rows))
            pooled[k][0] += corr
            pooled[k][1] += len(rows)

    out = []
    out.append(f"  Batches: {len(result_files)}   seeds: {seeds}")
    out.append(f"  {'LLM':<8} | {'mean':>7} | {'std':>7} | {'min':>7} | "
               f"{'max':>7} | {'pooled':>7} | {'pooled 95% CI':>18} | {'N':>5}")
    out.append(f"  {'-'*84}")
    for k, accs in per_llm.items():
        if not accs:
            continue
        c, n = pooled[k]
        lo, hi = wilson_ci(c, n)
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        out.append(f"  {k:<8} | {statistics.mean(accs):>7.4f} | {sd:>7.4f} | "
                   f"{min(accs):>7.4f} | {max(accs):>7.4f} | {c/n:>7.4f} | "
                   f"[{lo:>6.4f}, {hi:>6.4f}] | {n:>5}")
    out.append("\n  Report the pooled accuracy with its interval, and the")
    out.append("  across-batch std as the stability measure.")
    return "\n".join(out)


def _expand(args: List[str]) -> List[str]:
    """
    Expand wildcards internally.

    Windows cmd.exe does not expand globs before launching a program, so
    "...\\*.json" arrives here as a literal string and open() fails. Expanding
    here makes the same command work on Windows, PowerShell and POSIX shells.
    """
    import glob as _glob
    out: List[str] = []
    for a in args:
        if any(ch in a for ch in "*?["):
            hits = sorted(_glob.glob(a))
            if not hits:
                print(f"  [WARN] no files matched: {a}")
            out.extend(hits)
        else:
            out.append(a)
    return out


if __name__ == "__main__":
    import sys
    args = _expand(sys.argv[1:])
    if not args:
        print(__doc__)
        print("Aggregate several runs (wildcards are expanded internally):")
        print("  python llm_eval_metrics_v29_1.py results/triple_llm_v6_v29_1_*.json")
        print("  python llm_eval_metrics_v29_1.py runA.json runB.json runC.json")
    else:
        print(f"  Aggregating {len(args)} file(s):")
        for a in args:
            print(f"    {a}")
        print()
        print(aggregate_runs(args))
