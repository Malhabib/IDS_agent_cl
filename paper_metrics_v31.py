# paper_metrics_v31.py
"""
Paper-ready metrics, tables and figures for EV-IDS-Agent V31.

Reads the results JSON files the runner writes and produces every table and
figure as CSV + LaTeX and PNG + PDF. It never calls a model and never touches
the detection pipeline: everything here is computed after the fact from what a
run saved, so it costs no GPU time and works on runs already finished.

    python paper_metrics_v31.py results/triple_llm_v6_v31_*.json
    python paper_metrics_v31.py run_a.json run_b.json --out paper_out

Mapping to the IDS-Agent paper (Li et al.) -- see RELEASE_NOTES_V31.md:

  Tables 6/7  per-class F-score of each classifier, the majority vote and the
              agent. Classes here are Normal and Attack (binary task; attack
              sub-types are deliberately not broken out).
  Table 8     detection sensitivity: Aggressive / Balanced / Conservative.
              (a) post-hoc operating points from one run's confidence, or
              (b) prompt-instructed runs exactly as in the paper's A.4.
  Table 9     the long-term-memory retrieval weights lambda1 (recency) and
              lambda2 (similarity) of the paper's Eq. 1.
  Figure 8    confidence distributions per classifier. The paper's split is
              in-distribution vs unseen attack types. With one dataset and no
              held-out attack family that split is not available, so this
              figure splits by difficulty zone and by correct vs incorrect and
              SAYS SO in its title. It is not an out-of-distribution result.

Everything else is the robustness evidence a reviewer will ask for: MCC,
balanced accuracy, specificity, FAR/FNR, macro- and weighted-F1, ROC-AUC and
PR-AUC with curves, bootstrap confidence intervals, McNemar against the
majority vote / best single classifier / delivery-ratio rule, Fleiss' kappa
across the three LLMs, calibration (ECE, Brier, reliability diagram),
risk-coverage, per-zone results, efficiency, decision provenance, memory and
RAG usage, explanation agreement, error analysis, and cross-run seed /
ablation / temperature tables.
"""

import csv, glob, json, math, os, sys
from collections import Counter, OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from llm_eval_metrics_v31 import (mcnemar_test, cohens_kappa,
                                  expected_calibration_error, xai_faithfulness,
                                  format_compliance, llm_error_rate)

ATTACK = 'Malicious'
NORMAL = 'Normal'

# Paper column order (RF, LR, KNN, MLP, DT, SVC) plus our seventh classifier.
ML_ORDER = ['Random Forest', 'Logistic Regression', 'K-Nearest Neighbors',
            'MLP', 'Decision Tree', 'Support Vector Classifier',
            'Gradient Boosting']
ML_SHORT = {'Random Forest': 'RF', 'Logistic Regression': 'LR',
            'K-Nearest Neighbors': 'KNN', 'MLP': 'MLP', 'Decision Tree': 'DT',
            'Support Vector Classifier': 'SVC', 'Gradient Boosting': 'GB'}
AGENT_KEYS = ('llama', 'qwen', 'glm')

EASY_ZONES = {'EASY_NORMAL', 'EASY_ATTACK'}

# Default LTM weights. The agent hard-codes 0.3*recency + 0.7*similarity in
# LongTermMemory.retrieve_similar; runs that do not record a lambda used these.
DEFAULT_LTM_LAMBDA = (0.3, 0.7)

# Sensitivity, option (a). The paper's own low-confidence threshold (A.5:
# "low confidence (e.g., below a threshold of 0.7)") is used for the composite
# score so that no threshold is tuned on the test labels.
SENSITIVITY_LEVELS = ('aggressive', 'balanced', 'conservative')
SENS_ABBREV = {'aggressive': 'Aggr.', 'balanced': 'Bal.', 'conservative': 'Cons.'}
DEFAULT_TAU = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════
def pretty_model(model_id: str, key: str) -> str:
    """'llama3:latest' -> 'Llama3', 'glm-4.7-flash:latest' -> 'GLM-4.7-flash'."""
    if not model_id:
        return key.capitalize()
    base = model_id.split(':')[0] if model_id.endswith(':latest') else model_id
    if base.lower().startswith('glm'):
        return 'GLM' + base[3:]
    return base[:1].upper() + base[1:]


def scenario_letter(meta: Dict) -> str:
    k, m = bool(meta.get('use_knowledge')), bool(meta.get('use_memory'))
    return 'A' if (k and m) else 'B' if k else 'C' if m else 'D'


class Run:
    """One results JSON: metadata plus each agent's per-session rows."""

    def __init__(self, path: str, data: Dict):
        self.path = path
        self.meta = {k: v for k, v in data.items() if not k.startswith('results_')}
        llms = data.get('llms') or {}
        self.agents: "OrderedDict[str, List[Dict]]" = OrderedDict()
        for key in AGENT_KEYS:
            rows = data.get(f'results_{key}') or []
            if rows:
                self.agents[pretty_model(llms.get(key, ''), key)] = rows

    # Grouping keys -----------------------------------------------------------
    @property
    def letter(self) -> str:
        return scenario_letter(self.meta)

    @property
    def xai(self) -> bool:
        return bool(self.meta.get('use_xai', True))

    @property
    def seed(self):
        return self.meta.get('sample_seed')

    @property
    def sensitivity(self) -> str:
        return self.meta.get('sensitivity') or 'none'

    @property
    def ltm_lambda(self) -> Tuple[float, float]:
        lam = self.meta.get('ltm_lambda')
        return tuple(lam) if lam else DEFAULT_LTM_LAMBDA

    @property
    def tag(self) -> str:
        return (f"{self.letter}{'' if self.xai else '-noXAI'}"
                f"_seed{self.seed}"
                + ('' if self.sensitivity == 'none' else f"_{self.sensitivity}")
                + ('' if self.ltm_lambda == DEFAULT_LTM_LAMBDA
                   else f"_lam{self.ltm_lambda[0]:g}-{self.ltm_lambda[1]:g}"))

    @property
    def is_standard(self) -> bool:
        """A normal run: not one cell of a sensitivity or lambda sweep."""
        return self.sensitivity == 'none' and not self.meta.get('ltm_lambda_sweep')

    def common_indices(self) -> List[int]:
        """Sessions every agent in this run completed, in first-agent order."""
        sets = [{r['index'] for r in rows} for rows in self.agents.values()]
        if not sets:
            return []
        shared = set.intersection(*sets)
        first = next(iter(self.agents.values()))
        return [r['index'] for r in first if r['index'] in shared]


def load_runs(paths: Sequence[str]) -> Tuple[List[Run], List[Tuple[str, Dict]]]:
    """Returns (main runs, temperature sweeps)."""
    runs, sweeps = [], []
    for p in paths:
        try:
            with open(p, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [skip] {p}: {e}")
            continue
        if 'sweep' in data and 'temperatures' in data:
            sweeps.append((p, data))
        elif any(k.startswith('results_') for k in data):
            run = Run(p, data)
            if run.agents:
                runs.append(run)
            else:
                print(f"  [skip] {p}: no agent results")
    return runs, sweeps


def expand_paths(args: Sequence[str]) -> List[str]:
    """cmd.exe does not expand '*', so expand globs here as well."""
    out = []
    for a in args:
        hits = sorted(glob.glob(a))
        out.extend(hits if hits else [a])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PER-SESSION VIEWS
# ══════════════════════════════════════════════════════════════════════════════
def y_of(label: str) -> int:
    return 1 if label == ATTACK else 0


def agent_view(rows: List[Dict]) -> Dict[str, list]:
    """
    Truth, verdict, confidence and an attack score for an LLM agent.

    `confidence` is the agent's composite confidence IN ITS OWN VERDICT. The
    attack score used for ROC/PR maps it to 0.5 +/- 0.5*confidence, so every
    Attack verdict ranks above every Normal verdict and confidence orders the
    sessions within each side. The verdict is therefore never contradicted by
    the score, and AUC measures how well confidence ranks within verdicts.
    """
    y, yhat, conf, score, ok = [], [], [], [], []
    for r in rows:
        c = float(r['result'].get('confidence', 0.5) or 0.5)
        pred = r['predicted']
        y.append(y_of(r['ground_truth']))
        yhat.append(y_of(pred))
        conf.append(c)
        score.append(0.5 + 0.5 * c if pred == ATTACK else 0.5 - 0.5 * c)
        ok.append(pred == r['ground_truth'])
    return {'y': y, 'yhat': yhat, 'conf': conf, 'score': score, 'correct': ok}


def ml_view(rows: List[Dict], model: str) -> Dict[str, list]:
    """
    The same view for one ML classifier. Its stored confidence is the
    probability of its predicted class, so P(attack) is recovered exactly.
    """
    y, yhat, conf, score, ok = [], [], [], [], []
    for r in rows:
        p = (r.get('model_predictions') or {}).get(model) or {}
        pred = p.get('prediction', NORMAL)
        c = float(p.get('confidence', 0.5))
        y.append(y_of(r['ground_truth']))
        yhat.append(y_of(pred))
        conf.append(c)
        score.append(c if pred == ATTACK else 1.0 - c)
        ok.append(pred == r['ground_truth'])
    return {'y': y, 'yhat': yhat, 'conf': conf, 'score': score, 'correct': ok}


def majority_view(rows: List[Dict]) -> Dict[str, list]:
    """Majority vote of the seven classifiers; score = fraction voting Attack."""
    y, yhat, score, ok = [], [], [], []
    for r in rows:
        preds = [((r.get('model_predictions') or {}).get(m) or {}).get('prediction')
                 for m in ML_ORDER]
        preds = [p for p in preds if p]
        frac = sum(p == ATTACK for p in preds) / len(preds) if preds else 0.0
        mv = r.get('majority_vote') or (ATTACK if frac > 0.5 else NORMAL)
        y.append(y_of(r['ground_truth']))
        yhat.append(y_of(mv))
        score.append(frac)
        ok.append(mv == r['ground_truth'])
    return {'y': y, 'yhat': yhat, 'conf': [max(s, 1 - s) for s in score],
            'score': score, 'correct': ok}


def oracle_view(rows: List[Dict], threshold: float = 1.3) -> Dict[str, list]:
    """The single rule 'delivered/requested > 1.3 => Attack'."""
    y, yhat, score, ok = [], [], [], []
    for r in rows:
        ratio = float(r['result'].get('delivery_ratio', 0) or 0)
        pred = ATTACK if ratio > threshold else NORMAL
        y.append(y_of(r['ground_truth']))
        yhat.append(y_of(pred))
        score.append(ratio)
        ok.append(pred == r['ground_truth'])
    return {'y': y, 'yhat': yhat, 'conf': [1.0] * len(y), 'score': score,
            'correct': ok}


def ensemble_view(agent_rows: Dict[str, List[Dict]], indices: List[int]):
    """Majority of the LLM agents' verdicts on the shared sessions."""
    by = {name: {r['index']: r for r in rows} for name, rows in agent_rows.items()}
    y, yhat, score, ok = [], [], [], []
    for i in indices:
        votes = [by[n][i]['predicted'] for n in by]
        frac = sum(v == ATTACK for v in votes) / len(votes)
        pred = ATTACK if frac > 0.5 else NORMAL
        truth = next(iter(by.values()))[i]['ground_truth']
        y.append(y_of(truth))
        yhat.append(y_of(pred))
        score.append(frac)
        ok.append(pred == truth)
    return {'y': y, 'yhat': yhat, 'conf': [max(s, 1 - s) for s in score],
            'score': score, 'correct': ok}


def restrict(rows: List[Dict], indices: List[int]) -> List[Dict]:
    by = {r['index']: r for r in rows}
    return [by[i] for i in indices if i in by]


# ══════════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════
def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y: Sequence[int], yhat: Sequence[int]) -> Dict[str, float]:
    """
    Every threshold metric for a binary IDS, Attack = positive class.

    Undefined ratios (e.g. precision with no positive predictions) are reported
    as 0.0, the same convention as scikit-learn's zero_division=0.
    """
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    n = tp + tn + fp + fn

    p_att, r_att = _div(tp, tp + fp), _div(tp, tp + fn)
    p_nor, r_nor = _div(tn, tn + fn), _div(tn, tn + fp)
    f_att = _div(2 * p_att * r_att, p_att + r_att)
    f_nor = _div(2 * p_nor * r_nor, p_nor + r_nor)
    n_att, n_nor = tp + fn, tn + fp

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0

    return {
        'n': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy':          _div(tp + tn, n),
        'balanced_accuracy': (r_att + r_nor) / 2,
        'precision':         p_att,              # attack class
        'recall':            r_att,              # = detection rate / TPR
        'specificity':       r_nor,              # = TNR
        'far':               _div(fp, fp + tn),  # false alarm rate / FPR
        'fnr':               _div(fn, fn + tp),  # miss rate
        'f1':                f_att,
        'precision_normal':  p_nor,
        'recall_normal':     r_nor,
        'f1_normal':         f_nor,
        'macro_precision':   (p_att + p_nor) / 2,
        'macro_recall':      (r_att + r_nor) / 2,
        'macro_f1':          (f_att + f_nor) / 2,
        'weighted_f1':       _div(f_att * n_att + f_nor * n_nor, n),
        'mcc':               mcc,
    }


def roc_auc(y: Sequence[int], score: Sequence[float]) -> float:
    """Mann-Whitney AUC with ties counted as one half. NaN if one class only."""
    pos = [s for a, s in zip(y, score) if a == 1]
    neg = [s for a, s in zip(y, score) if a == 0]
    if not pos or not neg:
        return float('nan')
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def average_precision(y: Sequence[int], score: Sequence[float]) -> float:
    """PR-AUC as average precision (the step-wise definition sklearn uses)."""
    n_pos = sum(y)
    if n_pos == 0 or n_pos == len(y):
        return float('nan')
    order = sorted(range(len(y)), key=lambda i: -score[i])
    ap, tp, fp, i = 0.0, 0, 0, 0
    prev_recall = 0.0
    while i < len(order):
        # Treat tied scores as one threshold, as sklearn does.
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if y[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return ap


def roc_points(y, score):
    """(fpr, tpr) curve points over every distinct threshold."""
    n_pos, n_neg = sum(y), len(y) - sum(y)
    if not n_pos or not n_neg:
        return [0.0, 1.0], [0.0, 1.0]
    thresholds = sorted(set(score), reverse=True)
    fpr, tpr = [0.0], [0.0]
    for t in thresholds:
        tp = sum(1 for a, s in zip(y, score) if s >= t and a == 1)
        fp = sum(1 for a, s in zip(y, score) if s >= t and a == 0)
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
    return fpr, tpr


def pr_points(y, score):
    """(recall, precision) curve points over every distinct threshold."""
    n_pos = sum(y)
    if not n_pos:
        return [0.0, 1.0], [1.0, 0.0]
    rec, prec = [0.0], [1.0]
    for t in sorted(set(score), reverse=True):
        tp = sum(1 for a, s in zip(y, score) if s >= t and a == 1)
        fp = sum(1 for a, s in zip(y, score) if s >= t and a == 0)
        rec.append(tp / n_pos)
        prec.append(tp / (tp + fp) if tp + fp else 1.0)
    return rec, prec


def brier_confidence(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """
    Brier score of the confidence a model states in its own verdict:
    mean (confidence - 1[correct])^2. It is the squared-error companion of ECE
    and is defined identically for the classifiers and the LLM agents.
    """
    if not conf:
        return float('nan')
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(conf, correct)) / len(conf)


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]):
    """
    Selective prediction: keep only the most confident k sessions. Returns the
    curve (coverage, risk), the area under it (AURC, lower is better) and the
    accuracy retained at 50% and 80% coverage.
    """
    n = len(conf)
    if not n:
        return {'coverage': [], 'risk': [], 'aurc': float('nan'),
                'acc_at_50': float('nan'), 'acc_at_80': float('nan')}
    order = sorted(range(n), key=lambda i: -conf[i])
    cov, risk, errors = [], [], 0
    for k, i in enumerate(order, 1):
        errors += 0 if correct[i] else 1
        cov.append(k / n)
        risk.append(errors / k)

    def acc_at(c):
        k = max(1, int(math.ceil(c * n)))
        return 1.0 - risk[k - 1]

    return {'coverage': cov, 'risk': risk, 'aurc': sum(risk) / n,
            'acc_at_50': acc_at(0.5), 'acc_at_80': acc_at(0.8)}


def bootstrap_ci(y, yhat, metric: str, n_boot: int = 2000, seed: int = 12345,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap over sessions (paired: truth and verdict resampled together)."""
    n = len(y)
    if n < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    y, yhat = np.asarray(y), np.asarray(yhat)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(binary_metrics(y[idx].tolist(), yhat[idx].tolist())[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def fleiss_kappa(label_lists: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa for m raters over the same N items, binary labels."""
    m = len(label_lists)
    if m < 2 or not label_lists[0]:
        return float('nan')
    n_items = len(label_lists[0])
    p1 = 0.0
    p_bar = 0.0
    for i in range(n_items):
        ones = sum(lst[i] for lst in label_lists)
        zeros = m - ones
        p1 += ones
        p_bar += (ones * ones + zeros * zeros - m) / (m * (m - 1))
    p_bar /= n_items
    p_one = p1 / (n_items * m)
    pe = p_one ** 2 + (1 - p_one) ** 2
    if pe >= 1.0:
        return float('nan')
    return (p_bar - pe) / (1 - pe)


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY — OPTION (a), POST-HOC
# ══════════════════════════════════════════════════════════════════════════════
def apply_sensitivity(rows: List[Dict], level: str, source: str = 'stated',
                      tau: float = DEFAULT_TAU) -> List[int]:
    """
    Re-read one run's verdicts at a different operating point, without
    re-running anything. The paper's three instructions are mapped onto the
    agent's own confidence:

      aggressive    "discover the attack as the priority"
                    -> alert on every Attack verdict, AND on a Normal verdict
                       the agent was not sure of
      balanced      the verdict exactly as deployed
      conservative  "do not alert unless you are very sure"
                    -> alert only on an Attack verdict the agent was sure of

    source='stated'     "sure" = the LLM stated CONFIDENCE: high
    source='composite'  "sure" = composite confidence >= tau (default 0.7, the
                        paper's own low-confidence threshold from its A.5)

    Neither source uses the labels, so nothing is tuned on the test set.
    """
    out = []
    for r in rows:
        res = r['result']
        attack = r['predicted'] == ATTACK
        if source == 'composite':
            sure = float(res.get('confidence', 0) or 0) >= tau
        else:
            sure = (res.get('llm_confidence') or '').lower() == 'high'
        if level == 'aggressive':
            out.append(1 if attack or not sure else 0)
        elif level == 'conservative':
            out.append(1 if attack and sure else 0)
        else:
            out.append(1 if attack else 0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TABLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(v, nd=4):
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        if math.isnan(v):
            return 'n/a'
        return f"{v:.{nd}f}"
    return str(v)


def _ci(lo_hi, nd=3):
    lo, hi = lo_hi
    if math.isnan(lo):
        return 'n/a'
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _tex_escape(s: str) -> str:
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                 ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\^{}')):
        s = s.replace(a, b)
    return s


class Table:
    """A table that writes itself as CSV, LaTeX (booktabs) and console text."""

    def __init__(self, name: str, caption: str, headers: List[str],
                 rows: List[List], notes: Optional[List[str]] = None):
        self.name, self.caption = name, caption
        self.headers, self.rows = headers, rows
        self.notes = notes or []

    def text(self) -> str:
        cells = [self.headers] + [[str(c) for c in r] for r in self.rows]
        widths = [max(len(row[i]) for row in cells) for i in range(len(self.headers))]
        line = lambda r: "  " + " | ".join(str(c).rjust(w) if i else str(c).ljust(w)
                                            for i, (c, w) in enumerate(zip(r, widths)))
        out = [f"\n  {self.caption}", line(self.headers),
               "  " + "-+-".join('-' * w for w in widths)]
        out += [line(r) for r in self.rows]
        out += [f"    {n}" for n in self.notes]
        return "\n".join(out)

    def write(self, out_dir: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{self.name}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(self.headers)
            w.writerows(self.rows)
        tex_path = os.path.join(out_dir, f"{self.name}.tex")
        cols = 'l' + 'r' * (len(self.headers) - 1)
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{{_tex_escape(self.caption)}}}",
                 rf"\label{{tab:{self.name}}}",
                 r"\small",
                 rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
                 " & ".join(_tex_escape(str(h)) for h in self.headers) + r" \\",
                 r"\midrule"]
        lines += [" & ".join(_tex_escape(str(c)) for c in r) + r" \\" for r in self.rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
        if self.notes:
            lines.append(r"\par\smallskip\footnotesize "
                         + " ".join(_tex_escape(n) for n in self.notes))
        lines.append(r"\end{table}")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return [csv_path, tex_path]


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def _models_on(run: Run, idx: List[int]):
    """Every system to compare, each as a view over the same sessions."""
    first_rows = restrict(next(iter(run.agents.values())), idx)
    systems = OrderedDict()
    for m in ML_ORDER:
        systems[ML_SHORT[m]] = ('ml', ml_view(first_rows, m))
    systems['Majority Vote'] = ('baseline', majority_view(first_rows))
    systems['Ratio rule (>1.3)'] = ('baseline', oracle_view(first_rows))
    if len(run.agents) >= 2:
        systems['LLM ensemble'] = ('baseline', ensemble_view(run.agents, idx))
    for name, rows in run.agents.items():
        systems[f"{name} Agent"] = ('agent', agent_view(restrict(rows, idx)))
    return systems, first_rows


def best_ml(systems) -> str:
    """
    The strongest single classifier, by Macro-F1. One definition is used by the
    statistics table and the ROC/PR figure alike, so the two never name
    different "best" baselines.
    """
    ml = {k: v for k, (kind, v) in systems.items() if kind == 'ml'}
    return max(ml, key=lambda k: binary_metrics(ml[k]['y'], ml[k]['yhat'])['macro_f1'])


def table_per_class_f1(run: Run, idx, systems) -> Table:
    """Tables 6/7: per-class F-score with a macro average, paper layout."""
    cols = [k for k, (kind, _) in systems.items() if kind in ('ml', 'agent')
            or k == 'Majority Vote']
    rows = []
    for cls, key in ((NORMAL, 'f1_normal'), ('Attack', 'f1'), ('Macro Avg', 'macro_f1')):
        r = [cls]
        for c in cols:
            v = systems[c][1]
            r.append(_fmt(binary_metrics(v['y'], v['yhat'])[key], 2))
        rows.append(r)
    return Table('table6_per_class_fscore',
                 f"F-score of each classifier, the majority vote and the IDS agent "
                 f"(Scenario {run.letter}, n={len(idx)}). Paper Tables 6/7 layout.",
                 ['Class'] + cols, rows,
                 ["Binary task: Normal vs Attack. Macro Avg weights both classes equally."])


def table_detection(run: Run, idx, systems, n_boot) -> Table:
    rows = []
    for name, (kind, v) in systems.items():
        m = binary_metrics(v['y'], v['yhat'])
        rows.append([name, m['n'], m['tp'], m['tn'], m['fp'], m['fn'],
                     _fmt(m['accuracy']), _fmt(m['balanced_accuracy']),
                     _fmt(m['precision']), _fmt(m['recall']), _fmt(m['specificity']),
                     _fmt(m['far']), _fmt(m['fnr']), _fmt(m['f1']),
                     _fmt(m['macro_f1']), _fmt(m['weighted_f1']), _fmt(m['mcc']),
                     _fmt(roc_auc(v['y'], v['score'])),
                     _fmt(average_precision(v['y'], v['score']))])
    notes = ["Recall = detection rate (TPR); FAR = false alarm rate (FPR); "
             "FNR = miss rate.",
             "Agent ROC/PR scores rank sessions by verdict, then by confidence."]
    # Sessions the model never answered are labelled by the ML majority. They
    # are kept so n matches the run, but the numbers then mix model behaviour
    # with transport failure, and must not be quoted without saying so.
    for name, rows_a in run.agents.items():
        err = llm_error_rate(restrict(rows_a, idx))
        if err['errors']:
            notes.append(f"WARNING: {name} has {err['errors']} session(s) with no LLM "
                         f"answer, labelled by the ML majority. Its row is not a pure "
                         f"measurement of the model.")
    return Table('table_detection_metrics',
                 f"Detection performance, Scenario {run.letter} (Attack = positive class)",
                 ['Model', 'n', 'TP', 'TN', 'FP', 'FN', 'Acc', 'BalAcc', 'Prec',
                  'Recall/DR', 'Spec', 'FAR', 'FNR', 'F1', 'Macro-F1', 'Wtd-F1',
                  'MCC', 'ROC-AUC', 'PR-AUC'], rows, notes)


def table_statistics(run: Run, idx, systems, n_boot) -> Table:
    """Bootstrap CIs and paired significance against the obvious baselines."""
    ml_only = {k: v for k, (kind, v) in systems.items() if kind == 'ml'}
    best = best_ml(systems)
    maj = systems['Majority Vote'][1]
    orc = systems['Ratio rule (>1.3)'][1]
    rows = []
    for name, (kind, v) in systems.items():
        if kind != 'agent':
            continue
        m = binary_metrics(v['y'], v['yhat'])
        row = [name, len(v['y']),
               f"{m['accuracy']:.3f} {_ci(bootstrap_ci(v['y'], v['yhat'], 'accuracy', n_boot))}",
               f"{m['macro_f1']:.3f} {_ci(bootstrap_ci(v['y'], v['yhat'], 'macro_f1', n_boot))}",
               f"{m['mcc']:.3f} {_ci(bootstrap_ci(v['y'], v['yhat'], 'mcc', n_boot))}"]
        for base in (maj, ml_only[best], orc):
            t = mcnemar_test(v['correct'], base['correct'])
            row.append(f"{t['b']}/{t['c']} p={t['p_value']:.4f}"
                       + (" *" if t['significant'] else ""))
        rows.append(row)
    return Table('table_statistics',
                 f"Bootstrap 95% CIs and McNemar tests, Scenario {run.letter}",
                 ['Agent', 'n', 'Accuracy [95% CI]', 'Macro-F1 [95% CI]',
                  'MCC [95% CI]', 'vs Majority', f'vs best ML ({best})',
                  'vs Ratio rule'], rows,
                 [f"Bootstrap: {n_boot} paired resamples of sessions, percentile intervals.",
                  "McNemar exact test, cells b/c = agent right & baseline wrong / "
                  "agent wrong & baseline right; * = p < 0.05.",
                  "Best ML = highest Macro-F1 on these same sessions (also used in "
                  "the ROC/PR figure); choosing it on the test data favours the "
                  "baseline, so the comparison is conservative for the agent."])


def table_agreement(run: Run, idx) -> Optional[Table]:
    names = list(run.agents)
    if len(names) < 2:
        return None
    views = {n: agent_view(restrict(run.agents[n], idx)) for n in names}
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = views[names[i]], views[names[j]]
            t = mcnemar_test(a['correct'], b['correct'])
            k = cohens_kappa([str(x) for x in a['yhat']], [str(x) for x in b['yhat']])
            rows.append([f"{names[i]} vs {names[j]}", len(idx), t['b'], t['c'],
                         _fmt(t['p_value']), 'yes' if t['significant'] else 'no',
                         _fmt(k, 3)])
    notes = []
    if len(names) >= 3:
        fk = fleiss_kappa([views[n]['yhat'] for n in names])
        notes.append(f"Fleiss' kappa across all {len(names)} LLM agents: {_fmt(fk, 3)}")
    return Table('table_llm_agreement',
                 f"Agreement between LLM agents, Scenario {run.letter}",
                 ['Pair', 'n', 'b', 'c', 'McNemar p', 'Significant', "Cohen's kappa"],
                 rows, notes)


def table_calibration(run: Run, idx, systems) -> Tuple[Table, Dict]:
    rows, curves = [], {}
    for name, (kind, v) in systems.items():
        if kind not in ('agent', 'ml'):
            continue
        ece = expected_calibration_error(v['conf'], v['correct'], n_bins=10)
        rc = risk_coverage(v['conf'], v['correct'])
        rows.append([name, len(v['y']), _fmt(ece['ece']),
                     _fmt(brier_confidence(v['conf'], v['correct'])),
                     _fmt(rc['aurc']), _fmt(rc['acc_at_50']), _fmt(rc['acc_at_80']),
                     _fmt(sum(v['correct']) / len(v['correct']) if v['correct'] else float('nan'))])
        curves[name] = {'ece': ece, 'rc': rc, 'kind': kind}
    return Table('table_calibration',
                 f"Calibration and selective prediction, Scenario {run.letter}",
                 ['Model', 'n', 'ECE', 'Brier', 'AURC', 'Acc@50% cov',
                  'Acc@80% cov', 'Acc@100%'], rows,
                 ["ECE (10 bins) and Brier compare the confidence a model states "
                  "in its own verdict with whether that verdict was right.",
                  "AURC = area under the risk-coverage curve (lower is better); "
                  "Acc@x% = accuracy on the x% most confident sessions."]), curves


def table_sensitivity_posthoc(run: Run, idx, source, tau) -> List[Table]:
    """Table 8, option (a): one table per agent, per-class rows."""
    tables = []
    for name, rows in run.agents.items():
        rr = restrict(rows, idx)
        y = [y_of(r['ground_truth']) for r in rr]
        body = {cls: [cls] for cls in (NORMAL, 'Attack', 'Macro Avg')}
        for level in SENSITIVITY_LEVELS:
            m = binary_metrics(y, apply_sensitivity(rr, level, source, tau))
            for cls, keys in ((NORMAL, ('precision_normal', 'recall_normal', 'f1_normal')),
                              ('Attack', ('precision', 'recall', 'f1')),
                              ('Macro Avg', ('macro_precision', 'macro_recall', 'macro_f1'))):
                body[cls] += [_fmt(m[k], 2) for k in keys]
        src = ("stated LLM confidence ('high' = sure)" if source == 'stated'
               else f"composite confidence >= {tau:g}")
        safe = name.replace(' ', '_').replace('.', '_')
        tables.append(Table(
            f"table8a_sensitivity_posthoc_{safe}",
            f"Detection sensitivity, post-hoc operating points from one run "
            f"({name} Agent, Scenario {run.letter}, n={len(rr)}). Paper Table 8 layout.",
            ['Class', 'Aggr. P', 'Aggr. R', 'Aggr. F1', 'Bal. P', 'Bal. R',
             'Bal. F1', 'Cons. P', 'Cons. R', 'Cons. F1'],
            list(body.values()),
            [f"Option (a): no re-run. 'Sure' is decided by {src}. Aggressive also "
             f"alerts on unsure Normal verdicts; Conservative alerts only on sure "
             f"Attack verdicts; Balanced is the deployed verdict.",
             "No threshold is fitted to the test labels."]))
    return tables


def table_zones(run: Run, idx) -> Table:
    """
    Results by difficulty zone (delivered/requested ratio band).

    Reported as detection rate on the zone's attacks and false-alarm rate on its
    normals. Most zones contain only one class -- EASY_ATTACK has no normals,
    EASY_NORMAL no attacks -- so a macro-F1 there is capped at 0.5 by the empty
    class even when every verdict is correct, and would read as a failure.
    """
    first = restrict(next(iter(run.agents.values())), idx)
    zones = sorted({r['result'].get('difficulty_zone', 'UNKNOWN') for r in first})
    rows = []

    def dr_far(v):
        m = binary_metrics(v['y'], v['yhat'])
        n_att, n_nor = m['tp'] + m['fn'], m['tn'] + m['fp']
        return (_fmt(m['recall'], 3) if n_att else '—',
                _fmt(m['far'], 3) if n_nor else '—')

    for z in zones:
        zi = [r['index'] for r in first if r['result'].get('difficulty_zone') == z]
        zr = restrict(first, zi)
        n_att = sum(1 for r in zr if r['ground_truth'] == ATTACK)
        row = [z, len(zi), n_att, len(zi) - n_att]
        row += list(dr_far(majority_view(zr)))
        for name, rows_a in run.agents.items():
            row += list(dr_far(agent_view(restrict(rows_a, zi))))
        rows.append(row)
    heads = ['Zone', 'n', 'Attacks', 'Normals', 'Majority DR', 'Majority FAR']
    for name in run.agents:
        heads += [f"{name} DR", f"{name} FAR"]
    return Table('table_difficulty_zones',
                 f"Results by difficulty zone (delivered/requested ratio), Scenario {run.letter}",
                 heads, rows,
                 ["DR = detection rate on the zone's attacks; FAR = false-alarm rate "
                  "on its normals; '—' = the zone has no sessions of that class."])


def table_efficiency(run: Run, idx) -> Table:
    rows = []
    for name, rws in run.agents.items():
        rr = restrict(rws, idx)
        cx = [r['result'].get('complexity', {}) for r in rr]
        get = lambda k: [float(c.get(k, 0) or 0) for c in cx]
        lat = get('llm_latency_sec')
        rows.append([name, len(rr),
                     _fmt(float(np.mean(lat)) if lat else float('nan'), 1),
                     _fmt(float(np.median(lat)) if lat else float('nan'), 1),
                     _fmt(float(np.mean(get('stage1_latency'))) if rr else float('nan'), 1),
                     _fmt(float(np.mean(get('stage2_latency'))) if rr else float('nan'), 1),
                     _fmt(float(np.mean(get('xai_sec'))) if rr else float('nan'), 2),
                     _fmt(float(np.mean(get('session_total_sec'))) if rr else float('nan'), 1),
                     _fmt(float(np.sum(get('session_total_sec'))) / 3600 if rr else float('nan'), 2),
                     int(np.mean(get('prompt_chars'))) if rr else 0,
                     int(np.mean(get('response_words'))) if rr else 0,
                     _fmt(float(np.mean(get('tokens_per_sec'))) if rr else float('nan'), 2)])
    return Table('table_efficiency', f"Execution cost per session, Scenario {run.letter}",
                 ['Agent', 'n', 'LLM s (mean)', 'LLM s (median)', 'Stage1 s',
                  'Stage2 s', 'XAI s', 'Session s', 'Total h', 'Prompt chars',
                  'Response words', 'Words/s'], rows)


def table_integrity(run: Run, idx) -> Table:
    """
    Where each verdict came from, and whether memory and RAG were ever live.

    The LTM and RAG columns matter for interpreting every ablation: if a
    sentence-transformers model is missing, retrieval silently returns nothing
    and a "with memory" run is really a "without memory" run.
    """
    rows = []
    for name, rws in run.agents.items():
        rr = restrict(rws, idx)
        src = Counter(r['result'].get('verdict_source', 'stated') for r in rr)
        fc = format_compliance(rr)
        ltm = [int(r['result'].get('ltm_cases_referenced', 0) or 0) for r in rr]
        rag = [int(r['result'].get('rag_sources_used', 0) or 0) for r in rr]
        rows.append([name, len(rr), src.get('stated', 0), src.get('repaired', 0),
                     src.get('ml_fallback', 0), src.get('llm_error', 0),
                     _fmt(fc['rate'], 3),
                     'on' if run.meta.get('use_memory') else 'off',
                     _fmt(float(np.mean(ltm)) if ltm else 0.0, 2),
                     f"{sum(1 for x in ltm if x > 0)}/{len(ltm)}",
                     'on' if run.meta.get('use_knowledge') else 'off',
                     _fmt(float(np.mean(rag)) if rag else 0.0, 2),
                     f"{sum(1 for x in rag if x > 0)}/{len(rag)}"])
    notes = ["stated/repaired = the model's own verdict; ML fallback = answered "
             "without a verdict; LLM error = never answered."]
    for r in rows:
        if r[7] == 'on' and r[9].startswith('0/'):
            notes.append(f"WARNING: memory is ON for {r[0]} but no session retrieved "
                         f"a case. Memory was not live; treat this run as memory-off.")
        if r[10] == 'on' and r[12].startswith('0/'):
            notes.append(f"WARNING: RAG is ON for {r[0]} but no session retrieved a "
                         f"passage. RAG was not live.")
    return Table('table_provenance_and_retrieval',
                 f"Decision provenance and retrieval usage, Scenario {run.letter}",
                 ['Agent', 'n', 'Stated', 'Repaired', 'ML fallback', 'LLM error',
                  'Format compliance', 'LTM', 'LTM cases/session',
                  'Sessions with LTM', 'RAG', 'RAG passages/session',
                  'Sessions with RAG'], rows, notes)


def table_xai(run: Run, idx) -> Table:
    rows = []
    first = restrict(next(iter(run.agents.values())), idx)
    have_xai = any(r.get('xai_results') for r in first)
    for m in ML_ORDER:
        if not have_xai:
            rows.append([ML_SHORT[m], 'n/a', 'n/a', 'n/a'])
            continue
        both = agree = shap_ok = 0
        for r in first:
            x = (r.get('xai_results') or {}).get(m) or {}
            shap_ok += 1 if x.get('shap_ok') else 0
            if x.get('shap_top') and x.get('lime_top'):
                both += 1
                agree += 1 if x['shap_top'] == x['lime_top'] else 0
        rows.append([ML_SHORT[m], f"{shap_ok}/{len(first)}", both,
                     _fmt(agree / both if both else float('nan'), 3)])
    notes = []
    if not have_xai:
        notes.append("This run predates per-session XAI recording; re-run with the "
                     "V31 runner to fill this table.")
    for name, rws in run.agents.items():
        xf = xai_faithfulness(restrict(rws, idx))
        notes.append(f"{name} Agent cites the dominant SHAP feature in "
                     f"{_fmt(xf['rate'], 3)} of sessions "
                     f"(correct {_fmt(xf['rate_correct'], 3)}, "
                     f"wrong {_fmt(xf['rate_wrong'], 3)}).")
    return Table('table_xai_agreement',
                 f"Explanation reliability, Scenario {run.letter}",
                 ['Classifier', 'SHAP computed', 'Sessions with SHAP & LIME',
                  'SHAP-LIME top-feature agreement'], rows, notes)


def write_error_analysis(run: Run, idx, out_dir) -> str:
    path = os.path.join(out_dir, 'error_analysis.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['agent', 'session', 'truth', 'predicted', 'delivery_ratio',
                    'zone', 'verdict_source', 'llm_confidence', 'confidence',
                    'ml_majority', 'llm_overrode_ml'])
        for name, rws in run.agents.items():
            for r in restrict(rws, idx):
                if r['correct']:
                    continue
                res = r['result']
                w.writerow([name, r['index'], r['ground_truth'], r['predicted'],
                            res.get('delivery_ratio'), res.get('difficulty_zone'),
                            res.get('verdict_source'), res.get('llm_confidence'),
                            res.get('confidence'), res.get('ml_majority'),
                            res.get('llm_overrode_ml')])
    return path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir, name) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        paths.append(p)
    return paths


def figure_confidence_distributions(run: Run, idx, out_dir, split: str) -> List[str]:
    """
    Figure 8 layout: one row per classifier (and per LLM agent), two columns.

    split='zone'     left = easy zones, right = borderline/ambiguous zones
    split='correct'  left = correctly classified, right = misclassified

    The paper's columns are in-distribution vs unseen attack types. Neither
    split here is that, and the title says so.
    """
    plt = _plt()
    first = restrict(next(iter(run.agents.values())), idx)
    if split == 'zone':
        left_ids = {r['index'] for r in first
                    if r['result'].get('difficulty_zone') in EASY_ZONES}
        lt, rt = 'easy zones', 'borderline / ambiguous zones'
    systems = [(ML_SHORT[m] + ' (' + m + ')', 'ml', m) for m in ML_ORDER]
    systems += [(f"{n} Agent", 'agent', n) for n in run.agents]

    fig, axes = plt.subplots(len(systems), 2, figsize=(10, 2.1 * len(systems)))
    if len(systems) == 1:
        axes = np.array([axes])
    for row, (title, kind, key) in enumerate(systems):
        if kind == 'ml':
            v = ml_view(first, key)
            ids = [r['index'] for r in first]
        else:
            rr = restrict(run.agents[key], idx)
            v = agent_view(rr)
            ids = [r['index'] for r in rr]
        if split == 'zone':
            left = [c for c, i in zip(v['conf'], ids) if i in left_ids]
            right = [c for c, i in zip(v['conf'], ids) if i not in left_ids]
        else:
            left = [c for c, ok in zip(v['conf'], v['correct']) if ok]
            right = [c for c, ok in zip(v['conf'], v['correct']) if not ok]
            lt, rt = 'correctly classified', 'misclassified'
        for col, (data, sub, colour) in enumerate(((left, lt, 'tab:blue'),
                                                   (right, rt, 'tab:green'))):
            ax = axes[row, col]
            if data:
                ax.hist(data, bins=10, range=(0, 1), color=colour, alpha=0.75,
                        edgecolor='black', linewidth=0.4)
            else:
                ax.text(0.5, 0.5, 'no sessions', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='grey')
            ax.set_xlim(0, 1)
            ax.set_title(f"{title} — {sub} (n={len(data)})", fontsize=8)
            ax.set_xlabel('Confidence', fontsize=7)
            ax.set_ylabel('Frequency', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
    what = ('easy vs borderline/ambiguous difficulty zones' if split == 'zone'
            else 'correct vs incorrect verdicts')
    fig.suptitle(f"Confidence distributions: {what} (Scenario {run.letter}, n={len(idx)})\n"
                 f"NOT an out-of-distribution evaluation — no held-out attack family "
                 f"is available in this dataset", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    paths = _save(fig, out_dir, f"figure8_confidence_by_{split}")
    plt.close(fig)
    return paths


def figure_roc_pr(run: Run, systems, out_dir) -> List[str]:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ml = {k: v for k, (kind, v) in systems.items() if kind == 'ml'}
    best = best_ml(systems)
    shown = [(k, v) for k, (kind, v) in systems.items() if kind == 'agent']
    shown += [('Majority Vote', systems['Majority Vote'][1]), (f'{best} (best ML)', ml[best])]
    for name, v in shown:
        auc, ap = roc_auc(v['y'], v['score']), average_precision(v['y'], v['score'])
        fx, fy = roc_points(v['y'], v['score'])
        a1.plot(fx, fy, label=f"{name} (AUC={_fmt(auc, 3)})", linewidth=1.4)
        rx, ry = pr_points(v['y'], v['score'])
        a2.step(rx, ry, where='post', label=f"{name} (AP={_fmt(ap, 3)})", linewidth=1.4)
    a1.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    a1.set_xlabel('False alarm rate (FPR)'); a1.set_ylabel('Detection rate (TPR)')
    a1.set_title('ROC'); a1.legend(fontsize=7); a1.grid(alpha=0.3)
    base = sum(shown[0][1]['y']) / max(1, len(shown[0][1]['y']))
    a2.axhline(base, color='k', linestyle='--', linewidth=0.7)
    a2.set_xlabel('Recall'); a2.set_ylabel('Precision')
    a2.set_title('Precision-Recall'); a2.legend(fontsize=7); a2.grid(alpha=0.3)
    fig.suptitle(f"ROC and PR curves (Scenario {run.letter})", fontsize=10)
    fig.tight_layout()
    paths = _save(fig, out_dir, 'figure_roc_pr')
    plt.close(fig)
    return paths


def figure_calibration(run: Run, curves, out_dir) -> List[str]:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, c in curves.items():
        if c['kind'] != 'agent':
            continue
        bins = c['ece']['bins']
        if bins:
            a1.plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins],
                    'o-', label=f"{name} (ECE={c['ece']['ece']:.3f})")
        rc = c['rc']
        a2.plot(rc['coverage'], rc['risk'], label=f"{name} (AURC={_fmt(rc['aurc'], 3)})")
    a1.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    a1.set_xlabel('Stated confidence'); a1.set_ylabel('Observed accuracy')
    a1.set_title('Reliability diagram'); a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
    a1.legend(fontsize=7); a1.grid(alpha=0.3)
    a2.set_xlabel('Coverage (most confident first)'); a2.set_ylabel('Risk (error rate)')
    a2.set_title('Risk-coverage'); a2.legend(fontsize=7); a2.grid(alpha=0.3)
    fig.suptitle(f"Calibration of the LLM agents (Scenario {run.letter})", fontsize=10)
    fig.tight_layout()
    paths = _save(fig, out_dir, 'figure_calibration')
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def _agent_metrics(run: Run, name: str) -> Dict[str, float]:
    v = agent_view(run.agents[name])
    return binary_metrics(v['y'], v['yhat'])


def table_seed_robustness(runs: List[Run]) -> Optional[Table]:
    # Sweep cells are excluded: a lambda run at the default weights has the
    # same grouping key as a standard run and was being counted as an extra seed.
    groups: Dict[Tuple, List[Run]] = {}
    for r in runs:
        if r.is_standard:
            groups.setdefault((r.letter, r.xai), []).append(r)
    rows = []
    for (letter, xai), grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        seeds = sorted({str(r.seed) for r in grp})
        if len(seeds) < 2:
            continue
        for name in sorted({n for r in grp for n in r.agents}):
            ms = [_agent_metrics(r, name) for r in grp if name in r.agents]
            row = [f"{letter}{'' if xai else '-noXAI'}", name, len(ms), ",".join(seeds)]
            for key in ('accuracy', 'macro_f1', 'mcc', 'far'):
                vals = [m[key] for m in ms]
                row.append(f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}")
            rows.append(row)
    if not rows:
        return None
    return Table('table_seed_robustness', "Robustness across sampling seeds (mean ± std)",
                 ['Config', 'Agent', 'Runs', 'Seeds', 'Accuracy', 'Macro-F1',
                  'MCC', 'FAR'], rows,
                 ["Each seed draws an independent test sample; std is across runs."])


def table_ablation(runs: List[Run]) -> Optional[Table]:
    """Scenario A/B/C/D x XAI on/off, standard runs only, deltas vs D with XAI."""
    std = [r for r in runs if r.is_standard]
    configs = {(r.letter, r.xai) for r in std}
    if len(configs) < 2:
        return None
    by_cfg: Dict[Tuple, List[Run]] = {}
    for r in std:
        by_cfg.setdefault((r.letter, r.xai), []).append(r)
    base_key = ('D', True) if ('D', True) in by_cfg else sorted(by_cfg)[0]
    labels = {'A': 'RAG + LTM + XAI', 'B': 'RAG + XAI', 'C': 'LTM + XAI', 'D': 'XAI only'}
    names = sorted({n for r in std for n in r.agents})
    rows = []
    for key in sorted(by_cfg):
        letter, xai = key
        desc = labels[letter] if xai else labels[letter].replace(' + XAI', '').replace('XAI only', 'none') + ' (XAI off)'
        row = [f"{letter}{'' if xai else '-noXAI'}", desc, len(by_cfg[key])]
        for n in names:
            vals = [_agent_metrics(r, n)['macro_f1'] for r in by_cfg[key] if n in r.agents]
            base = [_agent_metrics(r, n)['macro_f1'] for r in by_cfg[base_key] if n in r.agents]
            if vals:
                d = np.mean(vals) - np.mean(base) if base else float('nan')
                row += [f"{np.mean(vals):.4f}", "—" if key == base_key else f"{d:+.4f}"]
            else:
                row += ['n/a', 'n/a']
        rows.append(row)
    heads = ['Config', 'Components', 'Runs']
    for n in names:
        heads += [f"{n} Macro-F1", f"{n} Δ"]
    return Table('table_ablation',
                 f"Ablation across scenarios (Δ vs {base_key[0]}"
                 f"{'' if base_key[1] else '-noXAI'})", heads, rows,
                 ["Mean over the standard runs (all seeds) in each configuration; "
                  "sweep cells are excluded. Check the retrieval "
                  "table: a configuration whose memory or RAG never retrieved "
                  "anything is not testing that component."])


def table_sensitivity_prompted(runs: List[Run]) -> List[Table]:
    """Table 8, option (b): one run per prompted sensitivity level."""
    prompted = [r for r in runs if r.sensitivity in SENSITIVITY_LEVELS]
    if not prompted:
        return []
    tables = []
    for name in sorted({n for r in prompted for n in r.agents}):
        level_runs = {r.sensitivity: r for r in prompted if name in r.agents}
        if len(level_runs) < 2:
            continue
        body = {cls: [cls] for cls in (NORMAL, 'Attack', 'Macro Avg')}
        heads = ['Class']
        for level in SENSITIVITY_LEVELS:
            if level not in level_runs:
                continue
            m = _agent_metrics(level_runs[level], name)
            ab = SENS_ABBREV[level]
            heads += [f"{ab} P", f"{ab} R", f"{ab} F1"]
            for cls, keys in ((NORMAL, ('precision_normal', 'recall_normal', 'f1_normal')),
                              ('Attack', ('precision', 'recall', 'f1')),
                              ('Macro Avg', ('macro_precision', 'macro_recall', 'macro_f1'))):
                body[cls] += [_fmt(m[k], 2) for k in keys]
        n = len(next(iter(level_runs.values())).agents[name])
        safe = name.replace(' ', '_').replace('.', '_')
        tables.append(Table(
            f"table8b_sensitivity_prompted_{safe}",
            f"Detection sensitivity instructed in the prompt ({name} Agent, n={n}). "
            f"Paper Table 8 layout.", heads, list(body.values()),
            ["Option (b): the paper's A.4 instruction 'Summarize the classification "
             "with {sensitivity} sensitivity, {details}' added to the Stage 2 question."]))
    return tables


def table_lambda(runs: List[Run]) -> List[Table]:
    """Table 9: LTM retrieval weights lambda1 (recency) and lambda2 (similarity)."""
    lam_runs = [r for r in runs if r.sensitivity == 'none' and r.meta.get('use_memory')
                and r.meta.get('ltm_lambda_sweep')]
    if not lam_runs:
        return []
    tables = []
    for name in sorted({n for r in lam_runs for n in r.agents}):
        rows = []
        for r in sorted(lam_runs, key=lambda x: x.ltm_lambda):
            if name not in r.agents:
                continue
            m = _agent_metrics(r, name)
            ltm = [int(x['result'].get('ltm_cases_referenced', 0) or 0)
                   for x in r.agents[name]]
            rows.append([f"{r.ltm_lambda[0]:g}", f"{r.ltm_lambda[1]:g}",
                         f"{100 * m['accuracy']:.1f}", f"{100 * m['macro_precision']:.1f}",
                         f"{100 * m['macro_recall']:.1f}", f"{100 * m['macro_f1']:.1f}",
                         m['n'], f"{sum(1 for x in ltm if x > 0)}/{len(ltm)}"])
        if len(rows) < 2:
            continue
        safe = name.replace(' ', '_').replace('.', '_')
        notes = ["λ1 weights recency and λ2 cosine similarity in long-term-memory "
                 "retrieval (paper Eq. 1). Precision and recall are macro-averaged.",
                 f"Default in V31: λ1={DEFAULT_LTM_LAMBDA[0]:g}, λ2={DEFAULT_LTM_LAMBDA[1]:g}."]
        if all(row[-1].startswith('0/') for row in rows):
            notes.append("WARNING: no session retrieved a memory case in any setting, "
                         "so λ cannot have had an effect. The memory was not live.")
        tables.append(Table(f"table9_ltm_lambda_{safe}",
                            f"Performance for different values of λ1 and λ2 ({name} Agent). "
                            f"Paper Table 9 layout.",
                            ['λ1', 'λ2', 'Accuracy (%)', 'Precision (%)', 'Recall (%)',
                             'F1 (%)', 'n', 'Sessions with LTM'], rows, notes))
    return tables


def tables_temperature(sweeps) -> Tuple[List[Table], List[str]]:
    tables, figs = [], []
    for path, data in sweeps:
        rows, series = [], {}
        llms = data.get('llms') or {}
        for slot, by_t in data['sweep'].items():
            name = pretty_model(llms.get(slot, ''), slot)
            for t in sorted(by_t, key=float):
                rr = by_t[t]
                if not rr:
                    continue
                v = agent_view(rr)
                m = binary_metrics(v['y'], v['yhat'])
                rows.append([name, f"{float(t):g}", m['n'], _fmt(m['accuracy']),
                             _fmt(m['macro_f1']), _fmt(m['mcc']), _fmt(m['far'])])
                series.setdefault(name, []).append((float(t), m['accuracy'], m['macro_f1']))
        if not rows:
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        tables.append(Table(f"table_temperature_{tag}",
                            "Effect of decoding temperature across LLM models",
                            ['Agent', 'Temperature', 'n', 'Accuracy', 'Macro-F1',
                             'MCC', 'FAR'], rows))
        series_by_tag = series
        figs.append((tag, series_by_tag))
    return tables, figs


def figure_temperature(tag, series, out_dir) -> List[str]:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, pts in series.items():
        pts = sorted(pts)
        a1.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=name)
        a2.plot([p[0] for p in pts], [p[2] for p in pts], 'o-', label=name)
    for ax, lab in ((a1, 'Accuracy'), (a2, 'Macro-F1')):
        ax.set_xlabel('Temperature'); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('Effect of temperature across LLM models', fontsize=10)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"figure_temperature_{tag}")
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════════
def analyse_run(run: Run, out_dir: str, n_boot: int, source: str, tau: float,
                verbose: bool = True) -> Dict[str, List[str]]:
    idx = run.common_indices()
    tdir, fdir = os.path.join(out_dir, 'tables'), os.path.join(out_dir, 'figures')
    written = {'tables': [], 'figures': [], 'other': []}
    if not idx:
        print(f"  [skip] {run.path}: the agents share no sessions")
        return written
    systems, _ = _models_on(run, idx)

    tables = [table_per_class_f1(run, idx, systems),
              table_detection(run, idx, systems, n_boot),
              table_statistics(run, idx, systems, n_boot)]
    agr = table_agreement(run, idx)
    if agr:
        tables.append(agr)
    cal, curves = table_calibration(run, idx, systems)
    tables += [cal]
    tables += table_sensitivity_posthoc(run, idx, source, tau)
    tables += [table_zones(run, idx), table_efficiency(run, idx),
               table_integrity(run, idx), table_xai(run, idx)]

    counts = {n: len(rws) for n, rws in run.agents.items()}
    if len(set(counts.values())) > 1:
        print(f"  [note] agents completed different numbers of sessions {counts}; "
              f"every table uses the {len(idx)} sessions all of them share.")
    for t in tables:
        written['tables'] += t.write(tdir)
        if verbose:
            print(t.text())

    for fn, args in ((figure_confidence_distributions, (run, idx, fdir, 'zone')),
                     (figure_confidence_distributions, (run, idx, fdir, 'correct')),
                     (figure_roc_pr, (run, systems, fdir)),
                     (figure_calibration, (run, curves, fdir))):
        try:
            written['figures'] += fn(*args)
        except Exception as e:
            print(f"  [figure skipped] {fn.__name__}: {e}")
    written['other'].append(write_error_analysis(run, idx, out_dir))
    return written


def generate_paper_outputs(paths: Sequence[str], out_dir: Optional[str] = None,
                           n_boot: int = 2000, source: str = 'stated',
                           tau: float = DEFAULT_TAU, verbose: bool = True) -> str:
    """Analyse each run, then every cross-run table the set of runs supports."""
    runs, sweeps = load_runs(expand_paths(paths))
    if not runs and not sweeps:
        print("  No usable results files.")
        return ''
    if out_dir is None:
        base = os.path.dirname(os.path.abspath(runs[0].path if runs else sweeps[0][0]))
        out_dir = os.path.join(base, 'paper', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    index = [f"# Paper outputs — EV-IDS-Agent V31",
             f"", f"Generated {datetime.now():%Y-%m-%d %H:%M} from {len(runs)} run(s)"
             f" and {len(sweeps)} temperature sweep(s).", "",
             "LaTeX tables use `\\usepackage{booktabs}`.", ""]

    # Per-run analysis. Sweep runs (option b, lambda) feed only their own
    # cross-run tables and are not analysed individually to avoid clutter.
    for run in runs:
        if not run.is_standard:
            continue
        sub = os.path.join(out_dir, run.tag)
        if verbose:
            print(f"\n{'=' * 100}\nPAPER METRICS — {run.tag} ({os.path.basename(run.path)})"
                  f"\n{'=' * 100}")
        w = analyse_run(run, sub, n_boot, source, tau, verbose)
        index.append(f"## {run.tag}  (`{os.path.basename(run.path)}`)")
        index += [f"- {os.path.relpath(p, out_dir)}" for p in w['tables'] + w['figures'] + w['other']]
        index.append("")

    cross = []
    for t in (table_seed_robustness(runs), table_ablation(runs)):
        if t:
            cross.append(t)
    cross += table_sensitivity_prompted(runs)
    cross += table_lambda(runs)
    tt, tfigs = tables_temperature(sweeps)
    cross += tt
    cdir = os.path.join(out_dir, 'cross_run')
    if cross or tfigs:
        index.append("## Cross-run")
    for t in cross:
        paths_w = t.write(os.path.join(cdir, 'tables'))
        index += [f"- {os.path.relpath(p, out_dir)}" for p in paths_w]
        if verbose:
            print(t.text())
    for tag, series in tfigs:
        try:
            for p in figure_temperature(tag, series, os.path.join(cdir, 'figures')):
                index.append(f"- {os.path.relpath(p, out_dir)}")
        except Exception as e:
            print(f"  [figure skipped] temperature: {e}")

    index += ["", "## Caveats", "",
              "- Figure 8 splits by difficulty zone and by correct/incorrect. It is "
              "not an out-of-distribution evaluation: no held-out attack family "
              "exists in this dataset.",
              "- Table 8a is a post-hoc operating-point analysis of one run; Table 8b "
              "(when present) re-runs the agent with the paper's prompt instruction.",
              "- 'Best ML' in the statistics table is selected on the test sessions, "
              "which favours the baseline.",
              "- Check `table_provenance_and_retrieval` before interpreting any "
              "memory or RAG result: retrieval that returned nothing means the "
              "component was not live."]
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(index) + "\n")
    if verbose:
        print(f"\n  Paper outputs written to: {out_dir}")
    return out_dir


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('files', nargs='+', help='results JSON files (globs allowed)')
    ap.add_argument('--out', default=None, help='output directory')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--sensitivity-source', choices=('stated', 'composite'),
                    default='stated',
                    help="option (a): decide 'sure' from the LLM's stated "
                         "confidence or from the composite confidence")
    ap.add_argument('--tau', type=float, default=DEFAULT_TAU,
                    help='composite-confidence threshold for option (a)')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)
    out = generate_paper_outputs(a.files, a.out, a.bootstrap, a.sensitivity_source,
                                 a.tau, verbose=not a.quiet)
    return 0 if out else 1


if __name__ == '__main__':
    sys.exit(main())
