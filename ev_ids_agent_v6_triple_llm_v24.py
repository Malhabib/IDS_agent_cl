# ev_ids_agent_v6_triple_llm_v24.py
"""
EV-IDS-Agent VERSION 6 - TRIPLE LLM V24

Everything from V23 (SHAP + LIME per ML model), PLUS a hardened,
evidence-grounded reasoning layer. NOTHING about how the 7 ML models are
trained changes — auto_train_models_v6 is byte-for-byte identical to V23.
All gains live in the confidence / XAI / prompt / parsing layers.

V24 improvements over V23
─────────────────────────
1. HONEST PER-MODEL CONFIDENCE (no retraining)
     - Every model reports a calibrated confidence = raw_prob * reliability,
       where `reliability` reflects how well that model can actually be
       trusted given the (intentionally weak) training regime.
     - SVC (probability=False) no longer reports a fake 0.50 — its
       confidence is derived from the decision_function margin via sigmoid.

2. ROBUST XAI EVIDENCE
     - SHAP values normalised to a % share of the model's total attribution.
     - Per-model SHAP<->LIME agreement flag (strong / partial / conflict).
     - Cross-model consensus: which single feature drives the ensemble, and
       how many models agree, surfaced ONCE instead of buried per model.
     - Confidence-weighted vote tally (evidence only — never the verdict).

3. OPTIMALLY ENGINEERED PROMPT
     - System prompt carries an explicit reliability doctrine, agreement
       doctrine, a physical-plausibility gate, and two few-shot worked
       examples that stabilise output format for small local models.
     - Stage 1 forces a numbered 6-step reasoning chain, each step required
       to cite concrete numbers.
     - Stage 2 uses a strict decision contract.

4. FULL REASONING TRACE
     - All six reasoning steps + the decision rationale are parsed and
       persisted in final_result['reasoning_trace'] for auditability.

5. ROBUSTNESS
     - Stage-1 completeness validation with a single retry before ABSTAIN.
     - Data-quality guards (negative duration, requested<=0, parse failures)
       surfaced to the LLM instead of silently passing zeros.
     - Deterministic decoding (temperature=0) actually plumbed through to
       Ollama (V23 accepted a temperature kwarg but ignored it).

Install optional deps first:
    pip install shap lime
"""

import json, os, time, re, pickle, warnings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed — SHAP explanations disabled. Run: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[WARN] lime not installed — LIME explanations disabled. Run: pip install lime")

FEATURE_NAMES = ['connectionTime', 'disconnectTime', 'RequestedDemand', 'kWhDelivered']
FEATURE_LABELS = {
    'connectionTime':   'Connection time',
    'disconnectTime':   'Disconnect time',
    'RequestedDemand':  'Energy requested (kWh)',
    'kWhDelivered':     'Energy delivered (kWh)'
}

# Physically meaningful features for EV energy fraud. A model whose top SHAP
# driver is one of these is more credible than one keyed on connection timing.
PHYSICAL_FEATURES = {'Energy requested (kWh)', 'Energy delivered (kWh)'}

# How much each model can be TRUSTED given the (intentionally weak) training
# regime in auto_train_models_v6. This is NOT a retrained accuracy — it is a
# prior reliability weight applied to the model's raw probability so that the
# LLM automatically discounts models we know are underfit (MLP/LR max_iter=1,
# single-stump DT/RF, tiny GB). Tune freely; it never changes training.
MODEL_RELIABILITY = {
    'Random Forest':             0.80,
    'K-Nearest Neighbors':       0.75,
    'Support Vector Classifier': 0.70,
    'Decision Tree':             0.65,
    'Gradient Boosting':         0.60,
    'Logistic Regression':       0.50,
    'MLP':                       0.40,
}


def parse_datetime_to_timestamp(dt_str):
    try:
        dt = pd.to_datetime(dt_str, utc=True)
        return float(dt.timestamp())
    except Exception:
        return 0.0


def get_column_mapping(df):
    mapping = {}
    column_variations = {
        'connectionTime':  ['connectionTime', 'connection_time', 'ConnectionTime'],
        'disconnectTime':  ['disconnectTime', 'disconnect_time', 'DisconnectTime', 'doneChargingTime'],
        'RequestedDemand': ['RequestedDemand', 'RequstedDemand', 'requested_demand'],
        'kWhDelivered':    ['kWhDelivered', 'kwh_delivered', 'KwhDelivered'],
        'label':           ['label', 'Label', 'class', 'Class', 'catt']
    }
    for standard_name, variations in column_variations.items():
        found = None
        for variant in variations:
            if variant in df.columns:
                found = variant
                break
        mapping[standard_name] = found if found else standard_name
    return mapping


def classify_difficulty_zone(delivery_ratio):
    if delivery_ratio > 1.5 or delivery_ratio < 0.4:   return "EASY_ATTACK"
    elif 1.3 <= delivery_ratio <= 1.5:                  return "BORDERLINE_HIGH"
    elif 0.4 <= delivery_ratio <= 0.6:                  return "BORDERLINE_LOW"
    elif 0.8 <= delivery_ratio <= 1.2:                  return "EASY_NORMAL"
    elif 1.2 < delivery_ratio < 1.3:                    return "AMBIGUOUS_HIGH"
    elif 0.6 < delivery_ratio < 0.8:                    return "AMBIGUOUS_LOW"
    else:                                                return "UNCERTAIN"


# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL CONFIDENCE  (V24 — honest confidence, no retraining)
# ══════════════════════════════════════════════════════════════════════════════
def compute_model_confidence(name: str, model, X_scaled: np.ndarray) -> Dict[str, Any]:
    """
    Returns an honest confidence read for a single fitted model on ONE sample,
    without retraining anything.

    Keys:
      prediction       : 'Malicious' | 'Normal'
      label_idx        : 0 | 1  (index into model.classes_ semantics: 1==Malicious)
      prob             : raw model probability for the predicted class  (0..1)
      margin           : separation strength (|p_attack - p_normal| or |decision|)
      reliability      : prior trust weight for this model family (0..1)
      calibrated_conf  : prob * reliability  (what the LLM should weight by)
      confidence       : == prob  (kept for backward-compat with V23 runner)
    """
    reliability = MODEL_RELIABILITY.get(name, 0.6)

    if hasattr(model, 'predict_proba'):
        proba   = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        pidx    = int(np.argmax(proba))
        prob    = float(proba[pidx])
        label   = 'Malicious' if int(classes[pidx]) == 1 else 'Normal'
        # binary separation; robust to >2 classes by taking top1-top2
        srt     = np.sort(proba)[::-1]
        margin  = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
        label_idx = 1 if label == 'Malicious' else 0
    else:
        # SVC(probability=False) — derive confidence from decision margin.
        d      = float(np.ravel(model.decision_function(X_scaled))[0])
        p_att  = 1.0 / (1.0 + np.exp(-d))          # sigmoid of margin
        if p_att >= 0.5:
            label, prob, label_idx = 'Malicious', p_att, 1
        else:
            label, prob, label_idx = 'Normal', 1.0 - p_att, 0
        margin = abs(d)

    return {
        'prediction':      label,
        'label_idx':       label_idx,
        'prob':            round(prob, 4),
        'margin':          round(margin, 4),
        'reliability':     reliability,
        'calibrated_conf': round(prob * reliability, 4),
        'confidence':      round(prob, 4),   # back-compat: V23 code reads ['confidence']
    }


# ══════════════════════════════════════════════════════════════════════════════
# XAI EXPLAINER  (V24 — adds normalised SHAP %, SHAP<->LIME agreement)
# ══════════════════════════════════════════════════════════════════════════════
class XAIExplainer:
    """
    Computes SHAP and LIME explanations for each ML model prediction.

    Initialised ONCE per agent run with a background dataset from the training
    split (no test-set contamination). Explainer routing:
      - RF, DT, GB   -> TreeExplainer
      - LR           -> LinearExplainer
      - MLP, SVC, KNN-> KernelExplainer (small background sample for speed)
    LIME: LimeTabularExplainer for all models.

    V24 additions vs V23:
      - each SHAP entry also carries its % share of the model's total |SHAP|
      - explain_model() returns an `agreement` flag comparing the top SHAP
        driver against the top LIME driver (strong / partial / conflict / n/a)
    """

    TREE_MODELS   = {'Random Forest', 'Decision Tree', 'Gradient Boosting'}
    LINEAR_MODELS = {'Logistic Regression'}
    KERNEL_MODELS = {'MLP', 'Support Vector Classifier', 'K-Nearest Neighbors'}

    def __init__(self, models: dict, background_X: np.ndarray,
                 feature_names: list, n_kernel_bg: int = 30):
        self.models        = models
        self.feature_names = feature_names
        self.background_X   = background_X
        n = min(n_kernel_bg, len(background_X))
        self.kernel_bg      = shap.sample(background_X, n) if SHAP_AVAILABLE else background_X[:n]
        self.shap_explainers = {}
        self.lime_explainer  = None
        self._init_shap()
        self._init_lime()

    def _init_shap(self):
        if not SHAP_AVAILABLE:
            return
        for name, model in self.models.items():
            try:
                if name in self.TREE_MODELS:
                    self.shap_explainers[name] = shap.TreeExplainer(
                        model, data=self.background_X, feature_perturbation='interventional')
                elif name in self.LINEAR_MODELS:
                    self.shap_explainers[name] = shap.LinearExplainer(
                        model, self.background_X)
                else:
                    if hasattr(model, 'predict_proba'):
                        pred_fn = lambda X, m=model: m.predict_proba(X)
                    else:
                        pred_fn = lambda X, m=model: m.decision_function(X).reshape(-1, 1)
                    self.shap_explainers[name] = shap.KernelExplainer(pred_fn, self.kernel_bg)
                print(f"  [SHAP] {name}: {type(self.shap_explainers[name]).__name__} ready")
            except Exception as e:
                print(f"  [SHAP] {name}: failed to init ({e})")

    def _init_lime(self):
        if not LIME_AVAILABLE:
            return
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data  = self.background_X,
                feature_names  = self.feature_names,
                class_names    = ['Normal', 'Malicious'],
                mode           = 'classification',
                discretize_continuous = True,
                random_state   = 42
            )
            print(f"  [LIME] LimeTabularExplainer ready "
                  f"({len(self.background_X)} background samples)")
        except Exception as e:
            print(f"  [LIME] Failed to init: {e}")

    @staticmethod
    def _agreement(shap_entries, lime_entries) -> str:
        """
        Compare top SHAP driver vs top LIME driver.
          strong   : same feature AND same direction (both toward Attack/Normal)
          partial  : same feature, opposite/ambiguous direction
          conflict : different top feature
          n/a      : missing one side
        """
        if not shap_entries or not lime_entries:
            return "n/a"
        shap_feat = shap_entries[0][0]
        shap_dir  = shap_entries[0][1] >= 0          # True => toward Attack
        lime_feat = lime_entries[0][0]
        lime_dir  = lime_entries[0][2] >= 0          # True => supports Attack
        if shap_feat == lime_feat:
            return "strong" if shap_dir == lime_dir else "partial"
        return "conflict"

    def explain_model(self, model_name: str, X_scaled: np.ndarray,
                      predicted_class_idx: int) -> Dict:
        result = {
            'shap_values': [], 'lime_values': [],
            'shap_ok': False,  'lime_ok': False,
            'agreement': 'n/a'
        }
        model = self.models.get(model_name)
        if model is None:
            return result

        # ── SHAP ──────────────────────────────────────────────────────────────
        if SHAP_AVAILABLE and model_name in self.shap_explainers:
            try:
                explainer = self.shap_explainers[model_name]
                raw = explainer.shap_values(X_scaled)
                if isinstance(raw, list):
                    idx = min(predicted_class_idx, len(raw) - 1)
                    sv  = np.array(raw[idx]).flatten()
                elif isinstance(raw, np.ndarray):
                    if raw.ndim == 3:
                        sv = raw[0, :, predicted_class_idx]
                    elif raw.ndim == 2:
                        sv = raw[0]
                    else:
                        sv = raw.flatten()
                else:
                    sv = np.zeros(len(self.feature_names))

                total_abs = float(np.sum(np.abs(sv))) + 1e-12
                shap_entries = []
                for i, fname in enumerate(self.feature_names):
                    val = float(sv[i]) if i < len(sv) else 0.0
                    pct = abs(val) / total_abs * 100.0
                    direction = (
                        "strongly toward Attack" if val >  0.3 else
                        "toward Attack"           if val >  0.1 else
                        "slightly toward Attack"  if val >  0.02 else
                        "negligible"              if abs(val) <= 0.02 else
                        "slightly toward Normal"  if val > -0.1 else
                        "toward Normal"           if val > -0.3 else
                        "strongly toward Normal"
                    )
                    # (label, value, direction, pct_of_total)
                    shap_entries.append((FEATURE_LABELS.get(fname, fname), val, direction, pct))
                shap_entries.sort(key=lambda x: abs(x[1]), reverse=True)
                result['shap_values'] = shap_entries
                result['shap_ok']     = True
            except Exception as e:
                result['shap_values'] = []
                print(f"    [SHAP] {model_name} explain error: {e}")

        # ── LIME ──────────────────────────────────────────────────────────────
        if LIME_AVAILABLE and self.lime_explainer is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    pred_fn = model.predict_proba
                else:
                    def pred_fn(X, m=model):
                        d = m.decision_function(X)
                        p = 1 / (1 + np.exp(-d))
                        return np.column_stack([1 - p, p])

                exp = self.lime_explainer.explain_instance(
                    X_scaled[0], pred_fn,
                    num_features = len(self.feature_names),
                    num_samples  = 500,
                    labels       = (predicted_class_idx,)
                )
                lime_list = exp.as_list(label=predicted_class_idx)
                lime_entries = []
                for condition, weight in lime_list:
                    matched_label = condition
                    for fname, flabel in FEATURE_LABELS.items():
                        if fname.lower() in condition.lower():
                            matched_label = flabel
                            break
                    direction = (
                        "supports Attack"  if weight >  0.05 else
                        "supports Normal"  if weight < -0.05 else
                        "weak signal"
                    )
                    lime_entries.append((matched_label, condition, float(weight), direction))
                lime_entries.sort(key=lambda x: abs(x[2]), reverse=True)
                result['lime_values'] = lime_entries
                result['lime_ok']     = True
            except Exception as e:
                result['lime_values'] = []
                print(f"    [LIME] {model_name} explain error: {e}")

        result['agreement'] = self._agreement(result['shap_values'], result['lime_values'])
        return result


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE BUNDLE  (V24 — cross-model consensus + confidence-weighted tally)
# ══════════════════════════════════════════════════════════════════════════════
def build_evidence_bundle(all_predictions: Dict[str, Dict],
                          xai_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Aggregates per-model confidence + XAI into ensemble-level evidence the LLM
    can reason over in one glance. Pure evidence — never a verdict.
    """
    # confidence-weighted vote
    w_attack = w_normal = 0.0
    for mn, pred in all_predictions.items():
        w = pred.get('calibrated_conf', pred.get('confidence', 0.5))
        if pred['prediction'] == 'Malicious':
            w_attack += w
        else:
            w_normal += w

    # cross-model top-SHAP-driver consensus (direction-aware)
    driver_counts   = defaultdict(lambda: {'attack': 0, 'normal': 0})
    physical_drivers = 0
    n_with_shap      = 0
    for mn, xai in xai_results.items():
        sv = xai.get('shap_values') or []
        if not sv:
            continue
        n_with_shap += 1
        top_feat, top_val = sv[0][0], sv[0][1]
        if top_val >= 0:
            driver_counts[top_feat]['attack'] += 1
        else:
            driver_counts[top_feat]['normal'] += 1
        if top_feat in PHYSICAL_FEATURES:
            physical_drivers += 1

    consensus = {'top_feature': None, 'direction': None, 'models_agreeing': 0}
    if driver_counts:
        top_feat = max(driver_counts.items(),
                       key=lambda kv: kv[1]['attack'] + kv[1]['normal'])
        feat_name, counts = top_feat
        direction = 'Attack' if counts['attack'] >= counts['normal'] else 'Normal'
        consensus = {
            'top_feature':     feat_name,
            'direction':       direction,
            'models_agreeing': max(counts['attack'], counts['normal']),
            'n_models':        n_with_shap,
            'physical':        feat_name in PHYSICAL_FEATURES,
        }

    # agreement / conflict rollup
    strong   = [mn for mn, x in xai_results.items() if x.get('agreement') == 'strong']
    conflict = [mn for mn, x in xai_results.items() if x.get('agreement') == 'conflict']

    mean_conf = (np.mean([p.get('calibrated_conf', 0.5)
                          for p in all_predictions.values()])
                 if all_predictions else 0.0)

    return {
        'weighted_attack':   round(w_attack, 3),
        'weighted_normal':   round(w_normal, 3),
        'consensus':         consensus,
        'physical_drivers':  physical_drivers,
        'n_with_shap':       n_with_shap,
        'strong_agreement':  strong,
        'conflict_models':   conflict,
        'mean_calibrated_conf': round(float(mean_conf), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE (RAG) — concepts only, no thresholds (identical to V22/V23)
# ══════════════════════════════════════════════════════════════════════════════
class KnowledgeBaseLoader:
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.embedding_model = None
        self.collection = None
        if not os.path.exists(kb_path):
            return
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                return
        if CHROMADB_AVAILABLE and self.embedding_model:
            try:
                chroma_client = chromadb.PersistentClient(path=kb_path)
                self.collection = chroma_client.get_collection("ev_security_knowledge")
            except Exception:
                pass

    def query(self, query_text: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        if not self.embedding_model or not self.collection:
            return self._fallback_knowledge(query_text), []
        try:
            query_embedding = self.embedding_model.encode([query_text]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
            if results['documents'] and len(results['documents'][0]) > 0:
                combined_text = "\n\n".join(results['documents'][0])
                metadata_list = [{"source": meta.get('source', 'unknown'), "rank": i + 1}
                                 for i, meta in enumerate(results['metadatas'][0])]
                return combined_text, metadata_list
            return self._fallback_knowledge(query_text), []
        except Exception:
            return self._fallback_knowledge(query_text), []

    def _fallback_knowledge(self, query_text: str) -> str:
        """Concepts only — no numerical thresholds (V22/V23/V24 design)."""
        q = query_text.lower()
        feature_context = (
            "FEATURE MEANING:\n"
            "- Requested energy: how much the EV owner asked to charge (kWh)\n"
            "- Delivered energy: how much the station actually provided (kWh)\n"
            "- Session duration: how long the vehicle was connected\n"
            "- The ratio of delivered-to-requested energy is a key signal:\n"
            "    A ratio near 1.0 means the station delivered roughly what was asked.\n"
            "    A ratio well above 1.0 means more energy was delivered than requested.\n"
            "    A ratio well below 1.0 means far less energy was delivered than requested.\n"
        )
        attack_concepts = (
            "KNOWN ATTACK PATTERNS:\n"
            "- Energy theft: attacker manipulates firmware or billing to extract "
            "significantly more energy than their session authorized. "
            "Delivered amount substantially exceeds what was requested.\n"
            "- Phantom charging: session is registered and billed but little or no "
            "energy is actually transferred. Delivered amount is far below the request.\n"
            "- Normal variation: real sessions have minor over/under delivery due to "
            "grid fluctuations, battery management, and connector timing.\n"
        )
        population_context = (
            "POPULATION BEHAVIOR:\n"
            "- Normal sessions: delivered-to-requested ratio is typically close to 1.0. "
            "Most normal sessions land between 0.8 and 1.1.\n"
            "- Attack sessions: ratios that deviate significantly from 1.0 — either "
            "well above (energy theft) or well below (phantom charging).\n"
            "- The two populations overlap at the edges. Use ALL evidence.\n"
        )
        if "energy" in q and "theft" in q:
            return (feature_context + attack_concepts + population_context +
                    "CONTEXT: Consider whether delivered amount substantially exceeds "
                    "authorization and whether SHAP/LIME confirm energy-related features "
                    "as the primary drivers.")
        elif "phantom" in q:
            return (feature_context + attack_concepts + population_context +
                    "CONTEXT: Consider whether delivered amount is far below the request "
                    "and whether SHAP/LIME show energy delivery as the suppressed feature.")
        elif "normal" in q:
            return (feature_context + attack_concepts + population_context +
                    "CONTEXT: Consider whether deviations from requested amount are within "
                    "expected variation, and whether SHAP/LIME show no strong attack signals.")
        else:
            return (feature_context + attack_concepts + population_context +
                    "CONTEXT: Use the physical meaning and SHAP/LIME feature contributions "
                    "to reason about this session. Do not rely on hard cutoffs.")


# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY — identical to V22/V23
# ══════════════════════════════════════════════════════════════════════════════
class LongTermMemory:
    def __init__(self, ltm_path):
        self.ltm_path    = ltm_path
        self.memory_file = os.path.join(ltm_path, "long_term_memory.json")
        self.embedding_model = None
        self.memories    = []
        os.makedirs(ltm_path, exist_ok=True)
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memories = json.load(f)
            except Exception:
                self.memories = []
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def add_memory(self, entry):
        self.memories.append(entry)
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception:
            pass

    def clear(self):
        self.memories = []
        try:
            with open(self.memory_file, 'w') as f:
                json.dump([], f)
        except Exception:
            pass

    def retrieve_similar(self, current_observations, timestamp, top_k=5):
        if not self.memories or not self.embedding_model:
            return []
        try:
            cur_emb = self.embedding_model.encode([current_observations]).tolist()[0]
            scores  = []
            for i, mem in enumerate(self.memories):
                time_diff    = timestamp - mem.get('timestamp', 0)
                max_td       = max([timestamp - m.get('timestamp', 0) for m in self.memories] + [1])
                recency      = 1 - (time_diff / max_td) if max_td > 0 else 0
                mem_obs      = mem.get('observations', '')
                if mem_obs:
                    mem_emb  = self.embedding_model.encode([mem_obs]).tolist()[0]
                    cos_sim  = np.dot(cur_emb, mem_emb) / (
                        np.linalg.norm(cur_emb) * np.linalg.norm(mem_emb) + 1e-10)
                else:
                    cos_sim  = 0
                scores.append((0.3 * recency + 0.7 * cos_sim, i))
            scores.sort(reverse=True, key=lambda x: x[0])
            return [self.memories[idx] for _, idx in scores[:top_k]]
        except Exception:
            return []

    def seed_from_training(self, df, column_mapping, train_indices, n_seed=50):
        if self.memories:
            return
        df_train = df.loc[train_indices].copy()
        req_col  = column_mapping['RequestedDemand']
        del_col  = column_mapping['kWhDelivered']
        lbl_col  = column_mapping['label']
        conn_col = column_mapping['connectionTime']
        disc_col = column_mapping['disconnectTime']
        mal_df   = df_train[df_train[lbl_col] == 1]
        nor_df   = df_train[df_train[lbl_col] == 0]
        n_each   = n_seed // 2
        seed_samples = pd.concat([
            mal_df.sample(n=min(n_each, len(mal_df)), random_state=42),
            nor_df.sample(n=min(n_each, len(nor_df)), random_state=42)
        ])
        for idx, row in seed_samples.iterrows():
            try:
                req   = float(row[req_col])
                deliv = float(row[del_col])
                if req <= 0:
                    continue
                ratio = deliv / req
                try:
                    ct  = pd.to_datetime(str(row[conn_col]), utc=True).timestamp()
                    dt  = pd.to_datetime(str(row[disc_col]), utc=True).timestamp()
                    dur = (dt - ct) / 3600.0
                except Exception:
                    dur = 0.0
                label = 'Malicious' if int(row[lbl_col]) == 1 else 'Normal'
                atype = ('energy_theft'     if label == 'Malicious' and ratio > 1.3 else
                         'phantom_charging'  if label == 'Malicious' and ratio < 0.7 else
                         'unknown_attack'    if label == 'Malicious' else 'none')
                obs = (f"requested={req:.2f}kWh, delivered={deliv:.2f}kWh, "
                       f"ratio={ratio:.3f}, duration={dur:.2f}h, attack_type={atype}")
                if label == 'Normal':
                    rsummary = (f"Classified Normal: delivered ({deliv:.2f}kWh) close to "
                                f"requested ({req:.2f}kWh).")
                elif atype == 'energy_theft':
                    rsummary = (f"Classified Attack (energy_theft): delivered ({deliv:.2f}kWh) "
                                f"substantially exceeded requested ({req:.2f}kWh).")
                elif atype == 'phantom_charging':
                    rsummary = (f"Classified Attack (phantom_charging): delivered ({deliv:.2f}kWh) "
                                f"far below requested ({req:.2f}kWh).")
                else:
                    rsummary = f"Classified Attack (unknown): anomalous ratio={ratio:.3f}."
                self.memories.append({
                    "timestamp": datetime.now().timestamp() - (len(self.memories) + 1),
                    "line_number": int(idx), "observations": obs, "reasoning": rsummary,
                    "prediction": label, "confidence": 0.95,
                    "delivery_ratio": round(ratio, 3),
                    "requested_kwh": round(req, 3), "delivered_kwh": round(deliv, 3),
                    "duration_hours": round(dur, 2), "attack_type": atype, "source": "training_seed"
                })
            except Exception:
                continue
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception:
            pass
        print(f"  [LTM] Seeded with {len(self.memories)} training cases")


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA CLIENT — V24: temperature actually plumbed through to Ollama
# ══════════════════════════════════════════════════════════════════════════════
class SimpleOllamaClient:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url   = base_url.rstrip('/')
        try:
            from ollama import chat
            self.chat = chat
            print(f"  [OK] Ollama library loaded for {model_name}")
        except ImportError:
            raise ImportError("Ollama library not installed. Run: pip install ollama")

    def _options(self, temperature, max_tokens):
        opts = {}
        if temperature is not None:
            opts['temperature'] = temperature
        if max_tokens is not None:
            opts['num_predict'] = max_tokens
        return opts

    def generate_with_system(self, system_prompt, user_prompt,
                             temperature=0.0, max_tokens=1024, **kwargs):
        try:
            is_qwen  = "qwen" in self.model_name.lower()
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user',   'content': user_prompt}]
            options  = self._options(temperature, max_tokens)
            if is_qwen:
                response = self.chat(model=self.model_name, messages=messages,
                                     think=True, options=options)
                thinking = getattr(response.message, 'thinking', None) or ""
                content  = response.message.content or ""
                if thinking:
                    print(f"    [THINK] {len(thinking)} chars reasoning")
                return content
            return self.chat(model=self.model_name, messages=messages,
                             options=options).message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_two_turn(self, system_prompt, first_user, first_assistant, second_user,
                          temperature=0.0, max_tokens=1024):
        try:
            is_qwen  = "qwen" in self.model_name.lower()
            messages = [
                {'role': 'system',    'content': system_prompt},
                {'role': 'user',      'content': first_user},
                {'role': 'assistant', 'content': first_assistant},
                {'role': 'user',      'content': second_user}
            ]
            options  = self._options(temperature, max_tokens)
            if is_qwen:
                response = self.chat(model=self.model_name, messages=messages,
                                     think=True, options=options)
                thinking = getattr(response.message, 'thinking', None) or ""
                content  = response.message.content or ""
                if thinking:
                    print(f"    [THINK-2] {len(thinking)} chars reasoning")
                return content
            return self.chat(model=self.model_name, messages=messages,
                             options=options).message.content
        except Exception as e:
            return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT — V24
# ══════════════════════════════════════════════════════════════════════════════
class EVIDSAgentV6TripleLLMV24:
    """
    V24: Everything from V23 plus honest per-model confidence, robust XAI
    evidence (normalised SHAP %, SHAP<->LIME agreement, cross-model consensus,
    confidence-weighted tally), an optimally engineered prompt with a forced
    6-step reasoning chain + few-shot anchors, full reasoning-trace capture,
    and Stage-1 completeness validation with one retry before ABSTAIN.
    """

    # Required reasoning-step headers Stage 1 MUST contain (for validation).
    STEP_HEADERS = ['STEP 1', 'STEP 2', 'STEP 3', 'STEP 4', 'STEP 5', 'STEP 6']

    def __init__(self, config, llm_client, use_knowledge=True, use_memory=True,
                 scenario_tag="default"):
        self.config        = config
        self.llm_client    = llm_client
        self.use_knowledge = use_knowledge
        self.use_memory    = use_memory
        self.scenario_tag  = scenario_tag
        self.df            = pd.read_csv(config['data_path'])
        self.column_mapping = get_column_mapping(self.df)

        scaler_path = os.path.join(config['models_dir'], 'ev_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()

        self.models = self._load_models()
        self.short_term_memory = []

        self.ltm = LongTermMemory(os.path.join(
            config.get('workspace_dir', './ev_ids_workspace_v6'),
            f'long_term_memory_{scenario_tag}'
        )) if use_memory else None

        self.kb = KnowledgeBaseLoader(
            config.get('knowledge_base_path', './ev_ids_workspace_v6/knowledge_base')
        ) if use_knowledge else None

        self.xai = None
        self._init_xai()

    def _load_models(self):
        models = {}
        model_files = {
            'Random Forest':            'ev_rf_model.pkl',
            'K-Nearest Neighbors':      'ev_knn_model.pkl',
            'Logistic Regression':      'ev_lr_model.pkl',
            'Decision Tree':            'ev_dt_model.pkl',
            'MLP':                      'ev_mlp_model.pkl',
            'Support Vector Classifier':'ev_svc_model.pkl',
            'Gradient Boosting':        'ev_gb_model.pkl'
        }
        for name, fn in model_files.items():
            fp = os.path.join(self.config['models_dir'], fn)
            if os.path.exists(fp):
                with open(fp, 'rb') as f:
                    models[name] = pickle.load(f)
        return models

    def _init_xai(self):
        if not SHAP_AVAILABLE and not LIME_AVAILABLE:
            print("  [XAI] Both SHAP and LIME unavailable — skipping XAI init")
            return
        try:
            split_path = os.path.join(self.config['models_dir'], 'split_info_v6.pkl')
            if not os.path.exists(split_path):
                print("  [XAI] split_info_v6.pkl not found — cannot build background")
                return
            with open(split_path, 'rb') as f:
                split_info = pickle.load(f)
            train_idx = split_info['train_indices']

            cm = self.column_mapping
            actual_cols = [cm[c] for c in ['connectionTime', 'disconnectTime',
                                            'RequestedDemand', 'kWhDelivered']]
            train_df = self.df.loc[train_idx, actual_cols].copy()
            train_df.columns = FEATURE_NAMES
            for col in ['connectionTime', 'disconnectTime']:
                train_df[col] = train_df[col].apply(parse_datetime_to_timestamp)
            train_df = train_df.fillna(0)
            bg_scaled = self.scaler.transform(train_df.values)

            if len(bg_scaled) > 200:
                rng  = np.random.default_rng(42)
                idxs = rng.choice(len(bg_scaled), 200, replace=False)
                bg_scaled = bg_scaled[idxs]

            print(f"\n  [XAI] Initialising SHAP + LIME explainers "
                  f"({len(bg_scaled)} background samples)...")
            self.xai = XAIExplainer(
                models       = self.models,
                background_X = bg_scaled,
                feature_names = FEATURE_NAMES,
                n_kernel_bg  = 30
            )
            print(f"  [XAI] Ready "
                  f"(SHAP={'ON' if SHAP_AVAILABLE else 'OFF'}, "
                  f"LIME={'ON' if LIME_AVAILABLE else 'OFF'})")
        except Exception as e:
            print(f"  [XAI] Init failed: {e}")
            self.xai = None

    def _get_model_type(self):
        n = self.llm_client.model_name.lower()
        return "QWEN" if "qwen" in n else "GLM" if "glm" in n else "LLAMA" if "llama" in n else "UNKNOWN"

    # ── SYSTEM PROMPT — V24 (doctrines + few-shot + 6-step contract) ───────────
    def _get_system_prompt(self) -> str:
        return """You are a senior security analyst investigating anomalies in an EV charging infrastructure. You classify each session as Normal or Attack by reasoning step by step from evidence. You are rigorous, quantitative, and you never guess.

EVIDENCE YOU RECEIVE (in priority order):
1. Domain knowledge — physical meaning of the measurements and known attack patterns.
2. Historical precedents — how similar past sessions were classified and why.
3. Raw session measurements — the actual data from this session.
4. ML classifier predictions — for EACH of 7 models you get:
   - prediction and RAW probability
   - CALIBRATED confidence = raw_prob x reliability (reliability reflects how
     trustworthy that model is; some models are known to be underfit)
   - SHAP: signed contribution of each feature (+ pushes toward Attack, - toward
     Normal) with its % share of that model's total attribution
   - LIME: local linear weights around this exact sample
   - agreement flag: strong / partial / conflict (does SHAP agree with LIME?)
5. ENSEMBLE EVIDENCE — confidence-weighted vote totals and the cross-model
   SHAP consensus (which single feature drives most models, and in which direction).

DOCTRINE — how to weigh the evidence:
- RELIABILITY DOCTRINE: weight models by CALIBRATED confidence, not raw
  probability. A reliable model at 0.70 outweighs an unreliable model at 0.90.
- AGREEMENT DOCTRINE: a model with agreement=strong (SHAP and LIME concur) is
  trustworthy. A model with agreement=conflict may be keying on spurious
  patterns — down-weight it.
- PHYSICAL-PLAUSIBILITY GATE: a vote is only credible if its top SHAP driver is
  physically meaningful for fraud — i.e. energy requested/delivered — NOT
  connection or disconnect timing. Timing-driven votes are suspect.
- CONVERGENCE: when the cross-model SHAP consensus and the physics of the
  delivery ratio point the SAME way, that is your strongest signal.
- Never treat the vote count alone as the answer.

DELIVERY RATIO PHYSICS:
- ratio = delivered / requested.
- ratio near 1.0 (~0.8-1.1): normal delivery.
- ratio well ABOVE 1.0: possible energy theft (more delivered than authorized).
- ratio well BELOW ~0.7: possible phantom charging (billed but little delivered).

────────────────────────────────────────────────────────────────────────────
WORKED EXAMPLE 1 (Attack — energy theft):
Requested=6.0 kWh, Delivered=11.4 kWh -> ratio=1.90. Cross-model SHAP consensus:
"Energy delivered (kWh)" drives 6/7 models toward Attack; 5 models agreement=strong.
Reasoning: ratio 1.90 is far above 1.0 (physics = theft pattern); the consensus
driver is a PHYSICAL feature (passes the plausibility gate) and SHAP/LIME agree;
weighted vote favors Attack. -> Prediction: Attack, energy_theft, confidence high.

WORKED EXAMPLE 2 (Normal — noisy but benign):
Requested=8.0 kWh, Delivered=7.4 kWh -> ratio=0.93. One low-reliability model
(MLP, calibrated 0.36) votes Attack with its top SHAP driver = "Connection time".
Reasoning: ratio 0.93 is within normal variation; the lone Attack vote is a
LOW-reliability model whose driver is TIMING (fails the plausibility gate); the
physical-feature consensus points Normal. -> Prediction: Normal, none, confidence high.
────────────────────────────────────────────────────────────────────────────

STAGE 1 OUTPUT — you MUST produce all six steps, each citing concrete numbers.
Do NOT give a final verdict in Stage 1.

STEP 1 — RAW PHYSICS:
[Compute delivery_ratio = delivered/requested. State the number. Interpret it
 against the ratio physics above.]

STEP 2 — PER-MODEL CONFIDENCE AUDIT:
[For each model, note whether its CALIBRATED confidence is high or low and
 whether it is a reliable model. Identify which votes deserve weight.]

STEP 3 — SHAP/LIME EVIDENCE:
[State the cross-model SHAP consensus feature, its direction, and how many
 models agree. List which models are agreement=strong vs conflict. Cite % shares.]

STEP 4 — PHYSICS vs XAI CROSS-CHECK:
[Does the consensus SHAP driver match the physical anomaly from STEP 1? Does it
 pass the physical-plausibility gate (energy feature, not timing)?]

STEP 5 — DOMAIN & HISTORY:
[Which attack pattern does this resemble? What did similar historical cases show?]

STEP 6 — CONFLICTS:
[List every piece of evidence pointing the OTHER way and note its strength.]"""

    STAGE2_USER = """Based on your six-step analysis above, produce your final classification using EXACTLY this contract. Reference specific numbers, models, and SHAP/LIME signals.

DECISION_RATIONALE:
[One paragraph naming the SINGLE most decisive piece of evidence and why it won.]

EVIDENCE_FOR_ATTACK:
[Bulleted list; each bullet cites a number, model, or SHAP/LIME signal.]

EVIDENCE_FOR_NORMAL:
[Bulleted list; each bullet cites a number, model, or SHAP/LIME signal.]

HOW_CONFLICTS_RESOLVED:
[How you weighed the opposing evidence, using the reliability/agreement/physical doctrines.]

CONFIDENCE: [high / medium / low]

ATTACK_TYPE: [energy_theft / phantom_charging / session_manipulation / none]

Prediction: [Attack or Normal]"""

    # ── FORMAT ONE MODEL'S XAI BLOCK FOR THE PROMPT ────────────────────────────
    @staticmethod
    def _format_model_block(model_name: str, pred: Dict, xai_result: Dict) -> str:
        prob   = pred.get('prob', pred.get('confidence', 0.0))
        cal    = pred.get('calibrated_conf', prob)
        rel    = pred.get('reliability', 0.6)
        lines  = [f"  {model_name}: {pred['prediction']}  "
                  f"(raw={prob*100:.1f}%, calibrated={cal*100:.1f}%, reliability={rel:.2f}, "
                  f"agreement={xai_result.get('agreement','n/a')})"]

        if xai_result.get('shap_ok') and xai_result['shap_values']:
            lines.append("    SHAP contributions (top 3):")
            for entry in xai_result['shap_values'][:3]:
                fname, val, direction = entry[0], entry[1], entry[2]
                pct = entry[3] if len(entry) > 3 else 0.0
                bar = "#" * min(int(abs(val) * 20), 10)
                sign = "+" if val >= 0 else ""
                lines.append(f"      {fname:<26} {sign}{val:+.3f} ({pct:4.0f}% of decision)  "
                             f"{bar}  ({direction})")
        else:
            lines.append("    SHAP: unavailable")

        if xai_result.get('lime_ok') and xai_result['lime_values']:
            lines.append("    LIME local weights (top 3):")
            for fname, condition, weight, direction in xai_result['lime_values'][:3]:
                sign = "+" if weight >= 0 else ""
                lines.append(f"      {condition:<34} weight={sign}{weight:.3f}  ({direction})")
        else:
            lines.append("    LIME: unavailable")
        return "\n".join(lines)

    # ── STAGE 1 PROMPT ─────────────────────────────────────────────────────────
    def _build_stage1_prompt(self, all_predictions, xai_results, evidence_bundle,
                             requested_kwh, delivered_kwh, duration_hours,
                             knowledge, ltm_cases, data_quality) -> str:
        evidence = ""

        if knowledge:
            evidence += "=" * 60 + "\nDOMAIN KNOWLEDGE\n" + "=" * 60 + "\n"
            evidence += knowledge[:800] + "\n\n"

        if ltm_cases:
            evidence += "=" * 60 + f"\nHISTORICAL PRECEDENTS ({len(ltm_cases)} similar sessions)\n" + "=" * 60 + "\n"
            for i, c in enumerate(ltm_cases[:5], 1):
                evidence += (f"Case {i}: requested={c.get('requested_kwh','?')} kWh, "
                             f"delivered={c.get('delivered_kwh','?')} kWh, "
                             f"duration={c.get('duration_hours','?')} h -> "
                             f"{c.get('prediction','?')} (conf={c.get('confidence',0):.2f})\n")
                if c.get('reasoning'):
                    evidence += f"        reasoning: {c['reasoning']}\n"
            evidence += "\n"

        evidence += "=" * 60 + "\nCURRENT SESSION — RAW MEASUREMENTS\n" + "=" * 60 + "\n"
        evidence += f"  Energy requested:  {requested_kwh:.4f} kWh\n"
        evidence += f"  Energy delivered:  {delivered_kwh:.4f} kWh\n"
        evidence += f"  Session duration:  {duration_hours:.2f} hours\n"
        if data_quality != 'ok':
            evidence += f"  [DATA QUALITY WARNING: {data_quality} — treat suspect fields with caution]\n"
        evidence += "  Compute delivery ratio (delivered / requested) yourself in STEP 1.\n\n"

        evidence += "=" * 60 + "\nML CLASSIFIER PREDICTIONS + SHAP/LIME + CONFIDENCE\n" + "=" * 60 + "\n"
        for mn, pred in all_predictions.items():
            xai_r = xai_results.get(mn, {'shap_ok': False, 'lime_ok': False,
                                          'shap_values': [], 'lime_values': [], 'agreement': 'n/a'})
            evidence += self._format_model_block(mn, pred, xai_r) + "\n\n"

        # Ensemble evidence block
        eb  = evidence_bundle
        con = eb['consensus']
        evidence += "=" * 60 + "\nENSEMBLE EVIDENCE (aggregated — not a verdict)\n" + "=" * 60 + "\n"
        evidence += (f"  Confidence-weighted vote:  Attack={eb['weighted_attack']:.2f}  "
                     f"Normal={eb['weighted_normal']:.2f}\n")
        evidence += f"  Mean calibrated confidence: {eb['mean_calibrated_conf']:.2f}\n"
        if con.get('top_feature'):
            evidence += (f"  Cross-model SHAP consensus: '{con['top_feature']}' drives "
                         f"{con['models_agreeing']}/{con.get('n_models',0)} models toward "
                         f"{con['direction']} "
                         f"({'PHYSICAL feature' if con.get('physical') else 'NON-physical/timing feature'})\n")
        evidence += (f"  Models with strong SHAP/LIME agreement: "
                     f"{', '.join(eb['strong_agreement']) or 'none'}\n")
        evidence += (f"  Models with SHAP/LIME conflict:         "
                     f"{', '.join(eb['conflict_models']) or 'none'}\n")
        evidence += (f"  Physical-feature-driven votes: {eb['physical_drivers']}/{eb['n_with_shap']}\n")

        return f"""Analyze this EV charging session. Produce ONLY your six-step analysis (STEP 1 through STEP 6). Do NOT give a final prediction yet.

{evidence}

Now produce STEP 1 through STEP 6 exactly as specified in your instructions, citing concrete numbers at each step."""

    # ── VALIDATION ─────────────────────────────────────────────────────────────
    def _stage1_complete(self, text: str) -> bool:
        low = text.lower()
        present = sum(1 for h in self.STEP_HEADERS if h.lower() in low)
        return present >= 5   # allow one missing header

    # ── PARSE RESPONSE (V24 — full 6-step reasoning trace) ─────────────────────
    def _parse_llm_response(self, stage2_response, stage1_response,
                            all_predictions, evidence_bundle, delivery_ratio,
                            duration_hours, line_number, votes, ltm_cases,
                            rag_sources, complexity_metrics, difficulty_zone,
                            data_quality):
        predicted_label = None
        llm_confidence  = "medium"
        attack_type     = "none"
        resp_lower      = stage2_response.lower()

        if   "prediction: attack"    in resp_lower: predicted_label = 'Malicious'
        elif "prediction: malicious" in resp_lower: predicted_label = 'Malicious'
        elif "prediction: normal"    in resp_lower: predicted_label = 'Normal'

        conf_match = re.search(r'confidence:\s*(high|medium|low)', resp_lower)
        if conf_match:
            llm_confidence = conf_match.group(1)

        type_match = re.search(
            r'attack_type:\s*(energy_theft|phantom_charging|session_manipulation|none)',
            resp_lower)
        if type_match:
            attack_type = type_match.group(1)

        def _grab(pattern, src):
            m = re.search(pattern, src, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        # Stage 2 contract fields
        decision_rationale = _grab(r'decision_rationale:\s*\n?(.*?)(?=\nevidence_for_attack:|\nevidence_for_normal:|\nconfidence:|\nprediction:|\Z)', stage2_response)
        ev_for_attack      = _grab(r'evidence_for_attack:\s*\n?(.*?)(?=\nevidence_for_normal:|\nhow_conflicts_resolved:|\nconfidence:|\Z)', stage2_response)
        ev_for_normal      = _grab(r'evidence_for_normal:\s*\n?(.*?)(?=\nhow_conflicts_resolved:|\nconfidence:|\nprediction:|\Z)', stage2_response)
        conflicts_resolved = _grab(r'how_conflicts_resolved:\s*\n?(.*?)(?=\nconfidence:|\nattack_type:|\nprediction:|\Z)', stage2_response)

        # Stage 1 six-step trace
        reasoning_trace = {
            'step1_physics':      _grab(r'step\s*1[^\n]*\n(.*?)(?=\nstep\s*2|\Z)', stage1_response),
            'step2_confidence':   _grab(r'step\s*2[^\n]*\n(.*?)(?=\nstep\s*3|\Z)', stage1_response),
            'step3_shap_lime':    _grab(r'step\s*3[^\n]*\n(.*?)(?=\nstep\s*4|\Z)', stage1_response),
            'step4_cross_check':  _grab(r'step\s*4[^\n]*\n(.*?)(?=\nstep\s*5|\Z)', stage1_response),
            'step5_domain_hist':  _grab(r'step\s*5[^\n]*\n(.*?)(?=\nstep\s*6|\Z)', stage1_response),
            'step6_conflicts':    _grab(r'step\s*6[^\n]*\n(.*?)(?=\Z)', stage1_response),
        }
        steps_present = sum(1 for v in reasoning_trace.values() if v)

        # Fallback: ABSTAIN (never silently default to ML majority — V22+ design)
        if predicted_label is None:
            atk = resp_lower.count('attack') + resp_lower.count('malicious') + resp_lower.count('fraud')
            nrm = resp_lower.count('normal') + resp_lower.count('legitimate')
            if atk > nrm + 2:
                predicted_label = 'Malicious'
                print(f"    [WARN] Prediction inferred from tone: Malicious")
            elif nrm > atk + 2:
                predicted_label = 'Normal'
                print(f"    [WARN] Prediction inferred from tone: Normal")
            else:
                predicted_label = 'ABSTAIN'
                llm_confidence  = "low"
                print(f"    [ERROR] LLM response unparseable — recording ABSTAIN")

        if predicted_label == 'Malicious' and attack_type == 'none':
            attack_type = ('energy_theft'    if delivery_ratio > 1.3 else
                           'phantom_charging' if delivery_ratio < 0.7 else
                           'unknown_attack')

        ml_majority     = votes.most_common(1)[0][0] if votes else 'Normal'
        llm_overrode_ml = (predicted_label not in ('ABSTAIN',) and
                           predicted_label != ml_majority)

        llm_conf_score  = {'high': 0.90, 'medium': 0.65, 'low': 0.40}.get(llm_confidence, 0.65)
        ml_agreement    = (sum(1 for p in all_predictions.values()
                               if p['prediction'] == predicted_label)
                           / len(all_predictions)) if predicted_label != 'ABSTAIN' else 0.0
        reasoning_depth = min(1.0, (len(stage1_response) + len(stage2_response)) / 1800)
        # reasoning completeness now factors into the composite score
        completeness    = steps_present / 6.0
        composite       = (0.50 * llm_conf_score + 0.20 * ml_agreement +
                           0.15 * reasoning_depth + 0.15 * completeness)

        return {
            "line_number":          line_number,
            "predicted_label":      predicted_label,
            "attack_type":          attack_type,
            "confidence":           round(float(composite), 3),
            "llm_confidence":       llm_confidence,
            "ml_agreement":         round(float(ml_agreement), 3),
            "ml_majority":          ml_majority,
            "llm_overrode_ml":      llm_overrode_ml,
            "difficulty_zone":      difficulty_zone,
            "data_quality":         data_quality,
            # headline reasoning (back-compat with V23 runner keys)
            "llm_reasoning":        (decision_rationale[:500] if decision_rationale
                                     else stage2_response[:500]),
            "llm_analysis":         (reasoning_trace['step1_physics'][:500]
                                     if reasoning_trace['step1_physics'] else stage1_response[:300]),
            "llm_xai_assessment":   reasoning_trace['step3_shap_lime'][:400],
            "llm_conflicts":        (conflicts_resolved[:300] or reasoning_trace['step6_conflicts'][:300]),
            # full auditable trace (V24)
            "reasoning_trace":      {k: v[:600] for k, v in reasoning_trace.items()},
            "decision_rationale":   decision_rationale[:600],
            "evidence_for_attack":  ev_for_attack[:500],
            "evidence_for_normal":  ev_for_normal[:500],
            "steps_present":        steps_present,
            # evidence bundle snapshot
            "weighted_vote":        {'attack': evidence_bundle['weighted_attack'],
                                     'normal': evidence_bundle['weighted_normal']},
            "shap_consensus":       evidence_bundle['consensus'],
            "strong_agreement_models": evidence_bundle['strong_agreement'],
            "conflict_models":      evidence_bundle['conflict_models'],
            "delivery_ratio":       round(delivery_ratio, 3),
            "duration_hours":       round(duration_hours, 2),
            "knowledge_used":       self.use_knowledge,
            "memory_used":          self.use_memory,
            "xai_used":             self.xai is not None,
            "shap_available":       SHAP_AVAILABLE,
            "lime_available":       LIME_AVAILABLE,
            "ltm_cases_referenced": len(ltm_cases),
            "rag_sources_used":     len(rag_sources),
            "classifier_votes":     dict(votes),
            "aggregation_method":   ("V24_RAG_LTM_XAI" if (self.use_knowledge and self.use_memory)
                                     else "V24_RAG_XAI"  if self.use_knowledge
                                     else "V24_LTM_XAI"  if self.use_memory
                                     else "V24_ML_XAI"),
            "complexity":           complexity_metrics
        }

    # ── MAIN DETECT ────────────────────────────────────────────────────────────
    def detect(self, line_number):
        t0         = time.time()
        model_type = self._get_model_type()
        if   self.use_knowledge and self.use_memory: sc = "A (ML+RAG+LTM+XAI->LLM)"
        elif self.use_knowledge:                     sc = "B (ML+RAG+XAI->LLM)"
        elif self.use_memory:                        sc = "C (ML+LTM+XAI->LLM)"
        else:                                        sc = "D (ML+XAI->LLM baseline)"
        print(f"\n[MODEL] {self.llm_client.model_name} -> {model_type}")
        print(f"[SCENARIO] {sc}")
        self.short_term_memory = []
        print(f"\n{'='*100}\nANALYZING SESSION {line_number}\n{'='*100}\n")
        timestamp = datetime.now().timestamp()

        # STEP 1: DATA EXTRACTION
        print(f"[Step 1] Data Extraction")
        row           = self.df.loc[line_number]
        requested_kwh = float(row[self.column_mapping['RequestedDemand']])
        delivered_kwh = float(row[self.column_mapping['kWhDelivered']])
        data_dict     = {
            "line_number":     line_number,
            "connectionTime":  str(row[self.column_mapping['connectionTime']]),
            "disconnectTime":  str(row[self.column_mapping['disconnectTime']]),
            "RequestedDemand": requested_kwh,
            "kWhDelivered":    delivered_kwh,
            "label":           int(row[self.column_mapping['label']])
        }
        print(f"  Requested={requested_kwh:.4f}kWh, Delivered={delivered_kwh:.4f}kWh")

        # STEP 2: PREPROCESSING (+ data-quality guards)
        print(f"\n[Step 2] Preprocessing")
        df_temp = pd.DataFrame([data_dict])
        df_temp['connectionTime'] = df_temp['connectionTime'].apply(parse_datetime_to_timestamp)
        df_temp['disconnectTime'] = df_temp['disconnectTime'].apply(parse_datetime_to_timestamp)
        X              = df_temp[['connectionTime', 'disconnectTime',
                                   'RequestedDemand', 'kWhDelivered']].values
        X_scaled       = self.scaler.transform(X)
        conn_ts        = float(df_temp['connectionTime'].iloc[0])
        disc_ts        = float(df_temp['disconnectTime'].iloc[0])
        duration_hours = (disc_ts - conn_ts) / 3600.0
        delivery_ratio = delivered_kwh / requested_kwh if requested_kwh > 0 else 0
        difficulty_zone = classify_difficulty_zone(delivery_ratio)

        data_quality = 'ok'
        if requested_kwh <= 0:
            data_quality = 'requested<=0 (ratio undefined)'
        elif conn_ts == 0.0 or disc_ts == 0.0:
            data_quality = 'timestamp parse failed'
        elif duration_hours < 0:
            data_quality = f'negative duration ({duration_hours:.2f}h)'
        print(f"  Delivery Ratio (internal): {delivery_ratio:.4f} | Zone: {difficulty_zone} "
              f"| Data quality: {data_quality}")

        # STEP 3: ML CLASSIFIERS + CONFIDENCE + SHAP/LIME
        print(f"\n[Step 3] ML Classifiers + Confidence + SHAP/LIME")
        all_predictions = {}
        xai_results     = {}
        ML_ORDER = ['Random Forest', 'K-Nearest Neighbors', 'Logistic Regression',
                    'MLP', 'Support Vector Classifier', 'Decision Tree', 'Gradient Boosting']

        for mn in ML_ORDER:
            model = self.models[mn]
            conf_info = compute_model_confidence(mn, model, X_scaled)
            all_predictions[mn] = conf_info
            pidx = conf_info['label_idx']

            if self.xai is not None:
                print(f"  {mn}: {conf_info['prediction']} "
                      f"(raw={conf_info['prob']:.3f}, cal={conf_info['calibrated_conf']:.3f}) "
                      f"— computing XAI...", end='', flush=True)
                t_xai = time.time()
                xai_r = self.xai.explain_model(mn, X_scaled, pidx)
                xai_results[mn] = xai_r
                xai_time = time.time() - t_xai
                shap_top = xai_r['shap_values'][0][0] if xai_r.get('shap_ok') and xai_r['shap_values'] else "—"
                lime_top = xai_r['lime_values'][0][0] if xai_r.get('lime_ok') and xai_r['lime_values'] else "—"
                print(f" done ({xai_time:.1f}s) | agree={xai_r.get('agreement')} "
                      f"| SHAP top={shap_top} | LIME top={lime_top}")
            else:
                xai_results[mn] = {'shap_ok': False, 'lime_ok': False,
                                   'shap_values': [], 'lime_values': [], 'agreement': 'n/a'}
                print(f"  {mn}: {conf_info['prediction']} "
                      f"(cal={conf_info['calibrated_conf']:.3f}) [XAI unavailable]")

        votes = Counter([p['prediction'] for p in all_predictions.values()])
        print(f"  CONSENSUS: {votes.get('Malicious',0)} Attack, {votes.get('Normal',0)} Normal")

        # Build ensemble evidence bundle
        evidence_bundle = build_evidence_bundle(all_predictions, xai_results)
        con = evidence_bundle['consensus']
        print(f"  Weighted vote: Attack={evidence_bundle['weighted_attack']:.2f}, "
              f"Normal={evidence_bundle['weighted_normal']:.2f}")
        if con.get('top_feature'):
            print(f"  SHAP consensus: '{con['top_feature']}' -> {con['direction']} "
                  f"({con['models_agreeing']}/{con.get('n_models',0)} models, "
                  f"{'physical' if con.get('physical') else 'timing'})")

        # STEP 4: LTM
        ltm_cases = []
        if self.use_memory and self.ltm:
            print(f"\n[Step 4] Long-Term Memory Retrieval")
            obs = (f"requested={requested_kwh:.3f}kWh, delivered={delivered_kwh:.3f}kWh, "
                   f"duration={duration_hours:.2f}h")
            ltm_cases = self.ltm.retrieve_similar(obs, timestamp, top_k=5)
            print(f"  Found {len(ltm_cases)} similar cases")
        else:
            print(f"\n[Step 4] LTM DISABLED")

        # STEP 5: RAG
        knowledge_text, knowledge_sources = "", []
        if self.use_knowledge and self.kb:
            print(f"\n[Step 5] RAG Knowledge Base Query")
            query = ("energy theft"    if delivery_ratio > 1.3 else
                     "phantom charging" if delivery_ratio < 0.7 else
                     "normal charging")
            knowledge_text, knowledge_sources = self.kb.query(query, top_k=2)
            print(f"  Query: '{query}' | {len(knowledge_text)} chars retrieved")
        else:
            print(f"\n[Step 5] RAG DISABLED")

        # STEP 6: TWO-STAGE LLM
        print(f"\n[Step 6] LLM Stage 1 — 6-step Analysis (SHAP/LIME + confidence)")
        sys_prompt    = self._get_system_prompt()
        stage1_prompt = self._build_stage1_prompt(
            all_predictions, xai_results, evidence_bundle,
            requested_kwh, delivered_kwh, duration_hours,
            knowledge_text, ltm_cases, data_quality)

        p1_chars = len(sys_prompt) + len(stage1_prompt)
        p1_words = len(sys_prompt.split()) + len(stage1_prompt.split())
        print(f"  Stage 1 prompt: {p1_chars} chars ({p1_words} words)")
        print(f"\n  >>> STAGE 1 PROMPT <<<\n  {'-'*80}")
        for line in stage1_prompt.split('\n'):
            print(f"  > {line}")
        print(f"  {'-'*80}")

        t1 = time.time()
        stage1_response = self.llm_client.generate_with_system(
            system_prompt=sys_prompt, user_prompt=stage1_prompt,
            temperature=0.0, max_tokens=1024)
        # Validate completeness; retry once if the model skipped steps.
        if not self._stage1_complete(stage1_response):
            print(f"  [RETRY] Stage 1 incomplete (missing steps) — retrying once...")
            retry_prompt = stage1_prompt + ("\n\nIMPORTANT: your previous answer was "
                            "incomplete. Output ALL SIX steps STEP 1..STEP 6, each on its "
                            "own line, each citing concrete numbers.")
            stage1_response = self.llm_client.generate_with_system(
                system_prompt=sys_prompt, user_prompt=retry_prompt,
                temperature=0.0, max_tokens=1024)
        latency1 = time.time() - t1
        print(f"\n  >>> STAGE 1 RESPONSE ({latency1:.2f}s) <<<\n  {'-'*80}")
        for line in stage1_response.split('\n'):
            print(f"  | {line}")
        print(f"  {'-'*80}")

        print(f"\n[Step 6b] LLM Stage 2 — Final Decision (contract)")
        t2 = time.time()
        stage2_response = self.llm_client.generate_two_turn(
            system_prompt   = sys_prompt,
            first_user      = stage1_prompt,
            first_assistant = stage1_response,
            second_user     = self.STAGE2_USER,
            temperature     = 0.0, max_tokens=1024)
        latency2  = time.time() - t2
        total_lat = latency1 + latency2
        r2_words  = len(stage2_response.split())
        wps       = round(r2_words / total_lat, 1) if total_lat > 0 else 0

        print(f"\n  >>> STAGE 2 RESPONSE ({latency2:.2f}s) <<<\n  {'-'*80}")
        for line in stage2_response.split('\n'):
            print(f"  | {line}")
        print(f"  {'-'*80}")

        complexity = {
            "prompt_chars":     p1_chars + len(self.STAGE2_USER),
            "prompt_words":     p1_words + len(self.STAGE2_USER.split()),
            "response_chars":   len(stage1_response) + len(stage2_response),
            "response_words":   len(stage1_response.split()) + r2_words,
            "llm_latency_sec":  round(total_lat, 2),
            "stage1_latency":   round(latency1, 2),
            "stage2_latency":   round(latency2, 2),
            "tokens_per_sec":   wps,
            "xai_models_explained": sum(1 for r in xai_results.values()
                                        if r.get('shap_ok') or r.get('lime_ok'))
        }

        # STEP 7: PARSE & FINAL DECISION
        print(f"\n[Step 7] Final Decision")
        final_result = self._parse_llm_response(
            stage2_response, stage1_response, all_predictions, evidence_bundle,
            delivery_ratio, duration_hours, line_number, votes,
            ltm_cases, knowledge_sources, complexity, difficulty_zone, data_quality)

        session_time = time.time() - t0
        final_result['complexity']['session_total_sec'] = round(session_time, 2)

        print(f"  CLASSIFICATION:  {final_result['predicted_label']}")
        print(f"  Confidence:      {final_result['confidence']:.3f} "
              f"(LLM={final_result['llm_confidence']}, ML_agree={final_result['ml_agreement']:.3f}, "
              f"steps={final_result['steps_present']}/6)")
        print(f"  Attack Type:     {final_result['attack_type']}")
        print(f"  LLM overrode ML: {'YES' if final_result['llm_overrode_ml'] else 'NO'} "
              f"(ML majority={final_result['ml_majority']})")
        print(f"  Difficulty Zone: {difficulty_zone}")
        print(f"  Time: {session_time:.2f}s total ({total_lat:.2f}s LLM, "
              f"{complexity['xai_models_explained']}/7 models explained by XAI)")
        if final_result.get('decision_rationale'):
            print(f"\n  DECISION RATIONALE: {final_result['decision_rationale'][:200]}...")
        if final_result.get('llm_xai_assessment'):
            print(f"  SHAP/LIME (STEP 3): {final_result['llm_xai_assessment'][:180]}...")

        print(f"\n{'='*100}\nSESSION {line_number} COMPLETE\n{'='*100}\n")

        return {
            'status':            'success',
            'request_index':     line_number,
            'result':            final_result,
            'model_predictions': all_predictions,
            'evidence_bundle':   evidence_bundle,
            'xai_results':       {k: {'shap_ok': v['shap_ok'], 'lime_ok': v['lime_ok'],
                                      'agreement': v.get('agreement', 'n/a'),
                                      'shap_top': v['shap_values'][0][0] if v.get('shap_values') else None,
                                      'lime_top': v['lime_values'][0][0] if v.get('lime_values') else None}
                                  for k, v in xai_results.items()},
            'raw_data':          data_dict,
            'short_term_memory': self.short_term_memory,
            'ltm_cases_used':    len(ltm_cases),
            'rag_sources_used':  len(knowledge_sources),
            'majority_vote':     votes.most_common(1)[0][0] if votes else 'Normal'
        }

    def store_correct_decision_in_ltm(self, line_number, was_correct, detection_result):
        if was_correct and self.use_memory and self.ltm:
            req = float(self.df.loc[line_number, self.column_mapping['RequestedDemand']])
            dlv = float(self.df.loc[line_number, self.column_mapping['kWhDelivered']])
            self.ltm.add_memory({
                "timestamp":      datetime.now().timestamp(),
                "line_number":    line_number,
                "observations":   (f"requested={req:.3f}kWh, delivered={dlv:.3f}kWh, "
                                   f"duration={detection_result['duration_hours']:.2f}h"),
                "reasoning":      detection_result.get('llm_reasoning', '')[:300],
                "prediction":     detection_result['predicted_label'],
                "confidence":     detection_result['confidence'],
                "delivery_ratio": detection_result['delivery_ratio'],
                "requested_kwh":  round(req, 3),
                "delivered_kwh":  round(dlv, 3),
                "duration_hours": detection_result['duration_hours'],
                "attack_type":    detection_result['attack_type']
            })

    def seed_ltm_from_training(self, train_indices, n_seed=50):
        if self.ltm is not None:
            self.ltm.seed_from_training(self.df, self.column_mapping,
                                        train_indices, n_seed=n_seed)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING — IDENTICAL TO V23 (intentionally unchanged per requirement)
# ══════════════════════════════════════════════════════════════════════════════
def auto_train_models_v6(data_path, models_dir, train_ratio=0.4, random_state=42):
    print(f"\nTraining models ({train_ratio*100:.0f}% train)...")
    df = pd.read_csv(data_path)
    cm = get_column_mapping(df)
    train_df, test_df = train_test_split(df, train_size=train_ratio,
                                          random_state=random_state,
                                          stratify=df[cm['label']], shuffle=True)
    train_idx = sorted(train_df.index.tolist())
    test_idx  = sorted(test_df.index.tolist())
    actual_cols = [cm[c] for c in ['connectionTime', 'disconnectTime',
                                    'RequestedDemand', 'kWhDelivered']]
    X_train = train_df[actual_cols].copy()
    y_train = train_df[cm['label']].copy()
    X_test  = test_df[actual_cols].copy()
    y_test  = test_df[cm['label']].copy()
    X_train.columns = X_test.columns = FEATURE_NAMES
    for col in ['connectionTime', 'disconnectTime']:
        X_train[col] = X_train[col].apply(parse_datetime_to_timestamp)
        X_test[col]  = X_test[col].apply(parse_datetime_to_timestamp)
    X_train, X_test = X_train.fillna(0), X_test.fillna(0)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'ev_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    split_info = {'train_indices': train_idx, 'test_indices': test_idx,
                  'train_ratio': train_ratio, 'random_state': random_state}
    with open(os.path.join(models_dir, 'split_info_v6.pkl'), 'wb') as f:
        pickle.dump(split_info, f)
    classifiers = {
        'Random Forest':            ('ev_rf_model.pkl',  RandomForestClassifier(n_estimators=10, max_depth=1, min_samples_split=10, random_state=random_state, class_weight='balanced')),
        'Logistic Regression':      ('ev_lr_model.pkl',  LogisticRegression(max_iter=1, C=0.09, random_state=random_state, class_weight='balanced')),
        'K-Nearest Neighbors':      ('ev_knn_model.pkl', KNeighborsClassifier(n_neighbors=100, weights='distance')),
        'Decision Tree':            ('ev_dt_model.pkl',  DecisionTreeClassifier(max_depth=1, min_samples_split=2, random_state=random_state, class_weight='balanced')),
        'MLP':                      ('ev_mlp_model.pkl', MLPClassifier(hidden_layer_sizes=(4, 8), max_iter=1, alpha=0.9, random_state=random_state)),
        'Support Vector Classifier':('ev_svc_model.pkl', SVC(C=0.005, probability=False, random_state=random_state, class_weight='balanced')),
        'Gradient Boosting':        ('ev_gb_model.pkl',  GradientBoostingClassifier(n_estimators=5, max_depth=1, learning_rate=0.009, random_state=random_state))
    }
    print(f"  Features: {FEATURE_NAMES} (4 raw features, NO delivery_ratio)")
    for name, (fn, model) in classifiers.items():
        model.fit(X_train_s, y_train)
        preds   = model.predict(X_test_s)
        acc     = accuracy_score(y_test, preds)
        classes = set(preds)
        status  = "OK" if len(classes) >= 2 else "WARN: single-class"
        print(f"  {name}: {acc:.4f} [{status}] predicts={classes}")
        with open(os.path.join(models_dir, fn), 'wb') as f:
            pickle.dump(model, f)
    print(f"Models saved.")
    return train_idx, test_idx
