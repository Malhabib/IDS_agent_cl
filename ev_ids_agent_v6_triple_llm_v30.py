# ev_ids_agent_v6_triple_llm_v30.py
"""
EV-IDS-Agent VERSION 6 - TRIPLE LLM V30

HEADLINE CHANGE IN V30 — GLM, AND THE END OF UNMEASURABLE RUNS
  GLM's poor and unstable scores were not a prompt problem, a parsing problem
  or a limitation of the model. They were two mechanical faults:

  1. STAGE 2 WAS RUNNING ON A 128-TOKEN BUDGET WITH THINKING ON.
     Ollama grants num_ctx as a REQUEST. Under VRAM pressure it silently
     granted 4096 against the requested 8192. Stage 2 replays the system
     prompt, the ~10,900-character Stage 1 prompt, the Stage 1 response and
     the Stage 2 question, so within a 4096 window the budget clamp collapsed
     to its 128-token floor. A thinking model handed 128 tokens spends every
     one of them on its reasoning channel and returns an EMPTY answer, so no
     verdict line was ever written. Raising GLM's num_predict could not help,
     because the request was never the binding constraint -- and it did make
     things worse, by pushing generation further past the window.

     V30 reads the context the server actually granted, keeps Stage 2 on the
     ANSWER channel (Stage 1 is where the reasoning belongs), and disables the
     reasoning channel outright below THINK_MIN_BUDGET.

  2. A TIMED-OUT CALL WAS SCORED AS A MODEL DECISION.
     The Ollama call had no timeout at all. A failure returned the string
     "Error: ...", no verdict could be parsed from it, and the session quietly
     became an ML-majority fallback that was then counted as the model's own
     answer. This is why two runs of identical code, on identical data, at
     temperature 0, disagreed (GLM 0.46 then 0.66; Llama 0.98 then 0.96):
     they differed only in how many calls happened to time out.

     V30 gives every call a deadline, records a missed deadline as a distinct
     'llm_error' provenance, excludes those sessions from behavioural metrics,
     and prints a NOT PUBLISHABLE banner when any are present.

  Supporting these, V30 adds three gates that run BEFORE a study, so a broken
  environment costs minutes instead of two days:
     selftest_v30.py         seconds  no GPU   decision-path assertions
     integration_test_v30.py ~1 min   no GPU   full pipeline on synthetic data
     healthcheck_v30.py      minutes  GPU      per-model cost and verdict rate
  and a circuit breaker inside the run itself, which stops after five sessions
  if calls are failing or the projected wall-clock is implausible.

CARRIED FORWARD FROM V29 — SHAP/LIME DIRECTION
  Corrected the SHAP/LIME direction convention. Attributions were taken with
  respect to the PREDICTED class while the prompt labelled every positive
  value "toward Attack". For binary classification the two classes' Shapley
  values are mirror images, so every model that predicted Normal had its
  direction inverted -- the prompt asserted "toward Attack" for evidence that
  actually pointed toward Normal. Since MLP, SVC and Gradient Boosting predict
  Normal on every session and KNN on most, the majority of SHAP blocks in each
  prompt were sign-flipped toward Attack. This is the direct cause of the
  false-alarm rate of 0.42-0.45 and of sessions with delivery ratio 1.000
  being classified Malicious by all three LLMs. Attributions are now taken
  against the Attack class, so the stated convention "+ = toward Attack" is
  literally true, and LIME uses the same basis. On true-normal sessions the
  share of explanations pointing toward Attack falls from 54% to 11%.
  The bug dates from V23 and affected every run up to and including V28.

Built on V23, whose pipeline, prompts, 7-step detection flow, scenarios and
step-by-step explainability output are UNCHANGED. Everything below is a fix
to how the framework talks to the models and measures the outcome — never a
change to the idea or the objective.

FIX HISTORY FOLDED INTO THIS VERSION
  Generation semantics (V26)
    V23 accepted temperature/max_tokens but never forwarded them, so it ran
    unbounded at Ollama's default temperature — and that is how its 95-98%
    was produced. V25 sent num_predict + temperature=0, which truncated the
    chain-of-thought (for thinking models num_predict also consumes thinking
    tokens) and flattened the reasoning. V30 keeps V23 semantics and uses a
    per-model policy purely as a runaway-loop guard.

  Verdict handling (V26)
    V23 matched only the bare substring "prediction: attack|normal", so
    "**Prediction:** Attack", "Prediction : Attack", "### Prediction:",
    "Final Prediction:", "Classification:" and a bare trailing "Attack" were
    all discarded as ABSTAIN — this is why GLM lost 24/50 sessions. V30
    extracts robustly, prefers the text after </think>, takes the last verdict
    when a model restates one, repairs a missing verdict with one short call,
    and otherwise falls back to the ML majority so every session is scored.

  Runaway generation and VRAM (V27 + this version)
    GLM averaged 40,053 words/session, exceeding num_ctx so Ollama evicted the
    system prompt and the SHAP/LIME evidence mid-generation. Per-model caps fix
    that. keep_alive="30m" then pinned all three models in VRAM at once and
    forced CPU offload (throughput fell ~8x); it is back to 5m with an explicit
    unload between models.

  Measurement correctness (this version)
    - detect() no longer hard-codes temperature=0.0, which had silently made
      every temperature-sweep cell identical.
    - The verdict-repair call gets a real token budget on thinking models
      (24 tokens could never produce an answer).
    - The shared XAI cache carries a fingerprint of the trained models and is
      discarded when they are retrained, so stale SHAP/LIME can never be fed
      into the prompt.

REQUIRES (same folder): simple_run_v6_triple_llm_v28.py, llm_eval_metrics_v30.py
    pip install shap lime ollama requests scikit-learn pandas numpy
"""

import json, os, time, re, pickle, warnings, threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
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

# ── GENERATION SEMANTICS ──────────────────────────────────────────────────────
# V23 accepted temperature/max_tokens kwargs but NEVER forwarded them to
# Ollama. So V23 actually ran UNBOUNDED at Ollama's DEFAULT temperature.
# Its 95-98% accuracy was produced under those settings.
#
# V25 "fixed" this by sending num_predict + temperature=0. That truncated the
# long chain-of-thought (for thinking models num_predict also consumes the
# thinking tokens) and flattened the reasoning => accuracy collapsed.
#
# V26 therefore RESTORES V23 semantics by default: no output cap, no forced
# temperature. Speed comes only from quality-neutral wins (XAI compute-once
# cache + keep_alive), never from throttling the model.
STAGE1_MAX_TOKENS = None      # None => unbounded, exactly like V23
STAGE2_MAX_TOKENS = None
# Measured on the identical 50-session stratified sample:
#     T=0.0            -> Llama 0.72  Qwen 0.78  GLM 0.68
#     Ollama default   -> Llama 0.58  Qwen 0.70  GLM 0.60
# Greedy decoding is materially better for this task, so it is the default.
# The temperature sweep overrides this per cell.
LLM_TEMPERATURE   = 0.0

# ── CONTEXT WINDOW ───────────────────────────────────────────────────────────
# The KV cache scales with num_ctx, so this is the one knob that changes how
# much VRAM a model needs WITHOUT changing which model is used. That matters:
# the study's three target models are fixed by the research question, and
# swapping one to fit a GPU would change what is being measured. Lowering
# num_ctx is the correct lever; substituting a model is not.
#
# Override for a tight card:   set EV_IDS_NUM_CTX=6144     (Windows)
#                              export EV_IDS_NUM_CTX=6144  (Linux/macOS)
#
# There is a floor. Stage 2 replays the system prompt, the Stage 1 prompt, the
# whole Stage 1 response and the Stage 2 question, then still needs room to
# answer. Below MIN_VIABLE_NUM_CTX the budget clamp collapses to its floor and
# the model returns reasoning with no verdict -- the original GLM failure.
# min_viable_num_ctx() computes the real floor from the real prompt sizes.
def _env_num(name, default, cast=int):
    """
    Read a numeric override, tolerating the ways shells leave one empty.

    'set EV_IDS_TIMEOUT=' on Windows leaves the variable defined and empty, and
    int('') raises at import time -- the module would fail to load before any
    diagnostic could run. A bad value falls back to the default with a warning
    rather than taking the whole framework down.
    """
    raw = (os.environ.get(name) or '').strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"[WARN] {name}={raw!r} is not a number — using {default}")
        return default


NUM_CTX = _env_num('EV_IDS_NUM_CTX', 8192)


def min_viable_num_ctx(stage1_prompt_chars: int = 8600,
                       system_chars: int = 3928,
                       stage2_question_chars: int = 523,
                       stage1_response_tokens: int = 2048,
                       answer_tokens: int = 512) -> int:
    """
    Smallest context window in which Stage 2 can still write a verdict.

    Stage 2 is the binding case, not Stage 1: it carries the system prompt, the
    Stage 1 prompt, the Stage 1 response and the question, and only what is left
    can hold the answer. Rounded up to the next multiple of 1024, since Ollama
    allocates the cache in blocks.
    """
    prompt_tokens = (system_chars + stage1_prompt_chars
                     + stage2_question_chars) // 4
    need = prompt_tokens + stage1_response_tokens + answer_tokens + 128
    return int(-(-need // 1024) * 1024)
# keep_alive: how long Ollama keeps a model resident after a request.
# V27 used "30m", which was a serious mistake for a THREE-model study: the
# runner works through Llama -> GLM -> Qwen, so with a 30-minute hold all
# three models stay pinned in VRAM simultaneously. Once VRAM is exhausted
# Ollama offloads layers to CPU, and per-token throughput collapses:
#     V26 (short hold): Llama 2.22 w/s, Qwen 0.57 w/s, GLM 1.86 w/s
#     V27 (30m hold):   Llama 0.29 w/s, Qwen 0.07 w/s, GLM 0.34 w/s   ~8x slower
# A short hold keeps the CURRENT model warm between its own sessions while
# letting the previous model be evicted before the next one loads.
KEEP_ALIVE        = "5m"
REPAIR_MAX_TOKENS = 24        # tiny follow-up that asks only for the verdict

# ── V30: HARD CALL DEADLINE ──────────────────────────────────────────────────
# Up to V29.2 the Ollama call had NO timeout. A model that had been placed
# partly on the CPU took roughly eleven hours for a single session, which is
# how a 50-session run became a two-day run, and a call that never returned
# blocked the whole study.
#
# Worse than the delay was the silence. A failed call returned the string
# "Error: ..." from generate_with_system(); the caller could not extract a
# verdict from it, so the session quietly fell back to the ML majority and was
# then SCORED as though the LLM had answered. Two runs of identical code on
# identical data at temperature 0 therefore disagreed (GLM 0.46 then 0.66),
# because they differed in how many calls happened to time out.
#
# V30 gives every call a deadline and, when it is missed, records the session
# as an LLM ERROR that is reported separately and never silently scored as a
# model decision. See LLM_CALL_TIMEOUT_SEC and the llm_error verdict source.
# Override for a deliberately slow configuration:
#   set EV_IDS_TIMEOUT=1200        (Windows)     0 disables the deadline
#
# Raising it is legitimate. Every version up to V29.2 had NO deadline at all,
# which is why those runs completed on hardware that could not hold the model:
# they simply took as long as they took, and that is where the multi-day
# runtimes came from. A longer deadline reproduces that behaviour knowingly.
# What it does not do is make the results faster to obtain or easier to defend:
# a model at partial residency is slow, and the per-call time varies with
# whatever else is using the GPU.
LLM_CALL_TIMEOUT_SEC = _env_num('EV_IDS_TIMEOUT', 300)
if LLM_CALL_TIMEOUT_SEC <= 0:
    LLM_CALL_TIMEOUT_SEC = None   # no deadline, exactly like V23 through V29.2
LLM_ERROR_PREFIX     = "Error:"

# A thinking model bills its reasoning channel against the SAME num_predict as
# its answer. Below this many tokens it reliably emits reasoning and no answer.
# Measured for GLM: Stage 2 was running on the 128-token floor with thinking on,
# so the answer channel came back empty on every single session.
THINK_MIN_BUDGET = 512

# Stage 2 is a commitment step, not a reasoning step. See generate_two_turn().
STAGE2_ANSWER_CHANNEL_ONLY = True

# Index of the Attack class in the label encoding. SHAP attributions are read
# against THIS class for every model, so that the prompt's "+ = toward Attack"
# legend is true regardless of what the model predicted. Module level so the
# self-test can assert the convention rather than trusting a comment.
ATTACK_CLASS = 1

# ── PER-MODEL GENERATION POLICY (V27) ────────────────────────────────────────
# V26 measurement, 50 sessions, Scenario A:
#     Llama  ~   397 words/session  (~  530 tokens)
#     Qwen   ~   767 words/session  (~1,030 tokens)
#     GLM    ~40,053 words/session  (~53,000 tokens)  <-- runaway loop
#
# GLM's generation exceeded num_ctx (8,192), so Ollama evicted the OLDEST
# tokens mid-generation: the system prompt and the SHAP/LIME evidence. GLM
# then lost both its output format and its grounding, which explains its 12
# unparseable sessions, its 9 false positives on ratio~1.000 sessions, and
# 94 minutes/session (78.6 h total, 92.6% of the whole run).
#
# The V25 mistake was a GLOBAL 900-token cap, which clipped Qwen's ~1,030
# tokens and destroyed its reasoning. The correct design is a cap generous
# enough to be invisible to healthy models but tight enough to stop a loop:
# at 2,048 tokens Llama and Qwen are untouched, GLM cannot run away.
DEFAULT_GEN_POLICY = {
    'num_predict':    2048,   # ~1,500 words: 4x Qwen's typical output
    'repeat_penalty': 1.15,   # breaks degenerate repetition loops
    'thinking':       False,
}
MODEL_GEN_POLICY = {
    # Substring matched against the model name (lowercased), FIRST MATCH WINS,
    # so more specific keys must come before more general ones.
    #
    # qwen2.5 is not a reasoning model and rejects think=True, the same as glm4.
    # Declaring it removes a refused request and a retry on every single call.
    # Must precede the generic 'qwen' key.
    'qwen2.5': {'num_predict': 3072, 'repeat_penalty': 1.10, 'thinking': False},
    'qwen': {'num_predict': 3072, 'repeat_penalty': 1.10, 'thinking': True},
    # glm4 is not a reasoning model and rejects think=True. Measured: the client
    # asked for the reasoning channel, the server refused, and the call was
    # retried without it -- correct behaviour, but a wasted round trip on every
    # single call. Declaring it here removes the round trip. Must precede the
    # generic 'glm' key.
    # 3072, not 2048. Measured on the healthcheck at num_ctx 8192: the Stage 2
    # prompt leaves roughly 4,780 tokens of room and NO budget clamp fires, yet
    # glm4 needed the repair call on 5 of 6 probes -- it was running out of
    # output before reaching the Prediction line, not out of window. This is the
    # opposite situation to V29.1, where raising the budget hurt because the
    # window was 4096 and generation overflowed it. Here the headroom is
    # measured, so the increase is safe; the healthcheck's direct-vs-repair
    # counter confirms or refutes it in about ten minutes.
    'glm4': {'num_predict': 3072, 'repeat_penalty': 1.15, 'thinking': False},
    # GLM measured at ~2,753 words/session against a 2,048-token cap
    # (~1,500 words/stage): it was being truncated BEFORE emitting the verdict
    # line, which is why format compliance sat at 0.26 with 28/50 fallbacks.
    # It needs room to think AND answer.
    # GLM consumed all 4096 tokens on the reasoning channel and returned an
    # empty answer, measured at ~5,000 words of reasoning per session. It needs
    # headroom to finish reasoning and still write the verdict.
    #
    # V30 NOTE — this number was never the binding constraint, so raising it in
    # V29.1 could not have helped and in fact hurt. The real limit was the
    # context window Ollama granted: at 4096 the Stage 2 prompt (system + Stage 1
    # prompt + Stage 1 response + the Stage 2 question) left the 128-token floor,
    # and a thinking model handed 128 tokens spends every one of them on its
    # reasoning channel and returns an empty answer. The fixes that matter are
    # a model that fits in VRAM (so 8192 is actually granted) and Stage 2 running
    # on the answer channel only. 1536 is retained because it is a proven
    # runaway-loop guard, not because it limits anything in a healthy run.
    #
    # repeat_penalty lowered 1.20 -> 1.05. GLM produced token-level corruption
    # of words it was quoting back from the prompt:
    #     "RequestedD emand"   "kWhDeli vered"   "Requeste dDemand"
    #     "Delivrerd Energy"   "Requestedmemand"
    # and then reasoned about the corruption it had itself produced ("It seems
    # to be split across lines", "I am seeing a pattern of hallucinations in my
    # thought trace"). That is the signature of an aggressive repetition
    # penalty, not of a confused model: GLM's reasoning channel quotes the
    # evidence block repeatedly, every repeated token is penalised, and the
    # sampler is pushed off the correct spelling onto a near-miss. 1.20 was the
    # highest penalty of any model here and was applied to the model that
    # quotes the most. The value was originally raised to break a runaway
    # generation loop, but num_predict already bounds that.
    'glm':  {'num_predict': 1536, 'repeat_penalty': 1.05, 'thinking': True},
    'llama': {'num_predict': 2048, 'repeat_penalty': 1.15, 'thinking': False},
}


def gen_policy_for(model_name: str) -> dict:
    n = (model_name or "").lower()
    for key, pol in MODEL_GEN_POLICY.items():
        if key in n:
            return dict(DEFAULT_GEN_POLICY, **pol)
    return dict(DEFAULT_GEN_POLICY)

_THINK_TAG_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


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
# LLM CLIENTS
# ══════════════════════════════════════════════════════════════════════════════
def _clean_content(content: str) -> str:
    """Strip inline <think>...</think> blocks some models embed in content."""
    if not content:
        return ""
    return _THINK_TAG_RE.sub('', content).strip()


# Verdict patterns, most explicit first. V23 matched only the bare substring
# "prediction: attack|normal|malicious", so any markdown emphasis, extra
# spacing or a trailing period made the whole session unparseable — this is
# why GLM abstained on 24/50 sessions in V23 despite reasoning correctly.
_VERDICT_RE = re.compile(
    r'(?:\*{0,2}|#{0,3})\s*'                     # **, ##, or nothing
    r'(?:final\s+)?(?:prediction|verdict|classification|answer)'
    r'\s*\*{0,2}\s*[:\-–]\s*\*{0,2}\s*'          # : - – with optional **
    r'(attack|malicious|normal|benign|legitimate)',
    re.IGNORECASE)


def extract_section(name: str, text: str, stops: List[str]) -> str:
    """
    Pull a named section out of an LLM response.

    The original patterns required a newline immediately after the header, so a
    model writing "SHAP_LIME_ASSESSMENT: the delivered energy ..." on one line,
    or emphasising it as "**SHAP_LIME_ASSESSMENT:**", parsed as absent. That is
    what drove the XAI-reference count to 0/50 while XAI faithfulness stayed at
    1.00 on the same responses — a reporting artefact, not a change in model
    behaviour. Markdown emphasis, heading markers and same-line content are all
    tolerated here.
    """
    if not text:
        return ""
    # The colon is OPTIONAL: models frequently emit the header as a markdown
    # heading ("### SHAP_LIME_ASSESSMENT") or bold label ("**SHAP_LIME_ASSESSMENT**")
    # with no colon at all. Requiring one is what kept the XAI-reference count
    # pinned at 0/50 even after same-line and bold formats were handled.
    num  = r"(?:\d+[\.\)]\s*)?"          # optional "2." / "3)" numbering
    stop = "|".join(rf"\n\s*\**\s*#*\s*{num}{t}" for t in stops) or r"\Z"
    pat  = rf"\**\s*#*\s*{num}{name}\s*\**\s*:?\s*\**\s*(.*?)(?={stop}|\Z)"
    m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_verdict(text: str) -> Optional[str]:
    """
    Robustly pull the final Normal/Attack verdict out of an LLM response.

    Strategy (in order):
      1. All explicit verdict lines -> take the LAST one (models often restate
         a provisional answer before committing to the final one).
      2. If the model emitted <think>...</think>, search the text AFTER the
         closing tag first, since that is the committed answer.
      3. Bare trailing line that is just "Attack" / "Normal".
    Returns 'Malicious' | 'Normal' | None.
    """
    if not text:
        return None

    def _norm(word: str) -> str:
        w = word.lower()
        return 'Malicious' if w in ('attack', 'malicious') else 'Normal'

    # Prefer the post-</think> region when present
    regions = []
    tail_split = re.split(r'</think>', text, flags=re.IGNORECASE)
    if len(tail_split) > 1:
        regions.append(tail_split[-1])
    regions.append(text)

    for region in regions:
        matches = _VERDICT_RE.findall(region)
        if matches:
            return _norm(matches[-1])

    # Bare final line, e.g. the model answers with just "Attack"
    for line in reversed([l.strip() for l in text.strip().split('\n') if l.strip()]):
        bare = re.fullmatch(r'\*{0,2}(attack|malicious|normal|benign|legitimate)\*{0,2}\.?',
                            line, re.IGNORECASE)
        if bare:
            return _norm(bare.group(1))
    return None


class OllamaClient:
    """
    Ollama backend. Same interface as V23's SimpleOllamaClient, plus:
      - options actually sent (temperature, num_predict, num_ctx)
      - keep_alive so the model stays loaded between calls
      - thinking capture for Qwen AND GLM (answer recovered when content
        is empty), with automatic fallback if the model rejects think=True
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434",
                 temperature=None, timeout=LLM_CALL_TIMEOUT_SEC):
        self.model_name = model_name
        self.base_url   = base_url.rstrip('/')
        self.policy     = gen_policy_for(model_name)
        # Fixed decoding temperature for this client (used by the temperature
        # sweep). None => Ollama default, which is what V23 effectively used.
        self.temperature = temperature
        # Thinking channel: enabled per-model policy. GLM is a reasoning model,
        # so separating its channels keeps the answer out of the reasoning dump.
        self.thinking_capable = self.policy.get('thinking', False)
        self.timeout = timeout
        # V30 counters, read by the runner so a run can be judged on whether
        # the calls actually succeeded rather than on the scores alone.
        self.n_calls = self.n_timeouts = self.n_errors = self.n_clamped = 0
        # The context window Ollama ACTUALLY granted. Filled in by
        # refresh_actual_ctx(); until then the requested value is assumed.
        self.actual_ctx = NUM_CTX
        try:
            from ollama import Client
            # A module-level ollama.chat() has no timeout at all. A Client
            # carries one into httpx, so a stalled generation raises instead
            # of blocking the study indefinitely.
            self._client = Client(host=self.base_url, timeout=self.timeout)
            self.chat = self._client.chat
            print(f"  [OK] Ollama client ready for {model_name} "
                  f"(thinking={'ON' if self.thinking_capable else 'OFF'}, "
                  f"num_predict={self.policy['num_predict']}, "
                  f"repeat_penalty={self.policy['repeat_penalty']}, "
                  f"temperature={'default' if temperature is None else temperature}, "
                  f"timeout={'none' if self.timeout is None else f'{self.timeout}s'})")
        except ImportError:
            raise ImportError("Ollama library not installed. Run: pip install ollama")

    def refresh_actual_ctx(self):
        """
        Ask the server what context window this model really got.

        NUM_CTX is a REQUEST. When VRAM is short Ollama silently grants less --
        on the target machine a 19 GB model reported CONTEXT 4096 against a
        request of 8192. Every budget calculation downstream used the requested
        figure, so it believed it had twice the room it had. Reading the granted
        value makes the clamp honest and lets the preflight refuse to run.
        """
        try:
            import requests
            ps = requests.get(f"{self.base_url}/api/ps", timeout=10).json()
            for m in ps.get('models', []):
                if m.get('model') == self.model_name or m.get('name') == self.model_name:
                    ctx = (m.get('context_length')
                           or (m.get('details') or {}).get('context_length'))
                    if ctx:
                        self.actual_ctx = int(ctx)
                        if self.actual_ctx < NUM_CTX:
                            print(f"    [CTX] requested {NUM_CTX}, granted "
                                  f"{self.actual_ctx} — VRAM is short")
                    return self.actual_ctx
        except Exception:
            pass
        return self.actual_ctx

    def _fit_budget(self, messages, num_ctx, requested):
        """
        Keep prompt + generation inside the context window.

        With a granted window of 4096, a ~10,900-character Stage 1 prompt and a
        Stage 1 response replayed into Stage 2, the room left for Stage 2 output
        collapses to the 128-token floor. A THINKING model handed 128 tokens
        spends all of them on its reasoning channel and returns an EMPTY answer,
        so no verdict line is ever written. That -- not any deficiency of the
        model -- is why GLM produced unparseable sessions and why raising its
        num_predict made things worse: the request was never the binding
        constraint, the window was.
        """
        approx_prompt = sum(len(m.get('content', '')) for m in messages) // 4
        room = num_ctx - approx_prompt - 128          # margin for chat scaffolding
        if room < 128:
            room = 128
        return max(64, min(requested, room))

    def _call(self, messages, temperature, max_tokens, force_no_think=False):
        # Per-model generation policy: a runaway-loop guard that is invisible to
        # models generating normal-length answers.
        ctx = self.actual_ctx or NUM_CTX
        options = {'num_ctx': NUM_CTX,
                   'num_predict': self.policy['num_predict'],
                   'repeat_penalty': self.policy['repeat_penalty']}
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            options['temperature'] = temp
        if max_tokens is not None:          # explicit override (repair call)
            options['num_predict'] = max_tokens
        fitted = self._fit_budget(messages, ctx, options['num_predict'])
        if fitted < options['num_predict']:
            self.n_clamped += 1
            print(f"    [BUDGET] num_predict {options['num_predict']} -> {fitted} "
                  f"so prompt + output fit in the granted num_ctx={ctx}")
        options['num_predict'] = fitted

        # A thinking model needs room for BOTH channels: the reasoning channel
        # is billed against the same num_predict as the answer. Below this floor
        # it reliably returns reasoning and no answer, so thinking is switched
        # off rather than allowed to consume the whole budget and emit nothing.
        want_think = self.thinking_capable and not force_no_think
        if want_think and fitted < THINK_MIN_BUDGET:
            print(f"    [THINK] budget {fitted} < {THINK_MIN_BUDGET} — disabling the "
                  f"reasoning channel for this call so the answer channel is used")
            want_think = False

        kwargs = dict(model=self.model_name, messages=messages,
                      options=options, keep_alive=KEEP_ALIVE)
        # force_no_think: for a trivial read-out ("Attack or Normal?") the
        # reasoning channel is pure cost. A thinking model given think=True
        # spends its whole num_predict budget reasoning and returns an EMPTY
        # content field, which is exactly why GLM's verdict repair kept
        # failing and 46% of its decisions ended up decided by word counting.
        self.n_calls += 1
        if want_think:
            try:
                return self.chat(think=True, **kwargs)
            except Exception as e:
                # Only a genuine "this model/client cannot do thinking" is
                # allowed to disable the channel. V29.2 caught every exception
                # here, so a TIMEOUT silently turned into a retry without
                # thinking and the failure never surfaced. A transport fault
                # must propagate to _guarded() and be recorded as an LLM error.
                msg = str(e).lower()
                unsupported = (isinstance(e, TypeError)
                               or 'think' in msg
                               or 'does not support' in msg)
                if not unsupported:
                    raise
                print(f"    [THINK] {self.model_name} does not support the "
                      f"reasoning channel — disabling it for this run")
                self.thinking_capable = False
        return self.chat(**kwargs)

    def _guarded(self, fn, *a, **kw):
        """Run one call, classifying failures instead of hiding them."""
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e)
            is_timeout = ('timeout' in msg.lower() or 'timed out' in msg.lower()
                          or isinstance(e, TimeoutError))
            if is_timeout:
                self.n_timeouts += 1
                print(f"    [TIMEOUT] {self.model_name} exceeded {self.timeout}s — "
                      f"this session is an LLM ERROR, not a model decision")
            else:
                self.n_errors += 1
                print(f"    [LLM ERROR] {self.model_name}: {msg[:160]}")
            return f"{LLM_ERROR_PREFIX} {'timeout' if is_timeout else msg}"

    def health(self) -> dict:
        return {'calls': self.n_calls, 'timeouts': self.n_timeouts,
                'errors': self.n_errors, 'budget_clamps': self.n_clamped,
                'granted_ctx': self.actual_ctx}

    def _extract(self, response, tag="THINK"):
        content  = _clean_content(response.message.content or "")
        thinking = getattr(response.message, 'thinking', None) or ""
        if thinking:
            print(f"    [{tag}] {len(thinking)} chars reasoning")
        # Only fall back to the thinking channel when the answer channel is
        # genuinely empty — otherwise a tentative mid-reasoning statement can
        # be mistaken for the final verdict.
        if not content and thinking:
            print(f"    [{tag}] answer channel EMPTY — the budget was consumed by "
                  f"reasoning; falling back to the reasoning text")
            content = thinking.strip()
        return content

    def generate_with_system(self, system_prompt, user_prompt,
                             temperature=LLM_TEMPERATURE,
                             max_tokens=STAGE1_MAX_TOKENS,
                             force_no_think=False, **kwargs):
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt}]
        out = self._guarded(self._call, messages, temperature, max_tokens,
                            force_no_think=force_no_think)
        return out if isinstance(out, str) else self._extract(out)

    def generate_two_turn(self, system_prompt, first_user, first_assistant,
                          second_user, temperature=LLM_TEMPERATURE,
                          max_tokens=STAGE2_MAX_TOKENS,
                          force_no_think=None, **kwargs):
        """
        Stage 2 of the V23 pipeline: the model is shown its own Stage 1 analysis
        and asked to commit to a verdict.

        Stage 2 defaults to the ANSWER channel only. Stage 1 is where the
        reasoning belongs and it keeps its thinking channel; Stage 2 is a
        commitment step whose entire product is a short structured block ending
        in a Prediction line. Letting a thinking model reason again here is what
        broke GLM: Stage 2 replays the Stage 1 prompt AND the Stage 1 response,
        so it is by far the longest prompt in the pipeline and leaves the least
        room, and the little room left was being spent on a second round of
        reasoning that ended before the verdict line was written. Pass
        force_no_think=False to restore the old behaviour for an ablation.
        """
        if force_no_think is None:
            force_no_think = STAGE2_ANSWER_CHANNEL_ONLY
        messages = [
            {'role': 'system',    'content': system_prompt},
            {'role': 'user',      'content': first_user},
            {'role': 'assistant', 'content': first_assistant},
            {'role': 'user',      'content': second_user}
        ]
        out = self._guarded(self._call, messages, temperature, max_tokens,
                            force_no_think=force_no_think)
        return out if isinstance(out, str) else self._extract(out, tag="THINK-2")


class VLLMClient:
    """
    OpenAI-compatible client for a vLLM server. vLLM implements
    PagedAttention and continuous batching server-side; running the runner
    with several parallel workers lets the server batch requests.

    Launch the server first, e.g.:
        vllm serve <model-id> --port 8000
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000",
                 temperature=None):
        import requests
        self._requests  = requests
        self.model_name = model_name
        self.base_url   = base_url.rstrip('/')
        self.temperature = temperature
        self.policy      = gen_policy_for(model_name)
        self.n_calls = self.n_timeouts = self.n_errors = 0
        self.actual_ctx = NUM_CTX
        print(f"  [OK] vLLM client ready for {model_name} at {self.base_url} "
              f"(PagedAttention + continuous batching server-side)")

    def _call(self, messages, temperature, max_tokens, force_no_think=False):
        temp = temperature if temperature is not None else self.temperature
        body = {'model': self.model_name, 'messages': messages,
                'max_tokens': max_tokens or self.policy['num_predict']}
        if temp is not None:
            body['temperature'] = temp
        self.n_calls += 1
        resp = self._requests.post(
            f"{self.base_url}/v1/chat/completions", json=body,
            timeout=LLM_CALL_TIMEOUT_SEC)
        resp.raise_for_status()
        msg = resp.json()['choices'][0]['message']
        content = _clean_content(msg.get('content') or "")
        # some servers put reasoning in a separate field
        reasoning = msg.get('reasoning_content') or ""
        if not content and reasoning:
            content = reasoning.strip()
        return content

    def generate_with_system(self, system_prompt, user_prompt,
                             temperature=0.0, max_tokens=STAGE1_MAX_TOKENS, **kwargs):
        try:
            return self._call(
                [{'role': 'system', 'content': system_prompt},
                 {'role': 'user',   'content': user_prompt}],
                temperature, max_tokens)
        except Exception as e:
            self.n_errors += 1
            return f"{LLM_ERROR_PREFIX} {str(e)}"

    def generate_two_turn(self, system_prompt, first_user, first_assistant,
                          second_user, temperature=0.0,
                          max_tokens=STAGE2_MAX_TOKENS, **kwargs):
        try:
            return self._call(
                [{'role': 'system',    'content': system_prompt},
                 {'role': 'user',      'content': first_user},
                 {'role': 'assistant', 'content': first_assistant},
                 {'role': 'user',      'content': second_user}],
                temperature, max_tokens)
        except Exception as e:
            self.n_errors += 1
            return f"{LLM_ERROR_PREFIX} {str(e)}"

    def refresh_actual_ctx(self):
        return NUM_CTX          # vLLM honours the window it was launched with

    def health(self) -> dict:
        return {'calls': self.n_calls, 'timeouts': self.n_timeouts,
                'errors': self.n_errors, 'budget_clamps': 0,
                'granted_ctx': NUM_CTX}


def unload_ollama_model(model_name: str, base_url: str = "http://localhost:11434"):
    """
    Force Ollama to evict a model from VRAM (keep_alive=0).

    Called when the runner switches to the next LLM so the incoming model gets
    the whole GPU instead of competing with the previous one.
    """
    try:
        import requests
        requests.post(f"{base_url.rstrip('/')}/api/generate",
                      json={'model': model_name, 'keep_alive': 0}, timeout=30)
        print(f"  [VRAM] Unloaded {model_name}")
    except Exception as e:
        print(f"  [VRAM] Could not unload {model_name}: {e}")


def make_llm_client(backend: str, model_name: str, base_url: str,
                    temperature=None, timeout=LLM_CALL_TIMEOUT_SEC):
    if backend == 'vllm':
        return VLLMClient(model_name, base_url, temperature=temperature)
    client = OllamaClient(model_name, base_url, temperature=temperature,
                          timeout=timeout)
    # Ask the server what context window it actually granted before the first
    # session runs, so the budget clamp works on the real number.
    client.refresh_actual_ctx()
    return client


# Backwards-compatible alias (V23 name)
SimpleOllamaClient = OllamaClient


# ══════════════════════════════════════════════════════════════════════════════
# XAI EXPLAINER — identical computation to V23, with cost knobs
# ══════════════════════════════════════════════════════════════════════════════
class XAIExplainer:
    """
    SHAP + LIME per ML model (V23 design, unchanged signals):
      RF/DT/GB -> TreeExplainer, LR -> LinearExplainer,
      MLP/SVC/KNN -> KernelExplainer; LIME for all.
    V25 knobs: kernel_nsamples / lime_num_samples bound the sampling cost.
    """

    TREE_MODELS   = {'Random Forest', 'Decision Tree', 'Gradient Boosting'}
    LINEAR_MODELS = {'Logistic Regression'}
    KERNEL_MODELS = {'MLP', 'Support Vector Classifier', 'K-Nearest Neighbors'}

    def __init__(self, models: dict, background_X: np.ndarray,
                 feature_names: list, n_kernel_bg: int = 30,
                 kernel_nsamples=None, lime_num_samples: int = 500):
        # V25 cut KernelSHAP to 200 samples and LIME to 200 for speed. SHAP and
        # LIME ARE the evidence the LLM reasons from, so that degraded the core
        # signal. V26 restores full fidelity: KernelSHAP default ('auto') and
        # LIME 500, exactly as V23. Speed is recovered by caching instead.
        self.models          = models
        self.feature_names   = feature_names
        self.background_X    = background_X
        self.kernel_nsamples = kernel_nsamples
        self.lime_num_samples = lime_num_samples
        n = min(n_kernel_bg, len(background_X))
        self.kernel_bg = shap.sample(background_X, n) if SHAP_AVAILABLE else background_X[:n]
        self.shap_explainers = {}
        self.lime_explainer  = None
        self.informative     = {}     # model -> does its SHAP vary per session?
        self._init_shap()
        self._init_lime()
        self._profile_informativeness()

    def _profile_informativeness(self, n_probe: int = 12, tol: float = 1e-6):
        """
        Flag models whose SHAP attribution is effectively CONSTANT.

        A depth-1 stump splits once on one feature, so its Shapley attribution
        is the same for every session. Presenting such an explanation as
        per-session evidence injects a constant into every prompt while the
        system prompt instructs the LLM to weight explanations above the votes
        -- i.e. it substitutes noise for signal. Models detected here are
        reported as uninformative instead of being dressed up as evidence.
        """
        if not SHAP_AVAILABLE or len(self.background_X) < 2:
            return
        probe = self.background_X[:min(n_probe, len(self.background_X))]
        for name in self.models:
            if name not in self.shap_explainers:
                continue
            try:
                tops = []
                for row in probe:
                    r = self.explain_model(name, row.reshape(1, -1), 1)
                    if r.get('shap_ok') and r['shap_values']:
                        tops.append(r['shap_values'][0][1])
                if len(tops) >= 2:
                    varies = float(np.std(tops)) > tol
                    self.informative[name] = varies
                    if not varies:
                        print(f"  [XAI] {name}: attribution is CONSTANT across "
                              f"sessions -> will be reported as uninformative")
            except Exception:
                self.informative[name] = True

    def _init_shap(self):
        if not SHAP_AVAILABLE:
            return
        for name, model in self.models.items():
            try:
                if name in self.TREE_MODELS:
                    self.shap_explainers[name] = shap.TreeExplainer(
                        model, data=self.background_X,
                        feature_perturbation='interventional')
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

    def explain_model(self, model_name: str, X_scaled: np.ndarray,
                      predicted_class_idx: int) -> Dict:
        result = {'shap_values': [], 'lime_values': [],
                  'shap_ok': False, 'lime_ok': False}
        model = self.models.get(model_name)
        if model is None:
            return result

        # ── SHAP ──────────────────────────────────────────────────────────────
        if SHAP_AVAILABLE and model_name in self.shap_explainers:
            try:
                explainer = self.shap_explainers[model_name]
                if isinstance(explainer, shap.KernelExplainer):
                    ns  = self.kernel_nsamples if self.kernel_nsamples else 'auto'
                    raw = explainer.shap_values(X_scaled, nsamples=ns, silent=True)
                else:
                    raw = explainer.shap_values(X_scaled)

                # SIGN CONVENTION (fixed here; wrong since V23).
                # The prompt states "+ = toward Attack", so the attribution
                # MUST be taken with respect to the Attack class. Earlier
                # versions indexed by the PREDICTED class instead. For binary
                # classification the two classes' Shapley values are mirror
                # images, so every model that predicted Normal had its
                # direction inverted: the prompt announced "toward Attack" for
                # evidence that actually pointed toward Normal. Because
                # MLP/SVC/GB predict Normal on every session, most SHAP blocks
                # in every prompt were sign-flipped toward Attack, which is
                # what drove the false-alarm rate to 0.42-0.45.
                if isinstance(raw, list):
                    idx = min(ATTACK_CLASS, len(raw) - 1)
                    sv  = np.array(raw[idx]).flatten()
                elif isinstance(raw, np.ndarray):
                    if raw.ndim == 3:
                        idx = min(ATTACK_CLASS, raw.shape[2] - 1)
                        sv  = raw[0, :, idx]
                    elif raw.ndim == 2:
                        sv = raw[0]
                    else:
                        sv = raw.flatten()
                else:
                    sv = np.zeros(len(self.feature_names))

                shap_entries = []
                for i, fname in enumerate(self.feature_names):
                    val = float(sv[i]) if i < len(sv) else 0.0
                    direction = (
                        "strongly toward Attack" if val >  0.3 else
                        "toward Attack"           if val >  0.1 else
                        "slightly toward Attack"  if val >  0.02 else
                        "negligible"              if abs(val) <= 0.02 else
                        "slightly toward Normal"  if val > -0.1 else
                        "toward Normal"           if val > -0.3 else
                        "strongly toward Normal"
                    )
                    shap_entries.append((FEATURE_LABELS.get(fname, fname), val, direction))
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

                # Same convention as SHAP: weights are reported against the
                # Attack class so "supports Attack" is literally true.
                exp = self.lime_explainer.explain_instance(
                    X_scaled[0], pred_fn,
                    num_features = len(self.feature_names),
                    num_samples  = self.lime_num_samples,
                    labels       = (1,)
                )
                lime_list = exp.as_list(label=1)
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

        return result


# ══════════════════════════════════════════════════════════════════════════════
# XAI STORE — compute once per sample, share across all three LLM agents
# ══════════════════════════════════════════════════════════════════════════════
_EMPTY_XAI = {'shap_values': [], 'lime_values': [], 'shap_ok': False, 'lime_ok': False}


class XAIStore:
    """
    V23 recomputed SHAP+LIME for every sample once PER AGENT (3x total).
    The explanations depend only on (sample, ML model) — never on the LLM —
    so V25 computes them once, keeps them in memory, and persists them to
    disk so scenario re-runs are free.
    """

    @staticmethod
    def models_fingerprint(models_dir: str) -> str:
        """
        Identity of the CURRENT trained ML models.

        SHAP/LIME values depend entirely on the fitted models, but the cache is
        keyed only by (sample, model_name). If the models are retrained (e.g.
        a different train ratio) a stale cache would silently feed the LLM
        explanations computed from the OLD models — corrupting any comparison
        between runs. The fingerprint detects that and invalidates the cache.
        """
        import hashlib
        h = hashlib.sha256()
        try:
            sp = os.path.join(models_dir, 'split_info_v6.pkl')
            if os.path.exists(sp):
                with open(sp, 'rb') as f:
                    si = pickle.load(f)
                h.update(repr((si.get('train_ratio'), si.get('random_state'),
                               len(si.get('train_indices', [])),
                               len(si.get('test_indices', [])))).encode())
            for fn in sorted(os.listdir(models_dir)):
                if fn.endswith('.pkl'):
                    st = os.stat(os.path.join(models_dir, fn))
                    h.update(f"{fn}:{st.st_size}:{int(st.st_mtime)}".encode())
        except Exception:
            return "unknown"
        return h.hexdigest()[:16]

    def __init__(self, explainer: Optional[XAIExplainer], cache_path: str,
                 fingerprint: str = ""):
        self.explainer  = explainer
        self.cache_path = cache_path
        self._lock      = threading.Lock()
        self._cache     = {}
        self.fingerprint = fingerprint
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    blob = pickle.load(f)
                cached_fp = blob.get('fingerprint') if isinstance(blob, dict) else None
                entries   = blob.get('entries') if isinstance(blob, dict) else None
                if entries is None or cached_fp != fingerprint:
                    print(f"  [XAI-STORE] Cache is from a different set of trained "
                          f"models — discarding and recomputing "
                          f"(cached={cached_fp}, current={fingerprint})")
                    self._cache = {}
                else:
                    self._cache = entries
                    n_samples = len({k[0] for k in self._cache})
                    print(f"  [XAI-STORE] Loaded cached explanations "
                          f"({n_samples} samples, {len(self._cache)} entries)")
            except Exception:
                self._cache = {}

    def get(self, line_number: int, model_name: str,
            X_scaled: np.ndarray, predicted_class_idx: int) -> Dict:
        key = (int(line_number), model_name)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if self.explainer is None:
            return dict(_EMPTY_XAI)
        with self._lock:   # SHAP/LIME explainers are not thread-safe
            if key in self._cache:
                return self._cache[key]
            result = self.explainer.explain_model(model_name, X_scaled, predicted_class_idx)
            self._cache[key] = result
        return result

    def save(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({'fingerprint': self.fingerprint,
                             'entries': self._cache}, f)
        except Exception as e:
            print(f"  [XAI-STORE] save failed: {e}")


def build_xai_store(config: dict, df: pd.DataFrame, column_mapping: dict,
                    scaler, models: dict) -> XAIStore:
    """Build the shared explainer + store from the TRAINING split only."""
    if not SHAP_AVAILABLE and not LIME_AVAILABLE:
        print("  [XAI] Both SHAP and LIME unavailable — store disabled")
        return XAIStore(None, "")
    try:
        split_path = os.path.join(config['models_dir'], 'split_info_v6.pkl')
        if not os.path.exists(split_path):
            print("  [XAI] split_info_v6.pkl not found — store disabled")
            return XAIStore(None, "")
        with open(split_path, 'rb') as f:
            split_info = pickle.load(f)
        train_idx = split_info['train_indices']

        actual_cols = [column_mapping[c] for c in
                       ['connectionTime', 'disconnectTime', 'RequestedDemand', 'kWhDelivered']]
        train_df = df.loc[train_idx, actual_cols].copy()
        train_df.columns = FEATURE_NAMES
        for col in ['connectionTime', 'disconnectTime']:
            train_df[col] = train_df[col].apply(parse_datetime_to_timestamp)
        train_df = train_df.fillna(0)
        bg_scaled = scaler.transform(train_df.values)
        if len(bg_scaled) > 200:
            rng  = np.random.default_rng(42)
            idxs = rng.choice(len(bg_scaled), 200, replace=False)
            bg_scaled = bg_scaled[idxs]

        print(f"\n  [XAI] Initialising shared SHAP + LIME explainers "
              f"({len(bg_scaled)} background samples)...")
        explainer  = XAIExplainer(models, bg_scaled, FEATURE_NAMES, n_kernel_bg=30)
        cache_path = os.path.join(
            config.get('workspace_dir', './ev_ids_workspace_v6'), 'xai_cache_v30.pkl')
        fp = XAIStore.models_fingerprint(config['models_dir'])
        print(f"  [XAI] Model fingerprint: {fp}")
        return XAIStore(explainer, cache_path, fingerprint=fp)
    except Exception as e:
        print(f"  [XAI] store init failed: {e}")
        return XAIStore(None, "")


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE (RAG) — identical to V23
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
        """Concepts only — no numerical thresholds (V22/V23/V25 design)."""
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
# LONG-TERM MEMORY — identical to V23
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
# MAIN AGENT — V25 (V23 flow and prompts, performance-fixed)
# ══════════════════════════════════════════════════════════════════════════════
class EVIDSAgentV6TripleLLMV30:
    """
    Same idea, objective, 7-step flow and prompts as V23.
    V25 differences are implementation-only:
      - shared XAIStore (explanations computed once per sample)
      - bounded, deterministic LLM generation (num_predict / temperature / num_ctx)
      - thinking-channel capture for GLM/Qwen
      - always-classify (ML-majority fallback instead of ABSTAIN)
      - `verbose=False` supported for parallel runs (echoing 8k-char prompts
        to the Windows console for every session is itself measurably slow)
    """

    # Zones where sessions are genuinely hard — self-consistency is applied
    # only here so the extra LLM calls stay bounded.
    AMBIGUOUS_ZONES = {'BORDERLINE_HIGH', 'BORDERLINE_LOW',
                       'AMBIGUOUS_HIGH', 'AMBIGUOUS_LOW', 'UNCERTAIN'}

    def __init__(self, config, llm_client, use_knowledge=True, use_memory=True,
                 scenario_tag="default", xai_store: Optional[XAIStore] = None,
                 verbose: bool = True, print_lock=None,
                 self_consistency_k: int = 1, use_xai: bool = True):
        self.config             = config
        self.llm_client         = llm_client
        self.self_consistency_k = max(1, self_consistency_k)
        # use_xai=False removes the SHAP/LIME blocks from the prompt (models
        # still show prediction + confidence, as in V20/V22). Everything else
        # -- prompts, flow, scenarios, display -- is identical, so with/without
        # runs on the same samples isolate the XAI contribution exactly.
        self.use_xai            = use_xai
        self.use_knowledge = use_knowledge
        self.use_memory    = use_memory
        self.scenario_tag  = scenario_tag
        self.verbose       = verbose
        self.print_lock    = print_lock
        # Decoding temperature for this agent. None => the client's own value
        # (which the temperature sweep sets per run). Must NOT be hard-coded in
        # detect(), or the sweep silently evaluates one temperature five times.
        _client_temp = getattr(llm_client, 'temperature', None)
        self.temperature = _client_temp if _client_temp is not None else LLM_TEMPERATURE
        self._out          = []
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

        # Shared XAI store (compute-once). Falls back to a private store.
        if xai_store is not None:
            self.xai_store = xai_store
        else:
            self.xai_store = build_xai_store(config, self.df, self.column_mapping,
                                             self.scaler, self.models)

        # Full per-session transcripts (prompts incl. SHAP/LIME evidence +
        # LLM responses) — explainability is a core contribution, so every
        # interaction is persisted even when console output is compact.
        self.constant_models = self._profile_constant_models()

        self.transcript_dir = os.path.join(
            config.get('workspace_dir', './ev_ids_workspace_v6'),
            'transcripts', scenario_tag)
        try:
            os.makedirs(self.transcript_dir, exist_ok=True)
        except Exception:
            self.transcript_dir = None

    def _profile_constant_models(self, n_probe: int = 200):
        """
        Identify classifiers that emit the SAME class for every session.

        Measured on this dataset, MLP, SVC and Gradient Boosting never predict
        Attack. Their vote is a constant, but each still occupies a line in the
        prompt and a share of the vote tally, so on a genuine attack the tally
        can read "2 Attack, 5 Normal" -- a majority-Normal signal on a real
        attack. Any LLM that defers to the tally inherits that bias: Llama
        agreed with the ensemble on 84% of sessions and missed 9 attacks, while
        Qwen largely ignored it and missed none. Flagging these models lets the
        prompt state plainly that their vote carries no information.
        """
        constant = set()
        try:
            split_path = os.path.join(self.config['models_dir'], 'split_info_v6.pkl')
            if not os.path.exists(split_path):
                return constant
            with open(split_path, 'rb') as f:
                train_idx = pickle.load(f)['train_indices']
            cm = self.column_mapping
            cols = [cm[c] for c in ['connectionTime', 'disconnectTime',
                                    'RequestedDemand', 'kWhDelivered']]
            probe = self.df.loc[train_idx[:n_probe], cols].copy()
            probe.columns = FEATURE_NAMES
            for c in ['connectionTime', 'disconnectTime']:
                probe[c] = probe[c].apply(parse_datetime_to_timestamp)
            Xp = self.scaler.transform(probe.fillna(0).values)
            for name, m in self.models.items():
                preds = set(np.asarray(m.predict(Xp)).ravel().tolist())
                if len(preds) < 2:
                    constant.add(name)
            if constant:
                print(f"  [ML] Non-discriminating classifiers (single class on "
                      f"{len(Xp)} training sessions): {', '.join(sorted(constant))}")
                print(f"       Their votes will be marked as carrying no information.")
        except Exception as e:
            print(f"  [ML] constant-model profiling skipped: {e}")
        return constant

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

    def _get_model_type(self):
        n = self.llm_client.model_name.lower()
        return "QWEN" if "qwen" in n else "GLM" if "glm" in n else "LLAMA" if "llama" in n else "UNKNOWN"

    def _repair_budget(self) -> int:
        """Token budget for the verdict-repair call.

        A thinking model spends num_predict on its reasoning channel before it
        writes any content, so a 24-token repair budget guarantees an empty
        answer. Thinking models therefore need room to think AND answer.
        """
        # The read-out runs with force_no_think, so the reasoning channel is not
        # in play and a small budget suffices for every model.
        return 64

    def _log(self, msg):
        """Buffer one console line for this session.

        The full V23 step-by-step trace (prompts with SHAP/LIME evidence and
        both LLM responses) is ALWAYS produced — explainability is a core
        contribution. Buffering keeps each session's block contiguous when
        several sessions run in parallel.
        """
        self._out.append(msg)

    def _flush(self):
        block = "\n".join(self._out)
        self._out = []
        if not self.verbose:
            return
        if self.print_lock is not None:
            with self.print_lock:
                print(block, flush=True)
        else:
            print(block, flush=True)

    # ── SYSTEM PROMPT — identical to V23 ───────────────────────────────────────
    def _get_system_prompt(self) -> str:
        return """You are a security analyst investigating anomalies in a monitored infrastructure system.

YOUR TASK:
Classify a session as Normal or Attack by reasoning from evidence.

EVIDENCE YOU WILL RECEIVE (in priority order):
1. Domain knowledge — physical meaning of the measurements and known attack patterns
2. Historical precedents — how similar past sessions were classified and why
3. Raw session measurements — the actual data from this session
4. ML classifier predictions — each with:
   - Confidence score (how certain the model is)
   - SHAP explanation (which features drove the prediction and by how much)
   - LIME explanation (local linear approximation of the decision boundary)

HOW TO USE SHAP AND LIME:
- SHAP values tell you the DIRECTION and MAGNITUDE of each feature's contribution.
  A positive SHAP value means that feature pushed the model toward Attack.
  A negative SHAP value means it pushed toward Normal.
  Large absolute SHAP values = that feature was decisive for this model.
- LIME weights confirm or contradict SHAP locally around this specific sample.
  If SHAP and LIME agree on the same features driving the prediction,
  that model's reasoning is more trustworthy.
  If they disagree, the model may be relying on spurious patterns.
- When SHAP and LIME across MULTIPLE models all point to the same feature
  (e.g., delivered energy is consistently the top driver toward Attack),
  that is strong converging evidence.

HOW TO REASON:
- Start with domain knowledge to understand what the measurements mean physically
- Compute the delivery ratio yourself from the raw values (delivered / requested)
- Check whether that ratio is consistent with known attack patterns
- Examine the ML predictions: do they agree? Do their SHAP/LIME explanations
  point to the same features? Are those features physically meaningful?
- Use historical cases to see how similar sessions were handled before
- Weigh all evidence and arrive at your own independent conclusion

CRITICAL RULES:
- A model's VOTE is its conclusion; its SHAP/LIME explanation shows WHAT DROVE
  that conclusion. Use the explanation to judge how much the vote deserves,
  not as a substitute for it.
- A vote driven by the energy features (requested / delivered) is credible.
  A vote driven by connection or disconnect timing is suspect, because timing
  has no physical bearing on whether energy was stolen.
- Some explanations are marked UNINFORMATIVE. Those models use a single fixed
  rule, so their attribution is the same for every session and tells you
  nothing about THIS one. Ignore their explanation and use their vote only.
- Some classifiers are marked NON-DISCRIMINATING: they output the same class
  for every session, so their vote is a constant and tells you nothing about
  THIS session. Exclude them from your reasoning and use the DISCRIMINATING
  vote tally where one is given.
- The delivery ratio you computed from the raw measurements is the single most
  reliable piece of evidence available to you. It OUTRANKS the vote tally. A
  ratio far above 1.0 indicates energy theft even if most classifiers voted
  Normal — several of them are known to miss attacks.
- Never treat the raw vote count alone as the final answer, and never treat a
  single explanation as decisive.
- Explain your reasoning so another analyst can follow it

STAGE 1 FORMAT (analysis only — no final prediction yet):
PHYSICAL_INTERPRETATION:
[What do the raw values tell you? Compute and interpret the delivery ratio.]

SHAP_LIME_ASSESSMENT:
[Which features are consistently driving the ML predictions?
 Do the explanations make physical sense for this session?
 Which models have credible vs suspicious explanations?]

DOMAIN_MATCH:
[Which attack pattern (if any) does this session resemble, and how closely?]

HISTORICAL_CONTEXT:
[What do historical cases suggest?]

UNCERTAINTY_FACTORS:
[What makes this case ambiguous? What evidence conflicts?]"""

    STAGE2_USER = """Based on your analysis above, provide your final classification.

FINAL DECISION FORMAT:
REASONING_SUMMARY:
[One paragraph: the single most important piece of evidence and why it drove your decision.
 Reference the specific SHAP/LIME signals that confirmed or contradicted your conclusion.]

CONFLICTING_SIGNALS:
[Any evidence that pointed the other way and how you resolved it]

CONFIDENCE: [high / medium / low]

ATTACK_TYPE: [energy_theft / phantom_charging / session_manipulation / none]

Prediction: [Attack or Normal]"""

    # ── FORMAT XAI FOR PROMPT — identical to V23 ───────────────────────────────
    @staticmethod
    def _format_xai_for_prompt(model_name: str, xai_result: Dict,
                               prediction: str, confidence: float,
                               is_constant: bool = False) -> str:
        tag = "   [NON-DISCRIMINATING: this model outputs the same class for " \
              "every session — its vote carries no information]" if is_constant else ""
        lines = [f"  {model_name}: {prediction} ({confidence*100:.1f}%){tag}"]

        if xai_result.get('shap_ok') and not xai_result.get('informative', True):
            lines.append("    SHAP/LIME: UNINFORMATIVE for this model — it applies a "
                         "single fixed split,")
            lines.append("               so its attribution is identical for every "
                         "session. Judge this")
            lines.append("               model on its vote and confidence only.")
            return "\n".join(lines)

        if xai_result.get('shap_ok') and xai_result['shap_values']:
            lines.append("    SHAP — feature contributions to this prediction:")
            for fname, val, direction in xai_result['shap_values'][:3]:
                bar = "#" * min(int(abs(val) * 20), 10)
                lines.append(f"      {fname:<28} {val:+.3f}  {bar}  ({direction})")
        else:
            lines.append("    SHAP: unavailable")

        if xai_result.get('lime_ok') and xai_result['lime_values']:
            lines.append("    LIME — local decision boundary:")
            for fname, condition, weight, direction in xai_result['lime_values'][:3]:
                lines.append(f"      {condition:<35} weight={weight:+.3f}  ({direction})")
        else:
            lines.append("    LIME: unavailable")

        return "\n".join(lines)

    # ── STAGE 1 PROMPT — identical to V23 ──────────────────────────────────────
    def _build_stage1_prompt(self, data_dict, all_predictions, xai_results,
                             requested_kwh, delivered_kwh, duration_hours,
                             knowledge, ltm_cases) -> str:
        evidence = ""

        if knowledge:
            evidence += "=" * 60 + "\n"
            evidence += "DOMAIN KNOWLEDGE\n"
            evidence += "=" * 60 + "\n"
            evidence += knowledge[:800] + "\n\n"

        if ltm_cases:
            evidence += "=" * 60 + "\n"
            evidence += f"HISTORICAL PRECEDENTS ({len(ltm_cases)} similar sessions)\n"
            evidence += "=" * 60 + "\n"
            for i, c in enumerate(ltm_cases[:5], 1):
                evidence += f"Case {i}:\n"
                evidence += f"  Requested: {c.get('requested_kwh','?')} kWh\n"
                evidence += f"  Delivered: {c.get('delivered_kwh','?')} kWh\n"
                evidence += f"  Duration:  {c.get('duration_hours','?')} hours\n"
                evidence += f"  Outcome:   {c.get('prediction','?')} (conf={c.get('confidence',0):.2f})\n"
                if c.get('reasoning'):
                    evidence += f"  Reasoning: {c['reasoning']}\n"
                evidence += "\n"

        evidence += "=" * 60 + "\n"
        evidence += "CURRENT SESSION — RAW MEASUREMENTS\n"
        evidence += "=" * 60 + "\n"
        evidence += f"  Energy requested:  {requested_kwh:.4f} kWh\n"
        evidence += f"  Energy delivered:  {delivered_kwh:.4f} kWh\n"
        evidence += f"  Session duration:  {duration_hours:.2f} hours\n\n"
        evidence += "  Compute the delivery ratio (delivered / requested) yourself\n"
        evidence += "  as part of your physical interpretation.\n\n"

        votes = Counter([p['prediction'] for p in all_predictions.values()])
        evidence += "=" * 60 + "\n"
        if self.use_xai:
            evidence += "ML CLASSIFIER PREDICTIONS WITH SHAP + LIME EXPLANATIONS\n"
            evidence += "=" * 60 + "\n"
            evidence += (
                "Each model shows:\n"
                "  - Its prediction and confidence score\n"
                "  - SHAP values: how much each feature contributed (+= toward Attack)\n"
                "  - LIME weights: local linear approximation of its decision\n\n"
            )
            for mn, pred in all_predictions.items():
                xai_r = xai_results.get(mn, dict(_EMPTY_XAI))
                evidence += self._format_xai_for_prompt(
                    mn, xai_r, pred['prediction'], pred['confidence'],
                    is_constant=(mn in self.constant_models))
                evidence += "\n\n"
        else:
            # XAI ABLATION: predictions + confidence only, no SHAP/LIME.
            evidence += "ML CLASSIFIER PREDICTIONS\n"
            evidence += "=" * 60 + "\n"
            for mn, pred in all_predictions.items():
                tag = ("   [NON-DISCRIMINATING: same class every session]"
                       if mn in self.constant_models else "")
                evidence += (f"  {mn}: {pred['prediction']} "
                             f"({pred['confidence']*100:.1f}%){tag}\n")
            evidence += "\n"

        disc = {m: p for m, p in all_predictions.items()
                if m not in self.constant_models}
        dv   = Counter([p['prediction'] for p in disc.values()])
        evidence += (f"Raw vote tally (all {len(all_predictions)} models): "
                     f"{votes.get('Malicious',0)} Attack, {votes.get('Normal',0)} Normal\n")
        if self.constant_models:
            evidence += (f"DISCRIMINATING vote tally (excluding the "
                         f"{len(self.constant_models)} non-discriminating models): "
                         f"{dv.get('Malicious',0)} Attack, {dv.get('Normal',0)} Normal\n")
            evidence += ("Use the DISCRIMINATING tally. The raw tally is skewed by "
                         "models that never change their answer.\n")
        if self.use_xai:
            evidence += ("Note: use the SHAP/LIME explanations to assess each model's "
                         "credibility, not just its vote.\n")

        return f"""Analyze this session. Do NOT give a final prediction yet — only your analysis.

{evidence}

Provide your structured analysis using the format in your instructions."""

    # ── PARSE RESPONSE — V23 parsing + always-classify fallback ────────────────
    def _parse_llm_response(self, stage2_response, stage1_response,
                            all_predictions, delivery_ratio, duration_hours,
                            line_number, votes, ltm_cases,
                            rag_sources, complexity_metrics, difficulty_zone,
                            xai_top_feature="", repair_used=False,
                            llm_error=False):
        predicted_label = None
        llm_confidence  = "medium"
        attack_type     = "none"
        used_fallback   = False
        resp_lower      = stage2_response.lower()

        # Robust extraction (V26). V23's bare-substring match lost any verdict
        # written as "**Prediction:** Attack", "Prediction : Attack", etc.
        predicted_label = extract_verdict(stage2_response)

        conf_match = re.search(r'confidence:\s*(high|medium|low)', resp_lower)
        if conf_match:
            llm_confidence = conf_match.group(1)

        type_match = re.search(
            r'attack_type:\s*(energy_theft|phantom_charging|session_manipulation|none)',
            resp_lower)
        if type_match:
            attack_type = type_match.group(1)

        reasoning_summary = extract_section(
            "REASONING_SUMMARY", stage2_response,
            ["CONFLICTING_SIGNALS", "CONFIDENCE", "PREDICTION"])

        conflicts_text = extract_section(
            "CONFLICTING_SIGNALS", stage2_response,
            ["CONFIDENCE", "ATTACK_TYPE", "PREDICTION"])

        xai_assessment = extract_section("SHAP_LIME_ASSESSMENT", stage1_response,
                                         ["DOMAIN_MATCH", "HISTORICAL"])

        physical_interp = extract_section("PHYSICAL_INTERPRETATION", stage1_response,
                                          ["SHAP_LIME", "DOMAIN_MATCH"])

        # Always-classify fallback: tone inference, then ML majority (no ABSTAIN)
        ml_majority = votes.most_common(1)[0][0] if votes else 'Normal'
        # llm_parsed marks sessions where the LLM itself gave a usable verdict.
        # V23 would have scored ONLY these (everything else was ABSTAIN), so the
        # runner can report a V23-comparable metric alongside always-classify.
        llm_parsed = predicted_label is not None

        # The tone heuristic that used to sit here has been REMOVED. It decided
        # by counting the literal words "attack" and "normal" in the response.
        # Every SHAP line in the prompt reads "(toward Attack)" or "(toward
        # Normal)", so a verbose model that quotes and discusses those lines
        # accumulates the token "attack" regardless of its actual conclusion.
        # On the last Scenario A run GLM produced 5,247 words per session and
        # 21 of its 50 decisions were taken by this word count -- with 14 more
        # falling back to the ML majority, 70% of its reported decisions never
        # came from a stated verdict. That is not a measurement of the model.
        # A verdict is now either stated, recovered by the one-word repair
        # request, or deferred to the ML majority, and which of the three is
        # recorded in verdict_source.
        if predicted_label is None:
            predicted_label = ml_majority
            llm_confidence  = "low"
            used_fallback   = True
            self._log(f"    [WARN] No verdict stated or recovered — "
                      f"deferring to ML majority: {ml_majority}")

        # V30: an LLM that never answered is NOT an LLM that answered badly.
        # Up to V29.2 a timed-out call produced the string "Error: ...", no
        # verdict could be extracted from it, and the session silently became an
        # ML fallback that was then scored as a model decision. Two runs of the
        # same code on the same data at temperature 0 disagreed for exactly this
        # reason. The distinction is now carried through to the metrics, which
        # refuse to publish a score for a model whose calls did not complete.
        verdict_source = ('llm_error'   if llm_error
                          else 'ml_fallback' if used_fallback
                          else 'repaired'    if repair_used
                          else 'stated')

        if predicted_label == 'Malicious' and attack_type == 'none':
            attack_type = ('energy_theft'    if delivery_ratio > 1.3 else
                           'phantom_charging' if delivery_ratio < 0.7 else
                           'unknown_attack')

        llm_overrode_ml = (predicted_label != ml_majority)

        llm_conf_score  = {'high': 0.90, 'medium': 0.65, 'low': 0.40}.get(llm_confidence, 0.65)
        ml_agreement    = (sum(1 for p in all_predictions.values()
                               if p['prediction'] == predicted_label)
                           / len(all_predictions))
        reasoning_depth = min(1.0, (len(stage1_response) + len(stage2_response)) / 1400)
        composite       = 0.60 * llm_conf_score + 0.25 * ml_agreement + 0.15 * reasoning_depth

        return {
            "line_number":          line_number,
            "predicted_label":      predicted_label,
            "attack_type":          attack_type,
            "confidence":           round(float(composite), 3),
            "llm_confidence":       llm_confidence,
            "ml_agreement":         round(float(ml_agreement), 3),
            "ml_majority":          ml_majority,
            "llm_overrode_ml":      llm_overrode_ml,
            "used_fallback":        used_fallback,
            "llm_parsed":           llm_parsed,
            "verdict_source":       verdict_source,
            "llm_error":            bool(llm_error),
            "xai_top_feature":      xai_top_feature,
            "difficulty_zone":      difficulty_zone,
            "llm_reasoning":        reasoning_summary[:500] if reasoning_summary else stage2_response[:500],
            "llm_analysis":         physical_interp[:500]  if physical_interp  else stage1_response[:300],
            "llm_xai_assessment":   xai_assessment[:400]   if xai_assessment   else "",
            "llm_conflicts":        conflicts_text[:300],
            "delivery_ratio":       round(delivery_ratio, 3),
            "duration_hours":       round(duration_hours, 2),
            "knowledge_used":       self.use_knowledge,
            "memory_used":          self.use_memory,
            "xai_used":             self.use_xai and self.xai_store.explainer is not None,
            "xai_enabled":          self.use_xai,
            "shap_available":       SHAP_AVAILABLE,
            "lime_available":       LIME_AVAILABLE,
            "ltm_cases_referenced": len(ltm_cases),
            "rag_sources_used":     len(rag_sources),
            "classifier_votes":     dict(votes),
            "aggregation_method":   ("V30_RAG_LTM_XAI" if (self.use_knowledge and self.use_memory)
                                     else "V30_RAG_XAI"  if self.use_knowledge
                                     else "V30_LTM_XAI"  if self.use_memory
                                     else "V30_ML_XAI"),
            "complexity":           complexity_metrics
        }

    # ── MAIN DETECT — V23 7-step flow ──────────────────────────────────────────
    def detect(self, line_number):
        t0 = time.time()
        if   self.use_knowledge and self.use_memory: sc = "A (ML+RAG+LTM+XAI->LLM)"
        elif self.use_knowledge:                     sc = "B (ML+RAG+XAI->LLM)"
        elif self.use_memory:                        sc = "C (ML+LTM+XAI->LLM)"
        else:                                        sc = "D (ML+XAI->LLM baseline)"
        self._log(f"\n[MODEL] {self.llm_client.model_name} -> {self._get_model_type()}")
        self._log(f"[SCENARIO] {sc}")
        self._log(f"\n{'='*100}\nANALYZING SESSION {line_number}\n{'='*100}\n")
        timestamp = datetime.now().timestamp()

        # STEP 1: DATA EXTRACTION
        self._log(f"[Step 1] Data Extraction")
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
        self._log(f"  Requested={requested_kwh:.4f}kWh, Delivered={delivered_kwh:.4f}kWh")

        # STEP 2: PREPROCESSING
        self._log(f"[Step 2] Preprocessing")
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
        self._log(f"  Delivery Ratio (internal): {delivery_ratio:.4f} | Zone: {difficulty_zone}")

        # STEP 3: ML CLASSIFIERS + shared XAI
        self._log(f"\n[Step 3] ML Classifiers + SHAP/LIME (shared store)")
        all_predictions = {}
        xai_results     = {}
        ML_ORDER = ['Random Forest', 'K-Nearest Neighbors', 'Logistic Regression',
                    'MLP', 'Support Vector Classifier', 'Decision Tree', 'Gradient Boosting']

        t_xai_total = 0.0
        for mn in ML_ORDER:
            model = self.models[mn]
            if hasattr(model, 'predict_proba'):
                probas  = model.predict_proba(X_scaled)[0]
                classes = model.classes_
                pidx    = int(np.argmax(probas))
                label   = 'Malicious' if int(classes[pidx]) == 1 else 'Normal'
                conf    = float(probas[pidx])
            else:
                pred    = model.predict(X_scaled)[0]
                label   = 'Malicious' if int(pred) == 1 else 'Normal'
                conf    = 0.5
                pidx    = 1 if label == 'Malicious' else 0
            all_predictions[mn] = {'prediction': label, 'confidence': conf}

            t_x = time.time()
            xai_r = self.xai_store.get(line_number, mn, X_scaled, pidx)
            xai_results[mn] = xai_r
            t_xai_total += time.time() - t_x
            shap_top = xai_r['shap_values'][0][0] if xai_r.get('shap_ok') and xai_r['shap_values'] else "—"
            lime_top = xai_r['lime_values'][0][0] if xai_r.get('lime_ok') and xai_r['lime_values'] else "—"
            self._log(f"  {mn}: {label} ({conf:.3f}) | SHAP top={shap_top} | LIME top={lime_top}")

        votes = Counter([p['prediction'] for p in all_predictions.values()])
        # Dominant SHAP driver across the ensemble — used later to measure
        # whether the LLM's reasoning actually cited the XAI evidence.
        _tops = [v['shap_values'][0][0] for v in xai_results.values()
                 if v.get('shap_ok') and v.get('shap_values')]
        xai_top_feature = Counter(_tops).most_common(1)[0][0] if _tops else ""
        self._log(f"  CONSENSUS: {votes.get('Malicious',0)} Attack, "
                  f"{votes.get('Normal',0)} Normal | XAI {t_xai_total:.1f}s"
                  f" | top SHAP driver: {xai_top_feature or '—'}")

        # STEP 4: LTM
        ltm_cases = []
        if self.use_memory and self.ltm:
            self._log(f"\n[Step 4] Long-Term Memory Retrieval")
            obs = (f"requested={requested_kwh:.3f}kWh, delivered={delivered_kwh:.3f}kWh, "
                   f"duration={duration_hours:.2f}h")
            ltm_cases = self.ltm.retrieve_similar(obs, timestamp, top_k=5)
            self._log(f"  Found {len(ltm_cases)} similar cases")
        else:
            self._log(f"\n[Step 4] LTM DISABLED")

        # STEP 5: RAG
        knowledge_text, knowledge_sources = "", []
        if self.use_knowledge and self.kb:
            self._log(f"\n[Step 5] RAG Knowledge Base Query")
            query = ("energy theft"    if delivery_ratio > 1.3 else
                     "phantom charging" if delivery_ratio < 0.7 else
                     "normal charging")
            knowledge_text, knowledge_sources = self.kb.query(query, top_k=2)
            self._log(f"  Query: '{query}' | {len(knowledge_text)} chars retrieved")
        else:
            self._log(f"\n[Step 5] RAG DISABLED")

        # STEP 6: TWO-STAGE LLM (bounded, deterministic)
        self._log(f"\n[Step 6] LLM Stage 1 — Analysis (with SHAP/LIME context)")
        sys_prompt    = self._get_system_prompt()
        stage1_prompt = self._build_stage1_prompt(
            data_dict, all_predictions, xai_results,
            requested_kwh, delivered_kwh, duration_hours,
            knowledge_text, ltm_cases)

        p1_chars = len(sys_prompt) + len(stage1_prompt)
        p1_words = len(sys_prompt.split()) + len(stage1_prompt.split())
        self._log(f"  Stage 1 prompt: {p1_chars} chars ({p1_words} words)")
        # V23-style full prompt echo — the SHAP/LIME evidence handed to the LLM
        # is a core contribution and must be visible for every session.
        self._log(f"\n  >>> STAGE 1 PROMPT (evidence incl. per-model SHAP/LIME) <<<")
        self._log(f"  {'-'*80}")
        for line in stage1_prompt.split('\n'):
            self._log(f"  > {line}")
        self._log(f"  {'-'*80}")

        t1 = time.time()
        stage1_response = self.llm_client.generate_with_system(
            system_prompt=sys_prompt, user_prompt=stage1_prompt,
            temperature=self.temperature, max_tokens=STAGE1_MAX_TOKENS)
        latency1 = time.time() - t1
        llm_error = stage1_response.startswith(LLM_ERROR_PREFIX)
        self._log(f"\n  >>> STAGE 1 RESPONSE ({latency1:.2f}s) <<<")
        self._log(f"  {'-'*80}")
        for line in stage1_response.split('\n'):
            self._log(f"  | {line}")
        self._log(f"  {'-'*80}")

        self._log(f"\n[Step 6b] LLM Stage 2 — Final Decision")
        t2 = time.time()
        stage2_response = self.llm_client.generate_two_turn(
            system_prompt   = sys_prompt,
            first_user      = stage1_prompt,
            first_assistant = stage1_response,
            second_user     = self.STAGE2_USER,
            temperature     = self.temperature, max_tokens=STAGE2_MAX_TOKENS)
        llm_error = llm_error or stage2_response.startswith(LLM_ERROR_PREFIX)

        # Verdict repair (V26): only when no verdict can be extracted at all.
        # Recovers sessions V23 would have thrown away as ABSTAIN.
        n_repair = 0
        repair_used = False
        # Never spend a repair call on a session whose LLM never answered: the
        # repair would be asked to summarise an error string, and a lucky guess
        # from it would be recorded as a model decision.
        if extract_verdict(stage2_response) is None and not llm_error:
            n_repair = 1
            self._log(f"  [REPAIR] No verdict found — requesting the final line explicitly")
            # MINIMAL repair. The earlier version replayed the whole ~10k-char
            # evidence prompt plus both responses, which simply invited a
            # verbose model to ramble again -- GLM averaged 5,637 words per
            # session and still failed to emit a verdict in 18 of 50 sessions.
            # This call carries only the analyst's own conclusion and the raw
            # physical facts, and demands a single word.
            tail = (stage2_response or stage1_response or "")[-900:]
            repair = self.llm_client.generate_with_system(
                system_prompt=("You output exactly one word and nothing else. "
                               "No reasoning, no punctuation, no explanation."),
                user_prompt=(
                    f"An analyst reviewed an EV charging session and wrote:\n\n"
                    f"{tail}\n\n"
                    f"Measured facts: requested {requested_kwh:.3f} kWh, "
                    f"delivered {delivered_kwh:.3f} kWh "
                    f"(delivered/requested = {delivery_ratio:.3f}).\n\n"
                    f"Answer with ONE word only — Attack or Normal:"),
                temperature=0.0, max_tokens=self._repair_budget(),
                force_no_think=True)
            if repair and not repair.startswith(LLM_ERROR_PREFIX):
                stage2_response = stage2_response + "\nPrediction: " + repair.strip()
                repair_used = extract_verdict(stage2_response) is not None
                self._log(f"  [REPAIR] Recovered: {repair.strip()[:60]}")

        # Self-consistency (V26 enhancement, pipeline unchanged): for AMBIGUOUS
        # sessions only, re-ask Stage 2 a few times and take the majority
        # verdict. Applied selectively so cost stays bounded — easy cases are
        # decided once, exactly as in V23.
        n_consistency = 0
        if (self.self_consistency_k > 1 and not llm_error
                and difficulty_zone in self.AMBIGUOUS_ZONES):
            votes_sc = [extract_verdict(stage2_response)]
            for _ in range(self.self_consistency_k - 1):
                extra = self.llm_client.generate_two_turn(
                    system_prompt   = sys_prompt,
                    first_user      = stage1_prompt,
                    first_assistant = stage1_response,
                    second_user     = self.STAGE2_USER,
                    temperature     = 0.7)   # diversity for the vote
                votes_sc.append(extract_verdict(extra))
                n_consistency += 1
            valid = [v for v in votes_sc if v]
            if valid:
                winner = Counter(valid).most_common(1)[0][0]
                self._log(f"  [SELF-CONSISTENCY] zone={difficulty_zone} "
                          f"votes={valid} -> {winner}")
                if extract_verdict(stage2_response) != winner:
                    stage2_response += f"\nPrediction: " \
                                       f"{'Attack' if winner == 'Malicious' else 'Normal'}"

        latency2  = time.time() - t2
        total_lat = latency1 + latency2
        r2_words  = len(stage2_response.split())
        wps       = round(r2_words / total_lat, 1) if total_lat > 0 else 0
        self._log(f"\n  >>> STAGE 2 PROMPT <<<")
        self._log(f"  {'-'*80}")
        for line in self.STAGE2_USER.split('\n'):
            self._log(f"  > {line}")
        self._log(f"  {'-'*80}")
        self._log(f"\n  >>> STAGE 2 RESPONSE ({latency2:.2f}s) <<<")
        self._log(f"  {'-'*80}")
        for line in stage2_response.split('\n'):
            self._log(f"  | {line}")
        self._log(f"  {'-'*80}")

        complexity = {
            "prompt_chars":     p1_chars + len(self.STAGE2_USER),
            "prompt_words":     p1_words + len(self.STAGE2_USER.split()),
            "response_chars":   len(stage1_response) + len(stage2_response),
            "response_words":   len(stage1_response.split()) + r2_words,
            "llm_latency_sec":  round(total_lat, 2),
            "stage1_latency":   round(latency1, 2),
            "stage2_latency":   round(latency2, 2),
            "tokens_per_sec":   wps,
            "xai_sec":          round(t_xai_total, 2),
            "xai_models_explained": sum(1 for r in xai_results.values()
                                        if r.get('shap_ok') or r.get('lime_ok'))
        }

        # STEP 7: PARSE & FINAL DECISION
        self._log(f"\n[Step 7] Final Decision")
        final_result = self._parse_llm_response(
            stage2_response, stage1_response, all_predictions,
            delivery_ratio, duration_hours, line_number, votes,
            ltm_cases, knowledge_sources, complexity, difficulty_zone,
            xai_top_feature, repair_used=repair_used, llm_error=llm_error)

        session_time = time.time() - t0
        final_result['complexity']['session_total_sec'] = round(session_time, 2)

        # V23-style detailed final block (explainability surfaced per session)
        self._log(f"  CLASSIFICATION:  {final_result['predicted_label']}")
        self._log(f"  Confidence:      {final_result['confidence']:.3f} "
                  f"(LLM={final_result['llm_confidence']}, "
                  f"ML_agree={final_result['ml_agreement']:.3f})")
        self._log(f"  Attack Type:     {final_result['attack_type']}")
        self._log(f"  LLM overrode ML: {'YES' if final_result['llm_overrode_ml'] else 'NO'} "
                  f"(ML majority={final_result['ml_majority']})")
        self._log(f"  Difficulty Zone: {difficulty_zone}")
        self._log(f"  Time: {session_time:.2f}s total ({total_lat:.2f}s LLM, "
                  f"{complexity['xai_models_explained']}/7 models explained by XAI)")
        if final_result.get('llm_xai_assessment'):
            self._log(f"\n  LLM XAI ASSESSMENT: {final_result['llm_xai_assessment'][:200]}...")
        if final_result.get('llm_analysis'):
            self._log(f"  PHYSICAL INTERP:    {final_result['llm_analysis'][:200]}...")
        if final_result.get('llm_conflicts'):
            self._log(f"  CONFLICTS:          {final_result['llm_conflicts'][:150]}...")
        self._log(f"\n{'='*100}\nSESSION {line_number} COMPLETE\n{'='*100}\n")

        # Persist the complete interaction (prompts + SHAP/LIME evidence +
        # responses + verdict) so explainability survives parallel/quiet runs.
        if self.transcript_dir:
            try:
                transcript = (
                    f"SESSION {line_number} | model={self.llm_client.model_name} | "
                    f"scenario={sc} | tag={self.scenario_tag}\n"
                    f"{'='*100}\n\n"
                    f"[SYSTEM PROMPT]\n{sys_prompt}\n\n"
                    f"{'='*100}\n"
                    f"[STAGE 1 PROMPT — evidence incl. per-model SHAP/LIME]\n"
                    f"{stage1_prompt}\n\n"
                    f"{'='*100}\n"
                    f"[STAGE 1 RESPONSE ({latency1:.2f}s)]\n{stage1_response}\n\n"
                    f"{'='*100}\n"
                    f"[STAGE 2 PROMPT]\n{self.STAGE2_USER}\n\n"
                    f"{'='*100}\n"
                    f"[STAGE 2 RESPONSE ({latency2:.2f}s)]\n{stage2_response}\n\n"
                    f"{'='*100}\n"
                    f"[FINAL DECISION]\n"
                    f"  predicted_label = {final_result['predicted_label']}\n"
                    f"  confidence      = {final_result['confidence']}\n"
                    f"  attack_type     = {final_result['attack_type']}\n"
                    f"  llm_overrode_ml = {final_result['llm_overrode_ml']}"
                    f" (ML majority={final_result['ml_majority']})\n"
                    f"  used_fallback   = {final_result['used_fallback']}\n"
                    f"  difficulty_zone = {difficulty_zone}\n"
                    f"  session_time    = {session_time:.2f}s\n"
                )
                with open(os.path.join(self.transcript_dir,
                                       f"session_{line_number}.txt"),
                          'w', encoding='utf-8') as f:
                    f.write(transcript)
            except Exception:
                pass

        # Emit this session's complete V23-style trace as one contiguous block
        self._flush()

        return {
            'status':            'success',
            'request_index':     line_number,
            'result':            final_result,
            'model_predictions': all_predictions,
            'xai_results':       {k: {'shap_ok': v['shap_ok'], 'lime_ok': v['lime_ok'],
                                      'shap_top': v['shap_values'][0][0] if v.get('shap_values') else None,
                                      'lime_top': v['lime_values'][0][0] if v.get('lime_values') else None}
                                  for k, v in xai_results.items()},
            'raw_data':          data_dict,
            'short_term_memory': self.short_term_memory,
            'ltm_cases_used':    len(ltm_cases),
            'rag_sources_used':  len(knowledge_sources),
            'majority_vote':     votes.most_common(1)[0][0] if votes else 'Normal',
            # Transport health at the moment this session finished, so a run can
            # be audited for whether the calls completed, not only for its score.
            'llm_health':        (self.llm_client.health()
                                  if hasattr(self.llm_client, 'health') else {})
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
# MODEL TRAINING — identical to V23 (intentionally unchanged)
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
