# selftest_v31.py
"""
Offline correctness gate for V31. No GPU, no Ollama, no dataset, no models.

Why this exists
---------------
Every wasted multi-day run in this project failed for a reason that was visible
in the code and would have been caught in seconds:

  V25   a global 900-token cap truncated Qwen's reasoning
  V26   a section parser required a colon, so XAI references reported 0/50
  V29   a patch renamed a helper at the call sites but not at the definition,
        producing a NameError that py_compile cannot see
  V29.1 GLM's Stage 2 ran on a 128-token floor with its thinking channel on,
        so its answer channel was empty on every session
  V29.2 a timed-out call became the string "Error: ...", no verdict could be
        parsed from it, and the session was silently scored as an ML fallback

None of those needed a GPU to detect. This script asserts the behaviour of the
decision path directly, with a scripted fake model, and runs in under a second.

    python selftest_v31.py

Exit code 0 means the decision path behaves as specified. It does NOT mean the
models are healthy -- that is what healthcheck_v31.py measures. Run this first
(seconds), the healthcheck second (minutes), the study last (hours).
"""

import sys
import numpy as np

import ev_ids_agent_v6_triple_llm_v31 as A

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append(f"{name}\n      expected: {want!r}\n      got     : {got!r}")


# ── 1. VERDICT EXTRACTION ────────────────────────────────────────────────────
# V23 matched only the bare substring "prediction: attack", so every one of
# these formats was thrown away as an abstention. GLM lost 24 of 50 sessions
# to this alone.
VERDICT_CASES = [
    ("Prediction: Attack",                         'Malicious'),
    ("**Prediction:** Attack",                     'Malicious'),
    ("**Prediction**: Normal",                     'Normal'),
    ("Final Prediction: Malicious",                'Malicious'),
    ("### Classification: Normal",                 'Normal'),
    ("Verdict - Attack",                           'Malicious'),
    ("Answer: benign",                             'Normal'),
    ("<think>maybe normal</think>\nAttack",        'Malicious'),
    ("blah\n\nNormal.",                            'Normal'),
    ("Prediction: Normal\nPrediction: Attack",     'Malicious'),   # last wins
    ("no verdict anywhere in this text",           None),
    ("",                                           None),
]

# ── 2. SECTION EXTRACTION ────────────────────────────────────────────────────
# The colon-required parser reported "XAI referenced 0/50" while XAI
# faithfulness on the same responses was 1.00 -- a contradiction that was a
# parsing artefact, not model behaviour.
SECTION_CASES = [
    ("SHAP_LIME_ASSESSMENT: the delivered energy dominates\nDOMAIN_MATCH: yes",
     "the delivered energy dominates"),
    ("### SHAP_LIME_ASSESSMENT\nthe delivered energy dominates\n### DOMAIN_MATCH\nyes",
     "the delivered energy dominates"),
    ("**SHAP_LIME_ASSESSMENT**\nthe delivered energy dominates\n**DOMAIN_MATCH**\nyes",
     "the delivered energy dominates"),
    ("2. SHAP_LIME_ASSESSMENT: the delivered energy dominates\n3. DOMAIN_MATCH: yes",
     "the delivered energy dominates"),
    ("nothing relevant here", ""),
]


# ── 3. A SCRIPTED MODEL ──────────────────────────────────────────────────────
class ScriptedClient:
    """Returns queued strings. Stands in for a real model at zero cost."""

    def __init__(self, script):
        self.model_name = "scripted:test"
        self.script = list(script)
        self.seen = []

    def _next(self):
        return self.script.pop(0) if self.script else "Prediction: Normal"

    def generate_with_system(self, system_prompt, user_prompt, **kw):
        self.seen.append(('single', kw))
        return self._next()

    def generate_two_turn(self, system_prompt, first_user, first_assistant,
                          second_user, **kw):
        self.seen.append(('two_turn', kw))
        return self._next()

    def health(self):
        return {'calls': len(self.seen), 'timeouts': 0, 'errors': 0,
                'budget_clamps': 0, 'granted_ctx': A.NUM_CTX}


class StubAgent:
    """Minimal object exposing what _parse_llm_response reads off self."""

    use_knowledge = use_memory = use_xai = False

    def __init__(self):
        self.xai_store = type('S', (), {'explainer': None})()
        self.logged = []

    def _log(self, msg):
        self.logged.append(msg)


def parse(stage2, votes, *, repair_used=False, llm_error=False, ratio=1.0):
    from collections import Counter
    stub = StubAgent()
    preds = {m: {'prediction': p, 'confidence': 0.6}
             for m, p in votes.items()}
    return A.EVIDSAgentV6TripleLLMV31._parse_llm_response(
        stub, stage2, "stage 1 text", preds, ratio, 2.0, 1,
        Counter(votes.values()), [], [], {}, "EASY_NORMAL",
        repair_used=repair_used, llm_error=llm_error)


def _paper_metrics_checks():
    import inspect, json, math, os, tempfile
    import paper_metrics_v31 as P
    import simple_run_v6_triple_llm_v31 as R

    # ── core metrics, hand-computed: TP=3 TN=4 FP=1 FN=2 ────────────────────
    m = P.binary_metrics([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
    want = {'accuracy': 0.7, 'precision': 0.75, 'recall': 0.6,
            'specificity': 0.8, 'far': 0.2, 'fnr': 0.4,
            'balanced_accuracy': 0.7, 'f1': 2 * 0.75 * 0.6 / 1.35,
            'mcc': (3 * 4 - 1 * 2) / math.sqrt(4 * 5 * 5 * 6)}
    for k, v in want.items():
        check(f"paper metric {k} matches hand calculation", round(m[k], 12), round(v, 12))
    check("macro-F1 is the mean of the two class F1s",
          round(m['macro_f1'], 12), round((m['f1'] + m['f1_normal']) / 2, 12))
    check("a constant-Normal classifier has MCC 0, not an error",
          P.binary_metrics([1, 0, 1, 0], [0, 0, 0, 0])['mcc'], 0.0)
    check("ROC-AUC of a perfect ranking is 1", P.roc_auc([0, 0, 1, 1], [.1, .2, .8, .9]), 1.0)
    check("ROC-AUC counts ties as one half", P.roc_auc([0, 1], [.5, .5]), 0.5)
    check("ROC-AUC is undefined with one class", math.isnan(P.roc_auc([1, 1], [.2, .8])), True)
    check("average precision of a perfect ranking is 1",
          P.average_precision([0, 1, 1], [.1, .8, .9]), 1.0)
    check("Fleiss' kappa is 1 under perfect agreement",
          P.fleiss_kappa([[1, 0, 1, 0]] * 3), 1.0)
    rc = P.risk_coverage([0.9, 0.8, 0.1], [True, True, False])
    check("risk-coverage keeps the confident correct sessions first",
          rc['risk'], [0.0, 0.0, 1 / 3])
    check("Brier of perfectly confident correct verdicts is 0",
          P.brier_confidence([1.0, 1.0], [True, True]), 0.0)

    # ── sensitivity option (a): the paper's three instructions ──────────────
    def srow(pred, conf):
        return {'predicted': pred, 'ground_truth': pred,
                'result': {'llm_confidence': conf, 'confidence': 0.8 if conf == 'high' else 0.5}}
    rows = [srow('Malicious', 'high'), srow('Malicious', 'low'),
            srow('Normal', 'high'), srow('Normal', 'low')]
    check("aggressive alerts on every Attack and on unsure Normals",
          P.apply_sensitivity(rows, 'aggressive'), [1, 1, 0, 1])
    check("balanced is the deployed verdict", P.apply_sensitivity(rows, 'balanced'), [1, 1, 0, 0])
    check("conservative alerts only on sure Attacks",
          P.apply_sensitivity(rows, 'conservative'), [1, 0, 0, 0])
    check("the composite source uses the paper's 0.7 threshold",
          P.apply_sensitivity(rows, 'conservative', 'composite'), [1, 0, 0, 0])

    # ── sensitivity option (b): the paper's A.4 sentence, nothing else ──────
    base = A.EVIDSAgentV6TripleLLMV31.STAGE2_USER
    for level, detail in R.SENSITIVITY_DETAILS.items():
        q = R.sensitivity_stage2(level)
        check(f"{level}: the paper's sentence is added",
              f"Summarize the classification with {level} sensitivity, {detail}." in q, True)
        check(f"{level}: the original Stage 2 question is otherwise intact",
              q.replace(f" Summarize the classification with {level} sensitivity, "
                        f"{detail}.", ""), base)
    check("building a sensitivity prompt does not alter the class attribute",
          A.EVIDSAgentV6TripleLLMV31.STAGE2_USER, base)

    # ── Table 9: the lambda sweep must reproduce the deployed retrieval ─────
    src = inspect.getsource(A.LongTermMemory.retrieve_similar)
    check("the agent still weights memory as 0.3*recency + 0.7*similarity",
          '0.3 * recency + 0.7 * cos_sim' in src, True)
    check("the runner records the agent's real lambdas", R.AGENT_LTM_LAMBDA, (0.3, 0.7))

    class FakeEncoder:
        """Deterministic stand-in for sentence-transformers."""
        def encode(self, texts):
            import numpy as np
            out = []
            for t in texts:
                v = np.zeros(16)
                for i, ch in enumerate(t):
                    v[(ord(ch) * (i + 1)) % 16] += 1.0
                out.append(v)
            return np.array(out)

    with tempfile.TemporaryDirectory() as td:
        ltm = A.LongTermMemory(td)
        ltm.embedding_model = FakeEncoder()
        texts = [f"requested={10 + i % 7:.3f}kWh, delivered={9 + (i * 3) % 11:.3f}kWh"
                 for i in range(25)]
        ltm.memories = [{'timestamp': 1000.0 + 37 * i, 'observations': t, 'id': i}
                        for i, t in enumerate(texts)]
        query, now = "requested=12.000kWh, delivered=14.500kWh", 2500.0
        deployed = [m['id'] for m in ltm.retrieve_similar(query, now, top_k=5)]
        swept = [m['id'] for m in R.weighted_retriever(ltm, 0.3, 0.7)(query, now, top_k=5)]
        check("lambda sweep at (0.3, 0.7) returns exactly the deployed retrieval",
              swept, deployed)
        recency_only = [m['id'] for m in R.weighted_retriever(ltm, 1.0, 0.0)(query, now, top_k=5)]
        check("lambda1=1 retrieves the most recent memories",
              recency_only, [24, 23, 22, 21, 20])
        check("the lambdas actually change what is retrieved",
              recency_only != deployed, True)

        class Agentish:
            pass
        ag = Agentish()
        ag.ltm = A.LongTermMemory(os.path.join(td, 'dead'))
        ag.ltm.embedding_model = None
        ag.ltm.memories = list(ltm.memories)
        live, why = R.ltm_is_live(ag)
        check("a memory with no embedding model is reported as not live", live, False)
        ag.ltm = ltm
        check("a working memory is reported as live", R.ltm_is_live(ag)[0], True)

    # ── the runner saves before it reports ─────────────────────────────────
    main_src = inspect.getsource(R.main)
    check("results are saved before the comparison is printed",
          main_src.index('json.dump(make_json_serializable(dict(')
          < main_src.index('print_triple_comparison('), True)
    check("a failing comparison cannot discard the run",
          'except Exception' in main_src[main_src.index('print_triple_comparison('):
                                         main_src.index('print_triple_comparison(') + 400],
          True)
    check("sweeps run unattended (no Enter prompt between settings)",
          'confirm=False' in inspect.getsource(R.run_paper_sweep), True)

    # ── grouping and end-to-end output on a tiny synthetic run ─────────────
    ML = list(P.ML_ORDER)

    def row(i, truth, pred):
        return {'index': i, 'ground_truth': truth, 'predicted': pred,
                'correct': truth == pred, 'majority_vote': 'Normal',
                'model_predictions': {mm: {'prediction': 'Normal', 'confidence': 0.6}
                                      for mm in ML},
                'xai_results': {mm: {'shap_ok': True, 'lime_ok': True,
                                     'shap_top': 'Energy delivered',
                                     'lime_top': 'Energy delivered'} for mm in ML},
                'result': {'confidence': 0.8, 'llm_confidence': 'high',
                           'verdict_source': 'stated', 'llm_error': False,
                           'difficulty_zone': 'EASY_ATTACK' if truth == 'Malicious'
                                              else 'EASY_NORMAL',
                           'delivery_ratio': 1.8 if truth == 'Malicious' else 1.0,
                           'xai_top_feature': 'Energy delivered',
                           'llm_xai_assessment': 'Energy delivered dominates',
                           'ltm_cases_referenced': 2, 'rag_sources_used': 0,
                           'complexity': {'llm_latency_sec': 10.0}}}

    truths = ['Malicious' if i % 2 else 'Normal' for i in range(12)]
    with tempfile.TemporaryDirectory() as td:
        def dump(name, **meta):
            d = {'use_knowledge': False, 'use_memory': True, 'use_xai': True,
                 'sample_seed': meta.pop('seed', 42),
                 'llms': {'llama': 'llama3:latest', 'qwen': 'qwen3.5:latest'},
                 'results_llama': [row(i, t, t) for i, t in enumerate(truths)],
                 'results_qwen': [row(i, t, t if i % 5 else 'Normal')
                                  for i, t in enumerate(truths)],
                 'results_glm': []}
            d.update(meta)
            p = os.path.join(td, name)
            with open(p, 'w') as f:
                json.dump(d, f)
            return p

        files = [dump('std42.json'), dump('std7.json', seed=7),
                 dump('lam.json', ltm_lambda=[0.3, 0.7], ltm_lambda_sweep=True)]
        runs, _ = P.load_runs(files)
        check("a lambda-sweep cell at default weights is not a standard run",
              [r.is_standard for r in runs], [True, True, False])
        seed_t = P.table_seed_robustness(runs)
        check("the seed table counts only the standard runs",
              {row_[2] for row_ in seed_t.rows}, {2})

        out = P.generate_paper_outputs(files, os.path.join(td, 'out'), n_boot=50,
                                       verbose=False)
        produced = set()
        for root, _, fs in os.walk(out):
            produced |= set(fs)
        for name in ('README.md', 'table6_per_class_fscore.csv',
                     'table6_per_class_fscore.tex', 'table_detection_metrics.csv',
                     'table_statistics.csv', 'table_calibration.csv',
                     'table_provenance_and_retrieval.csv', 'table_xai_agreement.csv',
                     'figure8_confidence_by_zone.png', 'figure8_confidence_by_correct.pdf',
                     'figure_roc_pr.png', 'figure_calibration.png', 'error_analysis.csv',
                     'table_seed_robustness.csv'):
            check(f"paper output {name} is written", name in produced, True)
        with open(os.path.join(out, 'README.md'), encoding='utf-8') as f:
            readme = f.read()
        check("the README states that Figure 8 is not out-of-distribution",
              'not an out-of-distribution evaluation' in readme, True)


def main():
    # 1 ── verdict formats
    for text, want in VERDICT_CASES:
        check(f"extract_verdict({text[:34]!r})", A.extract_verdict(text), want)

    # 2 ── section formats
    for text, want in SECTION_CASES:
        got = A.extract_section("SHAP_LIME_ASSESSMENT", text,
                                ["DOMAIN_MATCH", "HISTORICAL"])
        check(f"extract_section({text[:30]!r})", got, want)

    # 3 ── provenance. The whole point of V30: these four outcomes must stay
    #      distinguishable, because scoring an error as a decision is what made
    #      two identical runs disagree.
    all_normal = {'RF': 'Normal', 'DT': 'Normal', 'KNN': 'Normal'}
    check("verdict_source stated",
          parse("Prediction: Attack", all_normal)['verdict_source'], 'stated')
    check("verdict_source repaired",
          parse("Prediction: Attack", all_normal,
                repair_used=True)['verdict_source'], 'repaired')
    check("verdict_source ml_fallback",
          parse("I cannot decide.", all_normal)['verdict_source'], 'ml_fallback')
    check("verdict_source llm_error",
          parse("Error: timeout", all_normal,
                llm_error=True)['verdict_source'], 'llm_error')
    check("llm_error flag is carried",
          parse("Error: timeout", all_normal, llm_error=True)['llm_error'], True)
    check("an error must NOT be reported as a stated verdict",
          parse("Error: timeout", all_normal, llm_error=True)['llm_parsed'], False)

    # 4 ── the LLM must be able to override the ML majority, since the ML
    #      models are deliberately weak and that override is the contribution.
    check("LLM overrides a unanimous Normal vote",
          parse("Prediction: Attack", all_normal)['llm_overrode_ml'], True)
    check("fallback follows the ML majority",
          parse("no verdict", all_normal)['predicted_label'], 'Normal')

    # 5 ── generation budget. This is the GLM bug in one assertion: with the
    #      window Ollama actually granted, Stage 2 collapses to the floor.
    fit = A.OllamaClient._fit_budget
    dummy = type('D', (), {})()
    long_prompt = [{'content': 'x' * 10900}, {'content': 'x' * 8000}]
    check("Stage 2 hits the floor at the granted 4096 window",
          fit(dummy, long_prompt, 4096, 1536), 128)
    # 18,900 prompt chars is ~4,725 tokens, so an 8,192 window leaves ~3,339 --
    # more than GLM's 1,536 request, which is therefore granted in full. The
    # point of the pair is that the SAME prompt is starved at 4,096 and
    # comfortable at 8,192; the window, not the request, was the constraint.
    check("Stage 2 gets its full request at 8192",
          fit(dummy, long_prompt, 8192, 1536), 1536)
    check("a short prompt is never clamped",
          fit(dummy, [{'content': 'x' * 400}], 8192, 1536), 1536)
    check("the clamp still bites when the window is genuinely small",
          fit(dummy, long_prompt, 6000, 1536), 1147)
    check("THINK_MIN_BUDGET is above the floor",
          A.THINK_MIN_BUDGET > 128, True)
    # num_ctx is the lever that makes a TARGET model fit. Substituting a model
    # changes what the study measures; shrinking the window does not. But the
    # window has a real floor, because Stage 2 replays everything Stage 1 saw.
    floor = A.min_viable_num_ctx()
    check("the context floor leaves room for an answer",
          fit(dummy, [{'content': 'x' * 8600}, {'content': 'x' * 3928},
                      {'content': 'x' * 523}, {'content': 'x' * 8192}],
              floor, 1536) >= A.THINK_MIN_BUDGET, True)
    check("the default window is at or above the floor", A.NUM_CTX >= floor, True)
    check("the floor is a whole number of 1024-token blocks", floor % 1024, 0)
    check("Stage 2 defaults to the answer channel",
          A.STAGE2_ANSWER_CHANNEL_ONLY, True)

    # 5b ── per-model generation policy. MODEL_GEN_POLICY is substring matched
    #       with FIRST MATCH WINS, so a specific key placed after a general one
    #       is silently dead. glm4 and qwen2.5 reject think=True; resolving them
    #       to the generic 'glm'/'qwen' entries costs a refused request and a
    #       retry on every single call.
    check("glm4 does not request the reasoning channel",
          A.gen_policy_for('glm4:latest')['thinking'], False)
    check("glm-4.7-flash still requests it",
          A.gen_policy_for('glm-4.7-flash:latest')['thinking'], True)
    check("qwen2.5 does not request the reasoning channel",
          A.gen_policy_for('qwen2.5:7b')['thinking'], False)
    check("qwen3.5 still requests it",
          A.gen_policy_for('qwen3.5:latest')['thinking'], True)
    for specific, general in (('glm4', 'glm'), ('qwen2.5', 'qwen')):
        keys = list(A.MODEL_GEN_POLICY)
        check(f"{specific!r} is matched before {general!r}",
              keys.index(specific) < keys.index(general), True)

    # A specific key was raised to 3072 while its general sibling was left at
    # 1536, so glm-4.7-flash kept the old budget and kept truncating one line
    # before the verdict. Nothing flagged the divergence. Every model must have
    # room for its reasoning channel AND the five-section Stage 2 format.
    for name in ('glm-4.7-flash:latest', 'glm4:latest', 'qwen3.5:latest',
                 'qwen2.5:7b', 'llama3:latest'):
        pol = A.gen_policy_for(name)
        check(f"{name} has a workable Stage 2 budget",
              pol['num_predict'] >= 2048, True)
        check(f"{name} repeat_penalty is not aggressive enough to corrupt tokens",
              pol['repeat_penalty'] <= 1.15, True)
    # The repair read-out must outlast a model that reasons anyway.
    check("the repair budget survives an unrequested reasoning channel",
          A.EVIDSAgentV6TripleLLMV31._repair_budget(None) >= 256, True)
    # A repair that returns reasoning instead of one word must still be read.
    # Splicing the whole blob after "Prediction: " produced an unmatchable line,
    # so a repair that HAD concluded was recorded as a failure.
    for blob, want in [
        ("Attack", 'Malicious'),
        ("  Normal  ", 'Normal'),
        ("The ratio is 1.78, far above 1.0.\nPrediction: Attack", 'Malicious'),
        ("Given the delivery ratio, this is an attack.\nAttack", 'Malicious'),
        ("mumbling with no conclusion at all", None),
    ]:
        got = A.extract_verdict(blob) or A.extract_verdict(
            "Prediction: " + blob.strip())
        check(f"repair blob {blob[:34]!r} reads as {want}", got, want)

    # 6 ── SHAP sign convention. The prompt tells the model "+ = toward
    #      Attack", so attributions must be taken against the Attack class for
    #      every model. Indexing by the PREDICTED class instead inverted the
    #      sign on every Normal-predicting model, which was most of the prompt.
    raw_two_class = [np.array([0.3, -0.1]), np.array([-0.3, 0.1])]
    attack_side = np.array(raw_two_class[A.ATTACK_CLASS]).flatten()
    check("SHAP is read against the Attack class",
          float(attack_side[0]), -0.3)

    # 7 ── every call site the runner depends on still exists. A rename that
    #      updates callers but not definitions compiles cleanly and dies at run
    #      time; that has cost this project a full run before.
    for attr in ('detect', '_parse_llm_response', '_build_stage1_prompt',
                 '_get_system_prompt', '_repair_budget', 'seed_ltm_from_training'):
        check(f"agent.{attr} exists",
              hasattr(A.EVIDSAgentV6TripleLLMV31, attr), True)
    for fn in ('make_llm_client', 'unload_ollama_model', 'extract_verdict',
               'extract_section', 'gen_policy_for', 'auto_train_models_v6',
               'get_column_mapping', 'classify_difficulty_zone'):
        check(f"module.{fn} exists", hasattr(A, fn), True)
    # Both clients must answer the questions the runner asks of them.
    # _guarded and _fit_budget are Ollama-specific: vLLM enforces its own
    # window at launch and its transport already raises on timeout.
    for attr in ('refresh_actual_ctx', 'health', 'generate_with_system',
                 'generate_two_turn'):
        check(f"OllamaClient.{attr} exists",
              hasattr(A.OllamaClient, attr), True)
        check(f"VLLMClient.{attr} exists",
              hasattr(A.VLLMClient, attr), True)
    for attr in ('_guarded', '_fit_budget'):
        check(f"OllamaClient.{attr} exists",
              hasattr(A.OllamaClient, attr), True)

    # 8 ── the healthcheck probes must be internally consistent. The first
    #      version used ONE fixed evidence block for all six probes, so a normal
    #      session at ratio 1.004 was shown SHAP values pointing toward Attack
    #      and an attack at 1.780 was shown a vote tally of 0 Attack. That
    #      measured contradiction-resolution, not the task, and it produced a
    #      misleading accuracy column for Llama and GLM.
    try:
        import healthcheck_v31 as H

        def tally(ratio):
            line = [l for l in H.evidence_for(ratio).split('\n')
                    if 'vote tally' in l][0]
            return int(line.split(':')[1].split('Attack')[0].strip())

        def dlv_shap(ratio):
            line = [l for l in H.evidence_for(ratio).split('\n')
                    if l.strip().startswith('Random Forest')][1]
            return float(line.split('kWhDelivered')[1].split()[0])

        check("normal probe shows no Attack votes", tally(1.004), 0)
        check("attack probe shows Attack votes", tally(1.780) >= 2, True)
        # EVERY block must track the session, not just the SHAP one. Fixing the
        # SHAP block while leaving RAG and memory attack-flavoured still shows a
        # ratio-1.004 session two paragraphs about energy theft.
        check("normal probe retrieves normal-charging knowledge",
              'normal charging' in H.rag_for(1.004), True)
        check("attack probe retrieves energy-theft knowledge",
              'energy theft' in H.rag_for(1.780), True)
        check("normal probe recalls no attack precedents",
              H.ltm_for(1.004).count('Classified Attack'), 0)
        check("attack probe recalls attack precedents",
              H.ltm_for(1.780).count('Classified Attack') >= 2, True)
        check("phantom-charging probe retrieves attack knowledge",
              'Phantom charging' in H.rag_for(0.26), True)
        check("Stage 1 probe does not ask for a prediction",
              'prediction yet' in H.probe_prompt(12.0, 18.0), True)
        # Cloud models run remotely, so local VRAM checks are meaningless for
        # them and must be skipped rather than reported as a fault. They are the
        # only way to run a model whose weights exceed every local card.
        check("a :cloud model is recognised", H.is_cloud_model('glm-5:cloud'), True)
        check("a zero-size model is recognised as cloud",
              H.is_cloud_model('x:latest', 0), True)
        check("a local model is not treated as cloud",
              H.is_cloud_model('glm-4.7-flash:latest', 19e9), False)

        # The gate must judge OUTCOMES, not the residency proxy. These are the
        # real recorded numbers from a 6 GB machine at num_ctx 6144.
        def verdict(**kw):
            base = dict(frac=1.0, total=5.5e9, ctx=A.NUM_CTX, errors=0, n_run=6,
                        stated=6, correct=6, unrecovered=0, repaired=0, clamps=0,
                        projected_h=2.1, n_project=50, cloud=False,
                        transcript_path='t.txt')
            return H.assess(**{**base, **kw})

        v, env, _, cav = verdict(frac=0.76, projected_h=2.1)
        check("76% residency with clean outcomes is a GO", v, 'GO')
        check("...and is reported as a caveat", len(cav), 1)
        check("...with no environment failure", env, [])

        v, env, _, _ = verdict(frac=0.57, errors=2, n_run=2, stated=0, correct=0,
                               projected_h=16.5)
        check("timeouts still FAIL", v, 'FAIL')

        # 0 errors, full context, clean transport, but no verdict survives the
        # repair: that is the MODEL's behaviour, not a broken environment, so it
        # is WEAK rather than FAIL. The run is worth doing and reporting as a
        # documented negative result.
        v, env, behav, _ = verdict(frac=0.22, stated=0, correct=0, unrecovered=6,
                                   projected_h=11.3)
        check("silent model with a clean environment is WEAK, not FAIL", v, 'WEAK')
        check("...and no environment failure is claimed", env, [])
        check("...and the behaviour is recorded", len(behav) >= 1, True)

        v, env, _, cav = verdict(frac=None, total=0, cloud=True)
        check("a cloud model gets no residency caveat", cav, [])
        check("a cloud model is not failed for residency", v, 'GO')

        v, env, _, _ = verdict(ctx=4096)
        check("a short-granted window still FAILs", v, 'FAIL')

        # A SLOW run is not a BROKEN run. A clean Qwen run -- 5/5 correct, no
        # fallbacks, no errors, no timeouts, no clamps, full 8192 context -- was
        # killed at session 5 of 200 for projecting 12 h against an 8 h limit.
        # Sample size is the researcher's decision; only an unusable run is
        # worth abandoning.
        v, env, _, cav = verdict(projected_h=30.0, n_project=200)
        check("a healthy but slow model is a GO", v, 'GO')
        check("...with no environment failure", env, [])
        check("...and the cost reported as a caveat",
              any('cost, not a fault' in x for x in cav), True)
        v, env, _, _ = verdict(projected_h=30.0, errors=3, n_run=6)
        check("slow AND failing still FAILs", v, 'FAIL')
        check("both 1.5 probes agree", tally(1.5) == tally(22.80 / 15.20), True)
        check("SHAP points toward Attack on an attack", dlv_shap(1.78) > 0.1, True)
        check("SHAP is near zero on a normal", abs(dlv_shap(1.004)) < 0.01, True)
        # The prompt must be big enough to reproduce the Stage 2 squeeze.
        p = H.probe_prompt(12.0, 18.0)
        check("probe prompt is realistically sized",
              len(H.SYSTEM) + len(p) > 7000, True)
        check("healthcheck uses the framework's own system prompt",
              H.SYSTEM == A.EVIDSAgentV6TripleLLMV31._get_system_prompt(None), True)
        check("healthcheck uses the framework's own Stage 2 question",
              H.STAGE2_QUESTION == A.EVIDSAgentV6TripleLLMV31.STAGE2_USER, True)
    except Exception as e:
        FAIL.append(f"healthcheck import/probe failed: {e}")

    # 9 ── the runner imports what it says it imports
    try:
        import inspect
        import simple_run_v6_triple_llm_v31 as R
        for fn in ('preflight_gpu_residency', 'check_gpu_residency',
                   'verify_models_installed', 'calculate_metrics',
                   'run_classification', 'main'):
            check(f"runner.{fn} exists", hasattr(R, fn), True)

        # The circuit breaker was originally written into the serial branch
        # only, so a run at workers=8 with a 28% error rate went all 50
        # sessions and cost an hour before reporting itself unpublishable. A
        # guard that covers one code path is not a guard, and nothing but this
        # assertion would have noticed.
        src = inspect.getsource(R.run_classification)
        serial, parallel = src.split('    else:', 1)
        check("circuit breaker guards the serial path",
              '_breaker()' in serial, True)
        check("circuit breaker guards the PARALLEL path",
              '_breaker()' in parallel, True)
        # The breaker must ABORT only on unanswered calls. Aborting on duration
        # discarded a clean 200-session Qwen run at session 5.
        breaker = src[src.index('def _breaker'):src.index('if workers <= 1')]
        abort_lines = [l for l in breaker.splitlines()
                       if l.strip().startswith('return (')]
        check("the breaker's only abort condition is the error rate",
              len(abort_lines), 1)
        check("duration warns instead of aborting",
              'NOT STOPPING' in breaker, True)
        check("the error-rate abort is still present",
              'n_error / done > BREAKER_MAX_ERROR_RATE' in breaker, True)
        check("the parallel path cancels queued work when it trips",
              'cancel' in parallel, True)
        check("llm errors are counted in progress", 'llm_error' in src, True)

        # RAGGED RESULTS. The circuit breaker can stop one model early, so the
        # three result sets need not be the same length. Every cross-model
        # figure used to assume they were, and a run with 200 Llama sessions
        # and 5 Qwen crashed in confusion_matrix with "inconsistent numbers of
        # samples: [200, 5]" -- losing the two COMPLETE models' results too.
        import io, contextlib
        ML = ['Random Forest', 'K-Nearest Neighbors', 'Logistic Regression',
              'MLP', 'Support Vector Classifier', 'Decision Tree',
              'Gradient Boosting']

        def _row(i, truth, pred):
            return {'index': i, 'ground_truth': truth, 'predicted': pred,
                    'correct': truth == pred, 'majority_vote': 'Normal',
                    'model_predictions': {m: {'prediction': 'Normal',
                                              'confidence': 0.6} for m in ML},
                    'result': {'predicted_label': pred, 'confidence': 0.7,
                               'llm_confidence': 'high', 'used_fallback': False,
                               'llm_parsed': True, 'llm_error': False,
                               'verdict_source': 'stated',
                               'difficulty_zone': 'EASY_ATTACK' if truth == 'Malicious'
                                                  else 'EASY_NORMAL',
                               'delivery_ratio': 1.8 if truth == 'Malicious' else 1.0,
                               'xai_top_feature': 'kWhDelivered',
                               'llm_xai_assessment': 'kWhDelivered drives it',
                               'llm_reasoning': 'delivered', 'llm_analysis': 'ratio',
                               'knowledge_used': True, 'memory_used': True,
                               'xai_enabled': True, 'ltm_cases_referenced': 2,
                               'rag_sources_used': 2,
                               'complexity': {'llm_latency_sec': 10.0,
                                              'stage1_latency': 6.0,
                                              'stage2_latency': 4.0, 'xai_sec': 1.0,
                                              'prompt_chars': 9000,
                                              'response_words': 400,
                                              'tokens_per_sec': 2.0,
                                              'session_total_sec': 12.0}}}

        truths = ['Malicious' if i % 3 == 0 else 'Normal' for i in range(40)]
        full_a = [_row(i, truths[i], truths[i] if i % 11 else 'Normal')
                  for i in range(40)]
        full_c = [_row(i, truths[i], truths[i] if i % 7 else 'Normal')
                  for i in range(40)]
        short  = [_row(i, truths[i], truths[i]) for i in range(5)]

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    R.print_triple_comparison(full_a, short, full_c, td)
                ok, err = True, ''
            except Exception as exc:
                ok, err = False, f"{type(exc).__name__}: {exc}"
            out = buf.getvalue()
            check(f"a truncated model does not crash the comparison{' — ' + err if err else ''}",
                  ok, True)
            if ok:
                check("the incomplete run is announced",
                      'INCOMPLETE RUN' in out, True)
                check("each model keeps its own n",
                      '5' in out and '40' in out, True)
                check("paired tests report the shared n",
                      'n=5' in out, True)

            # And the equal-length path must be untouched.
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                R.print_triple_comparison(full_a, list(full_c), full_c, td)
            out2 = buf2.getvalue()
            check("a complete run shows no incomplete banner",
                  'INCOMPLETE RUN' in out2, False)
            check("a complete run shows no restriction note",
                  'restricted to' in out2, False)

        check("_common_indices intersects by session id",
              R._common_indices(full_a, short), [0, 1, 2, 3, 4])
        check("_restrict follows the requested order",
              [r['index'] for r in R._restrict(full_a, [3, 1])], [3, 1])
    except Exception as e:
        FAIL.append(f"runner import failed: {e}")

    # 10 ── PAPER METRICS. Every number in the paper tables comes from here, so
    #       the core metrics are pinned to hand-computed values, and the
    #       experiment modes are checked against the pipeline they must not alter.
    try:
        _paper_metrics_checks()
    except Exception as e:
        import traceback
        FAIL.append(f"paper metrics checks crashed: {type(e).__name__}: {e}\n"
                    + traceback.format_exc())

    # ── report ──────────────────────────────────────────────────────────────
    # Printed first so a stale copy is obvious before any result is read.
    print(f"\n{A.config_banner()}")
    print(f"\n  V31 OFFLINE SELF-TEST")
    print(f"  {'-'*66}")
    print(f"  passed : {len(PASS)}")
    print(f"  failed : {len(FAIL)}")
    if FAIL:
        print(f"\n  FAILURES")
        for f in FAIL:
            print(f"    - {f}")
        print(f"\n  Do NOT start a run. Fix these first.")
        return 1
    print(f"\n  The decision path behaves as specified.")
    print(f"  Next: python healthcheck_v31.py   (minutes, needs Ollama)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
