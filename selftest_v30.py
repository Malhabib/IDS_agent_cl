# selftest_v30.py
"""
Offline correctness gate for V30. No GPU, no Ollama, no dataset, no models.

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

    python selftest_v30.py

Exit code 0 means the decision path behaves as specified. It does NOT mean the
models are healthy -- that is what healthcheck_v30.py measures. Run this first
(seconds), the healthcheck second (minutes), the study last (hours).
"""

import sys
import numpy as np

import ev_ids_agent_v6_triple_llm_v30 as A

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
    return A.EVIDSAgentV6TripleLLMV30._parse_llm_response(
        stub, stage2, "stage 1 text", preds, ratio, 2.0, 1,
        Counter(votes.values()), [], [], {}, "EASY_NORMAL",
        repair_used=repair_used, llm_error=llm_error)


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
              hasattr(A.EVIDSAgentV6TripleLLMV30, attr), True)
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
        import healthcheck_v30 as H

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
        check("both 1.5 probes agree", tally(1.5) == tally(22.80 / 15.20), True)
        check("SHAP points toward Attack on an attack", dlv_shap(1.78) > 0.1, True)
        check("SHAP is near zero on a normal", abs(dlv_shap(1.004)) < 0.01, True)
        # The prompt must be big enough to reproduce the Stage 2 squeeze.
        p = H.probe_prompt(12.0, 18.0)
        check("probe prompt is realistically sized",
              len(H.SYSTEM) + len(p) > 7000, True)
        check("healthcheck uses the framework's own system prompt",
              H.SYSTEM == A.EVIDSAgentV6TripleLLMV30._get_system_prompt(None), True)
        check("healthcheck uses the framework's own Stage 2 question",
              H.STAGE2_QUESTION == A.EVIDSAgentV6TripleLLMV30.STAGE2_USER, True)
    except Exception as e:
        FAIL.append(f"healthcheck import/probe failed: {e}")

    # 9 ── the runner imports what it says it imports
    try:
        import inspect
        import simple_run_v6_triple_llm_v30 as R
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
        check("the parallel path cancels queued work when it trips",
              'cancel' in parallel, True)
        check("llm errors are counted in progress", 'llm_error' in src, True)
    except Exception as e:
        FAIL.append(f"runner import failed: {e}")

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n  V30 OFFLINE SELF-TEST")
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
    print(f"  Next: python healthcheck_v30.py   (minutes, needs Ollama)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
