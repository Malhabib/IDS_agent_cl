# integration_test_v30.py
"""
End-to-end test of the V30 seven-step pipeline. Synthetic data, real
classifiers, real SHAP, scripted model. No GPU, no Ollama, no real dataset.

Why this exists
---------------
selftest_v30.py checks the decision path in isolation. It cannot catch a bug in
how the steps are wired together -- and that is the class of bug that has cost
this project whole runs:

  * a helper renamed at the call sites but not at its definition compiles
    cleanly and dies at run time with a NameError, after the run has started
  * a SHAP array indexed by the predicted class instead of the Attack class
    silently inverts the sign of most of the evidence in every prompt
  * a Stage 2 budget clamped to its floor produces empty answers only when a
    real prompt of realistic size is used

This script runs detect() for real, over every step, and asserts on what comes
out. It takes about a minute.

    python integration_test_v30.py

Exit 0 = the pipeline is wired correctly end to end.
"""

import os, shutil, sys, tempfile
import numpy as np
import pandas as pd

import ev_ids_agent_v6_triple_llm_v30 as A

RESULTS = []


def check(name, got, want):
    RESULTS.append((name, got == want, got, want))


def make_dataset(path, n=400, seed=42):
    """
    Synthetic EV charging sessions with the same structure as the real data.

    Normal sessions deliver about what was requested; attacks deliver far more
    (energy theft) or far less (phantom charging). Deliberately separable, so
    any failure here is a wiring bug and not a hard-data problem.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        attack = i % 4 == 0
        req = float(rng.uniform(5, 25))
        if not attack:
            dlv = req * float(rng.uniform(0.97, 1.03))
        elif i % 8 == 0:
            dlv = req * float(rng.uniform(1.6, 2.4))      # energy theft
        else:
            dlv = req * float(rng.uniform(0.10, 0.35))    # phantom charging
        start = pd.Timestamp('2024-01-01', tz='UTC') + pd.Timedelta(hours=i * 3)
        rows.append({
            'connectionTime':  start.isoformat(),
            'disconnectTime':  (start + pd.Timedelta(hours=float(rng.uniform(1, 6)))).isoformat(),
            'RequestedDemand': round(req, 3),
            'kWhDelivered':    round(dlv, 3),
            'label':           int(attack),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class ScriptedClient:
    """A model whose reply for each session is chosen by the test."""

    def __init__(self, mode):
        self.model_name = f"scripted-{mode}:test"
        self.mode = mode
        self.temperature = 0.0
        self.calls = []
        self.n_calls = self.n_timeouts = self.n_errors = 0

    def _reply(self, kind):
        if self.mode == 'timeout':
            return f"{A.LLM_ERROR_PREFIX} timeout"
        if self.mode == 'silent':          # answers, but states no verdict
            return "I have considered the evidence but will not commit."
        if self.mode == 'attack':
            return ("SHAP_LIME_ASSESSMENT: kWhDelivered dominates every model.\n"
                    "PHYSICAL_INTERPRETATION: far more energy left the station "
                    "than was authorised.\n"
                    "REASONING_SUMMARY: the delivery ratio is the decisive "
                    "physical measurement.\n"
                    "CONFIDENCE: high\n"
                    "ATTACK_TYPE: energy_theft\n"
                    "**Prediction:** Attack")
        return ("SHAP_LIME_ASSESSMENT: kWhDelivered is the dominant driver and "
                "points toward Normal.\n"
                "PHYSICAL_INTERPRETATION: delivered matches requested.\n"
                "REASONING_SUMMARY: nothing anomalous.\n"
                "CONFIDENCE: high\n"
                "### Prediction: Normal")

    def generate_with_system(self, system_prompt, user_prompt, **kw):
        self.n_calls += 1
        self.calls.append(('stage1', system_prompt, user_prompt))
        return self._reply('stage1')

    def generate_two_turn(self, system_prompt, first_user, first_assistant,
                          second_user, **kw):
        self.n_calls += 1
        self.calls.append(('stage2', system_prompt, second_user))
        return self._reply('stage2')

    def health(self):
        return {'calls': self.n_calls, 'timeouts': self.n_timeouts,
                'errors': self.n_errors, 'budget_clamps': 0,
                'granted_ctx': A.NUM_CTX}


def build_agent(workspace, data_path, client, **kw):
    config = {
        'data_path':           data_path,
        'models_dir':          os.path.join(workspace, 'models'),
        'knowledge_base_path': os.path.join(workspace, 'knowledge_base'),
        'workspace_dir':       workspace,
    }
    return A.EVIDSAgentV6TripleLLMV30(
        config, client, scenario_tag='integration', verbose=False, **kw)


def main():
    ws = tempfile.mkdtemp(prefix='ev_ids_v30_it_')
    try:
        data_path = make_dataset(os.path.join(ws, 'synthetic.csv'))
        models_dir = os.path.join(ws, 'models')

        print("  training the seven classifiers on synthetic data...")
        A.auto_train_models_v6(data_path, models_dir, train_ratio=0.5,
                               random_state=42)
        check("split_info written",
              os.path.exists(os.path.join(models_dir, 'split_info_v6.pkl')), True)

        df = pd.read_csv(data_path)
        cm = A.get_column_mapping(df)
        ratio = df[cm['kWhDelivered']] / df[cm['RequestedDemand']]
        theft  = int(ratio.idxmax())          # clearest energy-theft session
        normal = int((ratio - 1.0).abs().idxmin())   # clearest normal session

        # ── the model says Attack ────────────────────────────────────────────
        print("  running detect() with a model that states Attack...")
        agent = build_agent(ws, data_path, ScriptedClient('attack'),
                            use_knowledge=False, use_memory=False)
        out = agent.detect(theft)
        r = out['result']
        check("status success", out['status'], 'success')
        check("stated verdict is honoured", r['predicted_label'], 'Malicious')
        check("provenance is 'stated'", r['verdict_source'], 'stated')
        check("no fallback on a stated verdict", r['used_fallback'], False)
        check("no llm_error on a healthy call", r['llm_error'], False)
        check("attack_type parsed", r['attack_type'], 'energy_theft')
        check("confidence parsed", r['llm_confidence'], 'high')
        check("XAI assessment section extracted",
              bool(r['llm_xai_assessment']), True)
        check("physical interpretation extracted",
              bool(r['llm_analysis']), True)
        check("SHAP ran for all seven models",
              out['result']['complexity']['xai_models_explained'], 7)
        check("a dominant SHAP feature was identified",
              bool(r['xai_top_feature']), True)
        check("transport health reported", 'llm_health' in out, True)

        # The evidence the model is shown is the contribution. Assert it is
        # actually in the prompt rather than trusting that it was built.
        stage1_prompt = agent.llm_client.calls[0][2]
        for token in ('SHAP', 'LIME', 'Energy requested', 'Energy delivered',
                      'Random Forest', 'Gradient Boosting', 'confidence'):
            check(f"prompt carries {token!r}", token in stage1_prompt, True)
        check("prompt states the sign convention",
              'toward Attack' in stage1_prompt, True)
        check("prompt reports the raw measurements to 4 dp",
              f"{float(df.loc[theft, cm['kWhDelivered']]):.4f}" in stage1_prompt,
              True)
        # Three of the seven classifiers return one class for every session.
        # The prompt must say so, or the vote tally reads as evidence when it
        # is an artefact -- this is what made Llama defer on 84% of sessions.
        check("prompt flags the non-discriminating classifiers",
              'no information' in stage1_prompt.lower()
              or 'NON-DISCRIMINATING' in stage1_prompt, True)

        # ── the model states nothing ─────────────────────────────────────────
        print("  running detect() with a model that states no verdict...")
        agent2 = build_agent(ws, data_path, ScriptedClient('silent'),
                             use_knowledge=False, use_memory=False)
        r2 = agent2.detect(normal)['result']
        check("silence falls back to the ML majority",
              r2['verdict_source'], 'ml_fallback')
        check("fallback is flagged", r2['used_fallback'], True)
        check("fallback is not an llm_error", r2['llm_error'], False)
        check("a label is still produced",
              r2['predicted_label'] in ('Normal', 'Malicious'), True)

        # ── the call never returns ───────────────────────────────────────────
        # This is the case that made two identical runs disagree. It must be
        # distinguishable from the model answering badly.
        print("  running detect() with a model that times out...")
        agent3 = build_agent(ws, data_path, ScriptedClient('timeout'),
                             use_knowledge=False, use_memory=False)
        r3 = agent3.detect(normal)['result']
        check("a timeout is recorded as an LLM error",
              r3['verdict_source'], 'llm_error')
        check("llm_error flag set", r3['llm_error'], True)
        check("a timeout is not counted as a parsed verdict",
              r3['llm_parsed'], False)
        check("no repair call is wasted on an error",
              agent3.llm_client.n_calls, 2)

        # ── XAI ablation ─────────────────────────────────────────────────────
        print("  running detect() with XAI disabled...")
        agent4 = build_agent(ws, data_path, ScriptedClient('normal'),
                             use_knowledge=False, use_memory=False, use_xai=False)
        out4 = agent4.detect(normal)
        check("xai_enabled is False", out4['result']['xai_enabled'], False)
        p4 = agent4.llm_client.calls[0][2]
        check("SHAP blocks are absent when XAI is off",
              'SHAP' in p4, False)
        check("predictions and confidence remain when XAI is off",
              'Random Forest' in p4, True)
        check("the XAI-off prompt is shorter",
              len(p4) < len(stage1_prompt), True)

        # ── metrics module on a mixed result set ─────────────────────────────
        import llm_eval_metrics_v30 as M
        rows = [{'correct': True,  'result': r},
                {'correct': False, 'result': r2},
                {'correct': True,  'result': r3}]
        er = M.llm_error_rate(rows)
        check("error rate counts the timeout", er['errors'], 1)
        check("a set containing an error is not valid", er['valid'], False)
        check("answered_only drops the error", len(M.answered_only(rows)), 2)
        fc = M.format_compliance(rows)
        check("format compliance excludes errors", fc['n'], 2)
        check("format compliance records the exclusion",
              fc['excluded_llm_errors'], 1)
        xf = M.xai_faithfulness([{'correct': True, 'result': r}])
        check("faithfulness with no wrong sessions reports n/a, not zero",
              xf['rate_wrong'], None)

    finally:
        shutil.rmtree(ws, ignore_errors=True)

    failed = [x for x in RESULTS if not x[1]]
    print(f"\n  V30 INTEGRATION TEST")
    print(f"  {'-'*66}")
    print(f"  passed : {len(RESULTS) - len(failed)}")
    print(f"  failed : {len(failed)}")
    if failed:
        print(f"\n  FAILURES")
        for name, _, got, want in failed:
            print(f"    - {name}\n        expected {want!r}, got {got!r}")
        return 1
    print(f"\n  The seven-step pipeline is wired correctly end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
