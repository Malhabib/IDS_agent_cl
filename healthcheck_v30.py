# healthcheck_v30.py
"""
Decide in MINUTES whether a V30 study is worth starting. Needs Ollama; does not
need the dataset or the trained classifiers.

The problem it solves
---------------------
Runs in this project have repeatedly consumed one to two days and then turned
out to be unusable, for reasons that were all measurable in the first three
minutes:

  * the model did not fit in VRAM, so Ollama split it across the CPU and a
    single session took hours instead of seconds
  * Ollama silently granted a 4,096-token context against a request of 8,192,
    which starved the Stage 2 generation budget down to its floor
  * at that floor a thinking model spent every token on its reasoning channel
    and returned an empty answer, so no verdict was ever written
  * calls timed out, the timeout became an "Error: ..." string, and the session
    was silently scored as though the model had answered -- which is why two
    runs of identical code on identical data at temperature 0 disagreed

This script measures each of those directly, per model, and multiplies the
measured per-session cost by the sample size you intend to use. If the answer
is "eleven hours per session", you learn it now rather than on Thursday.

Usage
-----
    python healthcheck_v30.py                 # 50 sessions, default models
    python healthcheck_v30.py --n 100         # project the cost for 100
    python healthcheck_v30.py --models glm4:latest
    python healthcheck_v30.py --timeout 120   # stricter per-call deadline

Exit code 0 = GO. Anything else = do not start the study.
"""

import argparse, json, sys, time

import ev_ids_agent_v6_triple_llm_v30 as A

DEFAULT_MODELS = ['llama3:latest', 'glm4:latest', 'qwen3.5:latest']

# Six unambiguous sessions: three at a delivery ratio of ~1.00 (normal) and
# three at >=1.5 (energy theft). A model that cannot separate these cannot do
# the task at all, so a low score here is a real finding and not a hard sample.
PROBES = [
    (14.50, 14.56, 'Normal'), (8.20,  8.19,  'Normal'), (19.80, 19.84, 'Normal'),
    (12.00, 18.00, 'Attack'), (9.50,  16.91, 'Attack'), (15.20, 22.80, 'Attack'),
]

# Use the framework's OWN system prompt and Stage 2 question, not an
# approximation of them. Both are what the study will actually send, and their
# length is part of what the healthcheck is measuring.
SYSTEM = A.EVIDSAgentV6TripleLLMV30._get_system_prompt(None)
STAGE2_QUESTION = A.EVIDSAgentV6TripleLLMV30.STAGE2_USER


def evidence_for(ratio: float) -> str:
    """
    Build an evidence block that is CONSISTENT with the session.

    The first version of this healthcheck used one fixed evidence block for all
    six probes. That was a defect: a normal session at ratio 1.004 was shown
    SHAP attributions pointing toward Attack, and an attack at ratio 1.780 was
    shown a vote tally of "0 Attack, 4 Normal". The correctness column then
    measured whether a model could resolve deliberately contradictory evidence,
    which is not what a healthcheck is for and is not what the real pipeline
    presents. In the real run SHAP is computed on the actual session, so its
    signs track the data.

    The deliberate weakness of the classifier ensemble IS reproduced, because it
    is a real property of the study: MLP, SVC and Gradient Boosting return the
    same class for every session, so only four votes carry information.
    """
    attackish = ratio > 1.3 or ratio < 0.7
    # Random Forest and Decision Tree are the two classifiers that actually
    # separate; KNN and Logistic Regression are weak but not constant.
    rf  = ('Attack', 0.71) if attackish else ('Normal', 0.64)
    dt  = ('Attack', 0.78) if attackish else ('Normal', 0.70)
    # 22.80/15.20 evaluates to 1.5000000000000002, so a strict > 1.5 made two
    # probes with the same intended ratio disagree. Compare against 1.45.
    knn = ('Attack', 0.55) if ratio > 1.45 else ('Normal', 0.72)
    lr  = ('Normal', 0.76)
    n_attack = sum(1 for p, _ in (rf, dt, knn, lr) if p == 'Attack')

    # SHAP magnitude scales with how far the session is from a ratio of 1.0,
    # and the sign follows the direction, exactly as it does on real data.
    d = ratio - 1.0
    s = lambda x: f"{x:+.3f}"
    dlv_shap = max(-0.40, min(0.40, d * 0.28))

    # LIME weights, on the same basis as SHAP, for the block below.
    lime_a, lime_b, lime_c = s(dlv_shap * 1.1), s(-d * 0.03), s(d * 0.02)
    lime_d, lime_e         = s(dlv_shap * 1.6), s(d * 0.02)
    lime_f, lime_g         = s(dlv_shap * 0.7), s(-d * 0.02)
    lime_h, lime_i         = s(dlv_shap * 0.8), s(d * 0.04)
    lime_j                 = s(dlv_shap * 0.1)
    lime_k                 = s(dlv_shap * 0.2)
    req_hi, req_lo         = 14.20, 9.80

    def rank(scale):
        """One model's four SHAP attributions, formatted as the real prompt does."""
        vals = [('kWhDelivered',    dlv_shap * scale),
                ('RequestedDemand', -d * 0.05 * scale),
                ('connectionTime',  -d * 0.02 * scale),
                ('disconnectTime',  d * 0.01 * scale)]
        vals.sort(key=lambda kv: -abs(kv[1]))
        return "  ".join(f"{k} {s(v)}" for k, v in vals)

    return f"""ML CLASSIFIER PREDICTIONS (7 models, with confidence)
  Random Forest        {rf[0]:<8} {rf[1]:.2f}
  K-Nearest Neighbors  {knn[0]:<8} {knn[1]:.2f}
  Logistic Regression  {lr[0]:<8} {lr[1]:.2f}
  MLP                  Normal   0.59      [NON-DISCRIMINATING]
  Support Vector       Normal   0.50      [NON-DISCRIMINATING]
  Decision Tree        {dt[0]:<8} {dt[1]:.2f}
  Gradient Boosting    Normal   0.52      [NON-DISCRIMINATING]
  DISCRIMINATING vote tally: {n_attack} Attack, {4-n_attack} Normal

SHAP PER-MODEL ATTRIBUTIONS, ALL FOUR FEATURES
(+ = toward Attack, - = toward Normal; ranked by magnitude within each model)
  Random Forest             {rank(1.0)}
  K-Nearest Neighbors       {rank(0.6)}
  Logistic Regression       {rank(0.9)}
  MLP                       {rank(0.1)}
  Support Vector Classifier {rank(0.1)}
  Decision Tree             {rank(1.8)}
  Gradient Boosting         {rank(0.2)}

LIME LOCAL EXPLANATIONS (local linear approximation, same sign convention)
  Random Forest        kWhDelivered > {req_hi:.2f}   {lime_a}
                       RequestedDemand <= {req_lo:.2f}   {lime_b}
                       connectionTime in bucket 3        {lime_c}
  Decision Tree        kWhDelivered > {req_hi:.2f}   {lime_d}
                       disconnectTime in bucket 2        {lime_e}
  K-Nearest Neighbors  kWhDelivered > {req_hi:.2f}   {lime_f}
                       RequestedDemand <= {req_lo:.2f}   {lime_g}
  Logistic Regression  kWhDelivered > {req_hi:.2f}   {lime_h}
                       connectionTime in bucket 3        {lime_i}
  MLP                  kWhDelivered > {req_hi:.2f}   {lime_j}
  Support Vector       RequestedDemand <= {req_lo:.2f}   -0.014
  Gradient Boosting    kWhDelivered > {req_hi:.2f}   {lime_k}

CRITICAL RULE
  The delivered/requested ratio is a direct physical measurement and OUTRANKS
  the classifier vote tally. Three of the seven classifiers return the same
  class for every session and carry no information."""


# The real Stage 1 prompt also carries the retrieved knowledge and the recalled
# cases. They are included here at realistic length because the point of the
# healthcheck is to load the context the study will load: it was the SIZE of
# this prompt, replayed into Stage 2, that collapsed GLM's generation budget to
# its floor. A short probe would measure latency and miss the actual failure.
#
# They must also TRACK THE SESSION. The real pipeline picks its RAG query from
# the delivery ratio -- "energy theft" above 1.3, "phantom charging" below 0.7,
# "normal charging" in between -- so a normal session is never shown two
# paragraphs about theft. The previous version of this file fixed the SHAP block
# to the session but left RAG and memory constant and attack-flavoured. That is
# a plausible reason Llama answered Attack on all six probes, including the
# three at ratio ~1.00, and it is the same defect as before in a second place.
RAG_ATTACK = """RETRIEVED DOMAIN KNOWLEDGE (RAG, top 2 passages, query 'energy theft')
  [1] Energy theft in EV charging infrastructure. Meter-tampering and
      protocol-manipulation attacks cause the station to deliver materially
      more energy than the session authorised. The signature is a
      delivered/requested ratio well above 1.0 sustained across the session,
      typically 1.5 or higher, with the connection duration unchanged. Because
      billing is derived from the authorised figure rather than the delivered
      figure, the operator absorbs the difference. Detection relies on the
      energy balance rather than on timing, since a competent attacker leaves
      the session envelope intact. Reported false-positive sources include
      legitimate top-up sessions and meter drift, both of which produce ratios
      within a few percent of 1.0 rather than above 1.3.
  [2] Phantom charging. The inverse signature: the station reports a session
      that delivered far less energy than requested, often below 0.4 of the
      authorised amount, while still occupying the connector for a normal
      duration. This is associated with denial-of-charging and with billing
      fraud where the customer is charged for energy never delivered. It is
      distinguished from an ordinary interrupted session by the absence of a
      corresponding reduction in connection time."""

RAG_NORMAL = """RETRIEVED DOMAIN KNOWLEDGE (RAG, top 2 passages, query 'normal charging')
  [1] Normal EV charging sessions. A session in which the station delivers
      substantially the energy that was authorised is the expected case. Meter
      tolerance, cable losses and rounding in the session record routinely
      produce a delivered/requested ratio a few tenths of a percent either side
      of 1.0, so a ratio between roughly 0.95 and 1.05 is unremarkable and
      carries no security significance on its own. Operators see this in the
      large majority of sessions. Treating small deviations around 1.0 as
      anomalous is the dominant source of false alarms in deployed detectors,
      because the deviation is instrumentation noise rather than signal.
  [2] Session envelope in normal operation. Connection and disconnect times in
      a normal session bracket a charging period consistent with the energy
      delivered. Timing features have no physical bearing on whether energy was
      diverted, so a detector that keys on them rather than on the energy
      balance will misclassify ordinary sessions. The energy balance is the
      discriminating measurement; timing is context."""


def rag_for(ratio: float) -> str:
    """Mirror the pipeline's ratio-driven RAG query selection."""
    if ratio > 1.3 or ratio < 0.7:
        return RAG_ATTACK
    return RAG_NORMAL


def ltm_for(ratio: float) -> str:
    """
    Three recalled cases, nearest to this session.

    Like the RAG block, this has to track the session: the real long-term
    memory returns cases similar to the one under review, so a normal session
    recalls mostly normal precedents.
    """
    if ratio > 1.3:
        cases = [(8.900, 15.664, 2.80, "Attack (energy_theft)",
                  "delivered 1.760x the authorised amount with an unchanged "
                  "session envelope, matching the documented meter-tampering "
                  "signature"),
                 (12.500, 19.375, 3.40, "Attack (energy_theft)",
                  "ratio 1.550, well outside meter tolerance, energy balance "
                  "decisive"),
                 (16.000, 16.080, 4.10, "Normal",
                  "ratio 1.005, within meter tolerance")]
    elif ratio < 0.7:
        cases = [(14.000, 3.640, 3.20, "Attack (phantom_charging)",
                  "ratio 0.260 with a full-length connection, matching the "
                  "denial-of-charging signature"),
                 (9.800, 2.744, 2.90, "Attack (phantom_charging)",
                  "ratio 0.280, energy delivered far below authorised"),
                 (11.400, 11.372, 3.10, "Normal",
                  "ratio 0.998, nothing anomalous")]
    else:
        cases = [(11.400, 11.372, 3.10, "Normal",
                  "delivered close to requested, ratio 0.998, no anomaly in the "
                  "energy balance and duration consistent with the delivered "
                  "amount"),
                 (17.200, 17.286, 4.40, "Normal",
                  "ratio 1.005, within meter tolerance; small deviations around "
                  "1.0 are instrumentation noise"),
                 (6.700, 6.674, 2.20, "Normal",
                  "ratio 0.996, unremarkable")]
    out = ["SIMILAR HISTORICAL CASES (long-term memory, 3 nearest)"]
    for i, (req, dlv, dur, label, why) in enumerate(cases, 1):
        out.append(f"  Case {i}  requested={req:.3f} kWh, delivered={dlv:.3f} kWh, "
                   f"duration={dur:.2f}h")
        out.append(f"          Classified {label}: {why}.")
    return "\n".join(out)


def probe_prompt(req, dlv):
    ratio = dlv / req
    # The trailing instruction names the STAGE 1 sections only. Stage 1 is
    # analysis; the verdict belongs to Stage 2, and asking for a prediction here
    # would contradict the framework's own system prompt.
    return (f"EV CHARGING SESSION\n"
            f"  RequestedDemand : {req:.3f} kWh\n"
            f"  kWhDelivered    : {dlv:.3f} kWh\n"
            f"  delivered/requested = {ratio:.3f}\n\n"
            f"{rag_for(ratio)}\n\n"
            f"{ltm_for(ratio)}\n\n"
            f"{evidence_for(ratio)}\n\n"
            f"Provide your Stage 1 analysis in the format given: "
            f"PHYSICAL_INTERPRETATION, SHAP_LIME_ASSESSMENT, DOMAIN_MATCH, "
            f"HISTORICAL_CONTEXT and UNCERTAINTY_FACTORS. Do not state a "
            f"prediction yet.")


def residency(model, base_url):
    """Return (fraction_in_vram, granted_context, total_bytes) from /api/ps."""
    import requests
    try:
        ps = requests.get(f"{base_url}/api/ps", timeout=10).json()
    except Exception:
        return None, None, None
    for m in ps.get('models', []):
        if m.get('model') == model or m.get('name') == model:
            total = float(m.get('size', 0) or 0)
            vram  = float(m.get('size_vram', 0) or 0)
            ctx   = (m.get('context_length')
                     or (m.get('details') or {}).get('context_length'))
            return (vram / total if total else None), ctx, total
    return None, None, None


def check_model(model, base_url, n_project, timeout):
    print(f"\n  {'='*72}\n  {model}\n  {'='*72}")
    row = {'model': model}

    try:
        client = A.make_llm_client('ollama', model, base_url,
                                   temperature=A.LLM_TEMPERATURE,
                                   timeout=timeout)
    except Exception as e:
        print(f"    cannot create a client: {e}")
        return dict(row, verdict='FAIL', reason='client construction failed')

    t0 = time.time()
    correct = stated = errors = 0
    repaired = unrecovered = 0
    latencies = []
    transcript = []
    consecutive_errors = 0
    n_run = 0
    for req, dlv, truth in PROBES:
        # Two consecutive failures already answer the question. Qwen at 2% VRAM
        # cost 12 timeouts at 300 s each -- an hour to learn what the first two
        # probes had established.
        if consecutive_errors >= 2:
            print(f"    stopping after {n_run} probes: two consecutive failures "
                  f"is enough to establish this model is not usable")
            break
        n_run += 1
        p = probe_prompt(req, dlv)
        t = time.time()
        s1 = client.generate_with_system(system_prompt=SYSTEM, user_prompt=p,
                                         temperature=A.LLM_TEMPERATURE,
                                         max_tokens=A.STAGE1_MAX_TOKENS)
        # The real pipeline is two-stage, and Stage 2 carries the longest
        # prompt in the run. Measuring only Stage 1 would miss the exact place
        # where GLM's budget collapsed, so both stages are exercised here.
        s2 = client.generate_two_turn(
            system_prompt=SYSTEM, first_user=p, first_assistant=s1,
            second_user=STAGE2_QUESTION,
            temperature=A.LLM_TEMPERATURE, max_tokens=A.STAGE2_MAX_TOKENS)
        latencies.append(time.time() - t)

        if s1.startswith(A.LLM_ERROR_PREFIX) or s2.startswith(A.LLM_ERROR_PREFIX):
            errors += 1
            consecutive_errors += 1
            print(f"    req={req:6.2f} dlv={dlv:6.2f}  ERROR")
            transcript.append((req, dlv, truth, 'ERROR', s1, s2, ''))
            continue
        consecutive_errors = 0

        v = A.extract_verdict(s2)
        rep = ''
        if v is None:
            # The pipeline does not give up here, so neither does the
            # healthcheck. Omitting the repair call made GLM look as though it
            # never answers, when the study would have recovered the verdict.
            repaired += 1
            rep = client.generate_with_system(
                system_prompt=("You output exactly one word and nothing else. "
                               "No reasoning, no punctuation, no explanation."),
                user_prompt=(f"An analyst reviewed an EV charging session and "
                             f"wrote:\n\n{(s2 or s1)[-900:]}\n\n"
                             f"Measured facts: requested {req:.3f} kWh, "
                             f"delivered {dlv:.3f} kWh "
                             f"(delivered/requested = {dlv/req:.3f}).\n\n"
                             f"Answer with ONE word only — Attack or Normal:"),
                temperature=0.0, max_tokens=64, force_no_think=True)
            if rep and not rep.startswith(A.LLM_ERROR_PREFIX):
                v = A.extract_verdict(rep + "\nPrediction: " + rep.strip())

        if v is not None:
            stated += 1
            hit = (v == 'Malicious') == (truth == 'Attack')
            correct += hit
            tag = " (via repair)" if rep else ""
            print(f"    req={req:6.2f} dlv={dlv:6.2f}  ratio={dlv/req:5.3f}  "
                  f"-> {v:<9} {'ok' if hit else 'WRONG'}{tag}  "
                  f"({latencies[-1]:5.1f}s)")
        else:
            unrecovered += 1
            print(f"    req={req:6.2f} dlv={dlv:6.2f}  NO VERDICT, REPAIR FAILED  "
                  f"({latencies[-1]:5.1f}s)")
            print(f"      Stage 2 ended: ...{s2[-220:].strip()!r}")
        transcript.append((req, dlv, truth, v or 'NONE', s1, s2, rep))

    # Always write the raw exchanges. Diagnosing "no verdict stated" from a
    # counter alone is guesswork; the responses say whether the model refused,
    # rambled past its budget, or answered in a format the parser missed.
    tpath = f"healthcheck_transcript_{model.replace(':', '_').replace('/', '_')}.txt"
    try:
        with open(tpath, 'w', encoding='utf-8') as f:
            for req, dlv, truth, verdict, s1, s2, rep in transcript:
                f.write(f"{'='*78}\nrequested={req} delivered={dlv} "
                        f"ratio={dlv/req:.3f} truth={truth} -> {verdict}\n"
                        f"{'='*78}\n\n[STAGE 1 RESPONSE]\n{s1}\n\n"
                        f"[STAGE 2 RESPONSE]\n{s2}\n\n"
                        f"[REPAIR]\n{rep}\n\n")
        print(f"\n    transcript written: {tpath}")
    except Exception as e:
        print(f"\n    could not write transcript: {e}")

    elapsed = time.time() - t0
    frac, ctx, total = residency(model, base_url)
    h = client.health()
    med = sorted(latencies)[len(latencies) // 2] if latencies else 0.0

    # Each study session runs the full pipeline, which is roughly the two
    # stages measured here plus XAI and occasionally a repair call.
    projected_h = med * n_project * 3 / 3600.0

    row.update({
        'in_vram':        None if frac is None else round(frac, 3),
        'granted_ctx':    ctx,
        'size_gb':        None if not total else round(total / 1024**3, 1),
        'probes_run':     n_run,
        'stated':         f"{stated}/{n_run}",
        'repaired':       repaired,
        'unrecovered':    unrecovered,
        'transcript':     tpath,
        'correct':        f"{correct}/{n_run}",
        'errors':         errors,
        'budget_clamps':  h['budget_clamps'],
        'median_sec':     round(med, 1),
        'probe_sec':      round(elapsed, 1),
        'projected_hours': round(projected_h, 1),
    })

    print(f"\n    resident in VRAM  : "
          f"{'unknown' if frac is None else f'{frac*100:.0f}%'}"
          f"{'' if not total else f'  ({total/1024**3:.1f} GB)'}")
    print(f"    granted context   : {ctx}  (requested {A.NUM_CTX})")
    print(f"    verdict obtained  : {stated}/{n_run}"
          f"  (direct {stated-max(0, repaired-unrecovered)}, "
          f"via repair {max(0, repaired-unrecovered)}, "
          f"unrecovered {unrecovered})")
    print(f"    correct           : {correct}/{n_run}")
    print(f"    LLM errors        : {errors}  (timeouts {h['timeouts']})")
    print(f"    budget clamps     : {h['budget_clamps']}")
    print(f"    median session    : {med:.1f}s")
    print(f"    projected {n_project:>3} sessions: {projected_h:.1f} h")

    # ── go / no-go ──────────────────────────────────────────────────────────
    # Two separate questions, kept apart deliberately.
    #
    #   ENVIRONMENT  is the machine capable of measuring this model at all?
    #                A failure here invalidates any number the study produces.
    #   BEHAVIOUR    given a healthy environment, is the model doing the task?
    #                A failure here is a RESULT. It may well be worth running
    #                and reporting -- a model that always answers Normal is a
    #                legitimate negative finding, provided the environment was
    #                sound when it was measured.
    env, behav = [], []
    if frac is not None and frac < 0.999:
        env.append(f"only {frac*100:.0f}% of the model is in VRAM; the rest runs "
                   f"on the CPU"
                   + (f" (the model is {total/1024**3:.1f} GB)" if total else ""))
    if ctx and ctx < A.NUM_CTX:
        env.append(f"Ollama granted a {ctx}-token context against a request of "
                   f"{A.NUM_CTX}, which starves the Stage 2 budget")
    if errors:
        env.append(f"{errors} of {n_run} probes failed or timed out")
    if h['budget_clamps'] > n_run:
        env.append(f"{h['budget_clamps']} generation-budget clamps: the prompt is "
                   f"crowding out the answer")
    if projected_h > 6:
        env.append(f"projected {projected_h:.1f} h for {n_project} sessions")

    answered = n_run - errors
    if answered and unrecovered:
        behav.append(f"{unrecovered} of {answered} answered probes produced no "
                     f"verdict even after the repair call; the study would score "
                     f"those as ML fallbacks (see {tpath})")
    elif answered and repaired:
        behav.append(f"{repaired} of {answered} needed the repair call to state a "
                     f"verdict; the direct Stage 2 format was not followed")
    if answered and correct <= answered // 2:
        behav.append(f"{correct}/{answered} correct on unambiguous sessions")
    if answered and correct == answered and answered >= 4:
        behav.append(f"{correct}/{answered} — note that these probes are "
                     f"deliberately easy, so this is a floor, not a score")

    row['verdict']     = 'FAIL' if env else ('WEAK' if behav and any(
        'correct' in b and 'floor' not in b for b in behav) else 'GO')
    row['environment'] = env
    row['behaviour']   = behav

    if env:
        print(f"\n    NO-GO — ENVIRONMENT (nothing measured here is trustworthy):")
        for p in env:
            print(f"      - {p}")
    if behav:
        caveat = ("unreliable while the environment is broken" if env
                  else "the environment was sound")
        print(f"\n    MODEL BEHAVIOUR ({caveat}):")
        for p in behav:
            print(f"      - {p}")
    if not env and not behav:
        print(f"\n    GO")
    elif not env:
        print(f"\n    Environment is sound. The behaviour note above is a RESULT, "
              f"not a fault;\n    running this model and reporting it is "
              f"legitimate.")

    A.unload_ollama_model(model, base_url)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=DEFAULT_MODELS)
    ap.add_argument('--url', default='http://localhost:11434')
    ap.add_argument('--n', type=int, default=50,
                    help='sample size you intend to run, for the cost projection')
    ap.add_argument('--timeout', type=int, default=A.LLM_CALL_TIMEOUT_SEC)
    args = ap.parse_args()

    # A gate that passes when it checked nothing is worse than no gate.
    if not args.models:
        print("\n  No models given, so nothing was checked. This is NOT a pass.")
        return 2

    sample = probe_prompt(12.0, 18.0)
    print(f"\n  V30 HEALTHCHECK")
    print(f"  {len(PROBES)} unambiguous probe sessions per model, full two-stage "
          f"pipeline.")
    print(f"  Prompts are the framework's own: system {len(SYSTEM)} chars, "
          f"Stage 1 evidence\n  {len(sample)} chars "
          f"(~{(len(SYSTEM)+len(sample))//4} tokens), Stage 2 replays both. "
          f"The evidence\n  block tracks each session, as SHAP does on real "
          f"data.")
    print(f"  Projecting the cost of {args.n} sessions.")

    rows = [check_model(m, args.url, args.n, args.timeout) for m in args.models]

    print(f"\n\n  SUMMARY")
    print(f"  {'-'*88}")
    print(f"  {'model':<20} {'VRAM':>6} {'ctx':>6} {'stated':>7} {'correct':>8} "
          f"{'err':>4} {'sec':>7} {'proj h':>7}  verdict")
    for r in rows:
        vram = 'n/a' if r.get('in_vram') is None else f"{r['in_vram']*100:.0f}%"
        print(f"  {r['model']:<20} {vram:>6} {str(r.get('granted_ctx','n/a')):>6} "
              f"{str(r.get('stated','-')):>7} {str(r.get('correct','-')):>8} "
              f"{str(r.get('errors','-')):>4} {str(r.get('median_sec','-')):>7} "
              f"{str(r.get('projected_hours','-')):>7}  {r['verdict']}")

    with open('healthcheck_v30.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Written: healthcheck_v30.json")

    broken = [r for r in rows if r['verdict'] == 'FAIL']
    weak   = [r for r in rows if r['verdict'] == 'WEAK']

    if broken:
        suggest_fitting_models([r['model'] for r in broken], args.url)
        print(f"\n  {len(broken)} of {len(rows)} models CANNOT BE MEASURED on this "
              f"machine.\n  Any number a study produces for them describes the GPU, "
              f"not the model.")
    if weak:
        print(f"\n  {len(weak)} model(s) ran cleanly but performed poorly. That is a "
              f"RESULT,\n  not a fault: "
              + ", ".join(r['model'] for r in weak)
              + f".\n  Running and reporting them as a documented negative result "
              f"is legitimate.")
    if broken:
        return 1
    print(f"\n  Environment is sound for every model checked. The study is worth "
          f"starting.")
    return 0


def suggest_fitting_models(broken_models, base_url):
    """
    Name concrete alternatives that fit, instead of only saying what does not.

    A model larger than the card can never be made to fit by closing
    applications; a model that merely exceeds the FREE VRAM can. The two cases
    need different advice, so both are computed here.
    """
    import requests
    try:
        tags = requests.get(f"{base_url}/api/tags", timeout=10).json()
    except Exception:
        return
    installed = [(m.get('name') or m.get('model'), float(m.get('size', 0) or 0))
                 for m in tags.get('models', [])]
    if not installed:
        return

    total_mib = free_mib = None
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        if out:
            total_mib, free_mib = (int(x) for x in out[0].split(','))
    except Exception:
        pass

    print(f"\n  MODELS INSTALLED, BY WEIGHT SIZE")
    if total_mib:
        print(f"  GPU: {total_mib/1024:.1f} GB total, {free_mib/1024:.1f} GB free "
              f"right now ({(total_mib-free_mib)/1024:.1f} GB held by other "
              f"processes)")
    # The figure /api/tags reports is the WEIGHTS on disk. What must fit in VRAM
    # is the weights PLUS the KV cache, and the KV cache scales with num_ctx.
    # qwen3.5 is 6.1 GB of weights but /api/ps reported a 10.3 GB footprint at
    # num_ctx 8192 -- so it is not "larger than the card", it is larger than the
    # free VRAM once its cache is included. Halving num_ctx roughly halves the
    # cache, which is a real option this advisor must not obscure.
    KV_AT_8K = 1.35        # observed multiplier: footprint / weights at 8k ctx
    KV_AT_4K = 1.18
    print(f"  'loaded' below = weights x {KV_AT_8K} for the KV cache at "
          f"num_ctx={A.NUM_CTX}.")
    free_gb = (free_mib / 1024) if free_mib else None
    card_gb = (total_mib / 1024) if total_mib else None

    for name, size in sorted(installed, key=lambda x: x[1]):
        gb = size / 1024 ** 3
        at8, at4 = gb * KV_AT_8K, gb * KV_AT_4K
        if card_gb and at4 > card_gb:
            note = "NEVER fits — exceeds the card even at num_ctx 4096"
        elif free_gb and at8 > free_gb and at4 <= free_gb:
            note = f"fits at num_ctx 4096 ({at4:.1f} GB), not at 8192 ({at8:.1f} GB)"
        elif free_gb and at8 > free_gb:
            note = f"needs {at8:.1f} GB loaded — free more VRAM"
        else:
            note = f"fits now ({at8:.1f} GB loaded)"
        mark = "  <- currently failing" if name in broken_models else ""
        print(f"    {name:<26} {gb:5.1f} GB weights   {note}{mark}")

    if free_gb:
        fits = [n for n, s in installed
                if (s / 1024**3) * KV_AT_8K <= free_gb]
        if fits:
            print(f"\n  Usable right now at num_ctx {A.NUM_CTX}: {', '.join(fits)}")
        print(f"  Re-check one with:  python healthcheck_v30.py --models <name> "
              f"--n 50")


if __name__ == "__main__":
    sys.exit(main())
