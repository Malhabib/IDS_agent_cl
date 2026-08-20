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

# A Stage 1 prompt of realistic size. The point of the healthcheck is to load
# the context the real run will load; a toy prompt would measure nothing.
EVIDENCE = """
ML CLASSIFIER PREDICTIONS (7 models, with confidence)
  Random Forest        Normal   0.64
  K-Nearest Neighbors  Normal   0.72
  Logistic Regression  Normal   0.76
  MLP                  Normal   0.59      [NON-DISCRIMINATING]
  Support Vector       Normal   0.50      [NON-DISCRIMINATING]
  Decision Tree        Normal   0.70
  Gradient Boosting    Normal   0.52      [NON-DISCRIMINATING]
  DISCRIMINATING vote tally: 0 Attack, 4 Normal

SHAP / LIME PER-MODEL ATTRIBUTIONS   (+ = toward Attack, - = toward Normal)
  Random Forest        kWhDelivered      +0.077   connectionTime   -0.018
  Decision Tree        kWhDelivered      +0.155   RequestedDemand  -0.041
  K-Nearest Neighbors  kWhDelivered      -0.053   disconnectTime   +0.012
  Logistic Regression  connectionTime    +0.079   kWhDelivered     -0.026
  MLP                  disconnectTime    +0.066   kWhDelivered     +0.004
  Support Vector       RequestedDemand   -0.031   kWhDelivered     +0.019
  Gradient Boosting    kWhDelivered      +0.088   connectionTime   -0.007

CRITICAL RULE
  The delivered/requested ratio is a direct physical measurement and OUTRANKS
  the classifier vote tally. Three of the seven classifiers return the same
  class for every session and carry no information.
""".strip()

SYSTEM = ("You are an expert EV charging security analyst. Analyse the "
          "evidence and end your reply with a line of the form "
          "'Prediction: Attack' or 'Prediction: Normal'.")


def probe_prompt(req, dlv):
    return (f"EV CHARGING SESSION\n"
            f"  RequestedDemand : {req:.3f} kWh\n"
            f"  kWhDelivered    : {dlv:.3f} kWh\n"
            f"  delivered/requested = {dlv/req:.3f}\n\n"
            f"{EVIDENCE}\n\n"
            f"Give SHAP_LIME_ASSESSMENT, PHYSICAL_INTERPRETATION, "
            f"REASONING_SUMMARY, CONFIDENCE and PREDICTION.")


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
    latencies = []
    for req, dlv, truth in PROBES:
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
            second_user=("State your final answer now, as a single line: "
                         "'Prediction: Attack' or 'Prediction: Normal'."),
            temperature=A.LLM_TEMPERATURE, max_tokens=A.STAGE2_MAX_TOKENS)
        latencies.append(time.time() - t)

        if s1.startswith(A.LLM_ERROR_PREFIX) or s2.startswith(A.LLM_ERROR_PREFIX):
            errors += 1
            print(f"    req={req:6.2f} dlv={dlv:6.2f}  ERROR")
            continue
        v = A.extract_verdict(s2)
        if v is not None:
            stated += 1
            hit = (v == 'Malicious') == (truth == 'Attack')
            correct += hit
            print(f"    req={req:6.2f} dlv={dlv:6.2f}  ratio={dlv/req:5.3f}  "
                  f"-> {v:<9} {'ok' if hit else 'WRONG'}  "
                  f"({latencies[-1]:5.1f}s)")
        else:
            print(f"    req={req:6.2f} dlv={dlv:6.2f}  NO VERDICT STATED  "
                  f"({latencies[-1]:5.1f}s)")

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
        'stated':         f"{stated}/{len(PROBES)}",
        'correct':        f"{correct}/{len(PROBES)}",
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
    print(f"    verdict stated    : {stated}/{len(PROBES)}")
    print(f"    correct           : {correct}/{len(PROBES)}")
    print(f"    LLM errors        : {errors}  (timeouts {h['timeouts']})")
    print(f"    budget clamps     : {h['budget_clamps']}")
    print(f"    median session    : {med:.1f}s")
    print(f"    projected {n_project:>3} sessions: {projected_h:.1f} h")

    # ── go / no-go ──────────────────────────────────────────────────────────
    problems = []
    if frac is not None and frac < 0.999:
        problems.append(f"only {frac*100:.0f}% of the model is in VRAM; the rest "
                        f"runs on the CPU")
    if ctx and ctx < A.NUM_CTX:
        problems.append(f"Ollama granted a {ctx}-token context against a request "
                        f"of {A.NUM_CTX}, which starves the Stage 2 budget")
    if errors:
        problems.append(f"{errors} of {len(PROBES)} probes failed or timed out")
    if h['budget_clamps'] > len(PROBES):
        problems.append(f"{h['budget_clamps']} generation-budget clamps: the "
                        f"prompt is crowding out the answer")
    if stated < len(PROBES):
        problems.append(f"only {stated} of {len(PROBES)} probes produced a stated "
                        f"verdict; the rest would be scored as fallbacks")
    if correct <= len(PROBES) // 2:
        problems.append(f"{correct}/{len(PROBES)} correct on deliberately "
                        f"unambiguous sessions")
    if projected_h > 6:
        problems.append(f"projected {projected_h:.1f} h for {n_project} sessions")

    row['verdict']  = 'FAIL' if problems else 'GO'
    row['problems'] = problems
    if problems:
        print(f"\n    NO-GO:")
        for p in problems:
            print(f"      - {p}")
    else:
        print(f"\n    GO")

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

    print(f"\n  V30 HEALTHCHECK")
    print(f"  {len(PROBES)} unambiguous probe sessions per model, full two-stage "
          f"pipeline,\n  realistic prompt size. Projecting the cost of "
          f"{args.n} sessions.")

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

    bad = [r for r in rows if r['verdict'] != 'GO']
    if bad:
        print(f"\n  {len(bad)} of {len(rows)} models are NOT ready. Starting the "
              f"study now would\n  reproduce the failure you have already paid "
              f"for twice. Fix the items above.")
        return 1
    print(f"\n  All models ready. The study is worth starting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
