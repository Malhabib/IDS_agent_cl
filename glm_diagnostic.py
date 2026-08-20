# glm_diagnostic.py
"""
Decide, in a few minutes, whether GLM's poor detection accuracy is caused by
THE MODEL or by THE PROMPT. Runs 12 trivial sessions at four prompt sizes.

Background
----------
In V29.2 Scenario D, GLM produced a stated verdict in 37 of 50 sessions and
still scored 0.46, with a false-alarm rate of 0.72: it labelled 21 of 29
sessions whose delivered/requested ratio was about 1.000 as attacks. Llama and
Qwen scored 0.98 and 1.00 on the identical sample. Those are stated verdicts,
not parsing artefacts, so the remaining question is whether GLM cannot do the
task at all or whether the ~9,700-character evidence prompt is what breaks it.

This script asks the same 12 sessions four ways:

  L1 raw          just the two numbers
  L2 +ratio       the numbers plus the computed ratio and the physics rule
  L3 +ML votes    L2 plus the seven classifier votes
  L4 +SHAP/LIME   L3 plus the explanation blocks (closest to the real prompt)

Reading the result
------------------
  L1/L2 high, dropping as context grows -> THE PROMPT. The model can do the
      task; the evidence block is what derails it. Shrinking or restructuring
      the GLM prompt is worth doing.
  L1/L2 already poor -> THE MODEL. No prompt engineering will rescue it.
      Report GLM as a documented negative result and move on.

Usage
-----
    python glm_diagnostic.py                       # default glm4
    python glm_diagnostic.py qwen3.5:latest        # sanity-check a good model

Run this only once `ollama ps` reports 100% GPU for the model under test. A
model that is partly on the CPU times out rather than answers, and every
timeout is scored here as a wrong answer, so a partly-resident model looks
exactly like an incapable one.
"""

import sys, time

# glm4 (5.5 GB), not glm-4.7-flash (19 GB): the larger variant does not fit in
# 8 GB of VRAM and its answers were timeouts, not judgements.
MODEL = sys.argv[1] if len(sys.argv) > 1 else "glm4:latest"

# 12 sessions: 6 clearly normal (ratio ~1.00), 6 clear attacks (ratio >= 1.5).
# Deliberately unambiguous -- a competent analyst should score 12/12.
CASES = [
    (14.50, 14.56, "Normal"), (8.20,  8.19,  "Normal"),
    (19.80, 19.84, "Normal"), (6.40,  6.38,  "Normal"),
    (11.10, 11.13, "Normal"), (16.75, 16.70, "Normal"),
    (12.00, 18.00, "Attack"), (9.50, 16.91,  "Attack"),
    (7.30, 14.59,  "Attack"), (15.20, 22.80, "Attack"),
    (10.00, 17.77, "Attack"), (13.40, 20.10, "Attack"),
]

PHYSICS = ("A delivery ratio near 1.0 means the station delivered what was "
           "requested and the session is normal. A ratio well above 1.0 means "
           "more energy was delivered than authorised, indicating energy theft.")

VOTES = ("ML classifier votes: Random Forest Normal (64%), K-Nearest Neighbors "
         "Normal (72%), Logistic Regression Normal (76%), MLP Normal (59%), "
         "Support Vector Classifier Normal (50%), Decision Tree Normal (70%), "
         "Gradient Boosting Normal (52%). Vote tally: 0 Attack, 7 Normal.\n"
         "Note: MLP, SVC and Gradient Boosting output the same class for every "
         "session, so their votes carry no information.")

SHAP = ("SHAP/LIME explanations:\n"
        "  Random Forest       Energy delivered  +0.077  (slightly toward Attack)\n"
        "                      Energy requested  -0.018  (negligible)\n"
        "  Decision Tree       Energy delivered  +0.155  (toward Attack)\n"
        "  K-Nearest Neighbors Energy delivered  -0.053  (slightly toward Normal)\n"
        "  Logistic Regression Connection time   +0.079  (slightly toward Attack)\n"
        "  MLP                 Disconnect time   +0.066  (slightly toward Attack)\n"
        "  [these attributions are near-constant across sessions]")


def build(level: int, req: float, dlv: float) -> str:
    p = (f"An EV charging session requested {req:.2f} kWh and was delivered "
         f"{dlv:.2f} kWh.\n")
    if level >= 2:
        p += f"Delivery ratio = {dlv/req:.3f}.\n{PHYSICS}\n"
    if level >= 3:
        p += VOTES + "\n"
    if level >= 4:
        p += SHAP + "\n"
    p += "\nIs this session Attack or Normal? Answer with ONE word only:"
    return p


def main():
    try:
        from ollama import chat
    except ImportError:
        print("pip install ollama"); return

    print(f"\nModel: {MODEL}")
    print("12 unambiguous sessions (6 normal at ratio~1.00, 6 attacks at ratio>=1.5)")
    print("A competent analyst scores 12/12 at every level.\n")
    print(f"  {'level':<26} | {'correct':>8} | {'normals':>8} | {'attacks':>8} | {'sec':>6}")
    print(f"  {'-'*70}")

    results = {}
    for level, name in [(1, "L1 raw numbers"), (2, "L2 + ratio & physics"),
                        (3, "L3 + ML votes"), (4, "L4 + SHAP/LIME")]:
        ok = n_ok = a_ok = 0
        t0 = time.time()
        for req, dlv, truth in CASES:
            try:
                r = chat(model=MODEL,
                         messages=[{'role': 'system',
                                    'content': "Reply with exactly one word: Attack or Normal."},
                                   {'role': 'user', 'content': build(level, req, dlv)}],
                         options={'temperature': 0.0, 'num_predict': 64,
                                  'num_ctx': 8192})
                txt = (r.message.content or "").strip().lower()
                if not txt:                       # thinking model with empty answer
                    txt = (getattr(r.message, 'thinking', '') or "").lower()
                pred = "Attack" if "attack" in txt.split()[:3] else "Normal"
            except Exception as e:
                print(f"    [error] {e}"); pred = "?"
            hit = (pred == truth)
            ok += hit
            if truth == "Normal": n_ok += hit
            else:                 a_ok += hit
        dt = time.time() - t0
        results[level] = ok
        print(f"  {name:<26} | {ok:>6}/12 | {n_ok:>6}/6 | {a_ok:>6}/6 | {dt:>6.0f}")

    print(f"\n  {'-'*70}")
    base, full = results.get(2, 0), results.get(4, 0)
    if base >= 10 and full <= 7:
        print("  VERDICT: THE PROMPT. The model handles the task on a short prompt")
        print("           and degrades as the evidence block grows. Shrinking or")
        print("           restructuring the GLM prompt is worth doing.")
    elif base <= 7:
        print("  VERDICT: THE MODEL. It fails even on two numbers and an explicit")
        print("           rule. No prompt engineering will rescue it -- report GLM")
        print("           as a documented negative result.")
    else:
        print("  VERDICT: MIXED. Compare the per-level normals column: if normals")
        print("           collapse as context grows, the evidence block is biasing")
        print("           it toward Attack.")


if __name__ == "__main__":
    main()
