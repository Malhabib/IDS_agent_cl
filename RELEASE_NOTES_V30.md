# EV-IDS-Agent V6 — Triple LLM, V30

Pipeline, prompts, the seven-step detection flow, the A/B/C/D scenarios and the
V23-style per-session explainability output are **unchanged**. Everything below
is a fix to how the framework talks to the models and to how it reports what
happened.

---

## 1. The GLM problem, and what it actually was

GLM was never failing at the task. Mechanical faults made it impossible for GLM
to answer, and made the resulting damage invisible.

### 1.1 Stage 2 ran on a 128-token budget with the reasoning channel on

`num_ctx` is a **request**. Under VRAM pressure Ollama silently grants less —
`ollama ps` reported `CONTEXT 4096` against the requested 8192.

Stage 2 replays the system prompt, the ~10,900-character Stage 1 prompt, the
Stage 1 response and the Stage 2 question. It is by far the longest prompt in
the pipeline. Within a 4096 window the budget clamp in `_fit_budget` collapses
to its floor:

| granted window | Stage 1 budget | Stage 2 budget |
|---|---|---|
| 4096 (what was granted) | 1243 | **128** |
| 8192 (what was requested) | 1536 | 1536 |

A thinking model handed 128 tokens spends every one of them on its reasoning
channel and returns an **empty answer**. No verdict line was ever written, so
the repair call fired, and when that also failed the session fell back to the
ML majority.

This also explains why raising GLM's `num_predict` in V29.1 made it *worse*
rather than better: the request was never the binding constraint. Pushing it up
only drove generation further past a window that had already been cut in half.

**Fixed by:** reading the granted context from `/api/ps` (`refresh_actual_ctx`)
and clamping against that rather than the requested value; running Stage 2 on
the answer channel only (`STAGE2_ANSWER_CHANNEL_ONLY`), since Stage 1 is where
the reasoning belongs and Stage 2 is a commitment step; and disabling the
reasoning channel outright below `THINK_MIN_BUDGET` (512) rather than letting it
consume a budget too small to finish in.

### 1.2 The granted context varies between runs — and that is the irreproducibility

| run | GLM | Llama |
|---|---|---|
| 19 Aug | 0.46, 37 stated | 0.98 |
| 20 Aug | 0.66, 14 stated | 0.96 |

Same code, same seed, same data, temperature 0. Greedy decoding cannot do that.

The V30 healthcheck, run with the desktop applications closed, reports
`granted context 8192` for `llama3` and `glm4`. Earlier `ollama ps` output on
the same machine reported `CONTEXT 4096` for the same `llama3`. **The window
Ollama grants depends on the VRAM free at load time, so it varies from run to
run** — and per §1.1 that single number swings GLM's Stage 2 budget between
1536 and 128. That is a sufficient mechanism for the observed instability, and
it is now directly evidenced rather than inferred.

An earlier draft of these notes attributed the instability to timeouts. That was
wrong: the V29.2 calls carried no deadline, so they did not time out — they
simply took as long as they took, which is where the two-day runtime came from.
Timeouts are a real hazard that V30 now bounds, but they were not the mechanism
behind the differing accuracies.

### 1.3 A failed call was scored as a model decision

The Ollama call had **no timeout at all**, and a failure returned the string
`"Error: ..."`. No verdict could be parsed from it, so the session silently
became an ML-majority fallback — counted as the model's own answer.

**Fixed by:** every call now carries a deadline (`LLM_CALL_TIMEOUT_SEC`, 300 s).
A missed deadline is recorded as a distinct `verdict_source = 'llm_error'`,
excluded from the behavioural metrics, and reported under its own column with a
**NOT PUBLISHABLE** banner when present.

### 1.4 A timeout was being disguised as a capability problem

`_call` caught *every* exception around `think=True` and retried without
thinking. So a timeout was silently reclassified as "this model can't do
thinking" and never surfaced. Only a genuine unsupported-feature error now
disables the channel; a transport fault propagates and is recorded.

### 1.5 Model selection

Measured on the target machine (Quadro M4000, 8 GB):

| model | size | residency | verdict |
|---|---|---|---|
| `llama3:latest` | 5.2 GB | 100% GPU, ctx 8192 | usable |
| `glm4:latest` | 5.3 GB | 100% GPU, ctx 8192 | usable |
| `glm-4.7-flash:latest` | 19 GB | 67%/33% CPU/GPU | 2.4x the card — never fits |
| `qwen3.5:latest` | 10.3 GB | **2% GPU** | larger than the card — never fits |

`qwen3.5` is the important one. At 10.3 GB it exceeds the entire 8 GB card, so
no amount of closing applications will make it fit. Every previous Qwen result,
including the 1.0000, was produced by a model running almost entirely on the
CPU. That does not make the outputs wrong — CPU inference is still inference —
but it is the direct source of the multi-day runtime, and it means Qwen was
never measured under the same conditions as the other two models.

The GLM default is now `glm4:latest`. A Qwen variant that fits in 8 GB must be
chosen before the study runs; `healthcheck_v30.py` lists the installed models by
size and marks which ones fit.

---

## 2. Three gates, so a broken run costs minutes rather than two days

Run them in order. Each must exit 0 before the next is worth running.

| gate | time | needs | catches |
|---|---|---|---|
| `python selftest_v30.py` | seconds | nothing | verdict/section parsing, provenance, budget arithmetic, SHAP sign, renamed-symbol `NameError`s |
| `python integration_test_v30.py` | ~1 min | nothing | the seven steps wired together: real classifiers, real SHAP, scripted model, 41 assertions |
| `python healthcheck_v30.py --n 50` | minutes | Ollama | VRAM residency, the context actually granted, verdict rate, error rate, and the **projected total hours** |

Neither of the first two needs Ollama, a GPU, or your dataset — the integration
test generates synthetic sessions and trains the seven classifiers on them.

Every unusable run this project has paid for was detectable by one of these
three within minutes of starting:

| version | what went wrong | which gate catches it now |
|---|---|---|
| V25 | a global 900-token cap truncated Qwen's reasoning | selftest (budget arithmetic) |
| V26 | the section parser required a colon → "XAI referenced 0/50" | selftest (section formats) |
| V29 | a helper renamed at call sites but not at its definition | selftest (symbol existence) |
| V29.1 | GLM's Stage 2 on a 128-token floor with thinking on | selftest + healthcheck |
| V29.2 | failed calls silently scored as ML fallbacks | integration test + healthcheck |
| all | granted context varying with free VRAM | healthcheck (granted ctx column) |
| all | model not resident in VRAM → hours per session | healthcheck (projected hours) |

### A note on the healthcheck's own first version

The first version of `healthcheck_v30.py` used one fixed evidence block for all
six probes. A normal session at ratio 1.004 was therefore shown SHAP values
pointing toward Attack, and an attack at 1.780 was shown a vote tally of
"0 Attack, 4 Normal". Its accuracy column measured whether a model could resolve
deliberately contradictory evidence — not whether it could do the task — and it
made Llama look worse (4/6) and GLM look constant-Normal (3/6) for reasons that
were the probe's fault. Its prompt was also 1,445 characters against the study's
~10,900, so it did not exercise the Stage 2 squeeze it exists to detect.

Both are fixed: the evidence block now tracks the session as SHAP does on real
data, the prompts are the framework's own system prompt and Stage 2 question,
and `selftest_v30.py` asserts the consistency so it cannot regress. Any
correctness figure from a healthcheck run before this fix should be discarded.

### The in-run circuit breaker

Even after the gates pass, `run_classification` stops the run after five
sessions if more than 20% are LLM errors, or if the measured per-session cost
projects past 8 hours for that model. It reports the measured figures and points
at the healthcheck. It does not stop a *slow* run — it stops a *broken* one.

---

## 3. Reporting changes

- **Decision provenance** now has four columns: `stated`, `repaired`,
  `ML fallback`, `LLM error`. Only the first two are the model's own judgement;
  the fourth means no model was involved at all.
- **Transport integrity** is printed *before* any accuracy table, because a
  non-zero error rate makes every number below it a measurement of the GPU.
- **Format compliance** excludes sessions that never reached the model —
  counting a timeout as a formatting failure blames the model for the hardware.
- **XAI faithfulness** reports `n/a` instead of `0.0000` for an empty split. A
  model that got everything right has no incorrect sessions to measure, and
  "faithfulness when wrong = 0.0000" read as a catastrophic finding when it was
  an absent denominator.

---

## 4. What is still open, and is not a code problem

These are experimental-design questions the code cannot settle:

1. **Reproducibility gate.** Re-run Scenario D, seed 42, workers 1, and confirm
   it matches. Until a run reproduces, no comparison between runs means anything.
2. **Qwen's 1.00.** The `ratio > 1.3` oracle also scores 1.0000 on the balanced
   sample. Qwen is tracking dataset separability, not demonstrating superiority.
   Seeds 7, 13 and 99 plus the aggregator are what retire a bare 1.00.
3. **Whether RAG and LTM contribute anything.** Scenario A and Scenario D
   produced byte-identical confusion matrices for Llama and Qwen across three
   runs despite 941 more characters of prompt in A. Before publishing that as a
   finding, confirm `ltm_cases_referenced` and `rag_sources_used` were non-zero —
   otherwise the finding is that they were never live.
4. **The XAI ablation.** `use_xai=False` on the same samples is the only direct
   evidence that SHAP/LIME cause the improvement. It has still not been run.
