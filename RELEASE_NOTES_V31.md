# EV-IDS-Agent V6 — Triple LLM, V31

Pipeline, prompts, the seven-step detection flow, the A/B/C/D scenarios and the
V23-style per-session explainability output are **unchanged**. Everything below
is a fix to how the framework talks to the models and to how it reports what
happened.

---

## 0. What changed in V31

V31 is V30 with three defects fixed, all of them found by running it on real
hardware rather than by reading it. The pipeline, prompts, seven-step flow,
scenarios and per-session explainability output are unchanged.

### 0.1 A truncated model no longer destroys the whole comparison

A 200-sample run stopped Qwen at session 5 and then died:

```
ValueError: Found input variables with inconsistent numbers of samples: [200, 5]
```

Llama and GLM had both completed all 200 sessions and their results were lost
with it, after about an hour of GPU time. Every cross-model figure assumed the
three result sets were the same length **and in the same order**. Two rules now
apply: each model's own accuracy is computed over the sessions *it* completed,
and anything that *compares* models uses only the sessions all three share.

The second rule matters more than the crash. McNemar and Cohen's kappa are
paired tests — zipping lists of different lengths pairs unrelated sessions and
still prints a p-value. The zone table had the same defect through positional
indexing: it compared one model's third row against another's third row by
*position* rather than by session id. Silently wrong is worse than loudly broken.

### 0.2 The circuit breaker no longer aborts a healthy run for being slow

Qwen was stopped at session 5 of 200 while reporting 5/5 correct, zero
fallbacks, zero LLM errors, zero timeouts, zero budget clamps and the full
8192-token context. Not one health signal was bad. It was killed for projecting
12.0 h against an 8 h limit — while the comment above it read *"It does not stop
a slow run, it stops a broken one."*

The breaker's only abort condition is now the LLM error rate, which is the one
case where continuing produces numbers describing the GPU rather than the model.
Projected duration prints a single notice with the measured per-session rate and
lets the run continue. Sample size and patience are the researcher's decisions;
a slow result is still a result. `healthcheck` likewise reports duration as a
cost, not a fault.

### 0.3 The XAI cache survives a version bump

The cache is validated by a fingerprint of the trained models, not by the
framework version, so it is now named `xai_cache.pkl` rather than carrying a
version number. An existing `xai_cache_v30.pkl` is migrated on first run. Bumping
the name every release would silently discard a valid cache and pay for a full
SHAP recompute — minutes of KernelExplainer work — for no benefit.

---

## 0b. Paper metrics (V31, metrics only — pipeline unchanged)

`ev_ids_agent_v6_triple_llm_v31.py` is byte-for-byte unchanged by this work, and
a standard run executes exactly the same detection pipeline as before. Every
addition reads the results JSON a run saves.

### `paper_metrics_v31.py`

Run automatically at the end of every run, or by hand on any set of results:

```
python paper_metrics_v31.py results/triple_llm_v6_v31_*.json
```

It writes every table as CSV **and** LaTeX (booktabs) and every figure as PNG
**and** PDF into `results/paper/<timestamp>/`, with a `README.md` index.

| Output | IDS-Agent paper | What it shows |
|---|---|---|
| `table6_per_class_fscore` | Tables 6/7 | F-score per class (Normal, Attack, Macro Avg) for RF, LR, KNN, MLP, DT, SVC, GB, the majority vote and each LLM agent |
| `table8a_sensitivity_posthoc_*` | Table 8, option (a) | Aggressive / Balanced / Conservative operating points from one run's own confidence. No re-run, no threshold fitted to the labels |
| `table8b_sensitivity_prompted_*` | Table 8, option (b) | Runs with the paper's A.4 prompt instruction (run mode 4) |
| `table9_ltm_lambda_*` | Table 9 | λ1 (recency) and λ2 (similarity) of the long-term-memory retrieval, the paper's Eq. 1 (run mode 5) |
| `figure8_confidence_by_zone`, `_by_correct` | Figure 8 | Confidence histograms per classifier **and per LLM agent** |
| `table_detection_metrics` | — | Acc, balanced acc, precision, recall/DR, specificity, FAR, FNR, F1, macro-F1, weighted-F1, MCC, ROC-AUC, PR-AUC, with TP/TN/FP/FN |
| `table_statistics` | — | Bootstrap 95% CIs for accuracy, macro-F1 and MCC; McNemar for agent vs majority vote, vs best single classifier, vs the ratio rule |
| `table_llm_agreement` | — | Pairwise McNemar and Cohen's κ, plus Fleiss' κ across the three LLMs |
| `table_calibration`, `figure_calibration` | — | ECE, Brier, reliability diagram, risk-coverage curve and AURC |
| `figure_roc_pr` | — | ROC and precision-recall curves |
| `table_difficulty_zones` | — | Detection rate and false-alarm rate per ratio band |
| `table_efficiency` | — | Latency per stage, total hours, prompt and response size |
| `table_provenance_and_retrieval` | — | Where each verdict came from, and whether memory and RAG ever retrieved anything |
| `table_xai_agreement` | — | SHAP–LIME top-feature agreement per classifier; XAI faithfulness |
| `error_analysis.csv` | — | Every misclassified session with its ratio, zone and verdict source |
| `table_seed_robustness` | — | Mean ± std across sampling seeds |
| `table_ablation` | — | Scenarios A/B/C/D × XAI on/off, Δ vs Scenario D |
| `table_temperature_*`, `figure_temperature_*` | — | Effect of decoding temperature |

Every core metric was checked against scikit-learn on 3,000 random cases,
degenerate ones included, and agrees to machine precision (≤ 2.2e-16); ROC-AUC
and PR-AUC agree with tied scores.

### Things to know before quoting these numbers

- **Figure 8 is not an out-of-distribution result.** The paper's split is known
  vs held-out attack types. This dataset has no held-out attack family, so the
  figure splits by difficulty zone and by correct vs incorrect, and its title
  says so.
- **Attack sub-types are not broken out.** Classes are Normal and Attack.
- **Table 8a is post-hoc.** "Sure" is the LLM's stated `CONFIDENCE: high`
  (or, with `--sensitivity-source composite`, composite confidence ≥ 0.7 —
  the paper's own low-confidence threshold from its A.5).
- **"Best ML"** is the classifier with the highest macro-F1 on the same test
  sessions, which favours the baseline; the McNemar comparison is therefore
  conservative for the agent.
- **Check the retrieval table before any memory or RAG claim.** If
  sentence-transformers is missing, retrieval silently returns nothing and a
  "with memory" run is really a "without memory" run. This is the most likely
  explanation of the earlier finding that Scenarios A and D gave identical
  results.
- **LLM-error sessions** stay in the tables so n matches the run, but the
  detection table warns whenever an agent has any.

### Runner changes (reporting and saving only)

- **Results are saved before the comparison prints**, and saved even if one
  model returns nothing. Previously a reporting crash, or one empty model,
  discarded every completed session of the other two.
- Each session row now also records `xai_results` (already returned by
  `detect()`, just not saved), for the SHAP–LIME agreement table.
- Run metadata records `sensitivity`, `ltm_lambda`, `temperature` and
  `self_consistency_k`, so the metrics can group runs without parsing filenames.
- **Run mode 4 (Table 8b)** adds the paper's sentence to the Stage 2 question on
  the agent *instance* only, for the three levels. **Run mode 5 (Table 9)**
  replaces the memory retriever on the agent *instance* with the same formula
  with λ1/λ2 as parameters, and refuses to start if memory cannot retrieve
  anything. Neither touches the agent module, and at (0.3, 0.7) the λ retriever
  is checked to return exactly what the agent's own method returns. Both run
  unattended, with one LLM by default as in the paper.

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

### 1.1b GLM's poor scores since V23 were configuration, not the model

Six identical probes, one machine, temperature 0. Every difference below comes
from settings in this repository:

| repeat_penalty | num_predict | stated | correct | what happened |
|---|---|---|---|---|
| 1.20 | 1536 | 0/6 | 0/6 | token corruption, model confused by its own output |
| 1.05 | 1536 | 3/6 | 3/6 | corruption gone; answers truncated |
| 1.05 | 3072 | **5/6** | **5/6** | all direct, no repair needed |

Two independent faults, both ours:

**repeat_penalty 1.20** was the highest of any model, applied to the model whose
reasoning channel quotes the evidence block most heavily. Every repeated token
was penalised and the sampler was pushed off the correct spelling onto a near
miss — `"RequestedD emand"`, `"kWhDeli vered"`, `"Delivrerd Energy"` — after
which GLM reasoned about corruption it had produced itself. It was raised to
1.20 to break a runaway generation loop that `num_predict` already bounds.

**num_predict 1536** was left behind when the sibling `glm4` key was raised to
3072, so `glm-4.7-flash` never received the fix. It binds because this model
emits its reasoning channel even when `think=True` is not requested, and that
reasoning is billed against the same budget as the answer. The probes divided
exactly on reasoning length, and every failure stopped after `ATTACK_TYPE` — the
section immediately before the `Prediction` line. The model ran out of budget
mid-format; it never declined to answer.

This is the single most consequential finding in the V30 work. GLM was reported
as the weak model in every version from V23 onward — 24/50 unparseable, 28/50
fallbacks, accuracy 0.46–0.66 — and those numbers measured a misconfiguration.
Any GLM comparison in earlier drafts should be withdrawn rather than reinterpreted.

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
| `qwen3.5:latest` | 6.1 GB weights / 10.3 GB loaded | **2% GPU** | weights fit; weights + 8k KV cache do not |

`qwen3.5` is the important one, and it needs care: `ollama list` reports
**6.1 GB**, which is the weights on disk, while `/api/ps` reported a **10.3 GB**
footprint — weights plus the KV cache at `num_ctx 8192`. The weights alone fit
in 8 GB; the loaded model does not fit in the 6.3 GB that was free. So this is
not "larger than the card": it is recoverable by freeing the remaining VRAM, or
by halving `num_ctx`, which roughly halves the cache.

What is not recoverable is the past data. Every previous Qwen result, including
the 1.0000, was produced by a model running at ~2% GPU residency. That does not
make the outputs wrong — CPU inference is still inference — but it is the direct
source of the multi-day runtime, and it means Qwen was never measured under the
same conditions as the other two models.

The advisor in `healthcheck_v31.py` now distinguishes weights from loaded
footprint and states which models fit at 8192, which fit only at 4096, and which
cannot fit at all.

---

## 1.6 The target models are fixed; num_ctx is the lever

The study's three models — `llama3`, `glm-4.7-flash`, `qwen3.5` — are set by the
research question. Results from different models are not comparable, so
swapping one to make a run fit a particular GPU changes what is being measured.
An earlier revision of this file defaulted to `glm4` and `qwen2.5:7b` because
they fit an 8 GB card. That was the wrong trade: it silently altered the study
to suit the hardware. The defaults are back to the targets.

Where a target does not fit, the levers are, in order:

1. **Free VRAM.** Desktop applications held 1.7–7.7 GB on the test machine.
2. **Lower `num_ctx`.** The KV cache scales with the window, so this changes
   the VRAM a model needs *without changing the model*.
   `set EV_IDS_NUM_CTX=6144`, or `healthcheck_v31.py --num-ctx 6144`.
3. **A larger GPU.** `glm-4.7-flash` is 19 GB and needs a 24 GB card.

`num_ctx` has a hard floor, computed by `min_viable_num_ctx()` from the actual
prompt sizes: Stage 2 replays the system prompt, the Stage 1 prompt, the whole
Stage 1 response and the question, and what remains must still hold a verdict.
That floor is **6144**. Below it the budget clamp collapses and the model
returns reasoning with no answer — the original GLM failure, reached by a
different route. Both the runner and the healthcheck refuse to run below it.

A substitution, if one is genuinely unavoidable, is a documented deviation to
be reported in the write-up — not a default.

---

## 2. Three gates, so a broken run costs minutes rather than two days

Run them in order. Each must exit 0 before the next is worth running.

| gate | time | needs | catches |
|---|---|---|---|
| `python selftest_v31.py` | seconds | nothing | verdict/section parsing, provenance, budget arithmetic, SHAP sign, renamed-symbol `NameError`s |
| `python integration_test_v31.py` | ~1 min | nothing | the seven steps wired together: real classifiers, real SHAP, scripted model, 41 assertions |
| `python healthcheck_v31.py --n 50` | minutes | Ollama | VRAM residency, the context actually granted, verdict rate, error rate, and the **projected total hours** |

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

### A note on the healthcheck's own probes

The healthcheck needed two rounds of correction, both the same defect in
different places, and both found by running it rather than by reading it.

**Round one.** One fixed evidence block was used for all six probes, so a normal
session at ratio 1.004 was shown SHAP values pointing toward Attack and an
attack at 1.780 was shown a vote tally of "0 Attack, 4 Normal". Its accuracy
column measured contradiction-resolution, not the task. Its prompt was also
1,445 characters against the study's ~10,900, so it did not exercise the Stage 2
squeeze it exists to detect.

**Round two.** The SHAP block was fixed to track the session, but the RAG and
long-term-memory blocks were left constant and attack-flavoured — so a
ratio-1.004 session was still being shown two paragraphs about energy theft and
a recalled case of meter tampering. The real pipeline picks its RAG query from
the ratio, so it never does this. Llama answered Attack on all six probes under
those conditions, which is not evidence about Llama.

Also corrected in round two: the healthcheck omitted the pipeline's repair call,
so a model that needed one was recorded as never answering; and the Stage 1
instruction asked for a prediction, contradicting the framework's own system
prompt, which reserves the verdict for Stage 2.

Every block now tracks the session, the repair call is included, and
`selftest_v31.py` asserts all of it — 75 assertions — so it cannot regress.
Raw Stage 1, Stage 2 and repair responses are written to
`healthcheck_transcript_<model>.txt` on every run, because diagnosing "no
verdict stated" from a counter alone is guesswork.

**Any correctness figure from a healthcheck run before these fixes should be
discarded.**

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
