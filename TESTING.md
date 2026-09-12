# Testing & Evaluation

Phase 3 (Week 5) deliverable: a documented set of test queries run
against the assistant, with the expected behavior, what actually
happened, and a verdict. This satisfies the plan's milestone of having
"test results documented — a list of queries attempted and whether the
responses were correct/appropriate."

Current system under test: `phi-4-mini` (chat) + `qwen3-embedding-0.6b`
(embeddings), `MIN_SCORE=0.45`, knowledge base = *Electric Circuits*,
10th Edition (Nilsson & Riedel), 4621 chunks.

## Answerable questions (in-scope, should retrieve and answer)

| # | Query | Result | Verdict |
|---|-------|--------|---------|
| 1 | What is Kirchhoff's voltage law? | Correct definition, matched the book's wording, cited the source | ✅ Pass |
| 2 | Thevenin equations? | Pulled the actual numeric VTh/ZTh values from context correctly | ✅ Pass |
| 3 | What is the power dissipated in a 10Ω resistor with 5A current? | Math correct (250 W, via p=i²R from context) | ⚠️ Pass with caveat — claimed "as shown in the context" for a worked example, but the context only contained the *formula*, not this specific example. Misattributed a self-derived calculation as a book example. |
| 4 | If I know the power and resistance in a circuit, how do I find the current? | Correctly derived I=√(P/R) from the context's power formula | ✅ Pass (legitimate inference from context, not outside knowledge) |
| 5 | What's the difference between mesh current and node voltage methods? | Context only partially covered this; model gave the partial answer and explicitly flagged what was missing rather than filling the gap | ✅ Pass |
| 6 | Explain inductors | Same pattern — partial context, model stayed within it and flagged the gap | ✅ Pass |
| 7 | What is a transistor? | Retrieval only found a tangential chunk (the book doesn't define transistors); model reported the weak match honestly instead of inventing a definition | ✅ Pass (retrieval limitation, not a model failure — see Known Limitations) |

## Unanswerable questions (out-of-scope, should decline)

| # | Query | Result | Verdict |
|---|-------|--------|---------|
| 8 | What is the capital of France? | Correctly declined — all retrieved scores well below MIN_SCORE | ✅ Pass |
| 9 | What programming languages does the Foundry Local SDK support? | Correctly declined once the knowledge base became book-only | ✅ Pass |

## Boundary / partially-covered questions

| # | Query | Result | Verdict |
|---|-------|--------|---------|
| 10 | how electricity works? | Model listed what the context *did* cover (wiring, devices, power) and explicitly declined to explain the underlying physics, since that wasn't in context | ✅ Pass |
| 11 | What is the difference between AC and DC current? | Grounded in the context's definition of DC as constant voltage; inferred AC's behavior from that (reasonable inference), but the inference was imprecise (described AC as "constant voltage that changes direction," which is not accurate) | ⚠️ Pass with caveat — no outside-knowledge injection, but a shaky inference |
| 12 | is capacitor use which circuits? | Correctly identified the context was insufficient and declined rather than guessing | ✅ Pass |

## Known limitations (not fixed, documented instead)

- **Turkish-language queries retrieve poorly.** "kapasitör hangi
  devrelerde kullanılır?" scored below MIN_SCORE even though the English
  equivalent ("is capacitor use which circuits?") retrieved successfully.
  The embedding model (`qwen3-embedding-0.6b`) doesn't cross-match
  Turkish queries against English book content reliably. Ask questions
  in English for best retrieval.
- **Character-based chunking can cut mid-formula or mid-example** for a
  dense technical text like this one (see README). Not something we
  changed, since paragraph-aware chunking would be a bigger rewrite of
  `ingest.py`.
- **No automated citation-accuracy check.** Test #3 shows the model can
  correctly compute an answer using a context-supplied formula but
  mis-describe it as a book example. There's no code-level check for
  this - `--debug` / the web UI's debug toggle (comparing the printed
  context against the answer) is the only way to catch it, and that's
  manual.
- **Small-model instruction-following is probabilistic, not
  guaranteed.** `phi-3.5-mini` showed clearer violations of "answer only
  from context" than `phi-4-mini` does now, but the underlying risk
  class wasn't eliminated - a future test batch could still catch a
  violation `phi-4-mini` didn't produce here.

## How to re-run these tests

```powershell
python main.py --debug
```

or use the web UI's debug toggle (`streamlit run app.py`). Re-ask the
queries above (or your own) and compare the debug output (retrieval
scores and full context) against the answer, the same way this table was
built.
