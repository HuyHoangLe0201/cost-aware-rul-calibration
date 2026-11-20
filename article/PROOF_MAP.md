# Proof map: where each result is proved and how it was checked

This note is for a reader who wants to verify the mathematics of the article. Section and result numbers refer to the article. "Proof" gives the block of Appendix A; "Depends on" lists what the proof uses; "Numerical checks" lists the scripts of `article/src` and their outputs in `article/work`.

| Result | Proof (Appendix A) | Depends on | Numerical checks |
|---|---|---|---|
| Proposition 1 (exact excess cost) | Proof of Proposition 1 | convexity of the pinball loss | `math_audit.py` item 1 |
| Proposition 2 (excess in probability coordinates), parts (a)-(c) | Proof of Proposition 2 | Proposition 1, layer-cake formula | `math_audit.py` item 2, `random_checks.py` (1500 and 978 random pairs), `robustness_checks.py` (A) |
| Theorem 1 (bound, asymptotic constant, exact form) | Proof of Theorem 1 | Proposition 2, Beta law of the conformal coverage, Chernoff-Hoeffding bound, Tonelli | `math_audit.py` items 3 and 13, `random_checks.py`, `verify_theory.py`, `early_stream_exact.py`, `early_stream_scan.py` |
| Corollary 1 (cumulative cost) | Proof of Corollary 1 | Theorem 1 and its exact form | `math_audit.py` item 16, `elementary_checks.py` |
| Proposition 3 (naive treatments of censoring; absorption) | Proof of Proposition 3 | elementary; the invariant of part (c) is an induction on the pool | `math_audit.py` item 4, `naive_lock.py`, `censor_eval.py` |
| Theorem 2(a) (no early step, exponential bound) | Proof of Theorem 2(a) | Chernoff-Hoeffding bound | `km_theory_check.py` |
| Lemma 1 (band increments) | Proof of Theorem 2, Lemma 1 | conditional moments of the counting process | `math_audit.py` item 6 |
| Lemma 2 (uniform control) | Proof of Theorem 2, Lemma 2 | Lemma 1, Freedman's inequality, dyadic peeling, union bound over a grid | `math_audit.py` items 7 and 12, `adversarial_lemmas.py`, `elementary_checks.py` |
| Lemma 3 (accuracy at well-supported levels), parts (a)-(b) | Proof of Theorem 2, Lemma 3 | Lemma 2(ii), Fubini on risk-set bands, Cauchy-Schwarz | `math_audit.py` item 8, `adversarial_lemmas.py` (A), `elementary_checks.py` |
| Lemma 4 (labels created at a low rank) | Proof of Theorem 2, Lemma 4 | potential argument, two supermartingales, Markov's inequality | `math_audit.py` items 9 and 12, `adversarial_lemmas.py` (B) |
| Lemma 5 (few uses of a label) | Proof of Theorem 2, Lemma 5 | Hoeffding's lemma, Ville's inequality | `math_audit.py` items 11 and 12, `adversarial_lemmas.py` (B) |
| Theorem 2(e) (late early steps) | Proof of Theorem 2(e) | Lemmas 2-5, Claims 1-2, inequality (E) | `math_audit.py` items 11-12, `lockin_tail.py`, `theorem2_constants.py`, `elementary_checks.py` |
| Theorem 2(b), (c), (d) | proofs following (e) | (b) from (e); (c) from Lemmas 2-3; (d) from (b), (c) | `lockin_check.py`, `km_theory_check.py`, `adversarial_init.py` |
| Theorem 2(f) (rate of the average cost) | Proof of Theorem 2(f), Steps 1-3 and Claim U | (e) at each distance (Step 1), Lemmas 2 and 3(a) at dyadic levels (Step 2) | `theorem2_constants.py` (threshold and constant), `elementary_checks.py`, `math_audit.py` item 14 (cumulative cost of the closed loop) |
| Proposition 4 (value of side information) | Proof of Proposition 4 | Theorem 1 | `prop4_check.py`, `math_audit.py` item 5 |
| Proposition 5 (drift and rolling window) | Proof of Proposition 5 | Hoeffding's inequality | `math_audit.py` item 10, `verify_theory.py` (C) |
| Proposition 6 (optimality of the order and the constant) | Proof of Proposition 6 | van Trees inequality (Gill and Levit, 1995) on a one-dimensional submodel | `math_audit.py` item 15 |
| Proposition 7 (dependence between prediction and ratio) | Proof of Proposition 7 | Propositions 1 and 2, Cauchy-Schwarz | `robustness_checks.py` (B), (C) |

## What the numerical checks can and cannot detect

* The Monte Carlo and grid checks detect a wrong formula or a wrong direction of an inequality. They do not verify the constants of Lemmas 2-5: the constants hold with a large margin in simulation (for example the bound of Lemma 4 is about 30 times the observed count), so a mistake of a modest factor in a constant would not be detected. The constants are verified only by the algebraic checks of `elementary_checks.py` and by the independent recomputation of `theorem2_constants.py`.
* `adversarial_lemmas.py` tests the lemmas that hold for any predictable choice of the censoring levels against six policies other than Kaplan-Meier, to avoid checking them only on the policy for which they were designed.
* The thresholds of Theorem 2(e) and (f) (about 3e20 and 1e24 in the log-normal example) are far beyond any simulation; the simulations of Section 5.4 test the behaviour, not the thresholds.

## Suggested order of reading for a reviewer

1. Lemma 2, Steps 1-4 (martingale bound and peeling): everything in Theorem 2 rests on it.
2. Lemma 3(a): the Fubini bound on risk-set bands and the passage from the Nelson-Aalen to the Kaplan-Meier estimate.
3. Claim 1, Claim 2 and the counting argument with (S1)-(S2) in the proof of (e).
4. Step 2 and Claim U in the proof of (f): the dyadic counting of the steps above the optimum.
5. The proof of Proposition 6 (the map from the submodel parameter to the quantile and the reduction to an estimator of the parameter).

## Known limits of the results

Theorem 1 needs i.i.d. ratios and a density that is positive near the quantile; Theorem 2 needs i.i.d. ratios with a continuous distribution of bounded support and a fallback multiplier above the optimal one; Proposition 6 is a local minimax bound; the rate of Theorem 2(f) is $n^{-1/2}\log n$, slower than the order $\log n/n$ of Corollary 1 that the simulations suggest. See the Discussion of the article.
