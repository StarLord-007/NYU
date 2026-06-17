# Data Augmentation Assessment

**Verdict up front: no augmentation method was implemented, because none of the
candidates addresses the actual failure mode of this dataset, and several would
actively damage scientific credibility.**

The Group-by-DOI collapse is driven by *between-paper distribution shift* and
*paper-level confounding* (see `GROUP_COLLAPSE_EXPLANATION.md`), not by a
shortage of rows. The dataset has 4,136 unique labelled rows but only **85
statistically independent units (papers)**, of which only 53 contain both
classes. Every augmentation method below manufactures new rows *inside the
support of the existing papers*. None of them creates a new paper, a new rig,
a new facility, or a new material system — which is precisely the axis along
which the model fails. Augmentation can therefore inflate random-split metrics
(the new rows are near-copies of training rows) while leaving unseen-paper
performance unchanged or degraded.

A second, structural argument: ~20% of the raw file already consists of exact
duplicate rows (see `AUDIT_REPORT.md` §2.1). The dataset has, in effect,
already been "bootstrap-augmented" once by the data-entry process, and the
measured consequence was +0.02 of *fake* stratified-CV ROC-AUC with zero gain
in grouped CV. That is a direct, in-corpus experiment showing what
within-distribution row multiplication does here.

---

## Method-by-method evaluation

### 1. Bootstrap (row resampling)

* **Benefit.** Variance estimation; basis for bagging ensembles.
* **Risk.** As *augmentation* (training on resampled rows) it adds nothing: the
  empirical distribution is unchanged. Rows within a paper are strongly
  correlated, so the i.i.d. bootstrap also underestimates uncertainty.
* **Expected effect on Group ROC-AUC: ~0.00.**
* **Disposition.** Not used as augmentation. The *cluster* (paper-level)
  bootstrap, which respects the dependence structure, **is** implemented — not
  as augmentation but as an ensembling strategy (`--paper-bagging` in
  `fable_train.py`) and is the statistically correct variant for clustered
  literature data.

### 2. Gaussian noise injection

* **Benefit.** Mild regularisation; can smooth decision boundaries.
* **Risk.** Physically blind. Isotropic noise on `oxygen_fraction` near the
  limiting oxygen concentration (LOC) flips true labels: at 101 kPa an opposed-
  flow PE-insulated wire sample sits within ~1–2 vol-% O₂ of its
  ignition/extinction boundary, while a "small" σ = 0.02 noise on the mole
  fraction is 2 vol-%. Label noise is injected exactly where the physics is
  most informative. Noise on `gravity_g` is even less defensible: the variable
  spans 6 decades and is bimodal (µg vs 1g); Gaussian perturbation produces
  facilities that do not exist.
* **Expected effect on Group ROC-AUC: 0.00 to −0.02.**
* **Disposition.** Rejected.

### 3. Physics-constrained noise injection

* **Benefit.** The only intellectually defensible noise scheme: perturb within
  measurement uncertainty (e.g. ±0.5 kPa pressure, ±0.25 vol-% O₂ as reported
  by the original facilities), never crossing a known flammability boundary.
* **Risk.** The required uncertainty budgets are not in the database (they
  would have to be re-extracted per paper from the primary literature), and
  the constraint "never cross the ignition boundary" presupposes knowledge of
  the boundary — which is the quantity the model is supposed to learn. Tree
  ensembles are additionally insensitive to sub-threshold jitter (splits are
  rank-based), so the expected gain is small even when done correctly.
* **Expected effect on Group ROC-AUC: +0.00 to +0.01,** at high implementation
  and review-defence cost.
* **Disposition.** Deferred. Defensible in principle, but only after per-paper
  uncertainty budgets are added to the database as columns. Without them it is
  Gaussian noise with extra steps.

### 4. SMOTE

* **Benefit.** Addresses minority-class scarcity in feature space.
* **Risk (severe, twofold).**
  1. SMOTE interpolates linearly between minority-class neighbours. The
     no-ignition class is concentrated in particular papers/campaigns; nearest
     neighbours of a no-ignition row are overwhelmingly rows *from the same
     paper* (the paper-identifiability probe reaches 98% accuracy — papers are
     compact clusters in feature space). Synthetic points therefore densify
     the *paper fingerprint*, amplifying exactly the campaign-memorisation
     the project needs to remove.
  2. Interpolation across categorical one-hots and across physically discrete
     variables (geometry, facility, µg vs 1g) creates impossible experiments,
     e.g. a sample 40% of the way between a wire and a flat sheet at 0.3 g in
     a half-drop-tower.
* **Expected effect on Group ROC-AUC: −0.01 to −0.03** (stratified CV would
  *rise*, which is how this mistake usually gets published).
* **Disposition.** Rejected.

### 5. Borderline-SMOTE

* Same mechanics as SMOTE but concentrated near the class boundary. The class
  boundary in this dataset *is* the physically interesting ignition limit, so
  synthetic label-noise lands on the most safety-critical region of the input
  space. Worse than vanilla SMOTE for this application.
* **Expected effect on Group ROC-AUC: −0.01 to −0.04.** **Rejected.**

### 6. KMeans-SMOTE

* Clusters first, interpolates within clusters. In this dataset, clusters ≈
  papers (98% probe accuracy), so KMeans-SMOTE is approximately
  "per-paper SMOTE": it reinforces campaign fingerprints by construction.
* **Expected effect on Group ROC-AUC: −0.01 to −0.03.** **Rejected.**

### 7. CTGAN

* **Benefit.** Can in principle model mixed-type tabular distributions,
  including conditional structure.
* **Risk.** GANs need orders of magnitude more independent samples than 85
  papers; with ~4k strongly clustered rows CTGAN will mode-collapse onto the
  large campaigns and replay them. Generated rows have no experimental
  provenance, which is fatal for a combustion-safety publication: a reviewer
  asking "which experiment is this row?" has no answer. There is also no
  mechanism to force the generator to respect flammability physics.
* **Expected effect on Group ROC-AUC: −0.02 to +0.01** (high variance, no
  reliable gain).
* **Disposition.** Rejected.

### 8. TVAE

* Same fundamental objections as CTGAN (provenance-free synthetic
  experiments, sample-size regime far below what deep generative models need,
  cluster structure dominates the latent space). VAEs additionally
  over-smooth, pulling synthetic points toward campaign centroids —
  again, fingerprint reinforcement.
* **Expected effect on Group ROC-AUC: −0.02 to 0.00.** **Rejected.**

---

## Summary table

| Method | Mechanism | Main risk here | Expected ΔGroup-AUC | Verdict |
|---|---|---|---|---|
| Bootstrap (row) | resample rows | no new information; ignores clustering | ±0.00 | not used (cluster bootstrap used for *bagging* instead) |
| Gaussian noise | jitter features | label noise at the LOC boundary | 0.00 … −0.02 | rejected |
| Physics-constrained noise | jitter within measurement uncertainty | uncertainty budgets absent; boundary unknown a priori | +0.00 … +0.01 | deferred until uncertainties are in the DB |
| SMOTE | interpolate minority rows | densifies paper fingerprints; impossible experiments | −0.01 … −0.03 | rejected |
| Borderline-SMOTE | interpolate at boundary | corrupts the physically critical ignition limit | −0.01 … −0.04 | rejected |
| KMeans-SMOTE | cluster-wise interpolation | clusters ≈ papers → per-campaign fingerprint amplification | −0.01 … −0.03 | rejected |
| CTGAN | deep generative replay | mode collapse on big campaigns; no provenance | −0.02 … +0.01 | rejected |
| TVAE | deep generative replay | over-smoothing toward campaign centroids | −0.02 … 0.00 | rejected |

## What would actually help instead

The binding constraint is **independent papers**, not rows. The highest-value
"data augmentation" is literature curation:

1. ingest more papers, prioritising **no-ignition-rich studies** (40 of 85
   papers currently contain zero no-ignition rows);
2. record **measurement uncertainties** per variable per paper (enables the
   one defensible noise scheme, §3);
3. record **material thermophysical properties** (thickness-normalised fuel
   load, ignition energy density) so that papers become comparable on physical
   axes rather than campaign axes.
