# Neq and Protein Block reliability analysis

## Why this analysis was needed

The models were trained to predict two residue classes:

- `Neq = 1`: only one Protein Block state was sampled
- `Neq > 1`: more than one Protein Block state was sampled

This boundary was questioned because a residue with `Neq = 1.001` is placed in
the same class as a residue with a much larger Neq. We therefore needed to
check whether the model had learned useful information about backbone-state
variation, rather than only fitting a convenient binary label.

`Neq = 1` has a clear meaning. Neq is calculated as:

```text
Neq = exp(Protein Block entropy)
```

When `Neq = 1`, Protein Block entropy is zero. When `Neq > 1`, entropy is
nonzero. The training target can therefore be described as single-state versus
multi-state local backbone behavior.

## What the script uses

The analysis script is:

```text
prediction_endpoint_analysis/analyze_neq_pb_reliability.py
```

It uses the saved test-set predictions from all 21 trained runs: seven model
conditions, each trained with three random seeds. It also reads the Protein
Block assignments from the three ATLAS simulation replicates.

The script does not train or change any model. The predicted probabilities are
kept fixed. It only asks new questions about those saved predictions.

## What the script does

### 1. It checks whether the prediction score follows Neq

The three seed probabilities are averaged within each model condition. The
script then compares this flexibility score with the continuous Neq value.

It calculates the score in ordered Neq ranges and the Spearman correlation
between score and Neq. It repeats the correlation using only residues with
`Neq > 1`. This second calculation asks whether the score still increases
among residues that are already in the multi-state class.

Answer: the median score increased across every Neq range for all seven model
conditions. The overall correlations were 0.652 to 0.715. Among residues with
`Neq > 1`, the correlations were still positive at 0.350 to 0.412. The model
score therefore contains information about the amount of variation, not only
which side of 1 a residue falls on.

### 2. It removes residues just above 1 and recalculates performance

For a chosen value `t`, this analysis uses:

- negative residues: `Neq = 1`
- positive residues: `Neq > t`
- excluded residues: `1 < Neq <= t`

The script tests `t = 1.01, 1.05, 1.10, 1.50`, and `2.00`. No prediction is
changed. Only the residues included in the AUROC calculation change.

Answer: the original AUROC was 0.864 to 0.898. After residues with
`1 < Neq <= 2` were excluded, it increased to 0.929 to 0.953. The model
separates single-state residues from clearly multi-state residues better than
it separates single-state residues from residues only slightly above 1.

### 3. It tests stricter labels without excluding intermediate residues

This is a separate analysis. For a chosen value `t`, it defines:

- negative residues: `Neq <= t`
- positive residues: `Neq > t`

For example, at `t = 2`, residues between 1 and 2 become negative. The model
was not trained using this target; the analysis only tests how well its saved
score transfers to it.

Answer: at `t = 2`, AUROC fell to 0.797 to 0.817. The current model can rank
Neq to some extent, but it should not be described as a model trained to
separate moderate from very high Neq.

### 4. It rebuilds the labels from each ATLAS replicate

For each residue, the script reads the Protein Block states separately from
the three simulation replicates. Unassigned `Z` frames are removed. A replicate
is called single-state if it contains one valid PB state and multi-state if it
contains at least two.

This produces three independent labels for each residue. The script can then
ask whether a residue is multi-state in any replicate, in at least two
replicates, or in all three replicates.

Answer: all three replicates were usable for 46,919 of the 47,751 test
residues. The rebuilt binary labels agreed with the original labels for
99.979% of these residues. The remaining 832 residues were the first and last
two positions of each protein, where PB states were unassigned in every
replicate. They were excluded from conclusions based on the replicate data.

### 5. It checks whether agreement between replicates matters

The fixed model scores are evaluated against two useful label definitions:

- any-replicate: a residue is multi-state in at least one replicate
- unanimous: a residue is multi-state in all three replicates

Answer: AUROC was 0.862 to 0.897 for the any-replicate labels and 0.891 to
0.926 for the unanimous labels. Every model condition performed better on the
unanimous labels. The model therefore predicts PB variation more reliably when
that variation is repeated across all three simulations.

### 6. It checks how often the second PB state was observed

A residue can have `Neq > 1` even when its second PB state appears rarely. The
script repeats the replicate analysis while requiring the second state to
occupy a minimum fraction of the trajectory.

Answer: requiring higher secondary-state occupancy did not improve AUROC. At
1% occupancy, unanimous-label AUROC decreased to 0.844 to 0.878. The model
detects repeatable multi-state behavior, but it is not specifically better at
detecting residues with a common second state.

### 7. It checks whether the model is only recognizing local structure

Each residue has a dominant Protein Block state. Within each of the 16 dominant
states, the script separately tests whether the model score distinguishes
single-state from multi-state residues.

Answer: the median AUROC within a dominant state was 0.726 to 0.802, and the
multi-state residues had higher mean scores in all 16 states. The prediction
cannot be explained only by recognizing a static local PB shape, although some
PB states were easier than others.

### 8. It measures uncertainty across test proteins

The test set contains 208 proteins. The script makes 1,000 new test samples by
randomly selecting 208 whole proteins with replacement. It recalculates the
main AUROCs and score-Neq correlations for every sample. The middle 95% of
these results gives a confidence interval.

Whole proteins are sampled because residues from the same protein are related
and should not be counted as independent experiments.

## What this analysis can conclude

The `Neq = 1` target has a direct mathematical meaning: zero versus nonzero
Protein Block entropy. The saved model scores also increase with Neq and are
more accurate for PB variation reproduced in all three simulations. The model
has therefore learned information about repeatable backbone-state variation,
not only an arbitrary cutoff at 1.

The analysis does not show that `Neq = 1` is the best possible training target.
It also does not compare a newly trained binary model with newly trained
regression or hurdle models. Those questions would require retraining.

## Running the analysis

From the repository root:

```bash
python3 -m prediction_endpoint_analysis.analyze_neq_pb_reliability
```

Results are written to:

```text
results/publication_comparable_v2/analysis_neq_pb_reliability/
```

The final run audit is:

```text
results/publication_comparable_v2/analysis_neq_pb_reliability/audits/complete_analysis_audit.json
```
