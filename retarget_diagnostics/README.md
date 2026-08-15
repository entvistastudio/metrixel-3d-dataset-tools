# Retarget solver diagnostics → TensorBoard

> **This reads output from a feature that is still in development.** The motion-retargeting
> refinement described here is unreleased, no shipped Metrixel version writes these files, and no
> release has been committed to. The format may change, and the feature may not ship. The tool is
> published ahead of it so the format can be reviewed and so this repo stays the source of truth for
> the bundled tools — not because the capability is available.

Metrixel's motion-retargeting optimizer emits per-iteration solver telemetry as newline-delimited
JSON. This converts it to TensorBoard event files so a retarget can be inspected with the same tools
you use for training runs.

## Getting the data

Enable diagnostics, then retarget as usual:

- **GUI** — Preferences → General → Diagnostics (on by default)
- **CLI** — `--retarget_diagnostics=true`

Each retargeted source writes `retarget_solver.jsonl` beside its animation:

```
<output>/assets/<id>/retarget_<source>/
    retarget_solver.jsonl
    retarget_metrics.md
```

## Converting

```bash
pip install -r requirements.txt

# one run
python retarget_to_tensorboard.py path/to/retarget_solver.jsonl

# a whole batch or parameter sweep — every file found becomes its own TensorBoard run
python retarget_to_tensorboard.py path/to/output --logdir runs

tensorboard --logdir runs
```

## What is in the file

Three record types, one JSON object per line.

| type | count | contents |
|---|---|---|
| `run` | 1, first | frames, bones, key vertices, every loss weight, clamp settings |
| `iteration` | one per solver iteration | `cost`, `cost_change`, `gradient_max_norm`, `step_norm`, `relative_decrease`, `trust_region_radius`, `successful`, `cumulative_time_s` |
| `summary` | 1, last | `drift_clamp_alpha`, final loss split by term, and standard-retarget-vs-refined metrics |

The header travels with the curve on purpose: a convergence plot without the weights that produced
it cannot be compared against another run.

## Reading it

**Start with `drift_clamp_alpha`.** The optimizer's result is bounded before it ships — the pose is
blended back toward the standard retarget until it sits inside a displacement budget, and `alpha` is
the fraction that survived. A solve can converge beautifully and still deliver almost none of its own
result. At `alpha = 0.2`, four fifths of what the cost curve describes was clamped away, and the
curve is not describing the animation you received.

**`cost` is the objective, not the quality.** It is a weighted sum of contact, smoothness and
regularization terms; the weights are in the header. A lower final cost on different weights is not
a better result, only a different problem.

**Rejected steps are normal.** `successful = 0` means Levenberg–Marquardt rejected a trial step and
shrank its trust region. A run where most steps are rejected is worth a look; a few are not.

**The summary metrics are physical, and partial.** `foot_sliding` and `jerk` are what the feature
targets. `contact_fidelity` is computed on joint origins and is dominated by pairwise distance, so it
cannot distinguish body shapes that share a skeleton — read it as near-neutral by construction rather
than as a result.

## Notes

The JSONL is plain text and self-describing, so TensorBoard is a convenience rather than a
requirement — `pandas.read_json(path, lines=True)` gives you the same data as a frame. Records are
flat for exactly that reason.

A cancelled or crashed solve leaves a valid prefix and one truncated line. The converter skips the
bad line and keeps going; those files are often the interesting ones.
