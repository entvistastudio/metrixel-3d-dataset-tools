# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2026 EntertainmentVista Studio Pte. Ltd.                       *
#  *  All rights reserved.                                                       *
#  *                                                                             *
#  *  Metrixel is a custom-built C/C++ application designed to read 3D           *
#  *  models in FBX format, with optional support for USD files. The application *
#  *  processes these inputs to generate a "Unified Data Representation" which   *
#  *  includes rendered images, textures, animations, and metadata for use in    *
#  *  training and evaluation.                                                   *
#  *                                                                             *
#  *  The above copyright notice and this permission notice shall be included in *
#  *  all copies or substantial portions of the Software.                        *
#  *                                                                             *
#  *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
#  *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
#  *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
#  *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
#  *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
#  *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
#  *  DEALINGS IN THE SOFTWARE.                                                  *
#  *                                                                             *
# *******************************************************************************


"""Convert Metrixel motion-retargeting solver diagnostics to TensorBoard.

Metrixel writes ``retarget_solver.jsonl`` beside each retargeted animation when solver
diagnostics are enabled (Preferences > General > Diagnostics, or ``--retarget_diagnostics``).
It is newline-delimited JSON with three record types:

``{"type": "run", ...}``
    One header per file: frame/bone counts, every loss weight, the clamp settings. Emitted so a
    convergence curve can be read together with the configuration that produced it — a curve
    without its weights is not a measurement.

``{"type": "iteration", ...}``
    One per Ceres iteration: ``cost``, ``cost_change``, ``gradient_max_norm``, ``step_norm``,
    ``relative_decrease``, ``trust_region_radius``, ``successful``, ``cumulative_time_s``.

``{"type": "summary", ...}``
    One per file: the standard-retarget-vs-refined metrics, and ``drift_clamp_alpha`` — the
    fraction of the solve that survived the acceptance clamp.

Read ``drift_clamp_alpha`` alongside the cost curve. A solve can converge beautifully and still
ship almost none of its own result: alpha is the fraction that was kept, so a low alpha means the
curve describes a pose the user did not receive.

Usage
-----
    python retarget_to_tensorboard.py <path-to-jsonl-or-directory> [--logdir runs]
    tensorboard --logdir runs

Point it at a directory to convert a whole sweep at once — each file becomes its own TensorBoard
run, named after its parent directory, so several configurations overlay on the same axes.

Requires ``tensorboard`` (or ``torch``, whose ``torch.utils.tensorboard`` wraps the same writer).
Both are already present in a typical ML environment; neither is needed by Metrixel itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Scalars promoted to their own TensorBoard tag. Everything else in a record is still written,
# grouped under its record type, so the file stays the source of truth rather than this list.
ITERATION_SCALARS = (
    "cost",
    "cost_change",
    "gradient_max_norm",
    "step_norm",
    "relative_decrease",
    "trust_region_radius",
    "cumulative_time_s",
)


def _writer(logdir: Path):
    """Return a SummaryWriter, preferring the standalone tensorboard package over torch."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: WPS433
        return SummaryWriter(log_dir=str(logdir))
    except ImportError:
        pass
    try:
        from tensorboardX import SummaryWriter  # noqa: WPS433
        return SummaryWriter(logdir=str(logdir))
    except ImportError:
        pass
    sys.exit(
        "Neither torch.utils.tensorboard nor tensorboardX is available.\n"
        "Install one of:  pip install torch   |   pip install tensorboardX tensorboard"
    )


def read_records(path: Path):
    """Yield parsed records, skipping malformed lines rather than aborting the run.

    A diagnostics file is written incrementally during a solve, so a cancelled or crashed run
    leaves a valid prefix and one truncated final line. That file is still worth reading — it is
    often the most interesting one.
    """
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"  ! {path.name}:{lineno}: unparseable line, skipped", file=sys.stderr)


def convert(jsonl: Path, logroot: Path, run_name: str | None = None) -> bool:
    records = list(read_records(jsonl))
    if not records:
        print(f"  ! {jsonl}: no records", file=sys.stderr)
        return False

    # Name the run after the containing directory (`retarget_<source>`), which is what distinguishes
    # one source clip from another in a batch, and fall back to the file stem.
    name = run_name or jsonl.parent.name or jsonl.stem
    writer = _writer(logroot / name)

    header = next((r for r in records if r.get("type") == "run"), {})
    summary = next((r for r in records if r.get("type") == "summary"), {})
    iterations = [r for r in records if r.get("type") == "iteration"]

    for rec in iterations:
        step = int(rec.get("global_iteration", rec.get("iteration", 0)))
        for key in ITERATION_SCALARS:
            if isinstance(rec.get(key), (int, float)):
                writer.add_scalar(f"solver/{key}", float(rec[key]), step)
        # A rejected LM step is not a failure — it is the trust region doing its job — but a run
        # that rejects most of its steps is worth seeing at a glance.
        if "successful" in rec:
            writer.add_scalar("solver/step_successful", float(rec["successful"]), step)

    for key, value in summary.items():
        if key != "type" and isinstance(value, (int, float)):
            writer.add_scalar(f"summary/{key}", float(value), 0)

    # The configuration goes in as hyperparameters when the summary provides metrics to pair with
    # them, so a sweep can be sorted by outcome in TensorBoard's HPARAMS tab.
    hparams = {k: v for k, v in header.items()
               if k not in ("type", "schema") and isinstance(v, (int, float, str))}
    metrics = {f"final/{k}": float(v) for k, v in summary.items()
               if k != "type" and isinstance(v, (int, float))}
    if hparams and metrics:
        try:
            writer.add_hparams(hparams, metrics)
        except Exception as exc:  # noqa: BLE001 - hparams support varies across writer versions
            print(f"  ! {name}: hparams not written ({exc})", file=sys.stderr)

    # Also as text, so the configuration is readable without the HPARAMS tab.
    if header:
        writer.add_text("config", "```json\n" + json.dumps(header, indent=2) + "\n```", 0)

    writer.close()

    alpha = summary.get("drift_clamp_alpha")
    note = ""
    if isinstance(alpha, (int, float)):
        note = f", alpha={alpha:.3f}"
        if alpha < 0.5:
            note += f" (only {alpha * 100:.0f}% of the solve was kept)"
    print(f"  {name}: {len(iterations)} iterations{note}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Metrixel retarget solver diagnostics (JSONL) to TensorBoard event files."
    )
    parser.add_argument("input", type=Path,
                        help="retarget_solver.jsonl, or a directory searched recursively for them")
    parser.add_argument("--logdir", type=Path, default=Path("runs"),
                        help="TensorBoard log root (default: ./runs)")
    parser.add_argument("--glob", default="retarget_solver.jsonl",
                        help="filename pattern when input is a directory")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"not found: {args.input}", file=sys.stderr)
        return 1

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(args.input.rglob(args.glob))
        if not files:
            print(f"no files matching '{args.glob}' under {args.input}", file=sys.stderr)
            return 1

    print(f"writing to {args.logdir.resolve()}")
    converted = sum(1 for f in files if convert(f, args.logdir))
    print(f"\n{converted}/{len(files)} converted.  tensorboard --logdir {args.logdir}")
    return 0 if converted else 1


if __name__ == "__main__":
    raise SystemExit(main())
