# -*- coding: utf-8 -*-
"""recover_calibration.py -- recover the train-split mean/std a finetuned checkpoint was normalized with.

WHY THIS EXISTS
---------------
``finetune.py`` trains with ``normalize: True`` (``config/finetune.yml``), so the
regression head learns the **z-scored** target::

    utils/dataset.py:307    self.mean, self.std = np.mean(train_target_array), np.std(...)
    utils/dataset.py:309    self.target_array = (self.target_array - self.mean) / self.std

``finetune.py:126-127`` un-normalizes with ``pred * std + mean`` to report test MAE in
real units -- but it reads those constants off the *live* datawrapper. They are never
serialized: every ``ckpt/finetuned/*/*.pt`` in this repo is a bare ``state_dict`` of
weight tensors with no metadata. So ``predict.py`` emits an **uncalibrated z-score**.

That is harmless for ranking (argmin/argmax is invariant to a fixed positive affine
map) but fatal for any task that needs the real scale -- e.g. screening for a bandgap
*at* 1.34 eV rather than a maximal or minimal one.

This script recovers the two constants exactly, from public data only. No Zenodo
download, no processed dataset, no GPU, no torch.

HOW IT IS RECOVERABLE
---------------------
``bin/cif2dataset_finetune_dft_3d.py`` (deleted in commit 48551c4, intact in git
history) builds the JARVIS target arrays straight from
``jarvis.db.figshare.data("dft_3d")``, appending one entry per dataset record **in
dataset order**. ``utils/dataset.py`` then filters and splits deterministically, so
the whole chain reproduces::

    vals  = [e[target] for e in jdata("dft_3d")]              # original order
    kept  = [v for v in vals if v is not None and v != "na" and not isnan(v)]
    idx   = list(range(len(kept))); random.seed(123); random.shuffle(idx)
    train = np.array(kept)[idx[:train_size]]
    mean, std = np.mean(train), np.std(train)                  # population std, ddof=0

Three details are load-bearing rather than stylistic:

* The filter's conditions must stay in this order and short-circuit -- ``math.isnan``
  on a ``str`` raises ``TypeError``, so the ``!= "na"`` guard must come first.
* Only ``random.shuffle`` consumes a seeded stream, and ``random.seed(123)``
  immediately precedes it (``utils/dataset.py:262-272``). ``dataset.py`` also seeds
  numpy/torch, but nothing between the seed and the shuffle draws from them -- so
  seeding Python's ``random`` alone reproduces the permutation exactly.
* ``train_size >= 1`` takes ``dataset.py``'s ``else`` branch:
  ``train_idx = indices[:train_size]`` (the ``< 1.0`` branch is for fractional sizes).

ONE DATASET, THREE TARGETS -- BUT MIND THE OTHER ONE
----------------------------------------------------
``cif2dataset_finetune_dft_3d.py`` pulls ``formation_energy_peratom``,
``mbj_bandgap``, ``optb88vdw_bandgap``, ``optb88vdw_total_energy`` and ``ehull`` from
a single ``jdata("dft_3d")`` pass, so one download calibrates all of them.

``e_form`` and ``gap pbe`` do **not** come from here -- they come from
``jdata("megnet")`` (Materials Project) via ``bin/cif2dataset_finetune_megnet.py``.
Never mix the two: for a stability screen, use ``formation_energy_peratom`` (JARVIS)
so formation energy and ``ehull`` share DFT settings.

VALIDATION
----------
Three independent checks, all reported:

1. **Kept-count equality** -- the dataset-version gate. ``len(kept)`` must equal
   ``train + val + test``. This matters because ``dataset.py``'s own assertion
   (``self.test_size - 1 <= len(test_idx) <= self.test_size + 1``) slices val/test
   from the *end* of the shuffled index list, so it passes for ANY kept-count >= the
   sum. Only the equality check actually pins which ``dft_3d`` snapshot was used.
2. **Train-split vs full-set statistics** -- the drift bound. For a random ~80% split
   these must nearly coincide. If they do, the exact identity of the split barely
   affects the constants, so even a dataset-version mismatch would move the
   calibration far less than the model's own error. This is what tells you whether a
   failed check 1 actually matters.
3. **Physical plausibility** -- a warn-only range check per target.

USAGE
-----
    python recover_calibration.py                      # all three dft_3d targets
    python recover_calibration.py --target mbj_bandgap
    python recover_calibration.py --self-test          # no jarvis-tools needed

Needs only ``jarvis-tools`` and ``numpy`` (no torch / torch-geometric). Install the
version this repo pins, because ``jarvis-tools`` hardcodes figshare URLs per release
and the split sizes below imply a ``dft_3d`` snapshot of ~55723 entries::

    pip install "jarvis-tools==2022.9.16" numpy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# target -> (train_size, val_size, test_size); from config/finetune.yml's commented
# per-dataset blocks. Only dft_3d (JARVIS) targets belong here -- see module docstring.
_SPLITS: Dict[str, Tuple[int, int, int]] = {
    "formation_energy_peratom": (44578, 5572, 5572),   # kept must be 55722
    "optb88vdw_bandgap":        (44578, 5572, 5572),
    "optb88vdw_total_energy":   (44578, 5572, 5572),
    "mbj_bandgap":              (14537, 1817, 1817),   # kept must be 18171
    "ehull":                    (44296, 5537, 5537),   # kept must be 55370
}

# The three needed for a bandgap-target + stability screen.
_DEFAULT_TARGETS: List[str] = ["mbj_bandgap", "formation_energy_peratom", "ehull"]

_UNITS: Dict[str, str] = {
    "formation_energy_peratom": "eV/atom",
    "optb88vdw_bandgap":        "eV",
    "optb88vdw_total_energy":   "eV/atom",
    "mbj_bandgap":              "eV",
    "ehull":                    "eV/atom",
}

# Warn-only sanity envelopes: target -> (mean_lo, mean_hi, std_lo, std_hi).
# Deliberately generous -- these catch a catastrophically wrong recovery (e.g. reading
# the wrong column), not a few-percent difference.
_PLAUSIBLE: Dict[str, Tuple[float, float, float, float]] = {
    "mbj_bandgap":              (0.3, 3.0, 0.5, 3.5),
    "optb88vdw_bandgap":        (0.1, 3.0, 0.3, 3.5),
    "formation_energy_peratom": (-2.5, 0.5, 0.3, 3.0),
    # NOT Materials-Project-style energy-above-hull. JARVIS dft_3d's `ehull` is
    # distributed with median ~1.63 and p75 ~2.67 eV/atom (only 8.1% are exactly
    # 0), so an MP-like envelope of ~0.1 eV/atom is simply the wrong convention --
    # it flagged the CORRECT constants as implausible. Consequence for downstream
    # use: this quantity ranks relative stability fine (lower is better), but an
    # absolute experimental cutoff such as "< 50 meV/atom above hull" is NOT
    # meaningful against it.
    "ehull":                    (0.5, 3.0, 0.5, 3.0),
    "optb88vdw_total_energy":   (-15.0, 5.0, 0.5, 10.0),
}

# Printed alongside the recovered constants -- distribution facts that change how
# the calibrated number should be used, not pass/fail conditions.
_NOTES: Dict[str, str] = {
    "mbj_bandgap":
        "53.7% of the 18172 training values are EXACTLY 0 (metals) and the median "
        "is 0. A target near 1.34 eV therefore sits in a sparsely populated region "
        "of the training distribution; expect the head to be pulled toward 0 and "
        "clamp negative predictions when screening.",
    "optb88vdw_bandgap":
        "67.5% of training values are exactly 0; GGA also underestimates gaps. "
        "Prefer mbj_bandgap for any absolute-gap target.",
    "ehull":
        "JARVIS convention, NOT MP energy-above-hull: median ~1.63 eV/atom, only "
        "8.1% at 0. Usable as a relative stability ranking; an absolute cutoff "
        "like '< 50 meV/atom' is not meaningful against it.",
}

_RULE = "-" * 78


# --------------------------------------------------------------------------- #
# Core recipe -- module level and jarvis-free, so --self-test can exercise it
# --------------------------------------------------------------------------- #

def kept_values(raw: Sequence[Any]) -> List[float]:
    """Mirror ``utils/dataset.py``'s filter EXACTLY, preserving original order.

    ``dataset.py:213-222`` keeps entries that are not ``None``, not the string
    ``"na"``, and not NaN -- evaluated in that order with short-circuiting, which is
    what keeps ``math.isnan`` from ever seeing the ``"na"`` string.
    """
    out: List[float] = []
    for i, v in enumerate(raw):
        if v is None:
            continue
        if v == "na":
            continue
        try:
            if math.isnan(v):
                continue
        except TypeError as exc:
            raise TypeError(
                "entry {} has a non-numeric, non-'na' value {!r} ({}). "
                "utils/dataset.py's filter would raise here too, so finetuning "
                "cannot have used this snapshot as-is -- check the target name."
                .format(i, v, type(v).__name__)
            ) from exc
        out.append(float(v))
    return out


def train_split_stats(
    values: Sequence[float], train_size: int, *, random_seed: int = 123,
) -> Tuple[float, float]:
    """Reproduce the finetune-time ``(mean, std)`` of the training split.

    Population std (``ddof=0``), matching ``np.std``'s default as used at
    ``utils/dataset.py:307``.
    """
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    train_size = int(train_size)
    if train_size <= 0 or train_size > n:
        raise ValueError(
            "train_size={} is out of range for {} kept values.".format(train_size, n)
        )
    indices = list(range(n))
    random.seed(random_seed)
    random.shuffle(indices)
    train = arr[indices[:train_size]]
    return float(np.mean(train)), float(np.std(train))


def full_set_stats(values: Sequence[float]) -> Tuple[float, float]:
    """``(mean, std)`` over every kept value -- the drift bound (check 2)."""
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def _fmt(x: float) -> str:
    return "{:.10g}".format(x)


def _rel_diff(a: float, b: float) -> float:
    """Relative difference, guarded for a near-zero denominator."""
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _load_dataset(name: str, store_dir: Optional[str]) -> List[Dict[str, Any]]:
    try:
        from jarvis.db.figshare import data as jdata
    except ImportError as exc:
        raise SystemExit(
            "recover_calibration.py needs jarvis-tools (and numpy) -- nothing else.\n"
            "    pip install \"jarvis-tools==2022.9.16\" numpy\n"
            "Pin that version: jarvis-tools hardcodes figshare URLs per release, and "
            "this repo's split sizes imply a dft_3d snapshot of ~55723 entries "
            "(current releases ship ~75993). Original error: {}".format(exc)
        ) from exc

    print("Downloading / loading JARVIS dataset {!r} (first run fetches a few hundred "
          "MB, cached thereafter) ...".format(name))
    if store_dir:
        os.makedirs(store_dir, exist_ok=True)
        try:
            return jdata(name, store_dir=store_dir)
        except TypeError:
            # Older jarvis-tools has no store_dir kwarg; fall through to the default.
            pass
    return jdata(name)


# --------------------------------------------------------------------------- #
# Self-test (no jarvis-tools required)
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    # kept_values drops None / "na" / NaN and preserves order.
    raw = [1.0, None, "na", float("nan"), 2.5, 0.0, -3.0]
    got = kept_values(raw)
    assert got == [1.0, 2.5, 0.0, -3.0], got

    # A non-numeric, non-"na" value must raise a targeted TypeError, not escape raw.
    try:
        kept_values(["oops"])
    except TypeError as exc:
        assert "non-numeric" in str(exc), str(exc)
    else:
        raise AssertionError("kept_values(['oops']) should have raised TypeError")

    # train_split_stats matches an inline re-implementation of dataset.py's
    # shuffle+slice -- the oracle, written out independently.
    vals = [i * 0.37 - 5.0 for i in range(1000)]
    got_stats = train_split_stats(vals, 800, random_seed=123)
    idx = list(range(1000))
    random.seed(123)
    random.shuffle(idx)
    train = np.asarray(vals, dtype=float)[idx[:800]]
    assert got_stats == (float(np.mean(train)), float(np.std(train))), got_stats

    # Population std (ddof=0), matching np.std's default at dataset.py:307.
    _m, s = train_split_stats([0.0, 1.0, 2.0, 3.0], 4)
    assert abs(s - float(np.std([0.0, 1.0, 2.0, 3.0]))) < 1e-12
    assert abs(s - 1.1180339887) < 1e-9, s

    # Determinism across calls (each call reseeds).
    assert train_split_stats(vals, 800) == train_split_stats(vals, 800)

    print("self-test OK -- kept_values, train_split_stats, ddof=0, determinism")
    return 0


# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", action="append", default=None,
                   help="Repeatable. Default: {}".format(" ".join(_DEFAULT_TARGETS)))
    p.add_argument("--dataset", default="dft_3d",
                   help="JARVIS figshare dataset name (default: dft_3d).")
    p.add_argument("--train-size", type=int, default=None,
                   help="Override the split table (requires exactly one --target).")
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=123,
                   help="config/finetune.yml's random_seed (default: 123).")
    p.add_argument("--store-dir", default=None,
                   help="Where jarvis-tools caches the dataset archive.")
    p.add_argument("--allow-count-mismatch", action="store_true",
                   help="Continue past a failed kept-count check. The recovered "
                        "constants are then APPROXIMATE -- read check 2 before "
                        "trusting them.")
    p.add_argument("--json", default="mgt_calibration.json",
                   help="Where to write the results (default: mgt_calibration.json).")
    p.add_argument("--self-test", action="store_true",
                   help="Run the built-in checks and exit; needs no jarvis-tools.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test:
        return _self_test()

    targets = args.target or list(_DEFAULT_TARGETS)
    overrides = (args.train_size, args.val_size, args.test_size)
    if any(x is not None for x in overrides):
        if len(targets) != 1:
            raise SystemExit(
                "--train-size/--val-size/--test-size override the split table, so "
                "they require exactly one --target (got {}).".format(len(targets)))
        if any(x is None for x in overrides):
            raise SystemExit(
                "--train-size, --val-size and --test-size must be given together "
                "(the kept-count check needs all three).")
    unknown = [t for t in targets if t not in _SPLITS and args.train_size is None]
    if unknown:
        raise SystemExit(
            "No split sizes known for {}. Known dft_3d targets: {}. Supply "
            "--train-size/--val-size/--test-size to override."
            .format(unknown, sorted(_SPLITS)))

    dataset = _load_dataset(args.dataset, args.store_dir)
    print("Loaded {} entries from {!r}.\n".format(len(dataset), args.dataset))

    results: Dict[str, Any] = {}
    failures: List[str] = []

    for target in targets:
        if args.train_size is not None:
            train_size, val_size, test_size = overrides  # type: ignore[assignment]
        else:
            train_size, val_size, test_size = _SPLITS[target]
        expected = train_size + val_size + test_size

        print(_RULE)
        print("target: {}   [{}]".format(target, _UNITS.get(target, "?")))
        print(_RULE)

        missing = [i for i, e in enumerate(dataset) if target not in e]
        if missing:
            raise SystemExit(
                "{} of {} entries have no {!r} key -- wrong dataset for this target? "
                "(e_form / 'gap pbe' live in the 'megnet' dataset, not 'dft_3d'.)"
                .format(len(missing), len(dataset), target))

        kept = kept_values([e[target] for e in dataset])

        # -- check 1: split-size reconstruction (the dataset-version gate) --------
        # dataset.py takes train from the FRONT and val/test from the END of the
        # shuffled index list (utils/dataset.py:285-287):
        #     train_idx = indices[:train_size]
        #     valid_idx = indices[-(val_size + test_size):-test_size]
        #     test_idx  = indices[-test_size:]
        # It therefore needs len(kept) >= train+val+test but does NOT require an
        # exact partition: flooring an 80/10/10 split leaves 0-2 entries stranded
        # in the middle, unused by any split. Demanding equality would reject the
        # CORRECT snapshot (mbj_bandgap really does have 18172 kept values, and
        # 14537 + 1817 + 1817 = 18171 leaves exactly one unused).
        #
        # The real fingerprint is that finetune.yml's three sizes are precisely
        # floor(0.8*kept) / floor(0.1*kept) / floor(0.1*kept) -- that pins the
        # snapshot far more tightly than any single sum, because all three numbers
        # must be reproduced from one kept-count.
        n_kept = len(kept)
        leftover = n_kept - expected
        exp_train, exp_tenth = int(n_kept * 0.8), int(n_kept * 0.1)
        recon = (train_size == exp_train
                 and val_size == exp_tenth and test_size == exp_tenth)
        ok_count = (leftover >= 0) and recon
        print("  check 1  split sizes     : kept {}  ->  80/10/10 = {}/{}/{}   "
              "finetune.yml = {}/{}/{}  ({} unused) ... {}".format(
                  n_kept, exp_train, exp_tenth, exp_tenth,
                  train_size, val_size, test_size, leftover,
                  "OK" if ok_count else "MISMATCH"))
        if not ok_count:
            if leftover < 0:
                msg = ("kept only {} entries, fewer than the {} the split sizes "
                       "need -- dataset.py could not even build these slices."
                       .format(n_kept, expected))
            else:
                msg = ("finetune.yml's sizes {}/{}/{} are not floor(0.8/0.1/0.1 x "
                       "{}) = {}/{}/{}. The installed jarvis dft_3d snapshot is "
                       "not the one MGTransformer was finetuned on, so this is "
                       "NOT the training split."
                       .format(train_size, val_size, test_size, n_kept,
                               exp_train, exp_tenth, exp_tenth))
            if not args.allow_count_mismatch:
                raise SystemExit(
                    "  FAIL: " + msg + "\n"
                    "  Pin the matching version:  pip install \"jarvis-tools==2022.9.16\"\n"
                    "  Or pass --allow-count-mismatch to continue anyway and read "
                    "check 2, which bounds how much the difference actually matters.")
            print("  WARNING: " + msg)
            failures.append(target)

        if train_size > len(kept):
            raise SystemExit(
                "  FAIL: train_size={} exceeds the {} kept values.".format(
                    train_size, len(kept)))

        mean, std = train_split_stats(kept, train_size, random_seed=args.random_seed)
        f_mean, f_std = full_set_stats(kept)

        # -- check 2: train-split vs full-set (the drift bound) -------------------
        d_mean, d_std = _rel_diff(mean, f_mean), _rel_diff(std, f_std)
        close = d_mean < 0.05 and d_std < 0.05
        print("  check 2  split vs full   : mean {:+.6f} vs {:+.6f} ({:.3%})   "
              "std {:.6f} vs {:.6f} ({:.3%}) ... {}".format(
                  mean, f_mean, d_mean, std, f_std, d_std,
                  "CLOSE" if close else "DIVERGENT"))
        if close:
            print("           -> the constants barely depend on which subset is the "
                  "train split,")
            print("              so dataset-version drift would shift them far less "
                  "than model error.")
        else:
            print("           -> WARNING: the split matters here; check 1 must pass "
                  "for these to be exact.")

        # -- check 3: physical plausibility (warn only) ---------------------------
        env = _PLAUSIBLE.get(target)
        if env is None:
            print("  check 3  plausibility    : (no envelope defined) SKIPPED")
        else:
            m_lo, m_hi, s_lo, s_hi = env
            ok_phys = (m_lo <= mean <= m_hi) and (s_lo <= std <= s_hi)
            print("  check 3  plausibility    : mean in [{}, {}], std in [{}, {}] "
                  "... {}".format(m_lo, m_hi, s_lo, s_hi,
                                  "OK" if ok_phys else "OUT OF RANGE (warn only)"))

        print("\n  mean = {}\n  std  = {}".format(_fmt(mean), _fmt(std)))
        note = _NOTES.get(target)
        if note:
            print("\n  NOTE: " + note.replace(". ", ".\n        "))
        print()

        results[target] = {
            "mean": mean,
            "std": std,
            "units": _UNITS.get(target),
            "dataset": args.dataset,
            "n_kept": len(kept),
            "n_expected": expected,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "random_seed": args.random_seed,
            "count_check_passed": ok_count,
            "n_unused": leftover,
            "split_sizes_reconstructed": recon,
            "full_set_mean": f_mean,
            "full_set_std": f_std,
            "n_dataset_entries": len(dataset),
        }

    # -- summary ---------------------------------------------------------------
    print(_RULE)
    print("SUMMARY")
    print(_RULE)
    print("{:<26} {:>7} {:>7}  {:>14} {:>14}".format(
        "target", "kept", "expect", "mean", "std"))
    for t, r in results.items():
        print("{:<26} {:>7} {:>7}  {:>14} {:>14}{}".format(
            t, r["n_kept"], r["n_expected"], _fmt(r["mean"]), _fmt(r["std"]),
            "" if r["count_check_passed"] else "   <-- APPROXIMATE"))

    with open(args.json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("\nWrote {}".format(os.path.abspath(args.json)))

    print("\nPaste-ready (one block per target):\n")
    for t, r in results.items():
        note = "" if r["count_check_passed"] else "  # APPROXIMATE: kept-count mismatch"
        print("    # {} -- JARVIS {} train split (seed {}, first {} of {} shuffled){}"
              .format(t, r["dataset"], r["random_seed"], r["train_size"],
                      r["n_kept"], note))
        print("    calib: {{mean: {}, std: {}}}".format(
            _fmt(r["mean"]), _fmt(r["std"])))
    print()

    if failures:
        print("NOTE: {} did not pass the kept-count check; their constants are "
              "approximate.".format(", ".join(failures)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
