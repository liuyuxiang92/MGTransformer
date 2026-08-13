# -*- coding: utf-8 -*-
"""calibrate.py — sanity gate for graph_builder.py / predict.py before trusting them.

Several of graph_builder.py's hyperparameters are unverifiable guesses (see its
module docstring): the neighbor cutoff/max_neighbors, the `atom_features` table,
and — the biggest unknown — which endpoint's neighbors fill the 3 triplet slots
and how atoms with fewer than 3 neighbors are padded. None of this is
recoverable from this repo (the offline script that built the training datasets
isn't shipped here), so it cannot be verified by inspection alone.

What this script does instead: scores four well-known, well-studied cubic-
aristotype perovskites (SrTiO3, BaTiO3, CaTiO3, LaFeO3) with the
`formation_energy_peratom` checkpoint (or any --target you pass), across a small
grid of the unverified hyperparameters, and reports:

  1. The ranking under each hyperparameter setting — a human should eyeball
     whether it's chemically implausible (e.g. a well-known stable perovskite
     scoring far worse than the others would be a red flag).
  2. Per-compound score SPREAD across the hyperparameter grid — large spread
     means the featurizer's guesses matter more than the chemistry does, which
     means the pipeline should not be trusted for a real campaign yet.

This script does NOT hardcode "expected" formation-energy values to compare
against — literature DFT numbers vary by functional/convention and inventing a
target here would just be a second unverified guess stacked on the first. Read
the printed table and judge it against your own domain knowledge (all four
compounds are stable, well-characterized ABO3 perovskites, so scores should be
in a broadly similar range with no wild outlier).

Structures are the IDEAL CUBIC ARISTOTYPE (Pm-3m) at literature-typical
pseudocubic lattice constants — not the true (often lower-symmetry, e.g.
orthorhombic Pnma) room-temperature structures, and NOT relaxed by any
potential unless --relax-model is given. This is a coarse sanity check on the
MGTransformer featurizer, not a materials-accuracy validation.

Usage::

    python calibrate.py --target formation_energy_peratom
    python calibrate.py --target formation_energy_peratom \\
        --relax-model /path/to/perovskite_dpa4.ckpt.pt --relax-head <head>
"""
from __future__ import annotations

import argparse
import itertools
import os
import tempfile
from typing import Any, Dict, List, Tuple

from predict import MGTPredictor

# name -> (A, B, pseudocubic lattice constant in Angstrom)
# Literature-typical values for the cubic (or pseudocubic) aristotype; see
# module docstring re: these are approximations, not fully relaxed structures.
_COMPOUNDS: List[Tuple[str, str, str, float]] = [
    ("SrTiO3", "Sr", "Ti", 3.905),
    ("BaTiO3", "Ba", "Ti", 4.00),
    ("CaTiO3", "Ca", "Ti", 3.795),
    ("LaFeO3", "La", "Fe", 3.93),
]

# The hyperparameters graph_builder.py cannot verify from this repo alone.
_GRID: Dict[str, List[Any]] = {
    "cutoff": [8.0],
    "max_neighbors": [12],
    "atom_features": ["cgcnn"],
    "triplet_endpoint": ["dst", "src"],
    "triplet_pad_mode": ["repeat", "zero"],
}


def _cubic_perovskite(A: str, B: str, a: float):
    """Ideal cubic ABO3 (Pm-3m), same site convention as rl-matdesign's
    perovskite.vasp fixture: A at (0.5,0.5,0.5), B at (0,0,0), O at the three
    B-centered face centers.
    """
    from ase import Atoms

    return Atoms(
        symbols=[A, B, "O", "O", "O"],
        scaled_positions=[
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
        ],
        cell=[a, a, a],
        pbc=True,
    )


def _maybe_relax(atoms, *, model: str = None, head: str = None,
                  fmax: float = 0.05, steps: int = 200):
    if not model:
        return atoms
    from ase.optimize import LBFGS
    from deepmd.calculator import DP as DPCalculator
    try:
        from ase.filters import UnitCellFilter
    except ImportError:  # pragma: no cover
        from ase.constraints import UnitCellFilter

    work = atoms.copy()
    kwargs = {"head": head} if head else {}
    work.calc = DPCalculator(model=model, **kwargs)
    opt = LBFGS(UnitCellFilter(work, scalar_pressure=0.0))
    opt.run(fmax=fmax, steps=steps)
    return work


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="formation_energy_peratom")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--relax-model", default=None,
                    help="Optional DeepMD checkpoint (e.g. perovskite_dpa4.ckpt.pt) "
                         "to relax each structure before scoring.")
    p.add_argument("--relax-head", default=None)
    args = p.parse_args()

    keys = list(_GRID)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*_GRID.values())]

    with tempfile.TemporaryDirectory() as tmp:
        results: Dict[Tuple, Dict[str, float]] = {}
        for name, A, B, a in _COMPOUNDS:
            atoms = _cubic_perovskite(A, B, a)
            atoms = _maybe_relax(atoms, model=args.relax_model, head=args.relax_head)
            poscar = os.path.join(tmp, f"{name}.vasp")
            from ase.io import write as ase_write
            ase_write(poscar, atoms, format="vasp")

            for combo in combos:
                key = tuple(sorted(combo.items()))
                predictor = MGTPredictor(
                    args.target, ckpt_path=args.ckpt, config_path=args.config,
                    device=args.device, graph_kwargs=combo,
                )
                score = predictor.predict(poscar)
                results.setdefault(key, {})[name] = score

        print("\n=== Ranking under each hyperparameter setting ===")
        for combo in combos:
            key = tuple(sorted(combo.items()))
            row = results[key]
            ranked = sorted(row.items(), key=lambda kv: kv[1])
            ranked_str = " < ".join(f"{n}({v:+.4f})" for n, v in ranked)
            print(f"{combo}: {ranked_str}")

        print("\n=== Per-compound spread across the hyperparameter grid ===")
        for name, _A, _B, _a in _COMPOUNDS:
            vals = [results[tuple(sorted(c.items()))][name] for c in combos]
            spread = max(vals) - min(vals)
            print(f"{name}: min={min(vals):+.4f} max={max(vals):+.4f} spread={spread:.4f}")

        print(
            "\nReminder: this script does not judge pass/fail for you. Large "
            "per-compound spread, or a chemically implausible ranking under the "
            "settings you intend to actually use, means graph_builder.py's "
            "assumptions don't match this checkpoint's training-time featurizer "
            "-- fall back to the DPA4-only formation-energy proxy (see the plan) "
            "rather than trusting these numbers for a real campaign."
        )


if __name__ == "__main__":
    main()
