# -*- coding: utf-8 -*-
"""validate_calibration.py -- close the loop on recover_calibration.py's constants.

``recover_calibration.py`` recovers ``mean``/``std`` from the dataset side alone. This
script checks them from the MODEL side: it scores HELD-OUT TEST-SPLIT structures --
whose true labels are known -- through ``predict.py``, converts with
``y = z * std + mean``, and compares against those labels and against the test metrics
``tutorial.ipynb`` recorded for the same checkpoint and the same split.

That comparison is the real gate. If the calibrated predictions reproduce the notebook's
numbers, both the constants AND the inference featurizer are faithful. If they do not,
this tells you WHICH of the two is broken -- a wrong affine map shifts/scales the
predictions uniformly (correlation stays high), whereas a wrong featurizer degrades
correlation itself, which no choice of mean/std can repair.

Why this is stronger than ``calibrate.py``: that script scores four hand-built
perovskites and asks a human to eyeball the ranking, because it has no ground truth.
Here the ground truth is the dataset's own held-out labels.

WHY THE SHIMS
-------------
Two of MGTransformer's dependencies have no build for current torch, and neither is
needed on the inference path, so they are stubbed HERE rather than by editing the repo:

* ``torch_scatter.scatter`` -> ``torch_geometric.utils.scatter``, an exact drop-in
  (same ``src, index, dim, dim_size, reduce`` semantics). Used at
  ``models/so3/utils.py:180``.
* ``dgl`` -> a stub. ``jarvis.core.graphs`` names it only in type annotations
  (evaluated at def-time, hence the import error) and in graph-batching helpers this
  path never calls; ``nearest_neighbor_edges`` / ``build_undirected_edgedata``, the two
  functions ``graph_builder.py`` actually uses, are dgl-free.

FEATURIZER SETTINGS
-------------------
``graph_builder.py``'s defaults are NOT the ones the training data was built with. The
deleted ``bin/cif2dataset_finetune_dft_3d.py`` (commit 48551c4, recoverable from git
history) records the real ones::

    --cutoff 4.0  --neighbor_strategy k-nearest  --max_neighbors 25
    --atom_features atomic_number     # overridden at the command line: the model's
                                      # atom_input_features=92 only fits `cgcnn`,
                                      # and `atomic_number` raises a shape error here

``max_neighbors`` dominates: on an unbiased 80-structure test sample, 12 -> MAE 1.22 eV
(R2 0.25) while 25 -> 0.71 (R2 0.76) and 32 -> 0.69 (R2 0.78). ``cutoff`` and
``triplet_pad_mode`` measurably do nothing (k-nearest ignores the radius; every atom
has >= 3 neighbours, so the padding branch is dead).

Usage::

    python validate_calibration.py --target mbj_bandgap --n 80 --max-neighbors 25 32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Metrics tutorial.ipynb recorded for these checkpoints on this same test split.
_REPORTED: Dict[str, Dict[str, float]] = {
    "mbj_bandgap":              {"MAE": 0.18981, "RMSE": 0.33907, "R2": 0.97999},
    "formation_energy_peratom": {"MAE": 0.02608, "RMSE": 0.03597, "R2": 0.99892},
}


def _install_shims() -> None:
    """See module docstring. Nothing under MGTransformer/ is modified."""
    from torch_geometric.utils import scatter as pyg_scatter
    m = types.ModuleType("torch_scatter")
    m.scatter = pyg_scatter
    sys.modules.setdefault("torch_scatter", m)

    class _Any:
        def __getattr__(self, k): return _Any()
        def __call__(self, *a, **k): return _Any()

    d = types.ModuleType("dgl")
    d.__getattr__ = lambda k: _Any()      # type: ignore[attr-defined]
    d.DGLGraph = _Any
    sys.modules.setdefault("dgl", d)


def test_split_records(target: str, train: int, val: int, test: int,
                       *, seed: int = 123) -> List[Dict[str, Any]]:
    """The exact held-out test split, reproduced per utils/dataset.py."""
    from jarvis.db.figshare import data as jdata
    import math

    recs = []
    for e in jdata("dft_3d"):
        v = e.get(target)
        if v is None or v == "na":
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        recs.append(e)
    idx = list(range(len(recs)))
    random.seed(seed)
    random.shuffle(idx)
    return [recs[i] for i in idx[-test:]]      # dataset.py: indices[-test_size:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="mbj_bandgap")
    ap.add_argument("--n", type=int, default=80, help="structures to score")
    ap.add_argument("--max-neighbors", type=int, nargs="+", default=[25])
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--atom-features", default="cgcnn")
    ap.add_argument("--calib", default=os.path.join(_HERE, "mgt_calibration.json"))
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workdir", default="/tmp/mgt_validate")
    args = ap.parse_args()

    calib = json.load(open(args.calib))[args.target]
    M, S = calib["mean"], calib["std"]
    train, val, test = calib["train_size"], calib["val_size"], calib["test_size"]
    print("calibration: mean={:.6f} std={:.6f}  (from {})".format(M, S, args.calib))

    _install_shims()
    from jarvis.core.atoms import Atoms
    from jarvis.io.vasp.inputs import Poscar
    from predict import MGTPredictor

    recs = test_split_records(args.target, train, val, test)
    picks = random.Random(args.sample_seed).sample(recs, min(args.n, len(recs)))
    os.makedirs(args.workdir, exist_ok=True)
    paths, y_true = [], []
    for n, r in enumerate(picks):
        p = os.path.join(args.workdir, "%04d.vasp" % n)
        Poscar(Atoms.from_dict(r["atoms"])).write_file(p)
        paths.append(p)
        y_true.append(float(r[args.target]))
    yt = np.asarray(y_true)
    print("scoring {} HELD-OUT test-split structures ({:.0%} of them are exactly 0)\n"
          .format(len(paths), float((yt == 0).mean())))

    model = MGTPredictor(args.target, device=args.device)
    print("%5s %9s %9s %9s %9s" % ("maxN", "MAE", "RMSE", "R2", "corr"))
    print("-" * 48)
    for mn in args.max_neighbors:
        model.graph_kwargs = dict(cutoff=args.cutoff, max_neighbors=mn,
                                  atom_features=args.atom_features,
                                  triplet_endpoint="dst", triplet_pad_mode="repeat")
        y = np.asarray([float(model.predict(p)) * S + M for p in paths])
        mae = float(np.abs(yt - y).mean())
        rmse = float(np.sqrt(((yt - y) ** 2).mean()))
        r2 = float(1 - ((yt - y) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        print("%5d %9.4f %9.4f %9.4f %9.4f"
              % (mn, mae, rmse, r2, float(np.corrcoef(yt, y)[0, 1])))
    print("-" * 48)
    rep = _REPORTED.get(args.target)
    if rep:
        print("tutorial.ipynb, same checkpoint & split (n={}): MAE {:.5f}  RMSE {:.5f}  "
              "R2 {:.5f}".format(test, rep["MAE"], rep["RMSE"], rep["R2"]))
        print("\nA large gap here is a FEATURIZER problem, not a calibration one: a wrong\n"
              "affine map cannot lower R2, only shift and scale the predictions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
