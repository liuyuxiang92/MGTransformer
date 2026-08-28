# -*- coding: utf-8 -*-
"""predict.py — run a finetuned MGTransformer checkpoint on a NEW structure.

Everything else in this repo (finetune.py, tutorial.ipynb) only scores structures
already baked into a pre-processed dataset file. This is the missing single-
structure inference path, built on top of ``graph_builder.py``.

Target-agnostic: the ``model:`` architecture block in every shipped
``config/*.yml`` is identical across every finetuned checkpoint (only the
checkpoint file and ``target`` name differ — see ``config/finetune.yml``'s own
comment listing every dataset's targets), so nothing here hardcodes a target.

IMPORTANT -- the model's RAW output is a Z-SCORE, not a physical unit.
``finetune.py`` trains with ``normalize: True``, so the head regresses
``(y - mean) / std`` where ``mean``/``std`` come from that target's TRAIN SPLIT
(``utils/dataset.py:307``). Those constants are never serialized: every
``ckpt/finetuned/*/*.pt`` here is a bare ``state_dict``.

``MGTPredictor`` therefore CALIBRATES BY DEFAULT: it recovers the target name
from the checkpoint path, looks it up in ``mgt_calibration.json`` (produced by
``recover_calibration.py``), and returns ``z * std + mean`` -- i.e. **real units**
(eV, eV/atom). A target may also carry a ``min``/``max`` field there, a physical
bound applied after conversion (e.g. a bandgap cannot be negative).

If the target has no calibration entry the raw z-score is returned and a warning
is emitted once, naming the target -- so the two conventions are never silently
mixed. Pass ``calibrate=False`` (``--raw`` on the CLI) to force raw output.

"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, Optional, Union

import torch
import yaml
from torch_geometric.data import Batch

from graph_builder import structure_to_graphs

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_REPO_DIR, "config", "finetune.yml")


def _load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Read the ``model:`` architecture block from a finetune-style YAML.

    Only the ``model:`` sub-block is used (identical across every target per
    the module docstring) plus forcing ``task: 'finetune'`` — none of the
    dataset/training keys (``root``, ``train_size``, ``learning_rate``, ...)
    matter for inference.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["task"] = "finetune"
    return cfg


def _resolve_ckpt_path(target: str, ckpt_path: Optional[str] = None) -> str:
    if ckpt_path:
        return ckpt_path
    # Matches the directory naming under ckpt/finetuned/ shipped in this repo,
    # e.g. ckpt/finetuned/formation_energy_peratom/formation_energy_peratom_checkpoint_best.pt
    # (some targets have spaces in the name, e.g. "bulk modulus" -- os.path.join
    # handles that literally, no sanitization needed).
    return os.path.join(_REPO_DIR, "ckpt", "finetuned", target, f"{target}_checkpoint_best.pt")


_CALIBRATION_FILE = os.path.join(_REPO_DIR, "mgt_calibration.json")

# Targets already warned about, so a long serve.py session logs at most one line
# per target rather than one per structure.
_WARNED_UNCALIBRATED: set = set()


def target_from_ckpt_path(ckpt_path: str) -> Optional[str]:
    """Recover the target name from a checkpoint path.

    The layout shipped in this repo is ``ckpt/finetuned/<target>/<target>_checkpoint_best.pt``,
    so the parent directory name IS the target -- and those directory names are
    exactly the keys ``mgt_calibration.json`` uses. Falls back to stripping the
    ``_checkpoint*`` suffix off the filename, then gives up (returns ``None``)
    rather than guessing.
    """
    if not ckpt_path:
        return None
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    if parent and parent not in ("", ".", "finetuned", "ckpt"):
        return parent
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return stem.split("_checkpoint")[0] or None


def load_calibration(target: Optional[str], path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return ``{mean, std[, min][, max]}`` for *target*, or ``None`` if unknown."""
    if not target:
        return None
    calib_path = path or _CALIBRATION_FILE
    if not os.path.exists(calib_path):
        return None
    try:
        with open(calib_path, "r") as f:
            table = json.load(f)
    except (OSError, ValueError):
        return None
    entry = table.get(target)
    if not isinstance(entry, dict) or "mean" not in entry or "std" not in entry:
        return None
    std = float(entry["std"])
    if not (std > 0.0):
        return None
    out: Dict[str, Any] = {"mean": float(entry["mean"]), "std": std}
    for bound in ("min", "max"):
        if entry.get(bound) is not None:
            out[bound] = float(entry[bound])
    return out


class MGTPredictor:
    """Loads a finetuned checkpoint once; call ``.predict(structure)`` many times.

    This is what ``serve.py`` wraps to avoid paying full model-load cost per
    structure.
    """

    def __init__(
        self,
        target: str,
        *,
        ckpt_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
        graph_kwargs: Optional[Dict[str, Any]] = None,
        calibrate: bool = True,
        calibration_path: Optional[str] = None,
    ) -> None:
        from models.mgt import MGTransformer

        self.target = target
        self.device = device
        self.graph_kwargs: Dict[str, Any] = dict(graph_kwargs or {})

        config = _load_model_config(config_path)
        config["device"] = device
        self.model = MGTransformer(config=config, config_model=config["model"]).to(device)

        resolved_ckpt = _resolve_ckpt_path(target, ckpt_path)
        state_dict = torch.load(resolved_ckpt, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.ckpt_path = resolved_ckpt

        # Calibration: z-score -> real units. The target name is taken from the
        # checkpoint path when it was not given explicitly, so pointing at a
        # checkpoint is enough -- callers never have to name the target twice.
        self.calibration: Optional[Dict[str, Any]] = None
        if calibrate:
            name = target or target_from_ckpt_path(resolved_ckpt)
            self.calibration = load_calibration(name, calibration_path)
            if self.calibration is None and name not in _WARNED_UNCALIBRATED:
                _WARNED_UNCALIBRATED.add(name)
                warnings.warn(
                    f"No calibration entry for target {name!r} in "
                    f"{calibration_path or _CALIBRATION_FILE}; returning the RAW "
                    "z-score for this target. Ranking is unaffected, but the value "
                    "is NOT in eV. Run recover_calibration.py to add it.",
                    RuntimeWarning, stacklevel=2,
                )

    def predict(self, structure: Union[str, Any]) -> float:
        """Score *structure*, in REAL UNITS when this target is calibrated.

        *structure* is a POSCAR/CONTCAR path or an already-loaded
        ``jarvis.core.atoms.Atoms``. Returns ``z * std + mean`` (then clamped to
        any ``min``/``max`` bound) when a calibration entry exists, else the raw
        z-score -- see the module docstring.
        """
        se3_graph, so3_graph = structure_to_graphs(structure, **self.graph_kwargs)
        se3_batch = Batch.from_data_list([se3_graph]).to(self.device)
        so3_batch = Batch.from_data_list([so3_graph]).to(self.device)
        with torch.no_grad():
            y_pred = self.model(se3_batch, so3_batch)
        value = float(y_pred.detach().cpu().reshape(-1)[0].item())
        return self._calibrate(value)

    def _calibrate(self, value: float) -> float:
        """Un-normalize one raw model output, then apply any physical bound."""
        c = self.calibration
        if c is None:
            return value
        value = value * c["std"] + c["mean"]
        if "min" in c:
            value = max(c["min"], value)
        if "max" in c:
            value = min(c["max"], value)
        return value


def predict_one(
    structure: Union[str, Any],
    target: Optional[str] = None,
    *,
    ckpt_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
    graph_kwargs: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
) -> float:
    """One-shot convenience wrapper — loads the model fresh every call.

    Prefer :class:`MGTPredictor` (or ``serve.py``) when scoring more than a
    handful of structures; this pays full model-load cost every call.
    """
    predictor = MGTPredictor(
        target, ckpt_path=ckpt_path, config_path=config_path,
        device=device, graph_kwargs=graph_kwargs, calibrate=calibrate,
    )
    return predictor.predict(structure)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poscar", required=True, help="Path to a POSCAR/CONTCAR structure file.")
    p.add_argument("--target", default=None,
                    help="Finetuned checkpoint name under ckpt/finetuned/ "
                         "(e.g. formation_energy_peratom, ehull, 'bulk modulus'). "
                         "Optional when --ckpt is given: the target is then read "
                         "off the checkpoint path.")
    p.add_argument("--ckpt", default=None, help="Override the resolved checkpoint path.")
    p.add_argument("--config", default=None, help="Override the finetune-style YAML (model: block only).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--max-neighbors", type=int, default=12)
    p.add_argument("--atom-features", default="cgcnn")
    p.add_argument("--triplet-endpoint", default="dst", choices=["dst", "src"])
    p.add_argument("--triplet-pad-mode", default="repeat", choices=["repeat", "zero"])
    p.add_argument("--raw", action="store_true",
                    help="Return the uncalibrated z-score instead of real units.")
    args = p.parse_args()
    if not args.target and not args.ckpt:
        p.error("give --target (a name under ckpt/finetuned/) or --ckpt (a path).")
    return args


def main() -> None:
    args = _parse_args()
    graph_kwargs = dict(
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        atom_features=args.atom_features,
        triplet_endpoint=args.triplet_endpoint,
        triplet_pad_mode=args.triplet_pad_mode,
    )
    score = predict_one(
        args.poscar, args.target, ckpt_path=args.ckpt, config_path=args.config,
        device=args.device, graph_kwargs=graph_kwargs, calibrate=not args.raw,
    )
    print(f"{score!r}")


if __name__ == "__main__":
    main()
