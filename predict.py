# -*- coding: utf-8 -*-
"""predict.py — run a finetuned MGTransformer checkpoint on a NEW structure.

Everything else in this repo (finetune.py, tutorial.ipynb) only scores structures
already baked into a pre-processed dataset file. This is the missing single-
structure inference path, built on top of ``graph_builder.py``.

Target-agnostic: the ``model:`` architecture block in every shipped
``config/*.yml`` is identical across every finetuned checkpoint (only the
checkpoint file and ``target`` name differ — see ``config/finetune.yml``'s own
comment listing every dataset's targets), so nothing here hardcodes a target.

IMPORTANT -- raw output is a Z-SCORE, not a physical unit. ``finetune.py`` trains
with ``normalize: True``, so the head regresses ``(y - mean) / std`` where
``mean``/``std`` come from that target's TRAIN SPLIT (``utils/dataset.py:307``).
``finetune.py:126-127`` un-normalizes with ``pred * std + mean`` to report real
units, but it reads those constants off the live datawrapper -- they are never
serialized, and every ``ckpt/finetuned/*/*.pt`` here is a bare ``state_dict``.

Since ``std > 0`` this is a fixed monotonic affine transform, so ranking / argmin
across candidates is unaffected. Do NOT report this number as eV without
converting it first.

The constants ARE recoverable for the JARVIS (``dft_3d``) targets -- run
``recover_calibration.py``, which replays the deterministic filter+shuffle+slice
from public data and validates the result by reconstructing ``config/finetune.yml``'s
own split sizes. Recovered values live in ``mgt_calibration.json``; convert with
``y = z * std + mean``.

"""
from __future__ import annotations

import argparse
import os
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

    def predict(self, structure: Union[str, Any]) -> float:
        """Return the RAW (uncalibrated, relative) model output for *structure*.

        *structure* is a POSCAR/CONTCAR path or an already-loaded
        ``jarvis.core.atoms.Atoms``. See module docstring re: units.
        """
        se3_graph, so3_graph = structure_to_graphs(structure, **self.graph_kwargs)
        se3_batch = Batch.from_data_list([se3_graph]).to(self.device)
        so3_batch = Batch.from_data_list([so3_graph]).to(self.device)
        with torch.no_grad():
            y_pred = self.model(se3_batch, so3_batch)
        return float(y_pred.detach().cpu().reshape(-1)[0].item())


def predict_one(
    structure: Union[str, Any],
    target: str,
    *,
    ckpt_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
    graph_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """One-shot convenience wrapper — loads the model fresh every call.

    Prefer :class:`MGTPredictor` (or ``serve.py``) when scoring more than a
    handful of structures; this pays full model-load cost every call.
    """
    predictor = MGTPredictor(
        target, ckpt_path=ckpt_path, config_path=config_path,
        device=device, graph_kwargs=graph_kwargs,
    )
    return predictor.predict(structure)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poscar", required=True, help="Path to a POSCAR/CONTCAR structure file.")
    p.add_argument("--target", required=True,
                    help="Finetuned checkpoint name under ckpt/finetuned/ "
                         "(e.g. formation_energy_peratom, ehull, 'bulk modulus').")
    p.add_argument("--ckpt", default=None, help="Override the resolved checkpoint path.")
    p.add_argument("--config", default=None, help="Override the finetune-style YAML (model: block only).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--max-neighbors", type=int, default=12)
    p.add_argument("--atom-features", default="cgcnn")
    p.add_argument("--triplet-endpoint", default="dst", choices=["dst", "src"])
    p.add_argument("--triplet-pad-mode", default="repeat", choices=["repeat", "zero"])
    return p.parse_args()


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
        device=args.device, graph_kwargs=graph_kwargs,
    )
    print(f"{score!r}")


if __name__ == "__main__":
    main()
