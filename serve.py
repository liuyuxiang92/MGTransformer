# -*- coding: utf-8 -*-
"""serve.py — persistent MGTransformer inference server (stdin/stdout, JSON lines).

Loads a finetuned checkpoint ONCE, then scores one structure per input line for
the rest of the process's life. Avoids paying full torch/model-load startup
cost per structure — this repo's own model-loading pattern (``finetune.py``'s
``if __name__`` block, or calling ``predict.py`` fresh each time) is a full
cold start every call, and a caller like rl-matdesign's ``MGTransformerPredictor``
may issue thousands of calls across a method/budget/seed sweep.

Protocol (line-delimited, stdout is JSON-only — nothing else may write there):

* Startup: once the model is loaded, writes one line to stdout:
  ``{"status": "ready"}``
* Per request: caller writes one POSCAR/CONTCAR path per line to stdin.
* Per response: writes one line to stdout, either
  ``{"score": <float>}`` or, if that structure failed (e.g. unreadable POSCAR),
  ``{"error": "<message>"}`` — a single bad structure does not crash the
  server; it keeps serving subsequent lines.
* Shutdown: close stdin (EOF) — the server exits 0 after draining.

Usage::

    python serve.py --target formation_energy_peratom [--ckpt ...] [--device cuda:0] \\
        [--cutoff 8.0] [--max-neighbors 12] [--atom-features cgcnn] \\
        [--triplet-endpoint dst] [--triplet-pad-mode repeat]
"""
from __future__ import annotations

import argparse
import json
import sys

from predict import MGTPredictor


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True,
                    help="Finetuned checkpoint name under ckpt/finetuned/.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default=None)
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
    predictor = MGTPredictor(
        args.target, ckpt_path=args.ckpt, config_path=args.config,
        device=args.device, graph_kwargs=graph_kwargs,
    )

    sys.stdout.write(json.dumps({"status": "ready", "ckpt": predictor.ckpt_path}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        poscar_path = line.strip()
        if not poscar_path:
            continue
        try:
            score = predictor.predict(poscar_path)
            resp = {"score": score}
        except Exception as exc:  # noqa: BLE001 - one bad structure must not kill the server
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
