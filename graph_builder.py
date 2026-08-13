# -*- coding: utf-8 -*-
"""graph_builder.py — the missing raw-structure -> model-input featurizer.

Nothing in this repo (finetune.py / pretraining.py / tutorial.ipynb / utils/dataset.py)
turns a raw structure into the ``se3_graph`` / ``so3_graph`` PyG ``Data`` objects the
model consumes — everything here only ever loads a pre-built ``*_processed.pt``
dataset via ``CrystalDataLoader``. This module fills that gap so a *new* structure
(not one of the training datasets) can be scored.

Ground truth for the field contract (verified by reading ``models/mgt.py``,
``models/se3/{utils,layers}.py``, ``models/so3/{utils,atoms}.py``,
``utils/dataset.py``'s ``CrystalDataset.__getitem__``) — see each field below.
The one thing that is NOT recoverable from this repo, because the offline script
that built the training datasets isn't included: which endpoint's neighbors fill
the "3 slots" of ``edge_nei_len``/``edge_nei_angle``, and how atoms with fewer
than 3 other bonds are padded. Both are exposed as parameters here rather than
buried as constants — see ``predict.py``'s calibration mode, which sweeps them.

Verified field contract (from ``CrystalDataset.__getitem__``, finetune-mode branch):

    se3_graph = Data(x, edge_index, edge_attr, edge_nei_angle, edge_nei_len)
    so3_graph = Data(x, edge_index, edge_attr)

* ``x`` — ``[n_atoms, atom_input_features]`` (both graphs, same value). Node
  features. ``atom_input_features: 92`` in every shipped ``config/*.yml`` matches
  CGCNN's 92-dim ``atom_init.json`` table exactly, and jarvis's own
  ``get_node_attributes(..., atom_features="cgcnn")`` is confirmed (fetched from
  the jarvis-tools source) to load from that exact table — the strongest
  available evidence, but still not verified against this specific checkpoint's
  training run (no config in this repo names the table explicitly).
* ``edge_index`` — ``[2, n_edges]`` long (both graphs, same value).
* ``edge_attr`` — ``[n_edges, 3]`` float (both graphs, same value): the RAW
  Cartesian displacement vector. Confirmed twice independently: (1)
  ``se3/layers.py``'s ``SE3_GraphEncoder.forward`` does
  ``self.edge_embedding(-0.75 / torch.norm(data.edge_attr, dim=1))`` — a scalar
  derived from the vector, so the vector itself must be raw (the model applies
  the -0.75/length transform internally, once); (2) ``so3/utils.py``'s
  ``UpdateConvEqui.forward`` does ``edge_vec = data.edge_attr`` then feeds it
  RAW into ``o3.spherical_harmonics(sh, edge_vec, ...)``, which needs the actual
  vector, not a scalar. Producible directly via jarvis's own
  ``build_undirected_edgedata`` (returns exactly this).
* ``edge_nei_len`` — se3 ONLY, ``[n_edges, 3]`` float. Fed directly into the
  SAME ``edge_embedding`` module used for the primary edge
  (``se3/layers.py:134``: ``self.edge_embedding(data.edge_nei_len.reshape(-1))``)
  with NO length-to-feature transform applied first — unlike ``edge_attr``,
  which the model transforms internally. So values here must ALREADY be in the
  ``-0.75/length`` domain when we build them; the model does not do it again.
* ``edge_nei_angle`` — se3 ONLY, ``[n_edges, 3]`` float, confirmed range
  ``[-1, 1]`` (the ``angle_embedding`` RBF's ``vmin=-1.0, vmax=1.0`` in
  ``se3/layers.py``) — bond-angle COSINES, not degrees/radians.
* ``batch`` — added by ``torch_geometric.data.Batch.from_data_list([...])``,
  not built here.

No noise fields (``edge_nei_len_noise`` etc.) — those exist only for
``task='pretraining'`` per ``dataset.py``'s finetune-mode ``__getitem__`` branch,
which builds ``se3_graph_prompt``/``so3_graph_prompt`` straight from the fields
above with no noise terms at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data


def _load_atoms(structure: Union[str, Any]):
    """Accept a POSCAR path or an already-loaded ``jarvis.core.atoms.Atoms``."""
    from jarvis.core.atoms import Atoms

    if isinstance(structure, Atoms):
        return structure
    return Atoms.from_poscar(str(structure))


def _build_triplets(
    u: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    *,
    endpoint: str = "dst",
    pad_mode: str = "repeat",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed-arity (3 slots) per-edge neighbor length/angle features.

    For each directed edge ``e = (u[e] -> v[e])`` with displacement ``r[e]``,
    find the up-to-3 *shortest other* edges sharing the chosen endpoint atom
    (``endpoint="dst"`` -> group by ``v``, i.e. the atom the edge points INTO;
    ``endpoint="src"`` -> group by ``u``). This choice is UNVERIFIED against the
    original training-time featurizer (not present in this repo) — the default
    (``dst``) follows the convention that a message-passing edge's "neighborhood
    context" naturally lives at the atom receiving it, but ``predict.py``'s
    calibration mode sweeps both.

    Returns ``(edge_nei_len, edge_nei_angle)``, each ``[n_edges, 3]``, in the
    domains the model expects (see module docstring): length as
    ``-0.75/bond_length``, angle as the cosine to the primary edge ``e``.

    ``pad_mode``: when an atom has fewer than 3 *other* edges, ``"repeat"``
    duplicates the closest available neighbor to fill the remaining slots (never
    introduces a value that didn't occur in the real local environment);
    ``"zero"`` fills missing slots with length-code ``0.0`` (``-0.75/length`` ->
    0 as length -> inf, i.e. "no bond") and angle ``0.0``. Also unverified —
    exposed for the calibration sweep, not decided here.
    """
    if endpoint not in ("dst", "src"):
        raise ValueError(f"endpoint must be 'dst' or 'src', got {endpoint!r}")
    if pad_mode not in ("repeat", "zero"):
        raise ValueError(f"pad_mode must be 'repeat' or 'zero', got {pad_mode!r}")

    n_edges = r.shape[0]
    lengths = torch.norm(r, dim=1)
    group_key = (v if endpoint == "dst" else u).tolist()

    # atom -> [edge indices incident at that atom via the chosen endpoint],
    # sorted by bond length ascending (so "the 3 nearest" is a slice).
    by_atom: Dict[int, List[int]] = {}
    for e in range(n_edges):
        by_atom.setdefault(group_key[e], []).append(e)
    for a in by_atom:
        by_atom[a].sort(key=lambda e: lengths[e].item())

    nei_len = torch.zeros((n_edges, 3), dtype=r.dtype)
    nei_angle = torch.zeros((n_edges, 3), dtype=r.dtype)

    for e in range(n_edges):
        atom = group_key[e]
        candidates = [e2 for e2 in by_atom[atom] if e2 != e]
        chosen = candidates[:3]
        if pad_mode == "repeat" and chosen:
            while len(chosen) < 3:
                chosen.append(chosen[-1])
        # pad_mode == "zero" (or no candidates at all): leave remaining slots
        # at the pre-initialized 0.0 -- length-code 0 means "no bond" (length
        # -> inf), angle 0 means perpendicular; both are the RBF domain
        # midpoints, the closest thing to a neutral filler.
        for slot, e2 in enumerate(chosen):
            length2 = lengths[e2].item()
            nei_len[e, slot] = -0.75 / length2 if length2 > 0 else 0.0
            cos = torch.dot(r[e], r[e2]) / (lengths[e] * lengths[e2] + 1e-12)
            nei_angle[e, slot] = torch.clamp(cos, -1.0, 1.0)

    return nei_len, nei_angle


def structure_to_graphs(
    structure: Union[str, Any],
    *,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "cgcnn",
    triplet_endpoint: str = "dst",
    triplet_pad_mode: str = "repeat",
) -> Tuple[Data, Data]:
    """Build ``(se3_graph, so3_graph)`` for one structure — see module docstring.

    Parameters
    ----------
    structure:
        A POSCAR/CONTCAR path, or an already-loaded ``jarvis.core.atoms.Atoms``.
    cutoff, max_neighbors:
        Passed straight to jarvis's ``nearest_neighbor_edges`` (its own
        defaults: 8 Å / 12). Not confirmed to match the values used to build
        this checkpoint's training data — the calibration gate should sweep
        these.
    atom_features:
        Passed straight to jarvis's ``get_node_attributes``. ``"cgcnn"`` matches
        the checkpoint's ``atom_input_features: 92`` numerically (see module
        docstring) but is not independently confirmed.
    triplet_endpoint, triplet_pad_mode:
        See :func:`_build_triplets`.
    """
    from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
    from jarvis.core.specie import get_node_attributes

    atoms = _load_atoms(structure)

    edges = nearest_neighbor_edges(
        atoms=atoms, cutoff=cutoff, max_neighbors=max_neighbors, use_canonize=True,
    )
    u, v, r = build_undirected_edgedata(atoms, edges)
    edge_index = torch.stack([u, v], dim=0).long()
    edge_attr = r.float()

    feats = [get_node_attributes(sym, atom_features=atom_features) for sym in atoms.elements]
    x = torch.tensor(np.asarray(feats, dtype=np.float32))

    nei_len, nei_angle = _build_triplets(
        u, v, edge_attr, endpoint=triplet_endpoint, pad_mode=triplet_pad_mode,
    )

    se3_graph = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        edge_nei_angle=nei_angle, edge_nei_len=nei_len,
    )
    so3_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return se3_graph, so3_graph
