# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""DDTree construction for DTED training.

Ported from ``ddtree/ddtree.py::build_ddtree_tree`` (production inference
code) with a training-oriented API:

* Removed CUDA-timing subtimers (the training path already tracks its own
  per-step metrics; per-anchor timing would dominate CPU noise).
* Returns tensors directly on the caller-requested device to avoid an
  extra host->device copy at the call site.
* No dependency on the ``ddtree`` package -- SpecForge owns this copy so
  the two projects can evolve independently.

The heap-based construction is intentionally kept identical to
``ddtree/ddtree.py`` so training-time trees exactly match what inference
would produce given the same draft logits and budget (Corollary 7.1 in
docs/21 relies on this equivalence).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

__all__ = ["DDTreeStructure", "build_ddtree_tree", "build_ddtree_tree_from_topk"]


@dataclass(frozen=True)
class DDTreeStructure:
    """Materialized tree produced from one anchor's draft logits.

    Fields:
      node_token_ids: LongTensor (N,) — token id chosen at each non-root
        node ``u``. Root (index 0) is implicit and not stored.
      node_depths: LongTensor (N,) — depth of each non-root node in
        [1, K]. ``depth == 1`` means direct child of root.
      parent_indices: LongTensor (N+1,) — ``parent_indices[0] == -1``;
        for u >= 1, ``parent_indices[u]`` is the index of u's parent in
        [0, u-1] (the parent may be the root, encoded as 0).

    All tensors live on the requested device (default CPU) and use
    ``torch.long`` dtype so they can be used directly as gather indices.
    """

    node_token_ids: torch.Tensor
    node_depths: torch.Tensor
    parent_indices: torch.Tensor


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
    *,
    device: torch.device | str | None = None,
) -> DDTreeStructure:
    """Build a DDTree from one anchor's draft logits (Corollary 7.1).

    Args:
      draft_logits: FloatTensor of shape (K, V). K is the draft horizon
        (= block_size - 1 in DFlash). Values may live on any device and
        dtype; the function detaches implicitly by calling ``.float()``
        on a CPU copy for heap arithmetic.
      budget: max number of non-root nodes to expand. ``budget <= 0``
        yields an empty tree (root only).
      device: destination device for the returned tensors. Defaults to
        ``draft_logits.device`` so downstream gather ops need no move.

    Returns:
      DDTreeStructure with N = min(budget, reachable_nodes).

    Semantics:
      * Every heap entry is ``(neg_log_w, ranks_tuple, parent_idx, depth,
        rank, log_w)``, exactly matching ddtree/ddtree.py so tree order
        (and therefore behavior) is identical.
      * The tree is *deterministic given the draft logits* — no sampling.

    Perf note:
      * For batched chunk-level tree building, prefer
        ``build_ddtree_tree_from_topk``: it takes precomputed top-k
        log-probs / token-ids (typically produced by one batched GPU
        ``torch.topk`` over an entire chunk of anchors) and skips the
        per-anchor (K, V) GPU→CPU copy that dominates this function's
        cost (~10 ms/anchor at V=152k, ~150x more traffic than needed).
    """
    if device is None:
        device = draft_logits.device

    if budget <= 0 or draft_logits.shape[0] == 0:
        # Empty tree: only the (implicit) root exists.
        parents = torch.tensor([-1], dtype=torch.long, device=device)
        return DDTreeStructure(
            node_token_ids=torch.empty(0, dtype=torch.long, device=device),
            node_depths=torch.empty(0, dtype=torch.long, device=device),
            parent_indices=parents,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    # Move to CPU float32 for heap arithmetic. The tensor is detached
    # implicitly by ``.detach().cpu().float()``; upstream code is
    # responsible for supplying ``draft_logits`` that does not require
    # grad if it wants zero autograd overhead.
    logits_cpu = draft_logits.detach().to(device="cpu", dtype=torch.float32)
    top_logits, top_token_ids = torch.topk(logits_cpu, k=topk, dim=-1)
    log_z = torch.logsumexp(logits_cpu, dim=-1, keepdim=True)
    top_log_probs_np = (top_logits - log_z).numpy()
    top_token_ids_np = top_token_ids.numpy().astype(np.int64, copy=False)

    return build_ddtree_tree_from_topk(
        top_log_probs_np=top_log_probs_np,
        top_token_ids_np=top_token_ids_np,
        budget=budget,
        depth_limit=depth_limit,
        device=device,
    )


def build_ddtree_tree_from_topk(
    *,
    top_log_probs_np: np.ndarray,   # (K, budget) fp32, per-depth top-k log q
    top_token_ids_np: np.ndarray,   # (K, budget) int64, per-depth top-k token ids
    budget: int,
    depth_limit: int,
    device: torch.device | str,
) -> DDTreeStructure:
    """Heap-only variant of ``build_ddtree_tree``.

    Skips per-anchor GPU→CPU logits transfer + topk + logsumexp; the
    caller is expected to have already computed those in batched form
    (see ``build_dted_chunk_tree_tensors``).

    ``top_log_probs_np[k, r]`` = ``log q(rank r at depth k+1)`` (already
    normalized by ``logsumexp`` over the full vocab).
    """
    if budget <= 0 or depth_limit == 0 or top_log_probs_np.shape[0] == 0:
        parents = torch.tensor([-1], dtype=torch.long, device=device)
        return DDTreeStructure(
            node_token_ids=torch.empty(0, dtype=torch.long, device=device),
            node_depths=torch.empty(0, dtype=torch.long, device=device),
            parent_indices=parents,
        )

    topk = top_log_probs_np.shape[1]

    first_logw = float(top_log_probs_np[0, 0])
    heap: List[Tuple[float, Tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int64)
    parents_np[0] = -1
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        node_count += 1

        # Sibling: same parent, next rank at the same depth.
        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        # First child: this node becomes the parent for its top-1 child.
        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (
                    -child_logw,
                    child_ranks,
                    current_index,
                    depth + 1,
                    0,
                    child_logw,
                ),
            )

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count]).to(device)
    node_depths = torch.from_numpy(node_depths_np[:node_count]).to(device)
    parent_indices = torch.from_numpy(parents_np[: node_count + 1]).to(device)

    return DDTreeStructure(
        node_token_ids=node_token_ids,
        node_depths=node_depths,
        parent_indices=parent_indices,
    )
