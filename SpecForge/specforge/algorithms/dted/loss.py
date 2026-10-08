# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""DTED loss primitives (Phase 4.5 fast-path).

The previous per-anchor implementation (docs/22 §Perf-notes v0.3) fanned
out to N_valid independent ``lm_head`` calls, each with its own autograd
subgraph. That was ~30x slower per anchor than DFlash. This rewrite
mirrors the DFlash/DSpark architecture:

  * ``build_dted_chunk_tree_tensors``  -- batched CPU tree build,
    returning FLAT gather indices spanning every tree node in a
    whole chunk-of-anchors.
  * ``dted_chunk_loss_from_flat_logits`` -- vectorized loss over those
    flat tensors: one ``F.cross_entropy`` call, one mul, one sum.

The heavy lm_head / log_softmax matmuls stay on the caller (chunk fn)
so we can share their outputs across every tree node touching the
same (batch, anchor, depth) slot. Backward is a single well-connected
subgraph rather than 128 stacked ones -- matches DFlash's autograd
shape and unlocks its ``checkpointed_chunk_reduce`` machinery.

Mathematical objective (unchanged from v0.3):
    L_DTED = - sum_{v in T} weight(v) * log q_theta(x_v | par(v))
where weight(v) in {P(v), w(v) * depth(v)} depending on ``weight_type``,
and P/w are computed from a stop-gradient target distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from specforge.algorithms.dted.ddtree_build import (
    build_ddtree_tree,
    build_ddtree_tree_from_topk,
)

__all__ = [
    "DTEDChunkTreeTensors",
    "DTEDChunkDiagnostics",
    "build_dted_chunk_tree_tensors",
    "dted_chunk_loss_from_flat_logits",
    # Kept for backwards compatibility with the v0.3 unit tests:
    "DTEDAnchorLoss",
    "dted_loss_per_anchor",
]


# =====================================================================
# Batched (chunk-level) primitives -- the fast path used by model.py.
# =====================================================================


@dataclass
class DTEDChunkTreeTensors:
    """Flat per-tree-node arrays plus a per-row weight-sum table.

    Layout for a chunk of ``B*N_chunk`` anchors:
      * ``row_indices``: (M_total,) long -- flat-row index of each
        tree node, ``row = b*(N_chunk*K) + a*K + parent_depth``.
      * ``x_v``: (M_total,) long -- token id chosen at each node.
      * ``weight``: (M_total,) fp32 -- detached weight(v). Grad-free.
      * ``partition_weight_per_row``: (M_flat,) fp32 -- pre-scattered
        ``sum_{v at row r} weight(v)``. Used by the partition (logsumexp)
        term of the loss.
      * ``per_anchor_diag``: telemetry aggregates for the caller.

    Not gathering the logits ourselves (that materializes ``(M_total,
    V)``); the loss function does two cheap operations on the flat
    logits tensor instead: one ``logsumexp`` along dim=1 and one
    ``(row_indices, x_v)`` advanced-index that returns a ``(M_total,)``
    scalar tensor.
    """

    row_indices: torch.Tensor           # (M_total,) long
    x_v: torch.Tensor                   # (M_total,) long
    weight: torch.Tensor                # (M_total,) fp32
    partition_weight_per_row: torch.Tensor  # (M_flat,) fp32
    per_anchor_diag: "DTEDChunkDiagnostics"


@dataclass
class DTEDChunkDiagnostics:
    """Pre-aggregated per-anchor telemetry for a chunk.

    All tensors are ``float32`` on ``device`` and have shape
    ``(N_valid_in_chunk,)`` unless otherwise noted. Because tree build
    is CPU-side and target-only, these are all detached.
    """

    n_valid_anchors: int              # int, python
    n_tree_nodes_total: int           # int, python
    tree_sizes: torch.Tensor          # per-anchor tree size (N_i)
    p_tgt_mean_per_anchor: torch.Tensor
    P_mean_per_anchor: torch.Tensor
    weight_used_mean_per_anchor: torch.Tensor
    exit_weight_sum_per_anchor: torch.Tensor
    expected_al_per_anchor: torch.Tensor


def build_dted_chunk_tree_tensors(
    draft_logits_chunk_detached: torch.Tensor,   # (B, N_chunk, K, V) fp16/bf16/fp32
    target_logits_chunk_detached: torch.Tensor,  # same shape as draft
    anchor_valid_chunk: torch.Tensor,            # (B, N_chunk) bool
    *,
    block_size: int,
    tree_budget: int,
    alpha: float,
    weight_type: Literal["exit", "P_tgt"] = "exit",
    device: torch.device,
) -> DTEDChunkTreeTensors:
    """Build DDTrees for every valid anchor in a chunk and pack them
    into a dense (M_flat, S) slot layout suitable for dspark-style
    fused ``F.cross_entropy`` loss.

    All heap / P / rho / weight arithmetic runs on **CPU numpy** so we
    do not thrash the CUDA allocator with thousands of 0-dim tensor
    allocations. Only the final dense output goes to ``device``.

    Layout details:
      * ``M_flat = B * N_chunk * K`` where ``K = block_size - 1``.
      * ``row = b * (N_chunk * K) + a * K + parent_depth``  identifies
        the (batch, anchor-in-chunk, depth) triple whose flat log-prob
        row hosts the tree node.
    """
    assert draft_logits_chunk_detached.dim() == 4, (
        f"expected (B, N_chunk, K, V), got {tuple(draft_logits_chunk_detached.shape)}"
    )
    bsz, n_chunk, K, vocab = draft_logits_chunk_detached.shape
    assert K == block_size - 1, (
        f"chunk logits K={K} must equal block_size-1={block_size - 1}"
    )
    assert target_logits_chunk_detached.shape == draft_logits_chunk_detached.shape

    M_flat = bsz * n_chunk * K

    # ---- Chunk-level batched topk + logsumexp (single GPU sweep) ----
    # This is the single biggest speedup vs the per-anchor pipeline
    # (v0.4). Before: each anchor did GPU->CPU on a (K, V) fp32 tensor
    # then ran ``torch.topk(cpu, k=budget)`` -- ~10 ms/anchor at V=152k.
    # Now: one batched topk on GPU across all anchors of the chunk
    # (B*N_chunk*K rows of V dims), producing a (B, N_chunk, K, budget)
    # tensor that is ~150x smaller than the raw logits. Only that
    # small tensor + the small logsumexp result travel across PCIe.
    with torch.no_grad():
        top_logits, top_token_ids = torch.topk(
            draft_logits_chunk_detached, k=min(tree_budget, vocab), dim=-1
        )                                                                    # (B, N_chunk, K, budget)
        log_z = torch.logsumexp(
            draft_logits_chunk_detached.float(), dim=-1, keepdim=True
        )                                                                    # (B, N_chunk, K, 1)
        top_log_probs = (top_logits.float() - log_z).contiguous()            # (B, N_chunk, K, budget)

        # Single H2D transfer for the whole chunk (bytes = B*N_chunk*K*budget*4).
        top_log_probs_all_np = top_log_probs.cpu().numpy()                   # (B, N_chunk, K, budget) fp32
        top_token_ids_all_np = top_token_ids.cpu().numpy().astype(
            np.int64, copy=False
        )                                                                    # (B, N_chunk, K, budget)

        # ---- Compute per-position target probabilities once for the chunk.
        target_probs = F.softmax(
            target_logits_chunk_detached.float(), dim=-1
        )                                                                    # (B, N_chunk, K, V), fp32

    valid_mask_np = anchor_valid_chunk.detach().to("cpu").numpy()
    n_valid_est = int(valid_mask_np.sum())

    # ---- Per-anchor diagnostic pieces (CPU floats -- ONE H2D at end) ----
    tree_sizes: List[int] = []
    p_tgt_mean_pieces: List[float] = []
    P_mean_pieces: List[float] = []
    weight_used_mean_pieces: List[float] = []
    exit_weight_sum_pieces: List[float] = []
    expected_al_pieces: List[float] = []
    n_tree_nodes_total = 0

    # ---- Per-node lists (CPU): row, x_v, weight ----
    all_rows: List[int] = []
    all_x_v: List[int] = []
    all_weights: List[float] = []

    for b in range(bsz):
        for a in range(n_chunk):
            if not valid_mask_np[b, a]:
                continue

            target_probs_a = target_probs[b, a]                              # (K, V) fp32

            # Slice pre-computed top-k from the batched GPU pass.
            top_log_probs_np = top_log_probs_all_np[b, a]                    # (K, budget) fp32
            top_token_ids_np = top_token_ids_all_np[b, a]                    # (K, budget) int64

            # CPU heap tree build (no GPU->CPU logits transfer needed).
            tree = build_ddtree_tree_from_topk(
                top_log_probs_np=top_log_probs_np,
                top_token_ids_np=top_token_ids_np,
                budget=tree_budget,
                depth_limit=K,
                device=device,
            )
            n_nodes = int(tree.node_token_ids.numel())
            if n_nodes == 0:
                continue

            # Move tree structure to CPU (small: N <= tree_budget).
            parent_depth_cpu = (tree.node_depths - 1).clamp_min(0).cpu().numpy().astype(np.int64)
            x_v_cpu = tree.node_token_ids.cpu().numpy().astype(np.int64)
            parent_ids_cpu = tree.parent_indices[1:].cpu().numpy().astype(np.int64)
            depths_cpu = tree.node_depths.cpu().numpy().astype(np.float32)

            # Per-node raw and floored target probs -- pulled to CPU
            # once, all subsequent P/rho/weight math is pure numpy.
            p_tgt_raw = target_probs_a[
                tree.node_depths - 1, tree.node_token_ids
            ].cpu().numpy().astype(np.float32)
            p_tgt_floored = np.maximum(p_tgt_raw, np.float32(alpha))

            # P(v): sequential product along tree order.
            # P has room for the implicit root (index 0), children live in [1, N].
            P = np.empty(n_nodes + 1, dtype=np.float32)
            P[0] = 1.0
            for u in range(n_nodes):
                P[u + 1] = P[parent_ids_cpu[u]] * p_tgt_floored[u]
            P_v = P[1:]                                                      # (N,)

            # rho(v): 1 - sum_{c in Ch(v)} p_tgt_raw(x_c).
            children_prob_sum = np.zeros(n_nodes + 1, dtype=np.float32)
            np.add.at(children_prob_sum, parent_ids_cpu, p_tgt_raw)
            rho_v = np.clip(1.0 - children_prob_sum[1:], 0.0, 1.0)           # (N,)

            w_v = P_v * rho_v                                                # (N,)
            if weight_type == "exit":
                weight_np = w_v * depths_cpu
            else:  # "P_tgt"
                weight_np = P_v

            # Global row indices for the flat (M_flat, V) view.
            row_base = (b * n_chunk + a) * K
            local_rows = row_base + parent_depth_cpu                         # (N,) int64

            all_rows.extend(local_rows.tolist())
            all_x_v.extend(x_v_cpu.tolist())
            all_weights.extend(weight_np.astype(np.float64).tolist())

            # Diagnostics: pure python floats, batched to GPU at the end.
            p_tgt_mean_pieces.append(float(p_tgt_floored.mean()))
            P_mean_pieces.append(float(P_v.mean()))
            weight_used_mean_pieces.append(float(weight_np.mean()))
            exit_weight_sum_pieces.append(float(w_v.sum()))
            expected_al_pieces.append(float(P_v.sum()))
            tree_sizes.append(n_nodes)
            n_tree_nodes_total += n_nodes

    n_valid_anchors = len(tree_sizes)

    # -------------------------------------------------------------
    # Assemble flat (M_total,) arrays + partition weight per row.
    # -------------------------------------------------------------
    if not all_rows:
        # No supervised tree nodes at all.
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_f = torch.empty(0, dtype=torch.float32, device=device)
        partition_weight_zero = torch.zeros(M_flat, dtype=torch.float32, device=device)
        empty_scalar_f = torch.empty(0, dtype=torch.float32, device=device)
        diag = DTEDChunkDiagnostics(
            n_valid_anchors=n_valid_anchors,
            n_tree_nodes_total=0,
            tree_sizes=empty_scalar_f,
            p_tgt_mean_per_anchor=empty_scalar_f,
            P_mean_per_anchor=empty_scalar_f,
            weight_used_mean_per_anchor=empty_scalar_f,
            exit_weight_sum_per_anchor=empty_scalar_f,
            expected_al_per_anchor=empty_scalar_f,
        )
        return DTEDChunkTreeTensors(
            row_indices=empty_i,
            x_v=empty_i,
            weight=empty_f,
            partition_weight_per_row=partition_weight_zero,
            per_anchor_diag=diag,
        )

    all_rows_np = np.asarray(all_rows, dtype=np.int64)
    all_x_v_np = np.asarray(all_x_v, dtype=np.int64)
    all_weights_np = np.asarray(all_weights, dtype=np.float32)

    # Pre-scatter the per-row sum of weights on CPU (numpy add.at
    # handles the "same row multiple times" case correctly).
    partition_weight_np = np.zeros(M_flat, dtype=np.float32)
    np.add.at(partition_weight_np, all_rows_np, all_weights_np)

    row_indices_t = torch.from_numpy(all_rows_np).to(device)
    x_v_t = torch.from_numpy(all_x_v_np).to(device)
    weight_t = torch.from_numpy(all_weights_np).to(device)
    partition_weight_t = torch.from_numpy(partition_weight_np).to(device)

    diag = DTEDChunkDiagnostics(
        n_valid_anchors=n_valid_anchors,
        n_tree_nodes_total=n_tree_nodes_total,
        tree_sizes=torch.tensor(tree_sizes, dtype=torch.float32, device=device),
        p_tgt_mean_per_anchor=torch.tensor(
            p_tgt_mean_pieces, dtype=torch.float32, device=device
        ),
        P_mean_per_anchor=torch.tensor(
            P_mean_pieces, dtype=torch.float32, device=device
        ),
        weight_used_mean_per_anchor=torch.tensor(
            weight_used_mean_pieces, dtype=torch.float32, device=device
        ),
        exit_weight_sum_per_anchor=torch.tensor(
            exit_weight_sum_pieces, dtype=torch.float32, device=device
        ),
        expected_al_per_anchor=torch.tensor(
            expected_al_pieces, dtype=torch.float32, device=device
        ),
    )
    return DTEDChunkTreeTensors(
        row_indices=row_indices_t,
        x_v=x_v_t,
        weight=weight_t,
        partition_weight_per_row=partition_weight_t,
        per_anchor_diag=diag,
    )


def dted_chunk_loss_from_flat_logits(
    flat_draft_logits: torch.Tensor,   # (M_flat, V) -- WITH GRAD
    *,
    row_indices: torch.Tensor,         # (M_total,) long   -- row of each tree node
    x_v: torch.Tensor,                 # (M_total,) long   -- token id of each tree node
    weight: torch.Tensor,              # (M_total,) fp32   -- weight(v), detached
    partition_weight_per_row: torch.Tensor,  # (M_flat,) fp32 -- pre-scattered sum of weight
) -> torch.Tensor:
    """Return the un-normalized DTED loss numerator without ever
    materializing an ``(M_total, V)`` gathered-logits tensor.

    Uses the per-row decomposition:

        loss_num
          = sum_v w_v * (-log q(x_v | par(v)))
          = sum_v w_v * (logsumexp(logits[row_v]) - logits[row_v, x_v])
          = sum_r partition_weight[r] * logsumexp(logits[r])
              - sum_v w_v * logits[row_v, x_v]

    Both terms above are cheap:

      * ``logsumexp`` yields a ``(M_flat,)`` vector via
        ``torch.logsumexp(flat_draft_logits, dim=1)`` -- a single fused
        reduction over the vocab dim; no materialized log-probs.
      * ``logits[row_v, x_v]`` picks ``M_total`` scalars via 2D
        advanced indexing; the output is ``(M_total,)``, not
        ``(M_total, V)``.

    Backward for both paths reduces to sparse scatter-adds into
    ``flat_draft_logits.grad``, matching dspark's memory profile.
    """
    if x_v.numel() == 0:
        return flat_draft_logits.sum() * 0.0

    # ---- Partition term: (M_flat,) logsumexp weighted by row totals.
    # Keep the operand in its native dtype (bf16 in practice) so
    # logsumexp does NOT allocate a (M_flat, V) fp32 intermediate.
    # The final scalar is cast back to whatever type the caller uses.
    lse_per_row = torch.logsumexp(flat_draft_logits, dim=1)                  # (M_flat,)
    partition_part = (partition_weight_per_row * lse_per_row.float()).sum()

    # ---- Target-score term: (M_total,) advanced index picks scalars.
    logits_at_v = flat_draft_logits[row_indices, x_v]                        # (M_total,)
    target_part = (weight * logits_at_v.float()).sum()

    return partition_part - target_part


# =====================================================================
# Legacy per-anchor path (kept for v0.3 unit tests only -- not used in
# the training hot path anymore).
# =====================================================================


@dataclass
class DTEDAnchorLoss:
    loss: torch.Tensor
    num_nodes: int
    p_tgt_mean: torch.Tensor
    P_mean: torch.Tensor
    exit_weight_sum: torch.Tensor
    expected_al: torch.Tensor
    weight_used_mean: torch.Tensor


def dted_loss_per_anchor(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    tree_budget: int,
    alpha: float = 1e-4,
    eps: float = 1e-8,
    weight_type: Literal["exit", "P_tgt"] = "exit",
    device: Optional[torch.device] = None,
) -> DTEDAnchorLoss:
    """Legacy single-anchor loss (DO NOT USE in training).

    Kept because the v0.3 unit tests validate this shape. The
    replacement fast-path lives above; call it via the model's
    ``_dted_objective_chunk_terms``. Left intentionally simple.
    """
    if device is None:
        device = draft_logits.device
    assert draft_logits.dim() == 2, "draft_logits must be (K, V)"
    assert target_logits.dim() == 2, "target_logits must be (K, V)"

    with torch.no_grad():
        tree = build_ddtree_tree(draft_logits.detach(), tree_budget, device=device)
    n_nodes = int(tree.node_token_ids.numel())

    if n_nodes == 0:
        zero_scalar = torch.zeros((), device=device)
        return DTEDAnchorLoss(
            loss=(draft_logits.sum() * 0.0),
            num_nodes=0,
            p_tgt_mean=zero_scalar,
            P_mean=zero_scalar,
            exit_weight_sum=zero_scalar,
            expected_al=zero_scalar,
            weight_used_mean=zero_scalar,
        )

    draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)

    with torch.no_grad():
        target_probs = F.softmax(target_logits.float(), dim=-1)
        parent_depth = (tree.node_depths - 1).clamp_min(0)
        x_v = tree.node_token_ids
        p_tgt_raw = target_probs[parent_depth, x_v]
        p_tgt = p_tgt_raw.clamp_min(alpha)

        parent_ids = tree.parent_indices[1:]
        parent_ids_cpu = parent_ids.tolist()
        P_list = [torch.ones((), dtype=torch.float32, device=device)]
        for u_idx in range(n_nodes):
            par = parent_ids_cpu[u_idx]
            P_list.append(P_list[par] * p_tgt[u_idx])
        P_v = torch.stack(P_list[1:], dim=0)

        children_prob_sum = torch.zeros(n_nodes + 1, dtype=torch.float32, device=device)
        children_prob_sum = children_prob_sum.scatter_add(0, parent_ids, p_tgt_raw)
        rho_v = (1.0 - children_prob_sum[1:]).clamp(min=0.0, max=1.0)

        w_v = P_v * rho_v
        if weight_type == "exit":
            weight = w_v * tree.node_depths.to(torch.float32)
        elif weight_type == "P_tgt":
            weight = P_v
        else:
            raise ValueError(f"unknown weight_type: {weight_type!r}")

        p_tgt_mean_t = p_tgt.mean().detach()
        P_mean_t = P_v.mean().detach()
        weight_used_mean_t = weight.mean().detach()
        exit_weight_sum_t = w_v.sum().detach()
        expected_al_t = P_v.sum().detach()

    log_q_v = draft_log_probs[parent_depth, x_v]
    loss = -(weight * log_q_v).sum() / float(n_nodes)

    return DTEDAnchorLoss(
        loss=loss,
        num_nodes=n_nodes,
        p_tgt_mean=p_tgt_mean_t,
        P_mean=P_mean_t,
        exit_weight_sum=exit_weight_sum_t,
        expected_al=expected_al_t,
        weight_used_mean=weight_used_mean_t,
    )
