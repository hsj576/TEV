# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""DTED loss with full-tree-verify target scoring (ExitTrain).

The math is identical to :mod:`specforge.algorithms.dted.loss`; only
where ``p_target(x_v)`` comes from changes:

Base DTED (Markov approximation):
    ``p_target[v] = softmax(target_lm_head(target_h_teacher[depth(v)]))[x_v]``
    -- all tree nodes at depth ``k`` share the same target hidden
    (sampled along the teacher sequence).

ExitTrain (full-tree verify):
    ``p_target[v] = softmax(target_lm_head(target_h_verify[v]))[x_v]``
    where ``target_h_verify[v]`` is obtained by running the target
    model through ``[prefix | anchor | tree_nodes]`` with a tree
    attention mask -- so each ``target_h_verify[v]`` truly reflects
    the target's belief conditioned on the path ``root -> v``.

Concretely, ``target_h_verify[v]`` is the target's last-hidden at slot
``W + 1 + node_index_in_tree`` (0-indexed), producing the distribution
for **next token** given prefix + anchor + all tree ancestors + v.
That distribution is exactly what P/rho need.

Special case: p_target(depth-1 child) is scored using the target
hidden at slot ``W`` (the anchor's own position), since the depth-1
child's condition is "prefix + anchor".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "FullVerifyChunkDiagnostics",
    "FullVerifyChunkTreeTensors",
    "build_full_verify_chunk_tree_tensors",
]


@dataclass
class FullVerifyChunkDiagnostics:
    n_valid_anchors: int
    n_tree_nodes_total: int
    tree_sizes: torch.Tensor
    p_tgt_mean_per_anchor: torch.Tensor
    P_mean_per_anchor: torch.Tensor
    weight_used_mean_per_anchor: torch.Tensor
    exit_weight_sum_per_anchor: torch.Tensor
    expected_al_per_anchor: torch.Tensor


@dataclass
class FullVerifyChunkTreeTensors:
    """Loss-ready dense tensors, identical layout to the base DTED version.

    Fields
    ------
    row_indices : LongTensor (M_total,)
        Global row in the flat draft logits view (index into
        ``(B*N_chunk*K, V)``) for each tree node.
    x_v : LongTensor (M_total,)
        Token id chosen at each tree node (the label for CE-like
        chunk loss).
    weight : FloatTensor (M_total,)
        ``w(v) = P(v) * rho(v) * (depth(v) if weight_type == 'exit'
        else 1)`` (or just ``P(v)`` for weight_type ``'P_tgt'``).
        Normalised later by tree-node sum.
    partition_weight_per_row : FloatTensor (M_flat,)
        Sum of weights sharing the same row (needed for the Lemma-1
        partition of ``F.cross_entropy``).
    per_anchor_diag : FullVerifyChunkDiagnostics
        Per-anchor scalar diagnostics for logging.
    """

    row_indices: torch.Tensor
    x_v: torch.Tensor
    weight: torch.Tensor
    partition_weight_per_row: torch.Tensor
    per_anchor_diag: FullVerifyChunkDiagnostics


def build_full_verify_chunk_tree_tensors(
    *,
    tree_node_token_ids_list: List[torch.Tensor],   # each (n_v,) long, len=B*N_chunk
    tree_parent_indices_list: List[torch.Tensor],   # each (n_v + 1,) long
    tree_node_depths_list: List[torch.Tensor],      # each (n_v,) long
    target_probs_per_node_list: List[torch.Tensor], # each (n_v + 1, V) fp32 or bf16
    anchor_valid_chunk: torch.Tensor,               # (B, N_chunk) bool
    block_size: int,
    tree_budget: int,
    alpha: float,
    weight_type: str = "P_tgt",
    device: torch.device = None,
) -> FullVerifyChunkTreeTensors:
    """Compute per-node P/rho/weight using **per-node** target distributions.

    ``target_probs_per_node_list[i]`` has shape ``(n_v_i + 1, V)``:
    row 0 is the anchor's distribution (used to score depth-1
    children), rows 1..n_v_i score depth-(d+1) children of each
    tree node.

    The per-node distributions are precomputed by the caller from the
    target model's tree-attention forward and passed in on CPU (fp32).
    """
    bsz, n_chunk = anchor_valid_chunk.shape
    K = block_size - 1
    M_flat = bsz * n_chunk * K

    if device is None:
        device = anchor_valid_chunk.device

    if weight_type not in {"exit", "P_tgt"}:
        raise ValueError(f"weight_type must be 'exit' or 'P_tgt', got {weight_type!r}")

    tree_sizes: List[int] = []
    p_tgt_mean_pieces: List[float] = []
    P_mean_pieces: List[float] = []
    weight_used_mean_pieces: List[float] = []
    exit_weight_sum_pieces: List[float] = []
    expected_al_pieces: List[float] = []
    n_tree_nodes_total = 0

    all_rows: List[int] = []
    all_x_v: List[int] = []
    all_weights: List[float] = []

    valid_mask_np = anchor_valid_chunk.detach().cpu().numpy()

    for b in range(bsz):
        for a in range(n_chunk):
            flat_i = b * n_chunk + a
            if not valid_mask_np[b, a]:
                continue
            node_token_ids_t = tree_node_token_ids_list[flat_i]
            n_nodes = int(node_token_ids_t.numel())
            if n_nodes == 0:
                continue

            x_v_cpu = node_token_ids_t.detach().cpu().numpy().astype(np.int64)
            depths_cpu = tree_node_depths_list[flat_i].detach().cpu().numpy().astype(np.float32)
            parent_ids_cpu = (
                tree_parent_indices_list[flat_i][1:].detach().cpu().numpy().astype(np.int64)
            )  # (n_nodes,) -- parent of each real node in [0, n_nodes-1]; 0 == anchor.
            parent_depth_cpu = np.maximum(depths_cpu.astype(np.int64) - 1, 0)

            # target_probs_per_node: (n_nodes + 1, V). Row 0 = anchor.
            tgt_probs_np = (
                target_probs_per_node_list[flat_i].detach().cpu().float().numpy()
            )  # (n_nodes + 1, V)

            # --------------------------------------------------------
            # p_target(x_v) at the correct conditioning slot.
            # For a tree node u whose parent is ``parent_ids_cpu[u]``
            # (0 = anchor, k >= 1 = another tree node), the correct
            # distribution to score u's token is
            #   target_probs_per_node[parent_ids_cpu[u]]
            # -- i.e., the parent's ``next-token`` distribution.
            # --------------------------------------------------------
            p_tgt_raw = tgt_probs_np[
                parent_ids_cpu, x_v_cpu
            ].astype(np.float32)                    # (n_nodes,)
            p_tgt_floored = np.maximum(p_tgt_raw, np.float32(alpha))

            # P(v) chain product with floored p_target.
            P = np.empty(n_nodes + 1, dtype=np.float32)
            P[0] = 1.0
            for u in range(n_nodes):
                P[u + 1] = P[parent_ids_cpu[u]] * p_tgt_floored[u]
            P_v = P[1:]

            # rho(v): for each real node v, sum p_target_raw over its
            # actual children (tokens picked by the tree). Since the
            # children are a subset of the parent's distribution, and
            # the raw probs come from the parent's ``next-token``
            # distribution at index parent_ids_cpu[c], the sum is:
            children_prob_sum = np.zeros(n_nodes + 1, dtype=np.float32)
            np.add.at(children_prob_sum, parent_ids_cpu, p_tgt_raw)
            rho_v = np.clip(1.0 - children_prob_sum[1:], 0.0, 1.0)

            w_v = P_v * rho_v
            if weight_type == "exit":
                weight_np = w_v * depths_cpu
            else:  # "P_tgt"
                weight_np = P_v

            # Global row indices for the flat (M_flat, V) view.
            row_base = flat_i * K
            local_rows = row_base + parent_depth_cpu

            all_rows.extend(local_rows.tolist())
            all_x_v.extend(x_v_cpu.tolist())
            all_weights.extend(weight_np.astype(np.float64).tolist())

            p_tgt_mean_pieces.append(float(p_tgt_floored.mean()))
            P_mean_pieces.append(float(P_v.mean()))
            weight_used_mean_pieces.append(float(weight_np.mean()))
            exit_weight_sum_pieces.append(float(w_v.sum()))
            expected_al_pieces.append(float(P_v.sum()))
            tree_sizes.append(n_nodes)
            n_tree_nodes_total += n_nodes

    n_valid_anchors = len(tree_sizes)

    if not all_rows:
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_f = torch.empty(0, dtype=torch.float32, device=device)
        partition_weight_zero = torch.zeros(M_flat, dtype=torch.float32, device=device)
        empty_scalar_f = torch.empty(0, dtype=torch.float32, device=device)
        diag = FullVerifyChunkDiagnostics(
            n_valid_anchors=n_valid_anchors,
            n_tree_nodes_total=0,
            tree_sizes=empty_scalar_f,
            p_tgt_mean_per_anchor=empty_scalar_f,
            P_mean_per_anchor=empty_scalar_f,
            weight_used_mean_per_anchor=empty_scalar_f,
            exit_weight_sum_per_anchor=empty_scalar_f,
            expected_al_per_anchor=empty_scalar_f,
        )
        return FullVerifyChunkTreeTensors(
            row_indices=empty_i,
            x_v=empty_i,
            weight=empty_f,
            partition_weight_per_row=partition_weight_zero,
            per_anchor_diag=diag,
        )

    all_rows_np = np.asarray(all_rows, dtype=np.int64)
    all_x_v_np = np.asarray(all_x_v, dtype=np.int64)
    all_weights_np = np.asarray(all_weights, dtype=np.float32)

    partition_weight_np = np.zeros(M_flat, dtype=np.float32)
    np.add.at(partition_weight_np, all_rows_np, all_weights_np)

    diag = FullVerifyChunkDiagnostics(
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

    return FullVerifyChunkTreeTensors(
        row_indices=torch.from_numpy(all_rows_np).to(device),
        x_v=torch.from_numpy(all_x_v_np).to(device),
        weight=torch.from_numpy(all_weights_np).to(device),
        partition_weight_per_row=torch.from_numpy(partition_weight_np).to(device),
        per_anchor_diag=diag,
    )
