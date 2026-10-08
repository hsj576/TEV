# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Full-tree attention forward for ExitTrain (DTED full-verify) training.

Contract
--------
Given a batch of anchor trees (each anchor has its own DDTree with up
to ``tree_budget`` nodes), run one Qwen3-family forward pass over each
anchor's flattened sequence

    [prefix_window tokens] | [anchor token] | [tree node 1, ..., tree node N]

with a **tree attention mask** so each tree node attends only to
prefix + anchor + its own ancestor chain. Return per-node last-hidden
states which downstream loss code can feed through the shared
``lm_head`` to obtain

    p_target(next_token | prefix, path_to_v)

for every tree node ``v``. This is the honest conditional that the
base DTED code coarsely approximates with the teacher-path hidden.

Design decisions
----------------
1. **Per-anchor sequence, batched.** Every anchor's forward sequence
   has fixed length ``L = W + 1 + tree_budget`` (padding when the tree
   has fewer than ``tree_budget`` nodes). Batching across anchors gives
   an efficient ``(B_flat, L)`` forward on the target model.
2. **SDPA + 4D additive mask.** Qwen3's ``eager_attention_forward``
   does ``attn_weights + attention_mask``. Feeding a 4D float mask
   with ``0.0``/``-inf`` implements the tree attention for free -- no
   custom kernel needed.
3. **Position IDs.** Each token uses its **absolute position in the
   original input sequence**: prefix tokens keep their real positions,
   the anchor is at ``anchor_pos``, tree node ``v`` at
   ``anchor_pos + depth(v)``. RoPE therefore sees the correct relative
   offsets to grandparent/parent tokens regardless of layout.
4. **No gradient.** Target weights are frozen; entire forward runs
   under ``torch.no_grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


__all__ = [
    "TreeBatchLayout",
    "build_tree_batch_layout",
    "LocalFullTargetVerifier",
]


@dataclass
class TreeBatchLayout:
    """Dense, padded layout for one batch of anchor trees.

    All tensors have leading dim ``B_flat = sum(anchor_valid)``: only
    valid anchors contribute rows. Padded tree-node slots (when
    ``tree.node_count < tree_budget``) are marked in ``node_valid``
    and masked out of the attention.

    Fields
    ------
    input_ids : LongTensor ``(B_flat, L)``
        ``L = prefix_window + 1 + tree_budget``. Slots are filled as
        described in the module docstring. Padded slots use token id 0.
    position_ids : LongTensor ``(B_flat, L)``
        Absolute positions relative to the original sequence.
    attention_mask : FloatTensor ``(B_flat, 1, L, L)``
        Additive float mask (0.0 or ``-inf``) implementing the tree
        attention pattern.
    node_valid : BoolTensor ``(B_flat, tree_budget)``
        ``True`` for real tree node slots, ``False`` for padding.
    node_slot_indices : LongTensor ``(B_flat, tree_budget)``
        For each anchor, ``node_slot_indices[b, v] = prefix_window + 1 + v``
        (constant across ``b``). Handy for gathering per-node hidden
        after the forward.
    valid_anchor_flat_ids : LongTensor ``(B_flat,)``
        For each valid entry, its original ``b * N_chunk + a`` index
        in the pre-flattening ``(B, N_chunk)`` layout. Used to scatter
        per-node hidden back into the ``(B, N_chunk, ...)`` output
        tensor.
    node_token_ids : LongTensor ``(B_flat, tree_budget)``
        The token id chosen at each tree slot (0 for padding). This is
        redundant with ``input_ids[:, prefix_window + 1:]`` and kept
        here as a convenience.
    parent_node_indices : LongTensor ``(B_flat, tree_budget)``
        For each tree node, the index of its parent inside the same
        anchor's node list (``0..v-1``), or ``-1`` for padded slots.
        ``0`` means the root (anchor). Used by the loss to compute
        P(v) via cumulative product on GPU.
    node_depths : LongTensor ``(B_flat, tree_budget)``
        Depth of each tree node (``1..K``, or ``0`` for padding).
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    node_valid: torch.Tensor
    node_slot_indices: torch.Tensor
    valid_anchor_flat_ids: torch.Tensor
    node_token_ids: torch.Tensor
    parent_node_indices: torch.Tensor
    node_depths: torch.Tensor


def _build_single_tree_mask(
    prefix_len: int,
    tree_budget: int,
    parent_indices: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Construct the (L, L) additive attention mask for one anchor.

    Layout: rows/columns are ordered as [W prefix tokens | anchor | K
    tree node slots (padded to tree_budget)]. The prefix + anchor
    region is standard causal (lower-triangular). Tree rows may attend
    only to prefix + anchor + their own ancestors. Padded tree slots
    attend to nothing (all -inf); this both masks them out of the
    forward and keeps them from polluting other rows via ``value``.

    Parameters
    ----------
    prefix_len : int
        ``W + 1`` (window + anchor).
    tree_budget : int
        Padding budget.
    parent_indices : np.ndarray shape (node_count + 1,)
        ``parent_indices[0] == -1`` (root sentinel); for u >= 1,
        ``parent_indices[u]`` is the parent index inside the tree
        (0 == root == anchor).
    node_count : int
        Number of real tree nodes ``<= tree_budget``.
    """
    L = prefix_len + tree_budget
    NEG_INF = -1e9  # additive mask uses -inf; -1e9 avoids fp16 overflow.

    mask = np.full((L, L), NEG_INF, dtype=np.float32)

    # Prefix + anchor: standard causal (row i can attend to columns 0..i).
    for i in range(prefix_len):
        mask[i, : i + 1] = 0.0

    # Tree nodes: attend to prefix + anchor + own ancestors.
    # For every real tree node u (u >= 1 in the tree's own indexing),
    # its ancestor chain in the flattened layout is:
    #   [0..prefix_len-1]  (prefix + anchor)
    #   plus (prefix_len - 1 + a) for each proper ancestor ``a`` at
    #   tree-index >= 1.
    #
    # A tree node's "self" also needs to attend to itself.
    #
    # Note: we do NOT let tree nodes attend to each other unless
    # ancestor-related, so siblings are properly isolated.
    for v in range(1, node_count + 1):
        row = prefix_len + (v - 1)
        # Attend to prefix + anchor.
        mask[row, :prefix_len] = 0.0
        # Attend to all proper ancestors (walking up parent chain).
        u = v
        while True:
            parent = int(parent_indices[u])
            if parent <= 0:
                break
            # parent's slot in the flat layout is prefix_len - 1 + parent
            # BUT parent >= 1 refers to a tree node -> row prefix_len + (parent - 1)
            mask[row, prefix_len + (parent - 1)] = 0.0
            u = parent
        # Attend to self.
        mask[row, row] = 0.0

    return mask


def build_tree_batch_layout(
    *,
    input_ids: torch.Tensor,                    # (B, S) long
    anchor_positions: torch.Tensor,             # (B, N) long
    anchor_valid: torch.Tensor,                 # (B, N) bool
    tree_node_token_ids_list: List[torch.Tensor],  # each (n_v,) long or empty
    tree_parent_indices_list: List[torch.Tensor],  # each (n_v + 1,) long
    tree_node_depths_list: List[torch.Tensor],     # each (n_v,) long
    tree_budget: int,
    prefix_window: int,
    device: torch.device,
) -> TreeBatchLayout:
    """Flatten a chunk of anchor trees into a dense batched layout.

    The input lists are indexed by flat ``i = b * N + a`` and must
    have the same length ``B * N``. Empty (invalid) anchors are
    represented by tensors with ``.numel() == 0`` in the lists; they
    are skipped and do NOT contribute rows to the output.
    """
    bsz, seq_len = input_ids.shape
    _, num_anchors = anchor_positions.shape
    assert len(tree_node_token_ids_list) == bsz * num_anchors
    assert len(tree_parent_indices_list) == bsz * num_anchors
    assert len(tree_node_depths_list) == bsz * num_anchors

    W = int(prefix_window)
    L = W + 1 + int(tree_budget)

    # Collect valid indices only.
    valid_flat_ids: List[int] = []
    valid_flat_np = anchor_valid.reshape(-1).detach().cpu().numpy()
    for i in range(bsz * num_anchors):
        if valid_flat_np[i]:
            n_nodes = int(tree_node_token_ids_list[i].numel())
            if n_nodes > 0:
                valid_flat_ids.append(i)

    B_flat = len(valid_flat_ids)
    if B_flat == 0:
        empty_long = torch.empty(0, L, dtype=torch.long, device=device)
        empty_pos = torch.empty(0, L, dtype=torch.long, device=device)
        empty_mask = torch.empty(0, 1, L, L, dtype=torch.float32, device=device)
        empty_bool = torch.empty(0, int(tree_budget), dtype=torch.bool, device=device)
        empty_idx = torch.empty(0, int(tree_budget), dtype=torch.long, device=device)
        empty_valid = torch.empty(0, dtype=torch.long, device=device)
        return TreeBatchLayout(
            input_ids=empty_long,
            position_ids=empty_pos,
            attention_mask=empty_mask,
            node_valid=empty_bool,
            node_slot_indices=empty_idx,
            valid_anchor_flat_ids=empty_valid,
            node_token_ids=empty_idx,
            parent_node_indices=empty_idx,
            node_depths=empty_idx,
        )

    # Prepare numpy buffers on CPU (fast), copy once to device at the end.
    input_ids_cpu = input_ids.detach().cpu().numpy()               # (B, S)
    anchor_positions_cpu = anchor_positions.detach().cpu().numpy() # (B, N)

    out_input_ids_np = np.zeros((B_flat, L), dtype=np.int64)
    out_position_ids_np = np.zeros((B_flat, L), dtype=np.int64)
    out_mask_np = np.full((B_flat, L, L), -1e9, dtype=np.float32)
    out_node_valid_np = np.zeros((B_flat, tree_budget), dtype=bool)
    out_node_slot_indices_np = np.tile(
        np.arange(W + 1, W + 1 + tree_budget, dtype=np.int64), (B_flat, 1)
    )
    out_node_token_ids_np = np.zeros((B_flat, tree_budget), dtype=np.int64)
    out_parent_indices_np = np.full((B_flat, tree_budget), -1, dtype=np.int64)
    out_node_depths_np = np.zeros((B_flat, tree_budget), dtype=np.int64)

    for out_i, flat_i in enumerate(valid_flat_ids):
        b = flat_i // num_anchors
        a = flat_i - b * num_anchors
        anchor_pos = int(anchor_positions_cpu[b, a])
        # Prefix window: last W tokens before anchor (may be shorter
        # at the beginning of the sequence).
        left = max(0, anchor_pos - W)
        real_prefix_len = anchor_pos - left  # <= W
        pad_prefix_len = W - real_prefix_len  # left padding

        # ------ input_ids ------
        # slots [0 .. pad_prefix_len - 1] stay zero (padding).
        # slots [pad_prefix_len .. W - 1] hold the real prefix.
        if real_prefix_len > 0:
            out_input_ids_np[out_i, pad_prefix_len:W] = input_ids_cpu[
                b, left:anchor_pos
            ]
        # slot [W] holds the anchor token.
        out_input_ids_np[out_i, W] = input_ids_cpu[b, anchor_pos]

        # ------ position_ids ------
        # Real prefix keeps its true absolute positions.
        # Padded prefix slots get position 0 (they'll never be attended
        # anyway, but position must be a valid non-negative int).
        for p in range(pad_prefix_len, W):
            out_position_ids_np[out_i, p] = left + (p - pad_prefix_len)
        out_position_ids_np[out_i, W] = anchor_pos

        # ------ tree nodes ------
        node_token_ids_np = (
            tree_node_token_ids_list[flat_i].detach().cpu().numpy().astype(np.int64)
        )
        node_depths_np = (
            tree_node_depths_list[flat_i].detach().cpu().numpy().astype(np.int64)
        )
        # parent_indices has (n_nodes + 1,) shape with sentinel at [0].
        parent_indices_np = (
            tree_parent_indices_list[flat_i].detach().cpu().numpy().astype(np.int64)
        )
        n_nodes = node_token_ids_np.shape[0]
        assert n_nodes <= tree_budget

        # slots [W + 1 .. W + n_nodes] hold real tree node tokens.
        out_input_ids_np[out_i, W + 1 : W + 1 + n_nodes] = node_token_ids_np
        for v_idx in range(n_nodes):
            out_position_ids_np[out_i, W + 1 + v_idx] = (
                anchor_pos + int(node_depths_np[v_idx])
            )
        # Remaining slots [W + 1 + n_nodes ..] stay 0 (padded).

        out_node_valid_np[out_i, :n_nodes] = True
        out_node_token_ids_np[out_i, :n_nodes] = node_token_ids_np
        # ``parent_node_indices`` uses convention: ``0`` == root(anchor),
        # positive indices refer to earlier tree nodes (matches DDTree).
        out_parent_indices_np[out_i, :n_nodes] = parent_indices_np[1 : n_nodes + 1]
        out_node_depths_np[out_i, :n_nodes] = node_depths_np

        # ------ attention mask ------
        # Build the single-anchor (L, L) mask via helper.
        # Effective ``prefix_len`` for the mask helper is the FULL
        # ``W + 1`` (including padded slots that we simply mark as
        # unreachable below).
        single_mask = _build_single_tree_mask(
            prefix_len=W + 1,
            tree_budget=tree_budget,
            parent_indices=parent_indices_np,
            node_count=n_nodes,
        )
        # Kill attention to padded prefix slots on ALL rows.
        if pad_prefix_len > 0:
            single_mask[:, :pad_prefix_len] = -1e9
            single_mask[:pad_prefix_len, :] = -1e9
        # Padded tree-node rows: mask entire row (they exist only for
        # padding shape and shouldn't consume any attention).
        for v_idx in range(n_nodes, tree_budget):
            single_mask[W + 1 + v_idx, :] = -1e9
        out_mask_np[out_i] = single_mask

    # One H2D copy per tensor.
    layout = TreeBatchLayout(
        input_ids=torch.from_numpy(out_input_ids_np).to(device),
        position_ids=torch.from_numpy(out_position_ids_np).to(device),
        # (B_flat, 1, L, L) for broadcast over heads.
        attention_mask=torch.from_numpy(out_mask_np).to(device).unsqueeze(1),
        node_valid=torch.from_numpy(out_node_valid_np).to(device),
        node_slot_indices=torch.from_numpy(out_node_slot_indices_np).to(device),
        valid_anchor_flat_ids=torch.tensor(
            valid_flat_ids, dtype=torch.long, device=device
        ),
        node_token_ids=torch.from_numpy(out_node_token_ids_np).to(device),
        parent_node_indices=torch.from_numpy(out_parent_indices_np).to(device),
        node_depths=torch.from_numpy(out_node_depths_np).to(device),
    )
    return layout


class LocalFullTargetVerifier(nn.Module):
    """Per-rank local Qwen3-family target with tree-attention forward.

    Loads the target once from ``model_path`` in the requested dtype
    onto the current CUDA device (~16 GB for Qwen3-8B in bf16).
    ``state_dict`` is stubbed empty so training checkpoints stay lean.

    The forward path:
      1. Accept a :class:`TreeBatchLayout` with ``B_flat`` anchor
         sequences of length ``L``.
      2. Run ``target_model.model(...)`` (skipping the LM head to save
         one huge matmul) with the tree attention mask.
      3. Return per-tree-node hidden states shaped
         ``(B_flat, tree_budget, H_target)``. Padded slots are zero.

    Also exposes ``anchor_hidden`` (per-anchor last-hidden at the
    ``W``-th position) which the caller uses to score depth-1 nodes'
    ``p_target(x_c | prefix, anchor)``.
    """

    def __init__(
        self,
        model_path: str,
        prefix_window: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if prefix_window < 0:
            raise ValueError(f"prefix_window must be >= 0, got {prefix_window}")
        self.prefix_window = int(prefix_window)
        self._dtype = dtype

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.target_model = model

        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)

    @torch.no_grad()
    def forward_along_trees(
        self,
        layout: TreeBatchLayout,
    ) -> torch.Tensor:
        """Run one forward and return per-slot last-hidden.

        Returns
        -------
        hidden : Tensor ``(B_flat, L, H)`` in ``self._dtype``
            The target's post-final-norm last-hidden at every slot.
            The caller slices ``[:, W:]`` to get anchor + all tree
            node hiddens (``L_tree_nodes = 1 + tree_budget``); slot
            ``W`` scores depth-1 children (``next-token | prefix,
            anchor``), slot ``W + 1 + v`` scores depth-``depth(v)+1``
            children (``next-token | prefix, anchor, ancestors, v``).
        """
        if layout.input_ids.shape[0] == 0:
            return torch.empty(
                0,
                layout.input_ids.shape[1] if layout.input_ids.ndim >= 2 else 0,
                self.hidden_size,
                dtype=self._dtype,
                device=layout.input_ids.device,
            )
        # CRITICAL: cast the additive mask to the model's dtype.
        # PyTorch SDPA silently produces WRONG outputs when Q/K/V are
        # bfloat16 but ``attn_mask`` is float32 -- see diagnostic in
        # ``diagnose_tree_verify_4.py`` (max_abs_diff ~4.0 vs
        # reference is_causal path). Casting the mask to the same
        # dtype as the model activations fixes it. Additionally we
        # clamp large negative values so bf16's range (~-3.4e38) is
        # respected and downstream softmax still zero-masks properly.
        finfo_min = torch.finfo(self._dtype).min
        attention_mask = (
            layout.attention_mask.clamp_min(finfo_min).to(self._dtype)
        )
        # Qwen3Model.forward accepts a raw 4D tensor and short-circuits
        # ``create_causal_mask``, using our mask as-is for every layer.
        outputs = self.target_model.model(
            input_ids=layout.input_ids,
            attention_mask=attention_mask,       # (B_flat, 1, L, L) bf16
            position_ids=layout.position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state
