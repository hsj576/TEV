# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""OnlineExitTrainModel: DTED full-tree-verify training wrapper.

Compared to :class:`specforge.algorithms.dted.model.OnlineDTEDModel`
(hybrid CE + REINFORCE with a teacher-path Markov approximation),
this model:

  * Builds each anchor's DDTree in ``forward`` (same as the base
    DTED path but pulled out of the chunk kernel because the trees
    are needed up-front for the target verify pass).
  * Runs one target-model forward through
    :class:`specforge.algorithms.exittrain.tree_attention.LocalFullTargetVerifier`
    with a tree attention mask, yielding per-tree-node target
    hidden states.
  * Feeds those per-node hiddens into
    :func:`build_full_verify_chunk_tree_tensors`, which uses the
    honest per-node ``p_target(x | parent)`` distributions to compute
    P(v) and rho(v) (no Markov approximation).
  * Retains the hybrid CE + REINFORCE loss structure from base DTED.

Design notes
------------
* **Anchor count**: recommended ``num_anchors=32`` (fewer anchors than
  the base ``dted`` recipe) to keep target-verify cost tractable.
  See the ExitTrain README for details.
* **Per-rank target copy**: each trainer rank loads its own frozen
  bfloat16 target model. On a 143 GB HBM accelerator this adds ~16 GB
  on top of the ~40 GB base training footprint for an 8B target --
  comfortable.
* **Loss composition**: identical to base DTED::
    ``loss = ce_weight * CE + reinforce_weight * DTED``.
  The CE main loss still targets the ground-truth (gt) argmax and is
  unchanged; only the DTED aux loss changes p_target's source.
* **Backward through target**: none. Target params are frozen and
  everything runs under ``torch.no_grad``. Draft gradient flows via
  the ``flat_draft_logits`` term in ``dted_chunk_loss_from_flat_logits``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dted.ddtree_build import build_ddtree_tree_from_topk
from specforge.algorithms.dted.loss import dted_chunk_loss_from_flat_logits
from specforge.algorithms.exittrain.loss import (
    build_full_verify_chunk_tree_tensors,
)
from specforge.algorithms.exittrain.tree_attention import (
    LocalFullTargetVerifier,
    TreeBatchLayout,
    build_tree_batch_layout,
)


__all__ = ["OnlineExitTrainModel"]


class OnlineExitTrainModel(OnlineDFlashModel):
    """DTED online training wrapper with full-tree target verify."""

    def __init__(
        self,
        draft_model: Any,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 64,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        tree_budget: int = 64,
        dted_alpha: float = 1e-4,
        dted_eps: float = 1e-8,
        weight_type: str = "P_tgt",
        ce_weight: float = 1.0,
        dted_weight: float = 0.1,
        target_verifier: Optional[LocalFullTargetVerifier] = None,
        prefix_window: int = 64,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            loss_type="dflash",
            dpace_alpha=0.5,
        )
        if weight_type not in {"exit", "P_tgt"}:
            raise ValueError(f"weight_type must be 'exit' or 'P_tgt', got {weight_type!r}")
        if tree_budget <= 0:
            raise ValueError(f"tree_budget must be > 0, got {tree_budget}")
        if not 0.0 <= dted_alpha <= 1.0:
            raise ValueError(f"dted_alpha must be in [0, 1], got {dted_alpha}")
        if dted_eps <= 0.0:
            raise ValueError(f"dted_eps must be > 0, got {dted_eps}")
        if ce_weight < 0.0:
            raise ValueError(f"ce_weight must be >= 0, got {ce_weight}")
        if dted_weight < 0.0:
            raise ValueError(f"dted_weight must be >= 0, got {dted_weight}")

        self.tree_budget = int(tree_budget)
        self.dted_alpha = float(dted_alpha)
        self.dted_eps = float(dted_eps)
        self.weight_type = weight_type
        self.ce_weight = float(ce_weight)
        self.dted_weight = float(dted_weight)
        self.prefix_window = int(prefix_window)
        self.target_verifier = target_verifier
        if target_verifier is None:
            raise ValueError(
                "OnlineExitTrainModel requires a target_verifier; "
                "ExitTrain has no meaningful semantics without full-tree verify."
            )

    # ------------------------------------------------------------------
    # Build DDTrees for every valid anchor in the batch (CPU heap).
    # Same logic as build_dted_chunk_tree_tensors but returns raw
    # ``DDTreeStructure`` list (needed for the target-verify layout).
    # ------------------------------------------------------------------
    @staticmethod
    def _build_all_trees(
        draft_logits_all: torch.Tensor,    # (B, N, K, V) detached fp32/bf16
        anchor_valid: torch.Tensor,        # (B, N) bool
        tree_budget: int,
        depth_limit: int,
        device: torch.device,
    ) -> Tuple[
        List[torch.Tensor],  # node_token_ids per anchor
        List[torch.Tensor],  # parent_indices per anchor
        List[torch.Tensor],  # node_depths per anchor
    ]:
        bsz, num_anchors = anchor_valid.shape
        with torch.no_grad():
            top_logits, top_token_ids = torch.topk(
                draft_logits_all, k=min(tree_budget, draft_logits_all.shape[-1]), dim=-1
            )                                                          # (B, N, K, budget)
            log_z = torch.logsumexp(
                draft_logits_all.float(), dim=-1, keepdim=True
            )                                                          # (B, N, K, 1)
            top_log_probs = (top_logits.float() - log_z).contiguous()

            top_log_probs_all_np = top_log_probs.cpu().numpy()
            top_token_ids_all_np = top_token_ids.cpu().numpy().astype(np.int64, copy=False)

        valid_np = anchor_valid.detach().cpu().numpy()

        node_ids_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        depths_list: List[torch.Tensor] = []

        empty_i = torch.empty(0, dtype=torch.long, device=device)
        root_parents = torch.tensor([-1], dtype=torch.long, device=device)

        for b in range(bsz):
            for a in range(num_anchors):
                if not valid_np[b, a]:
                    node_ids_list.append(empty_i)
                    parents_list.append(root_parents)
                    depths_list.append(empty_i)
                    continue
                tree = build_ddtree_tree_from_topk(
                    top_log_probs_np=top_log_probs_all_np[b, a],
                    top_token_ids_np=top_token_ids_all_np[b, a],
                    budget=tree_budget,
                    depth_limit=depth_limit,
                    device=device,
                )
                node_ids_list.append(tree.node_token_ids)
                parents_list.append(tree.parent_indices)
                depths_list.append(tree.node_depths)
        return node_ids_list, parents_list, depths_list

    # ------------------------------------------------------------------
    # Chunk kernel: CE main loss + DTED aux loss with per-node
    # target probs already computed (passed in via layout tensors).
    # ------------------------------------------------------------------
    def _exittrain_chunk_terms(
        self,
        hidden: torch.Tensor,               # (B, N_chunk, bs, H) with grad
        anchor_valid: torch.Tensor,         # (B, N_chunk) bool
        gt_target_ids: torch.Tensor,        # (B, N_chunk, bs) long
        eval_mask: torch.Tensor,            # (B, N_chunk, bs) float
        ce_weight_mask: torch.Tensor,       # (B, N_chunk, bs) float
        row_indices: torch.Tensor,          # (M_chunk,) long: chunk-local rows for tree nodes in this chunk
        x_v: torch.Tensor,                  # (M_chunk,) long
        weight: torch.Tensor,               # (M_chunk,) fp32
        partition_weight_per_row: torch.Tensor,  # (M_chunk_flat,) fp32
        chunk_diag_pieces: torch.Tensor,    # (7,) fp32: pre-reduced diag sums for this chunk
    ) -> Tuple[torch.Tensor, ...]:
        """One-chunk hybrid loss (CE + DTED-with-full-verify) + diagnostics."""
        bsz, n_chunk, bs, H = hidden.shape
        K = self.block_size - 1
        assert bs == self.block_size

        # 1) One-shot lm_head over the chunk (WITH grad).
        pred_hidden = hidden[:, :, 1:, :].contiguous()
        draft_logits = self.lm_head(
            pred_hidden.reshape(bsz, n_chunk * K, H)
        ).reshape(bsz, n_chunk, K, -1)

        # 2) CE main loss (dflash-style, dense per-position).
        ce_label_ids = gt_target_ids[:, :, 1:]
        ce_pred_mask = ce_weight_mask[:, :, 1:]
        neg_log_q_flat = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.shape[-1]),
            ce_label_ids.reshape(-1),
            reduction="none",
        ).reshape_as(ce_label_ids)
        ce_loss_num = (neg_log_q_flat * ce_pred_mask).sum()
        ce_loss_den = ce_pred_mask.sum()

        # 3) DTED loss via row-decomposed formulation. Rows here refer
        # to the CHUNK-local flat (B * N_chunk * K, V) layout.
        flat_draft_logits = draft_logits.reshape(-1, draft_logits.shape[-1])
        dted_loss_num = dted_chunk_loss_from_flat_logits(
            flat_draft_logits,
            row_indices=row_indices,
            x_v=x_v,
            weight=weight,
            partition_weight_per_row=partition_weight_per_row,
        )

        # 4) Accuracy + gap-depth (no grad).
        with torch.no_grad():
            gt_pred_ids = gt_target_ids[:, :, 1:]
            eval_pred_mask = eval_mask[:, :, 1:]
            predicted_ids = draft_logits.argmax(dim=-1)
            valid_positions = (eval_pred_mask > 0.5) & anchor_valid.unsqueeze(-1)
            correct_num = ((predicted_ids == gt_pred_ids) & valid_positions).sum().float()
            accuracy_denom = valid_positions.sum().float()

            matches = predicted_ids == gt_pred_ids
            any_false_prefix = (~matches).cummax(dim=-1).values
            has_any_false = any_false_prefix[..., -1]
            first_false_idx = any_false_prefix.float().argmax(dim=-1)
            gap_full = torch.full_like(first_false_idx, K, dtype=torch.long)
            gap_depth_per_anchor = torch.where(
                has_any_false, first_false_idx.long(), gap_full
            ).float()
            gap_depth_sum = (gap_depth_per_anchor * anchor_valid.float()).sum()
            gap_depth_denom = anchor_valid.float().sum()

        # Diag pieces are precomputed by forward() (per whole chunk), we
        # simply forward them so ``checkpointed_chunk_reduce`` can sum
        # across chunks.
        (
            n_valid_t,
            n_tree_nodes_t,
            p_tgt_weighted_sum,
            P_weighted_sum,
            weight_used_weighted_sum,
            exit_weight_sum_total,
            expected_al_sum_total,
        ) = chunk_diag_pieces.unbind(0)

        return (
            ce_loss_num,                # 0
            ce_loss_den,                # 1
            dted_loss_num,              # 2
            n_valid_t,                  # 3
            n_tree_nodes_t,             # 4
            p_tgt_weighted_sum,         # 5
            P_weighted_sum,             # 6
            weight_used_weighted_sum,   # 7
            exit_weight_sum_total,      # 8
            expected_al_sum_total,      # 9
            correct_num,                # 10
            accuracy_denom,             # 11
            gap_depth_sum,              # 12
            gap_depth_denom,            # 13
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        # ``target_last_hidden_states`` is required to keep offline / SGLang
        # capture wire-format identical to base ``dted``, but ExitTrain does
        # NOT actually consume it (all target scoring goes through the local
        # verifier). We accept + ignore for backward compatibility.
        _ = target_last_hidden_states

        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # --- 1) Draft block-parallel forward.
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            max_valid_anchors=max_valid_anchors,
        )
        num_anchors_sampled = anchor_positions.shape[1]
        hidden_4d = output_hidden.reshape(
            bsz, num_anchors_sampled, self.block_size, -1
        )

        # --- 2) Labels + validity.
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        in_bounds = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        gt_target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_anchors_sampled, -1),
            2,
            safe_label_indices,
        )
        supervised = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_anchors_sampled, -1),
            2,
            safe_label_indices,
        ).float()
        eval_mask = in_bounds.float() * supervised * (label_offsets > 0).float()
        full_horizon = (
            eval_mask[:, :, 1:].sum(dim=-1) == float(self.block_size - 1)
        )
        anchor_valid = block_keep_mask & full_horizon                            # (B, N) bool

        ce_weight_mask = (
            block_keep_mask.float().unsqueeze(-1)
            * in_bounds.float()
            * (label_offsets > 0).float()
            * supervised
        )

        # --- 3) Build DDTrees for every valid anchor (uses DETACHED draft logits).
        K = self.block_size - 1
        with torch.no_grad():
            pred_hidden_all = hidden_4d[:, :, 1:, :].contiguous()
            draft_logits_full = self.lm_head(
                pred_hidden_all.reshape(-1, pred_hidden_all.shape[-1])
            ).reshape(bsz, num_anchors_sampled, K, -1)                           # (B, N, K, V)

        node_ids_list, parents_list, depths_list = self._build_all_trees(
            draft_logits_all=draft_logits_full,
            anchor_valid=anchor_valid,
            tree_budget=self.tree_budget,
            depth_limit=K,
            device=device,
        )

        # --- 4) Target verify: one forward through the target model
        #     across all valid anchors' flattened tree sequences.
        layout = build_tree_batch_layout(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            anchor_valid=anchor_valid,
            tree_node_token_ids_list=node_ids_list,
            tree_parent_indices_list=parents_list,
            tree_node_depths_list=depths_list,
            tree_budget=self.tree_budget,
            prefix_window=self.prefix_window,
            device=device,
        )
        # target hidden: (B_flat, L, H_target)
        target_hidden_flat = self.target_verifier.forward_along_trees(layout)

        # --- 5) Score anchor + each tree node through the target lm_head
        #     ONE TIME (no grad) to obtain per-node target probs.
        # Slice slots [W .. W + tree_budget] which are anchor + tree
        # nodes. Slot ``W`` = anchor (scores depth-1 children); slots
        # ``W + 1 .. W + tree_budget`` = tree nodes (each scores its
        # own next-depth children).
        W = self.prefix_window
        with torch.no_grad():
            # (B_flat, 1 + tree_budget, H_target)
            score_hidden = target_hidden_flat[:, W : W + 1 + self.tree_budget, :]
            score_hidden_flat = score_hidden.reshape(-1, score_hidden.shape[-1])
            # Feed through target lm_head (shared with draft's lm_head
            # -- both come from the target model's ``lm_head``, wired
            # in via ``target_lm_head`` on __init__).
            score_logits_flat = self.lm_head(score_hidden_flat)                  # (B_flat*(1+K), V)
            score_probs_flat = F.softmax(score_logits_flat.float(), dim=-1)

            B_flat = target_hidden_flat.shape[0]
            score_probs = score_probs_flat.reshape(
                B_flat, 1 + self.tree_budget, -1
            )                                                                    # (B_flat, 1+B, V)

        # --- 6) Build per-anchor target_probs_per_node lists.
        # For flat index ``i`` (valid): rows 0..n_nodes_i -> row 0 is
        # anchor, rows 1..n_nodes_i score children of tree nodes.
        # Invalid anchors get empty tensors.
        valid_flat_ids_cpu = layout.valid_anchor_flat_ids.cpu().tolist()
        target_probs_per_node_list: List[torch.Tensor] = []
        # Preallocate empty for every (B, N) slot; fill only valid ones.
        empty_probs = torch.empty(0, score_probs.shape[-1], device="cpu")
        target_probs_per_node_list = [empty_probs] * (bsz * num_anchors_sampled)
        # For each valid anchor, extract score rows [0 .. n_nodes] from
        # its ``score_probs[i_valid]``, where ``0`` = anchor row and
        # ``1..n_nodes`` = the first n_nodes tree slots (real ones).
        score_probs_cpu = score_probs.cpu()
        for out_i, flat_i in enumerate(valid_flat_ids_cpu):
            n_nodes = int(node_ids_list[flat_i].numel())
            if n_nodes == 0:
                continue
            # (n_nodes + 1, V) -- anchor + real tree nodes.
            per_anchor = score_probs_cpu[out_i, : 1 + n_nodes, :]
            target_probs_per_node_list[flat_i] = per_anchor

        # --- 7) Compute DTED tree tensors (row/x_v/weight) globally
        # over the whole batch. Since anchors are chunked below,
        # convert global row indices to be **absolute** across
        # ``B * N * K`` and let the chunk kernel offset locally.
        global_tree_tensors = build_full_verify_chunk_tree_tensors(
            tree_node_token_ids_list=node_ids_list,
            tree_parent_indices_list=parents_list,
            tree_node_depths_list=depths_list,
            target_probs_per_node_list=target_probs_per_node_list,
            anchor_valid_chunk=anchor_valid,
            block_size=self.block_size,
            tree_budget=self.tree_budget,
            alpha=self.dted_alpha,
            weight_type=self.weight_type,
            device=device,
        )
        diag = global_tree_tensors.per_anchor_diag
        n_valid_scalar = float(diag.n_valid_anchors)

        # --- 8) Degenerate: no valid anchors AND no CE supervision.
        n_ce_positions = float(ce_weight_mask.sum().detach().item())
        if n_valid_scalar == 0.0 and n_ce_positions == 0.0:
            zero = hidden_4d.sum() * 0.0
            return (
                zero,
                zero.new_zeros(()),
                {
                    "num_tree_nodes": zero.new_tensor(0.0),
                    "num_valid_anchors": zero.new_tensor(0.0),
                    "p_tgt_mean": zero.new_tensor(0.0),
                    "P_mean": zero.new_tensor(0.0),
                    "weight_used_mean": zero.new_tensor(0.0),
                    "exit_weight_sum_mean": zero.new_tensor(0.0),
                    "expected_al_mean": zero.new_tensor(0.0),
                    "gap_depth_mean": zero.new_tensor(0.0),
                    "ce_loss": zero.new_tensor(0.0),
                    "dted_loss": zero.new_tensor(0.0),
                    "accuracy_denom": zero.new_tensor(0.0),
                    "ratio_metrics": {},
                    "loss_terms": (zero, zero.new_tensor(1.0)),
                },
            )

        # --- 9) Chunked additive reduction along the anchor dim.
        # Instead of splitting tree tensors per chunk (complex), we
        # process the batch in a SINGLE pass. Given num_anchors=32-64
        # target-verify runs, using chunk size = num_anchors (single
        # chunk) keeps memory identical to the base ``dted`` recipe's
        # ``objective_chunk_blocks=128`` when num_anchors=256 (2 chunks);
        # a single chunk of 32-64 anchors is even lighter.
        # We still call the chunk kernel once for consistency of the
        # loss-terms surface.
        chunk_diag = torch.zeros(7, dtype=torch.float32, device=device)
        chunk_diag[0] = float(diag.n_valid_anchors)
        chunk_diag[1] = float(diag.n_tree_nodes_total)
        if diag.n_valid_anchors > 0:
            tree_sizes = diag.tree_sizes
            chunk_diag[2] = (diag.p_tgt_mean_per_anchor * tree_sizes).sum()
            chunk_diag[3] = (diag.P_mean_per_anchor * tree_sizes).sum()
            chunk_diag[4] = (diag.weight_used_mean_per_anchor * tree_sizes).sum()
            chunk_diag[5] = diag.exit_weight_sum_per_anchor.sum()
            chunk_diag[6] = diag.expected_al_per_anchor.sum()

        (
            ce_loss_num,
            ce_loss_den,
            dted_loss_num,
            n_valid_sum,
            n_tree_nodes_sum,
            p_tgt_weighted_sum,
            P_weighted_sum,
            weight_used_weighted_sum,
            exit_weight_sum_total,
            expected_al_sum_total,
            correct_num,
            accuracy_denom,
            gap_depth_sum,
            gap_depth_denom,
        ) = self._exittrain_chunk_terms(
            hidden=hidden_4d,
            anchor_valid=anchor_valid,
            gt_target_ids=gt_target_ids,
            eval_mask=eval_mask,
            ce_weight_mask=ce_weight_mask,
            row_indices=global_tree_tensors.row_indices,
            x_v=global_tree_tensors.x_v,
            weight=global_tree_tensors.weight,
            partition_weight_per_row=global_tree_tensors.partition_weight_per_row,
            chunk_diag_pieces=chunk_diag,
        )

        # --- 10) Assemble hybrid loss + metrics.
        ce_den_safe = ce_loss_den.clamp_min(1.0)
        dted_den_safe = n_tree_nodes_sum.clamp_min(1.0)
        ce_loss_mean = ce_loss_num / ce_den_safe
        dted_loss_mean = dted_loss_num / dted_den_safe
        loss = self.ce_weight * ce_loss_mean + self.dted_weight * dted_loss_mean

        combined_num = loss
        combined_den = loss.new_tensor(1.0)

        n_nodes_safe = n_tree_nodes_sum.clamp_min(1.0)
        n_valid_safe = n_valid_sum.clamp_min(1.0)
        mean_tree_size = n_tree_nodes_sum / n_valid_safe
        mean_p_tgt = p_tgt_weighted_sum / n_nodes_safe
        mean_P = P_weighted_sum / n_nodes_safe
        mean_weight_used = weight_used_weighted_sum / n_nodes_safe
        mean_exit_weight_sum = exit_weight_sum_total / n_valid_safe
        mean_expected_al = expected_al_sum_total / n_valid_safe
        mean_gap_depth = gap_depth_sum / gap_depth_denom.clamp_min(1.0)
        accuracy = correct_num / accuracy_denom.clamp_min(1.0)

        metrics: Dict[str, object] = {
            "num_tree_nodes": mean_tree_size.detach(),
            "num_valid_anchors": n_valid_sum.detach(),
            "p_tgt_mean": mean_p_tgt.detach(),
            "P_mean": mean_P.detach(),
            "weight_used_mean": mean_weight_used.detach(),
            "exit_weight_sum_mean": mean_exit_weight_sum.detach(),
            "expected_al_mean": mean_expected_al.detach(),
            "gap_depth_mean": mean_gap_depth.detach(),
            "ce_loss": ce_loss_mean.detach(),
            "dted_loss": dted_loss_mean.detach(),
            "accuracy_denom": accuracy_denom.detach(),
            "ratio_metrics": {
                "acc": (correct_num.detach(), accuracy_denom.detach()),
            },
            "loss_terms": (combined_num, combined_den),
        }
        return loss, accuracy, metrics
