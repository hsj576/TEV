# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""OnlineDTEDModel: DTED training wrapper for the DFlash draft.

Phase 5 architecture (hybrid CE + REINFORCE, matches DFlash/DSpark):

  * Primary loss = weighted cross-entropy against target argmax
    (identical to DFlash's ``loss_type="dflash"`` path). This gives
    dense, low-variance supervision that directly optimizes the
    draft's top-1 accuracy at every supervised position -- which is
    the necessary condition for tree accept.
  * Auxiliary loss = the original DTED REINFORCE objective
    (``sum_v w(v) * -log q_theta(x_v)``), scaled by ``dted_weight``
    (default 0.1). This retains the tree-shaped weighting signal that
    tells the draft "these tokens matter more".
  * Total loss = ``ce_weight * L_CE + dted_weight * L_DTED``.

Why this fixes the Phase 4 diagnosis (loss down but expected_al flat):
  * L_CE has dense per-position gradient (16k+ tokens/step) vs L_DTED's
    ~4k tree-node gradient, so variance shrinks ~4x.
  * L_CE directly optimizes ``argmax draft == argmax target`` which is
    the atomic operation tree verify performs. L_DTED alone only
    optimizes log-prob mass on tree nodes, not top-1 alignment.
  * ``weight_type`` default is now ``"P_tgt"`` (not "exit"): removes
    the depth-multiplier ``d(v)`` that pushed loss toward deep-chain
    nodes rarely accepted.

Speed vs. Phase 4.5: no regression -- the added CE term reuses the
already-computed ``draft_logits`` and is one fused
``F.cross_entropy(reshape, reshape)`` per chunk.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dted.loss import (
    build_dted_chunk_tree_tensors,
    dted_chunk_loss_from_flat_logits,
)
from specforge.core.chunking import checkpointed_chunk_reduce

__all__ = ["OnlineDTEDModel"]


class OnlineDTEDModel(OnlineDFlashModel):
    """DTED online training wrapper (Phase 5: hybrid CE + REINFORCE)."""

    def __init__(
        self,
        draft_model: Any,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 256,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        # ---- DTED-specific ---------------------------------------
        tree_budget: int = 64,
        dted_alpha: float = 1e-4,
        dted_eps: float = 1e-8,
        weight_type: str = "P_tgt",       # Phase 5: default off the depth mult
        ce_weight: float = 1.0,           # Phase 5: primary CE-to-target weight
        dted_weight: float = 0.1,         # Phase 5: auxiliary REINFORCE weight
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
            loss_type="dflash",  # unused; DTED reroutes forward
            dpace_alpha=0.5,     # unused
        )
        if weight_type not in {"exit", "P_tgt"}:
            raise ValueError(
                f"weight_type={weight_type!r}; must be 'exit' or 'P_tgt'"
            )
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

    # -----------------------------------------------------------------
    # Aligned target hidden (identical to OnlineDSparkModel).
    # -----------------------------------------------------------------
    def _aligned_target_hidden(
        self,
        target_last_hidden_states: torch.Tensor,
        safe_label_indices: torch.Tensor,
    ) -> torch.Tensor:
        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        batch_size = target_last_hidden_states.shape[0]
        hidden_size = target_last_hidden_states.shape[-1]
        gather_indices = target_pred_indices.reshape(batch_size, -1, 1).expand(
            -1, -1, hidden_size
        )
        return torch.gather(
            target_last_hidden_states,
            1,
            gather_indices,
        ).reshape(*safe_label_indices.shape, hidden_size)

    # -----------------------------------------------------------------
    # Chunk kernel -- called by ``checkpointed_chunk_reduce`` along the
    # anchor dimension. Every operation here works on **one chunk** of
    # anchors (``N_chunk = objective_chunk_blocks``).
    # -----------------------------------------------------------------
    def _dted_objective_chunk_terms(
        self,
        hidden: torch.Tensor,                    # (B, N_chunk, bs, H_draft)
        aligned_target_hidden: torch.Tensor,     # (B, N_chunk, bs, H_target)
        anchor_valid: torch.Tensor,              # (B, N_chunk) bool
        gt_target_ids: torch.Tensor,             # (B, N_chunk, bs) long
        eval_mask: torch.Tensor,                 # (B, N_chunk, bs) float
        ce_weight_mask: torch.Tensor,            # (B, N_chunk, bs) float, dflash-style
    ) -> Tuple[torch.Tensor, ...]:
        """One-chunk hybrid loss (CE + DTED) + diagnostics.

        The chunk emits *two* loss numerators (CE and DTED) and their
        respective denominators; the caller (``forward``) applies
        ``ce_weight`` and ``dted_weight`` after the checkpointed
        chunk-reduce sum, so both weights remain constant scalars in
        the autograd graph.
        """

        bsz, n_chunk, bs, H = hidden.shape
        K = self.block_size - 1                                        # 15
        assert bs == self.block_size

        # ------------------------------------------------------------
        # 1) One-shot lm_head over the whole chunk (WITH grad).
        # ------------------------------------------------------------
        # We only need logits at depths k = 1..K (predicting positions).
        # Slice depth first so the fused matmul does not waste FLOPs on
        # the anchor position (k = 0).
        pred_hidden = hidden[:, :, 1:, :].contiguous()                 # (B, N_chunk, K, H)
        draft_logits = self.lm_head(
            pred_hidden.reshape(bsz, n_chunk * K, H)
        ).reshape(bsz, n_chunk, K, -1)                                 # (B, N_chunk, K, V)

        # Target logits: no-grad, same shape.
        with torch.no_grad():
            tgt_pred_hidden = aligned_target_hidden[:, :, 1:, :].contiguous()
            target_logits = self.lm_head(
                tgt_pred_hidden.reshape(bsz, n_chunk * K, H)
            ).reshape(bsz, n_chunk, K, -1)                             # (B, N_chunk, K, V)

        # ------------------------------------------------------------
        # 2) Primary CE loss (distillation-style; hard label = argmax
        #    of target logits at each position).
        # ------------------------------------------------------------
        # We supervise draft to match target argmax at every position
        # covered by ``ce_weight_mask`` (block_valid & bounds & pos>0 &
        # loss_mask). This is dense per-position supervision -- one
        # gradient per supervised token -- and does NOT require the
        # anchor to have full-horizon supervision (unlike DTED).
        #
        # Using target ARGMAX (rather than the ground-truth text token)
        # matches the DFlash "target_ids" definition: target_ids are
        # the ground-truth NEXT tokens from ``input_ids``, which by
        # construction are already the tokens that target model would
        # emit greedy. So ``gt_target_ids`` is the correct label here.
        ce_label_ids = gt_target_ids[:, :, 1:]                             # (B, N_chunk, K)
        ce_pred_mask = ce_weight_mask[:, :, 1:]                            # (B, N_chunk, K)
        neg_log_q_flat = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.shape[-1]),
            ce_label_ids.reshape(-1),
            reduction="none",
        ).reshape_as(ce_label_ids)                                         # (B, N_chunk, K)
        ce_loss_num = (neg_log_q_flat * ce_pred_mask).sum()
        ce_loss_den = ce_pred_mask.sum()

        # ------------------------------------------------------------
        # 3) Build tree gather tensors from the DETACHED draft logits.
        # ------------------------------------------------------------
        # Note: tree structure depends only on ``draft_logits.detach()``.
        tree_tensors = build_dted_chunk_tree_tensors(
            draft_logits_chunk_detached=draft_logits.detach(),
            target_logits_chunk_detached=target_logits,           # already no-grad
            anchor_valid_chunk=anchor_valid,
            block_size=self.block_size,
            tree_budget=self.tree_budget,
            alpha=self.dted_alpha,
            weight_type=self.weight_type,
            device=hidden.device,
        )
        diag = tree_tensors.per_anchor_diag

        # ------------------------------------------------------------
        # 4) DTED loss numerator using the row-decomposed formulation.
        # ------------------------------------------------------------
        flat_draft_logits = draft_logits.reshape(-1, draft_logits.shape[-1])   # (M_flat, V)
        dted_loss_num = dted_chunk_loss_from_flat_logits(
            flat_draft_logits,
            row_indices=tree_tensors.row_indices,
            x_v=tree_tensors.x_v,
            weight=tree_tensors.weight,
            partition_weight_per_row=tree_tensors.partition_weight_per_row,
        )

        # ------------------------------------------------------------
        # 5) Accuracy + gap-depth (fully batched, no grad).
        # ------------------------------------------------------------
        with torch.no_grad():
            # Only supervised depths k in [1, K].
            gt_pred_ids = gt_target_ids[:, :, 1:]                              # (B, N_chunk, K)
            eval_pred_mask = eval_mask[:, :, 1:]                               # (B, N_chunk, K) float
            predicted_ids = draft_logits.argmax(dim=-1)                        # (B, N_chunk, K)

            # Accuracy over anchor_valid & eval_pred_mask positions.
            valid_positions = (
                eval_pred_mask > 0.5
            ) & anchor_valid.unsqueeze(-1)                                     # (B, N_chunk, K) bool
            correct_num = (
                (predicted_ids == gt_pred_ids) & valid_positions
            ).sum().float()                                                    # scalar
            accuracy_denom = valid_positions.sum().float()                     # scalar

            # gap_depth per valid anchor: length of the leading True
            # prefix of ``predicted_ids == gt_pred_ids`` along k.
            matches = (predicted_ids == gt_pred_ids)                           # (B, N_chunk, K) bool
            # Mask out invalid anchors so their contribution is 0.
            any_false_prefix = (~matches).cummax(dim=-1).values                # (B, N_chunk, K) bool
            has_any_false = any_false_prefix[..., -1]                          # (B, N_chunk)
            first_false_idx = any_false_prefix.float().argmax(dim=-1)          # (B, N_chunk) long
            gap_full = torch.full_like(first_false_idx, K, dtype=torch.long)
            gap_depth_per_anchor = torch.where(
                has_any_false, first_false_idx.long(), gap_full
            ).float()                                                          # (B, N_chunk)
            gap_depth_sum = (gap_depth_per_anchor * anchor_valid.float()).sum()
            gap_depth_denom = anchor_valid.float().sum()

        # ------------------------------------------------------------
        # 6) Per-anchor DTED-specific diagnostics -> chunk-level sums.
        # ------------------------------------------------------------
        n_valid_t = torch.tensor(
            float(diag.n_valid_anchors), device=hidden.device
        )
        n_tree_nodes_t = torch.tensor(
            float(diag.n_tree_nodes_total), device=hidden.device
        )
        if diag.n_valid_anchors > 0:
            tree_sizes = diag.tree_sizes
            p_tgt_weighted_sum = (
                diag.p_tgt_mean_per_anchor * tree_sizes
            ).sum()
            P_weighted_sum = (diag.P_mean_per_anchor * tree_sizes).sum()
            weight_used_weighted_sum = (
                diag.weight_used_mean_per_anchor * tree_sizes
            ).sum()
            exit_weight_sum_total = diag.exit_weight_sum_per_anchor.sum()
            expected_al_sum_total = diag.expected_al_per_anchor.sum()
        else:
            zero = torch.zeros((), device=hidden.device)
            p_tgt_weighted_sum = zero
            P_weighted_sum = zero
            weight_used_weighted_sum = zero
            exit_weight_sum_total = zero
            expected_al_sum_total = zero

        return (
            ce_loss_num,                    # 0: CE loss numerator (WITH grad)
            ce_loss_den,                    # 1: CE loss denominator (no grad)
            dted_loss_num,                  # 2: DTED loss numerator (WITH grad)
            n_valid_t,                      # 3: num_valid_anchors (int-as-float)
            n_tree_nodes_t,                 # 4: num_tree_nodes total
            p_tgt_weighted_sum,             # 5
            P_weighted_sum,                 # 6
            weight_used_weighted_sum,       # 7
            exit_weight_sum_total,          # 8
            expected_al_sum_total,          # 9
            correct_num,                    # 10
            accuracy_denom,                 # 11
            gap_depth_sum,                  # 12
            gap_depth_denom,                # 13
        )

    # -----------------------------------------------------------------
    # forward -- mirrors OnlineDSparkModel.forward, minus the DSpark
    # confidence head plumbing.
    # -----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        if target_last_hidden_states is None:
            raise ValueError(
                "OnlineDTEDModel.forward requires target_last_hidden_states"
            )
        if (
            self.attention_backend == "flex_attention"
            and not _flex_attention_available()
        ):
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # --- 1) Draft block-parallel forward (identical to DFlash/DSpark).
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

        # --- 2) Label indices + per-position validity.
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        in_bounds = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        gt_target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_anchors_sampled, -1),
            2,
            safe_label_indices,
        )                                                                      # (B, N, bs)

        supervised = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_anchors_sampled, -1),
            2,
            safe_label_indices,
        ).float()

        # Per-position eval mask (used for accuracy + full-horizon check).
        eval_mask = (
            in_bounds.float()
            * supervised
            * (label_offsets > 0).float()
        )                                                                      # (B, N, bs)

        # DTED requires the full K = block_size-1 horizon supervised.
        full_horizon = (
            eval_mask[:, :, 1:].sum(dim=-1) == float(self.block_size - 1)
        )
        anchor_valid = block_keep_mask & full_horizon                          # (B, N) bool

        # --- 2.5) CE weight mask (dflash-style: dense per-position).
        # Same construction as OnlineDFlashModel.forward's weight_mask:
        # block_valid * bounds * (pos>0) * loss_mask. Does NOT require
        # full-horizon; every supervised position contributes.
        ce_weight_mask = (
            block_keep_mask.float().unsqueeze(-1)                              # (B, N, 1)
            * in_bounds.float()
            * (label_offsets > 0).float()
            * supervised
        )                                                                      # (B, N, bs)

        # --- 3) Aligned target hidden (identical semantics to DSpark).
        aligned_target_hidden = self._aligned_target_hidden(
            target_last_hidden_states, safe_label_indices
        )                                                                      # (B, N, bs, H)

        # --- 4) Chunked additive reduction along the anchor dim.
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
        ) = checkpointed_chunk_reduce(
            self._dted_objective_chunk_terms,
            hidden_4d,
            aligned_target_hidden,
            anchor_valid,
            gt_target_ids,
            eval_mask,
            ce_weight_mask,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        # --- 5) Degenerate step: no supervised CE positions AND no
        # valid anchors. If either has any supervision, we can still
        # form a loss.
        n_valid_scalar = float(n_valid_sum.detach().item())
        ce_den_scalar = float(ce_loss_den.detach().item())
        if n_valid_scalar == 0.0 and ce_den_scalar == 0.0:
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

        # --- 6) Hybrid loss assembly.
        # CE part: mean over supervised positions.
        # DTED part: mean over tree nodes (unchanged from Phase 4.5).
        ce_den_safe = ce_loss_den.clamp_min(1.0)
        dted_den_safe = n_tree_nodes_sum.clamp_min(1.0)
        ce_loss_mean = ce_loss_num / ce_den_safe
        dted_loss_mean = dted_loss_num / dted_den_safe

        # Weighted combined loss.
        loss = self.ce_weight * ce_loss_mean + self.dted_weight * dted_loss_mean

        # ``loss_terms`` for the controller: keep numerator + denominator
        # of the COMBINED expression so cross-rank ratio-metric averaging
        # remains consistent. We construct an equivalent (num, den=1)
        # representation because the two sub-losses have different
        # denominators.
        combined_num = loss  # already-averaged scalar
        combined_den = loss.new_tensor(1.0)

        n_nodes_safe = n_tree_nodes_sum.clamp_min(1.0)
        mean_tree_size = n_tree_nodes_sum / n_valid_sum.clamp_min(1.0)
        mean_p_tgt = p_tgt_weighted_sum / n_nodes_safe
        mean_P = P_weighted_sum / n_nodes_safe
        mean_weight_used = weight_used_weighted_sum / n_nodes_safe
        mean_exit_weight_sum = exit_weight_sum_total / n_valid_sum.clamp_min(1.0)
        mean_expected_al = expected_al_sum_total / n_valid_sum.clamp_min(1.0)
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


def _flex_attention_available() -> bool:
    from specforge.algorithms.common.dflash_family_model import (
        FLEX_ATTENTION_AVAILABLE,
    )

    return FLEX_ATTENTION_AVAILABLE
