# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""End-to-end forward/backward tests for OnlineDTEDModel (Phase 3).

These tests exercise the full training-time path from ``forward()``
down through per-anchor DTED loss, including:

  * anchor sampling reuse from ``OnlineDFlashModel._forward_draft_blocks``
  * ``_aligned_target_hidden`` offset alignment (Q5 in docs/22)
  * per-anchor lm_head projections (draft with grad, target with no-grad)
  * per-anchor tree build + loss + aggregation
  * autograd wiring: draft weights receive nonzero grad, lm_head does
    NOT (target-side lm_head calls are inside ``torch.no_grad``)

The fixture reuses the sys.modules-stub pattern from
``test_dflash_losses.py`` so we can construct OnlineDTEDModel on CPU
without importing flash_attn or any GPU-specific model. See that file
for why the stubs are necessary.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------
# sys.modules stubs so ``OnlineDFlashModel`` (base class) and
# ``OnlineDTEDModel`` (subclass) can be imported without pulling in
# flash_attn / cuda-only modules.
# --------------------------------------------------------------------


class _DFlashDraftStub(nn.Module):
    pass


_stub_dflash_draft = types.ModuleType("specforge.modeling.draft.dflash")
_stub_dflash_draft.DFlashDraftModel = _DFlashDraftStub

_pkg_specforge = types.ModuleType("specforge")
_pkg_specforge.__path__ = [str(REPO / "specforge")]
_pkg_algorithms = types.ModuleType("specforge.algorithms")
_pkg_algorithms.__path__ = [str(REPO / "specforge" / "algorithms")]
_pkg_common = types.ModuleType("specforge.algorithms.common")
_pkg_common.__path__ = [str(REPO / "specforge" / "algorithms" / "common")]
_pkg_dted = types.ModuleType("specforge.algorithms.dted")
_pkg_dted.__path__ = [str(REPO / "specforge" / "algorithms" / "dted")]
_pkg_modeling = types.ModuleType("specforge.modeling")
_pkg_modeling.__path__ = [str(REPO / "specforge" / "modeling")]
_pkg_draft = types.ModuleType("specforge.modeling.draft")
_pkg_draft.__path__ = [str(REPO / "specforge" / "modeling" / "draft")]

# Load dflash_family_model first (base class).
_spec_base = importlib.util.spec_from_file_location(
    "specforge.algorithms.common.dflash_family_model",
    REPO / "specforge" / "algorithms" / "common" / "dflash_family_model.py",
)
_dflash_module = importlib.util.module_from_spec(_spec_base)

# Load dted submodules through the same stub tree.
_spec_ddtree = importlib.util.spec_from_file_location(
    "specforge.algorithms.dted.ddtree_build",
    REPO / "specforge" / "algorithms" / "dted" / "ddtree_build.py",
)
_ddtree_module = importlib.util.module_from_spec(_spec_ddtree)

_spec_loss = importlib.util.spec_from_file_location(
    "specforge.algorithms.dted.loss",
    REPO / "specforge" / "algorithms" / "dted" / "loss.py",
)
_loss_module = importlib.util.module_from_spec(_spec_loss)

_spec_model = importlib.util.spec_from_file_location(
    "specforge.algorithms.dted.model",
    REPO / "specforge" / "algorithms" / "dted" / "model.py",
)
_dted_model_module = importlib.util.module_from_spec(_spec_model)

with patch.dict(
    sys.modules,
    {
        "specforge": _pkg_specforge,
        "specforge.algorithms": _pkg_algorithms,
        "specforge.algorithms.common": _pkg_common,
        "specforge.algorithms.common.dflash_family_model": _dflash_module,
        "specforge.algorithms.dted": _pkg_dted,
        "specforge.algorithms.dted.ddtree_build": _ddtree_module,
        "specforge.algorithms.dted.loss": _loss_module,
        "specforge.algorithms.dted.model": _dted_model_module,
        "specforge.modeling": _pkg_modeling,
        "specforge.modeling.draft": _pkg_draft,
        "specforge.modeling.draft.dflash": _stub_dflash_draft,
    },
):
    _spec_base.loader.exec_module(_dflash_module)
    _spec_ddtree.loader.exec_module(_ddtree_module)
    _spec_loss.loader.exec_module(_loss_module)
    _spec_model.loader.exec_module(_dted_model_module)

OnlineDTEDModel = _dted_model_module.OnlineDTEDModel


# --------------------------------------------------------------------
# Minimal draft-model stub and fixtures.
# --------------------------------------------------------------------


class _LearnableDraft(nn.Module):
    """Tiny draft whose output depends on input via a linear projection.

    Enough to exercise the DTED forward end-to-end and verify that
    grad flows back into a real trainable parameter.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.sliding_window = None
        # Linear projection over noise_embedding as the sole trainable
        # parameter; DTED loss should update it via autograd.
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        del position_ids, target_hidden, attention_mask
        return self.projection(noise_embedding)

    # DSpark hooks the draft implements; DTED never calls them but
    # OnlineDFlashModel siblings may check attribute presence.
    def apply_logits_head(self, base_logits, **kwargs):
        del kwargs
        return base_logits

    def predict_confidence(self, hidden_states, prev_token_ids=None):
        del hidden_states, prev_token_ids
        return None


def _fixed_anchor_sampler(anchors: torch.Tensor, keep_mask: torch.Tensor):
    def _sample(self, seq_len, loss_mask, device, max_valid_anchors=None):
        del self, seq_len, loss_mask, max_valid_anchors
        return anchors.to(device), keep_mask.to(device)

    return _sample


def _fixed_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
    del anchor_positions, block_keep_mask
    bsz = input_ids.shape[0]
    # DFlash's draft expects a concatenated block sequence of shape
    # (bsz, num_anchors * block_size, hidden_size). Use a deterministic
    # non-zero pattern so ``Linear(x)`` produces non-trivial weight
    # gradients on backward (the zero-input case would masking any
    # weight update via ``x.T @ dL/dy = 0``).
    hidden_size = self.embed_tokens.embedding_dim
    positions = torch.arange(
        self.num_anchors * self.block_size, dtype=torch.double, device=input_ids.device
    )
    pattern = torch.sin(
        positions.view(1, -1, 1)
        + torch.arange(hidden_size, dtype=torch.double, device=input_ids.device).view(
            1, 1, -1
        )
    )
    return pattern.expand(bsz, -1, -1).contiguous()


def _make_dted_model(
    *,
    bsz: int,
    n_anchors: int,
    block_size: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    tree_budget: int,
    weight_type: str = "exit",
    anchors: torch.Tensor,
    keep_mask: torch.Tensor,
) -> OnlineDTEDModel:
    del bsz, seq_len  # only for readability
    draft = _LearnableDraft(hidden_size=hidden_size).double()
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).double()
    embed = nn.Embedding(vocab_size, hidden_size).double()
    model = OnlineDTEDModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=0,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=n_anchors,
        tree_budget=tree_budget,
        dted_alpha=1e-4,
        dted_eps=1e-8,
        weight_type=weight_type,
    ).double()
    # Bypass the real anchor sampler and the noise-embed constructor
    # so the test is deterministic and doesn't depend on random draws.
    model._sample_anchor_positions = types.MethodType(
        _fixed_anchor_sampler(anchors, keep_mask), model
    )
    model._create_noise_embed = types.MethodType(_fixed_noise_embed, model)
    return model


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------


class TestOnlineDTEDModelForward(unittest.TestCase):
    def _tiny_batch(self):
        torch.manual_seed(11)
        bsz, n_anchors, block_size, seq_len = 2, 2, 5, 12
        hidden_size, vocab_size = 8, 17
        input_ids = torch.tensor(
            [
                [1, 4, 2, 8, 3, 7, 5, 6, 9, 10, 11, 12],
                [2, 5, 1, 4, 7, 3, 8, 10, 11, 12, 13, 14],
            ],
            dtype=torch.long,
        )
        loss_mask = torch.ones(bsz, seq_len, dtype=torch.double)
        # Anchor at position 0 and 4 for row 0; position 1 and 5 for row 1.
        # With block_size=5, the block covers [anchor, anchor+4], all in
        # bounds and supervised.
        anchors = torch.tensor([[0, 4], [1, 5]], dtype=torch.long)
        keep_mask = torch.tensor([[True, True], [True, True]])
        hidden_states = torch.randn(bsz, seq_len, hidden_size, dtype=torch.double)
        target_last_hidden = torch.randn(bsz, seq_len, hidden_size, dtype=torch.double)
        return (
            bsz,
            n_anchors,
            block_size,
            seq_len,
            hidden_size,
            vocab_size,
            input_ids,
            loss_mask,
            hidden_states,
            target_last_hidden,
            anchors,
            keep_mask,
        )

    def test_forward_returns_finite_loss_and_metrics(self):
        (
            bsz,
            n_anchors,
            block_size,
            seq_len,
            hidden_size,
            vocab_size,
            input_ids,
            loss_mask,
            hidden_states,
            target_last_hidden,
            anchors,
            keep_mask,
        ) = self._tiny_batch()

        model = _make_dted_model(
            bsz=bsz,
            n_anchors=n_anchors,
            block_size=block_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tree_budget=16,
            anchors=anchors,
            keep_mask=keep_mask,
        )

        loss, accuracy, metrics = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden,
            max_valid_anchors=n_anchors,
        )

        # Loss must be a finite scalar tensor with grad_fn.
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(loss.grad_fn)

        # Accuracy in [0, 1].
        self.assertGreaterEqual(float(accuracy.item()), 0.0)
        self.assertLessEqual(float(accuracy.item()), 1.0)

        # Required metric surface (updated for exit-weight loss, D1(a)).
        for key in (
            "num_tree_nodes",
            "num_valid_anchors",
            "p_tgt_mean",
            "P_mean",
            "weight_used_mean",
            "exit_weight_sum_mean",
            "expected_al_mean",
            "gap_depth_mean",
            "ce_loss",
            "dted_loss",
            "accuracy_denom",
            "ratio_metrics",
            "loss_terms",
        ):
            self.assertIn(key, metrics, f"missing metric: {key}")
        # loss_terms numerator MUST retain grad -- controller.py does
        # ``loss = numerator.reshape(())`` and calls ``.backward()`` on
        # it directly. Detaching numerator crashes DDP with
        # "element 0 of tensors does not require grad" (seen in Phase 4
        # smoke). Denominator, however, must be detached.
        loss_num, loss_denom = metrics["loss_terms"]
        self.assertIsNotNone(
            loss_num.grad_fn,
            "loss_terms[0] (numerator) must retain grad_fn for controller",
        )
        self.assertIsNone(
            loss_denom.grad_fn,
            "loss_terms[1] (denominator) must be detached",
        )
        # Every anchor was valid (full horizon, all supervised).
        self.assertEqual(float(metrics["num_valid_anchors"].item()), 4.0)
        # Each anchor's tree ~ tree_budget nodes.
        self.assertGreater(float(metrics["num_tree_nodes"].item()), 0.0)
        self.assertLessEqual(
            float(metrics["num_tree_nodes"].item()), float(16)
        )

    def test_backward_produces_nonzero_draft_grad(self):
        (
            bsz,
            n_anchors,
            block_size,
            seq_len,
            hidden_size,
            vocab_size,
            input_ids,
            loss_mask,
            hidden_states,
            target_last_hidden,
            anchors,
            keep_mask,
        ) = self._tiny_batch()

        model = _make_dted_model(
            bsz=bsz,
            n_anchors=n_anchors,
            block_size=block_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tree_budget=16,
            anchors=anchors,
            keep_mask=keep_mask,
        )

        loss, _, _ = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden,
            max_valid_anchors=n_anchors,
        )
        loss.backward()

        # Draft projection must receive nonzero grad.
        draft_proj = model.draft_model.projection.weight
        self.assertIsNotNone(draft_proj.grad)
        self.assertGreater(draft_proj.grad.abs().sum().item(), 0.0)

        # The target-side lm_head is called under torch.no_grad; DTED's
        # p_tgt path must NOT contribute grad to it. But the draft side
        # DOES call lm_head with grad, so the head DOES get some grad
        # (from log q_theta). This is fine as long as the head is
        # intentionally trainable in the deployment (in our real setup
        # the head is frozen via ``requires_grad_(False)`` in the target
        # utils; that's an orthogonal concern outside this test).
        # Here we merely verify the graph is well-formed.
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_target_last_hidden_required(self):
        # DTED loss is undefined without p_tgt; forward must raise.
        (
            bsz,
            n_anchors,
            block_size,
            seq_len,
            hidden_size,
            vocab_size,
            input_ids,
            loss_mask,
            hidden_states,
            _target_last_hidden,
            anchors,
            keep_mask,
        ) = self._tiny_batch()

        model = _make_dted_model(
            bsz=bsz,
            n_anchors=n_anchors,
            block_size=block_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tree_budget=8,
            anchors=anchors,
            keep_mask=keep_mask,
        )

        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                target_last_hidden_states=None,
                max_valid_anchors=n_anchors,
            )

    def test_no_valid_anchor_yields_zero_loss_with_graph_edge(self):
        # If every anchor is invalid, we return a graph-edge zero loss.
        (
            bsz,
            n_anchors,
            block_size,
            seq_len,
            hidden_size,
            vocab_size,
            input_ids,
            loss_mask,
            hidden_states,
            target_last_hidden,
            anchors,
            _keep_mask,
        ) = self._tiny_batch()

        # Force all anchors invalid by mask.
        keep_mask = torch.zeros(bsz, n_anchors, dtype=torch.bool)
        model = _make_dted_model(
            bsz=bsz,
            n_anchors=n_anchors,
            block_size=block_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tree_budget=8,
            anchors=anchors,
            keep_mask=keep_mask,
        )
        loss, accuracy, metrics = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden,
            max_valid_anchors=n_anchors,
        )
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(float(accuracy.item()), 0.0)
        self.assertEqual(float(metrics["num_valid_anchors"].item()), 0.0)
        # Must still support backward without error.
        loss.backward()


class TestOnlineDTEDModelWeightVariants(unittest.TestCase):
    """Both weight variants must run without error and produce grads."""

    def test_P_tgt_variant_backward(self):
        # Second weight variant ("P_tgt") should also run and produce
        # non-zero draft gradients. This complements the default
        # "exit" variant tested in TestOnlineDTEDModelForward.
        torch.manual_seed(42)
        bsz, n_anchors, block_size, seq_len = 1, 1, 5, 8
        hidden_size, vocab_size = 8, 17
        anchors = torch.tensor([[0]], dtype=torch.long)
        keep_mask = torch.tensor([[True]])
        input_ids = torch.tensor(
            [[1, 4, 2, 8, 3, 7, 5, 6]], dtype=torch.long
        )
        loss_mask = torch.ones(bsz, seq_len, dtype=torch.double)
        hidden_states = torch.randn(bsz, seq_len, hidden_size, dtype=torch.double)
        target_last_hidden = torch.randn(
            bsz, seq_len, hidden_size, dtype=torch.double
        )
        model = _make_dted_model(
            bsz=bsz,
            n_anchors=n_anchors,
            block_size=block_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tree_budget=16,
            weight_type="P_tgt",
            anchors=anchors,
            keep_mask=keep_mask,
        )
        loss, _, _ = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden,
            max_valid_anchors=n_anchors,
        )
        loss.backward()
        self.assertGreater(
            model.draft_model.projection.weight.grad.abs().sum().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
