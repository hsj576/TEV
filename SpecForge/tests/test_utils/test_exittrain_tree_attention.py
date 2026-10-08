# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for :mod:`specforge.algorithms.exittrain.tree_attention`.

Everything runs on CPU. The verifier is exercised via a monkey-patched
stub causal LM (single embedding, no attention layers) so we can
assert shape/no-grad/state-dict semantics without loading a real
Qwen3-8B.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


def _make_stub_causal_lm(hidden_size: int = 32, vocab_size: int = 128):
    class _StubBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, attention_mask=None, position_ids=None,
                    use_cache=False, return_dict=True):
            hidden = self.embed(input_ids)
            if position_ids is not None:
                # Fold position ids so tests can detect them being used.
                hidden = hidden + position_ids.unsqueeze(-1).float() * 1e-3
            out = type("_Out", (), {})()
            out.last_hidden_state = hidden
            return out

    class _StubCausal(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _StubBase()
            self.config = type("Cfg", (), {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
            })()

        def eval(self):
            self.model.eval()
            return self

    return _StubCausal()


class TreeMaskShapeTests(unittest.TestCase):
    """Unit-test the tree mask construction helper directly."""

    def test_prefix_causal_and_tree_causal(self):
        from specforge.algorithms.exittrain.tree_attention import (
            _build_single_tree_mask,
        )
        # Anchor + 3 tree nodes with tree structure:
        #     root (anchor)
        #     ├── node1  (depth 1, parent=0=root)
        #     │    └── node2 (depth 2, parent=1)
        #     └── node3  (depth 1, parent=0=root)
        # After the sentinel, ``parent_indices = [-1, 0, 1, 0]``
        parent_indices = np.array([-1, 0, 1, 0], dtype=np.int64)
        # prefix_len = W + 1 = 2 (W=1 window + 1 anchor)
        mask = _build_single_tree_mask(
            prefix_len=2, tree_budget=3, parent_indices=parent_indices, node_count=3
        )
        # L = 2 + 3 = 5
        self.assertEqual(mask.shape, (5, 5))
        # Prefix rows: causal (lower triangular).
        self.assertEqual(mask[0, 0], 0.0)
        self.assertLess(mask[0, 1], -1e6)
        self.assertEqual(mask[1, 0], 0.0)
        self.assertEqual(mask[1, 1], 0.0)
        self.assertLess(mask[1, 2], -1e6)
        # Tree node1 (row 2): attends to prefix (0, 1) + self (2). Not sibling (4) nor its own child (3).
        self.assertEqual(mask[2, 0], 0.0)
        self.assertEqual(mask[2, 1], 0.0)
        self.assertEqual(mask[2, 2], 0.0)
        self.assertLess(mask[2, 3], -1e6)
        self.assertLess(mask[2, 4], -1e6)
        # Tree node2 (row 3): attends to prefix (0, 1) + parent node1 (2) + self (3). Not node3 (4).
        self.assertEqual(mask[3, 0], 0.0)
        self.assertEqual(mask[3, 1], 0.0)
        self.assertEqual(mask[3, 2], 0.0)
        self.assertEqual(mask[3, 3], 0.0)
        self.assertLess(mask[3, 4], -1e6)
        # Tree node3 (row 4): attends to prefix (0, 1) + self (4). Not node1 (2) or node2 (3).
        self.assertEqual(mask[4, 0], 0.0)
        self.assertEqual(mask[4, 1], 0.0)
        self.assertLess(mask[4, 2], -1e6)
        self.assertLess(mask[4, 3], -1e6)
        self.assertEqual(mask[4, 4], 0.0)


class BuildTreeBatchLayoutTests(unittest.TestCase):
    """Verify ``build_tree_batch_layout`` shape / masking invariants."""

    def _make_inputs(self, bsz=1, seq_len=16, num_anchors=2):
        # Two anchors, ``anchor_positions = [5, 10]``. Only the first
        # anchor has a valid tree with 3 nodes; the second is invalid
        # (empty tree tensors).
        input_ids = torch.arange(bsz * seq_len).long().reshape(bsz, seq_len)
        anchor_positions = torch.tensor([[5, 10]], dtype=torch.long)
        anchor_valid = torch.tensor([[True, False]])
        # Anchor 0: 3-node tree matching the mask test above.
        tree0_tokens = torch.tensor([101, 102, 103], dtype=torch.long)
        tree0_parents = torch.tensor([-1, 0, 1, 0], dtype=torch.long)
        tree0_depths = torch.tensor([1, 2, 1], dtype=torch.long)
        # Anchor 1: empty (invalid).
        empty_l = torch.empty(0, dtype=torch.long)
        empty_p = torch.tensor([-1], dtype=torch.long)
        return dict(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            anchor_valid=anchor_valid,
            tree_node_token_ids_list=[tree0_tokens, empty_l],
            tree_parent_indices_list=[tree0_parents, empty_p],
            tree_node_depths_list=[tree0_depths, empty_l],
            tree_budget=4,
            prefix_window=3,
            device=torch.device("cpu"),
        )

    def test_shape_and_padding(self):
        from specforge.algorithms.exittrain.tree_attention import (
            build_tree_batch_layout,
        )
        layout = build_tree_batch_layout(**self._make_inputs())
        # Only the first anchor is valid, so B_flat = 1.
        L = 3 + 1 + 4  # W + anchor + tree_budget = 8
        self.assertEqual(layout.input_ids.shape, (1, L))
        self.assertEqual(layout.position_ids.shape, (1, L))
        self.assertEqual(layout.attention_mask.shape, (1, 1, L, L))
        self.assertEqual(layout.node_valid.shape, (1, 4))
        # 3 real tree nodes, 1 padded.
        self.assertTrue(torch.equal(
            layout.node_valid[0], torch.tensor([True, True, True, False])
        ))
        # Anchor slot is at ``W`` = 3.
        # Real prefix positions should equal anchor_pos + slot_offset =
        # [-3, -2, -1] relative to anchor_pos=5, i.e., [2, 3, 4].
        self.assertEqual(layout.position_ids[0, 0].item(), 2)
        self.assertEqual(layout.position_ids[0, 3].item(), 5)  # anchor
        # Tree node slot positions: anchor_pos + depth(v).
        # Depths [1, 2, 1] -> positions [6, 7, 6].
        self.assertEqual(layout.position_ids[0, 4].item(), 6)
        self.assertEqual(layout.position_ids[0, 5].item(), 7)
        self.assertEqual(layout.position_ids[0, 6].item(), 6)

    def test_empty_batch_returns_zero_rows(self):
        """All-invalid batch yields zero-row tensors."""
        from specforge.algorithms.exittrain.tree_attention import (
            build_tree_batch_layout,
        )
        inputs = self._make_inputs()
        # Make BOTH anchors invalid.
        inputs["anchor_valid"] = torch.tensor([[False, False]])
        layout = build_tree_batch_layout(**inputs)
        self.assertEqual(layout.input_ids.shape[0], 0)
        self.assertEqual(layout.valid_anchor_flat_ids.numel(), 0)


class LocalFullTargetVerifierTests(unittest.TestCase):

    def _make_verifier(self):
        from specforge.algorithms.exittrain import target_verify_test_helpers  # local shim
        raise unittest.SkipTest("helper not used")

    def test_state_dict_exposes_target_params(self):
        """FSDP requires the target_verifier's params to appear in
        state_dict (paired with backend.py ignoring the whole submodule
        so they are not sharded). This test locks in that contract."""
        from specforge.algorithms.exittrain import tree_attention

        stub = _make_stub_causal_lm(hidden_size=32, vocab_size=128)
        with mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=stub,
        ):
            verifier = tree_attention.LocalFullTargetVerifier(
                model_path="/nonexistent",
                prefix_window=3,
                dtype=torch.float32,
                trust_remote_code=False,
            )
        sd = verifier.state_dict()
        # The target_model params (via the stub's embed) MUST be present.
        target_keys = [k for k in sd if "target_model" in k]
        self.assertGreater(len(target_keys), 0,
                           "target_model params must appear in state_dict "
                           "for FSDP compatibility")

    def test_frozen_params(self):
        from specforge.algorithms.exittrain import tree_attention

        stub = _make_stub_causal_lm(hidden_size=32, vocab_size=128)
        with mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=stub,
        ):
            verifier = tree_attention.LocalFullTargetVerifier(
                model_path="/nonexistent",
                prefix_window=3,
                dtype=torch.float32,
                trust_remote_code=False,
            )
        for p in verifier.target_model.parameters():
            self.assertFalse(p.requires_grad)

    def test_forward_along_trees_shape(self):
        from specforge.algorithms.exittrain import tree_attention

        stub = _make_stub_causal_lm(hidden_size=32, vocab_size=128)
        with mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=stub,
        ):
            verifier = tree_attention.LocalFullTargetVerifier(
                model_path="/nonexistent",
                prefix_window=3,
                dtype=torch.float32,
                trust_remote_code=False,
            )

        input_ids = torch.arange(16).long().unsqueeze(0)      # (1, 16)
        anchor_positions = torch.tensor([[5, 10]], dtype=torch.long)
        anchor_valid = torch.tensor([[True, False]])
        tree0_tokens = torch.tensor([101, 102, 103], dtype=torch.long)
        tree0_parents = torch.tensor([-1, 0, 1, 0], dtype=torch.long)
        tree0_depths = torch.tensor([1, 2, 1], dtype=torch.long)
        empty_l = torch.empty(0, dtype=torch.long)
        empty_p = torch.tensor([-1], dtype=torch.long)

        layout = tree_attention.build_tree_batch_layout(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            anchor_valid=anchor_valid,
            tree_node_token_ids_list=[tree0_tokens, empty_l],
            tree_parent_indices_list=[tree0_parents, empty_p],
            tree_node_depths_list=[tree0_depths, empty_l],
            tree_budget=4,
            prefix_window=3,
            device=torch.device("cpu"),
        )
        hidden = verifier.forward_along_trees(layout)
        # (B_flat=1, L=8, H=32)
        self.assertEqual(hidden.shape, (1, 8, 32))
        # It should be nonzero (stub embed + position id contribution).
        self.assertTrue(torch.any(hidden.abs() > 0))


if __name__ == "__main__":
    unittest.main()
