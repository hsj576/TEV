# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for DTED per-anchor loss and DDTree construction (exit-weight version).

Phase 1 tests from docs/22 §6, updated after the D1(a) decision to
switch the loss to the TEV exit-weight formulation (docs/20 §3.1,
docs/21 §3.2). Each test targets an invariant that should hold *by
construction* -- a regression here is a clear signal that either the
theory or the implementation drifted.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from specforge.algorithms.dted.ddtree_build import build_ddtree_tree
from specforge.algorithms.dted.loss import dted_loss_per_anchor


def _make_logits(
    K: int, V: int, seed: int = 0, temperature: float = 1.0
) -> torch.Tensor:
    """Deterministic random logits."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, V, generator=g) / temperature


class TestDDTreeBuild(unittest.TestCase):
    def test_empty_budget_returns_root_only(self):
        logits = _make_logits(K=5, V=100)
        tree = build_ddtree_tree(logits, budget=0)
        self.assertEqual(tree.node_token_ids.numel(), 0)
        self.assertEqual(tree.node_depths.numel(), 0)
        self.assertEqual(tree.parent_indices.tolist(), [-1])

    def test_budget_1_returns_only_top1_at_depth1(self):
        logits = _make_logits(K=5, V=32)
        tree = build_ddtree_tree(logits, budget=1)
        self.assertEqual(tree.node_token_ids.numel(), 1)
        self.assertEqual(int(tree.node_depths[0]), 1)
        self.assertEqual(int(tree.node_token_ids[0]), int(logits[0].argmax()))
        self.assertEqual(tree.parent_indices.tolist(), [-1, 0])

    def test_first_expanded_node_is_top1_at_depth1(self):
        K, V = 4, 8
        logits = _make_logits(K=K, V=V)
        tree = build_ddtree_tree(logits, budget=K)
        self.assertEqual(int(tree.node_depths[0]), 1)
        self.assertEqual(int(tree.node_token_ids[0]), int(logits[0].argmax()))
        self.assertEqual(int(tree.parent_indices[1]), 0)

    def test_all_depths_are_in_valid_range(self):
        K, V = 6, 32
        logits = _make_logits(K=K, V=V, seed=17)
        tree = build_ddtree_tree(logits, budget=32)
        depths = tree.node_depths.tolist()
        self.assertTrue(all(1 <= d <= K for d in depths))

    def test_parents_are_strictly_smaller_than_children(self):
        # Structural invariant that the O(N) forward sweep for P(v)
        # relies on: parent_index < current_index.
        logits = _make_logits(K=6, V=64, seed=7)
        tree = build_ddtree_tree(logits, budget=32)
        for u_idx in range(tree.node_token_ids.numel()):
            par = int(tree.parent_indices[u_idx + 1])
            self.assertLess(par, u_idx + 1)
            self.assertGreaterEqual(par, 0)

    def test_device_placement(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        logits = _make_logits(K=4, V=32).cuda()
        tree = build_ddtree_tree(logits, budget=8, device=logits.device)
        self.assertEqual(tree.node_token_ids.device.type, "cuda")
        self.assertEqual(tree.node_depths.device.type, "cuda")
        self.assertEqual(tree.parent_indices.device.type, "cuda")


class TestDTEDLossBasics(unittest.TestCase):
    def test_grad_flows_only_through_log_q(self):
        K, V = 4, 16
        draft = _make_logits(K, V, seed=1).clone().requires_grad_(True)
        target = _make_logits(K, V, seed=2)

        result = dted_loss_per_anchor(
            draft, target, tree_budget=8, alpha=1e-4
        )
        result.loss.backward()
        self.assertIsNotNone(draft.grad)
        # Target must never accumulate grad.
        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad)
        self.assertGreater(draft.grad.abs().sum().item(), 0.0)

    def test_empty_tree_yields_zero_loss_with_graph_edge(self):
        K, V = 4, 16
        draft = _make_logits(K, V, seed=3).clone().requires_grad_(True)
        target = _make_logits(K, V, seed=4)
        result = dted_loss_per_anchor(draft, target, tree_budget=0)
        self.assertEqual(result.num_nodes, 0)
        self.assertEqual(float(result.loss.item()), 0.0)
        # Backward must still be legal (graph edge preserved).
        result.loss.backward()

    def test_matching_distributions_yield_reasonable_p_tgt(self):
        # Under D1(a) exit-weight loss, ``draft == target`` no longer
        # gives accept_ratio == 1 (there's no accept ratio anymore).
        # Instead, ``p_tgt_mean`` measures the average per-step target
        # prob of tokens the tree selected. For random logits, this
        # should be modest (tree picks top-k tokens whose target probs
        # are still not close to 1). We only assert it's a valid
        # probability in (0, 1] and the loss is finite.
        K, V = 5, 32
        base = _make_logits(K, V, seed=11)
        draft = base.clone().requires_grad_(True)
        target = base.clone()

        result = dted_loss_per_anchor(draft, target, tree_budget=16, alpha=1e-4)
        self.assertGreater(result.p_tgt_mean, 0.0)
        self.assertLessEqual(result.p_tgt_mean, 1.0)
        # ``expected_al`` is sum_v P(v) and must be non-negative.
        self.assertGreaterEqual(result.expected_al, 0.0)
        self.assertTrue(math.isfinite(float(result.loss.item())))
        self.assertGreater(float(result.loss.item()), 0.0)


class TestExitWeightPartitionIdentity(unittest.TestCase):
    """Exit-weight sum invariants.

    Full-tree Lemma 1 (docs/20 §3.1) states ``sum_v w(v) = 1`` for the
    tree that covers the entire top-V branching at every depth. On our
    *budget-limited* DDTree this is generally NOT the case: nodes that
    were never expanded (because the heap ran out of budget) still
    consume target probability mass, and ``rho(v) > 0`` for internal
    nodes with unexpanded siblings among their would-be children.

    The correct interpretation is:

        sum_{v in T} w(v)   =  Pr[verify exits at some node inside T]
        1 - sum_v w(v)      =  Pr[verify exits at the root]
                              (target's first-step token is not among
                               root's tree children)

    So we assert two invariants:

      (I1) 0 <= sum_v w(v) <= 1  -- probability bound.
      (I2) A larger tree budget => larger sum (more coverage).
    """

    def _sum_exit_weights(self, draft_logits, target_logits, budget, alpha=1e-4):
        draft = draft_logits.clone().requires_grad_(True)
        target = target_logits.clone()
        result = dted_loss_per_anchor(
            draft, target,
            tree_budget=budget, alpha=alpha, weight_type="exit",
        )
        return float(result.exit_weight_sum), int(result.num_nodes)

    def test_partition_bounded_in_unit_interval(self):
        # (I1) sum_v w(v) is a probability mass; must live in [0, 1].
        for seed in (1, 7, 42, 123):
            with self.subTest(seed=seed):
                draft = _make_logits(K=5, V=32, seed=seed)
                target = _make_logits(K=5, V=32, seed=seed + 100)
                partition, n = self._sum_exit_weights(
                    draft, target, budget=64
                )
                self.assertGreater(n, 0)
                # Allow tiny numerical slack above 1.0 (float rounding).
                self.assertGreaterEqual(partition, 0.0)
                self.assertLessEqual(partition, 1.0 + 1e-5)

    def test_partition_grows_with_budget(self):
        # (I2) A larger DDTree covers more probability mass, so the
        # exit-weight sum should be monotone non-decreasing in budget.
        # (Strict monotone up to numerical noise -- if budget saturates
        # the reachable branching, both sums may be equal.)
        for seed in (5, 19):
            with self.subTest(seed=seed):
                draft = _make_logits(K=4, V=32, seed=seed)
                target = _make_logits(K=4, V=32, seed=seed + 200)
                small_part, _ = self._sum_exit_weights(
                    draft, target, budget=8
                )
                large_part, _ = self._sum_exit_weights(
                    draft, target, budget=64
                )
                self.assertLessEqual(small_part, large_part + 1e-5)


class TestDTEDManualRecomputation(unittest.TestCase):
    """Rebuild the exit-weight loss by hand and compare bit-for-bit."""

    def test_exit_loss_matches_hand_computation(self):
        K, V = 4, 8
        draft_l = _make_logits(K, V, seed=21, temperature=0.5)
        target_l = _make_logits(K, V, seed=22, temperature=0.5)
        draft = draft_l.clone().requires_grad_(True)
        target = target_l.clone()

        alpha = 1e-4
        result = dted_loss_per_anchor(
            draft, target,
            tree_budget=K, alpha=alpha, weight_type="exit",
        )
        tree = build_ddtree_tree(draft.detach(), K)
        N = tree.node_token_ids.numel()
        self.assertGreater(N, 0)

        # Manual computation of the exit-weight loss (docs/22 D1(a)).
        with torch.no_grad():
            log_q = F.log_softmax(draft.detach().float(), dim=-1)
            p = F.softmax(target.float(), dim=-1)
            parent_ids = tree.parent_indices.tolist()
            depths = tree.node_depths.tolist()
            tokens = tree.node_token_ids.tolist()

            # P(v) with per-step alpha floor.
            P_all = [1.0] * (N + 1)
            raw_p_tgt = [0.0] * (N + 1)
            for u_idx in range(N):
                d = depths[u_idx] - 1
                x_u = tokens[u_idx]
                par = parent_ids[u_idx + 1]
                p_raw = float(p[d, x_u])
                p_floored = max(p_raw, alpha)
                P_all[u_idx + 1] = P_all[par] * p_floored
                raw_p_tgt[u_idx + 1] = p_raw

            # rho(v): 1 - sum of children's raw p_tgt.
            children_prob_sum = [0.0] * (N + 1)
            for u_idx in range(N):
                par = parent_ids[u_idx + 1]
                children_prob_sum[par] += raw_p_tgt[u_idx + 1]
            rho_all = [0.0] * (N + 1)
            for u_idx in range(1, N + 1):
                rho_all[u_idx] = max(0.0, min(1.0, 1.0 - children_prob_sum[u_idx]))

            # w(v) * depth(v) as the loss weight.
            expected_loss = 0.0
            for u_idx in range(N):
                d = depths[u_idx] - 1
                x_u = tokens[u_idx]
                w = P_all[u_idx + 1] * rho_all[u_idx + 1]
                weight = w * depths[u_idx]
                expected_loss += -weight * float(log_q[d, x_u])
            expected_loss /= N

        self.assertAlmostEqual(
            float(result.loss.item()), expected_loss, places=5,
            msg=f"impl={result.loss.item():.6f}, manual={expected_loss:.6f}",
        )

    def test_P_tgt_variant_matches_hand_computation(self):
        # Same setup but weight_type="P_tgt": weight is just P(v).
        K, V = 4, 8
        draft_l = _make_logits(K, V, seed=51, temperature=0.5)
        target_l = _make_logits(K, V, seed=52, temperature=0.5)
        draft = draft_l.clone().requires_grad_(True)
        target = target_l.clone()

        alpha = 1e-4
        result = dted_loss_per_anchor(
            draft, target,
            tree_budget=K, alpha=alpha, weight_type="P_tgt",
        )
        tree = build_ddtree_tree(draft.detach(), K)
        N = tree.node_token_ids.numel()

        with torch.no_grad():
            log_q = F.log_softmax(draft.detach().float(), dim=-1)
            p = F.softmax(target.float(), dim=-1)
            parent_ids = tree.parent_indices.tolist()
            depths = tree.node_depths.tolist()
            tokens = tree.node_token_ids.tolist()

            P_all = [1.0] * (N + 1)
            expected_loss = 0.0
            for u_idx in range(N):
                d = depths[u_idx] - 1
                x_u = tokens[u_idx]
                par = parent_ids[u_idx + 1]
                P_all[u_idx + 1] = P_all[par] * max(float(p[d, x_u]), alpha)
                expected_loss += -P_all[u_idx + 1] * float(log_q[d, x_u])
            expected_loss /= N

        self.assertAlmostEqual(
            float(result.loss.item()), expected_loss, places=5,
        )


class TestDTEDLossWeightVariants(unittest.TestCase):
    def test_exit_and_P_tgt_agree_gradient_direction(self):
        # Prop. 4: sum_u P(u) == sum_u w(u) * depth(u) in expectation
        # (both equal E[AL]). For a single anchor at finite budget the
        # gradients need not be identical, but their directions should
        # be positively correlated.
        K, V = 5, 64
        base_draft = _make_logits(K, V, seed=31)
        target = _make_logits(K, V, seed=32)

        draft_A = base_draft.clone().requires_grad_(True)
        draft_B = base_draft.clone().requires_grad_(True)

        res_A = dted_loss_per_anchor(
            draft_A, target, tree_budget=64, weight_type="exit"
        )
        res_B = dted_loss_per_anchor(
            draft_B, target, tree_budget=64, weight_type="P_tgt"
        )
        res_A.loss.backward()
        res_B.loss.backward()
        gA = draft_A.grad.flatten()
        gB = draft_B.grad.flatten()

        self.assertGreater(gA.abs().sum().item(), 0.0)
        self.assertGreater(gB.abs().sum().item(), 0.0)

        cos = F.cosine_similarity(gA.unsqueeze(0), gB.unsqueeze(0)).item()
        self.assertGreater(
            cos, 0.5,
            f"exit vs P_tgt gradient cosine should be positive; got {cos:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
