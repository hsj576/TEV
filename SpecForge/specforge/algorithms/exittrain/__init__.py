"""SpecForge built-in ExitTrain algorithm (a.k.a. DTED full-tree-verify).

Same tree-expected-decoding objective as :mod:`specforge.algorithms.dted`,
but with a full-tree-verify upgrade: the target model performs a
**full-tree attention forward** over each anchor's DDTree, producing
the true per-node conditional distribution
``p_target(x | prefix, path_to_v)`` that scores each node of the tree.

This removes the Markov approximation used by the base ``dted`` recipe
(which reuses the teacher-sequence hidden for every tree node
regardless of branch) at the cost of one extra target-model forward
per training step.
"""

from __future__ import annotations

from specforge.algorithms.exittrain.providers import create_registration

__all__ = ["create_registration"]
