"""DTED (Draft-Tree-Expected-Decoding) algorithm registration.

Only ``create_registration`` is re-exported here so that
``builtin_algorithm_registry`` can build the immutable catalog without
importing torch or the training stack (see
``tests.test_algorithms.test_builtin_providers.
test_building_catalog_does_not_import_training_or_torch``).

Runtime consumers (loss unit tests, training model construction) should
import from the concrete submodules directly, e.g.::

    from specforge.algorithms.dted.ddtree_build import build_ddtree_tree
    from specforge.algorithms.dted.loss import dted_loss_per_anchor
    from specforge.algorithms.dted.model import OnlineDTEDModel
"""

from specforge.algorithms.dted.providers import create_registration

__all__ = ["create_registration"]
