# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Built-in DTED (Draft-Tree-Expected-Decoding) registration and providers.

DTED trains the DFlash draft with a tree-expected acceptance objective
(docs/21 §5, docs/22). Wire-format-wise it is identical to DSpark
(needs ``target_last_hidden_states`` transmitted from SGLang alongside
``hidden_states``), but architecturally it uses the plain
``DFlashDraftModel`` so warm-start from any DFlash checkpoint works.

Reused verbatim from other algorithms:
  * ``capture_method="dflash"`` (SGLang-side capture is unchanged)
  * ``draft architecture = DFlashDraftModel``
  * DSpark's ``build_dspark_collator`` / offline reader / normalizer
    (identical data layout: input_ids, loss_mask, hidden_states,
    target_last_hidden_states)

DTED-specific:
  * ``build_training_model`` -> ``build_dted_model`` (constructs
    ``OnlineDTEDModel`` with tree_budget / alpha / eps / weight_type).
  * ``resume_contract`` persists ``dted_*`` fields.
"""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.hidden_states_data import (
    DSPARK_NORMALIZER_ID,
    build_dspark_collator,
    build_dspark_offline_normalizer,
    build_dspark_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    TargetDerivedDraftDefaults,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens

ALGORITHM_NAME = "dted"
DRAFT_ARCHITECTURE = "DFlashDraftModel"  # reuse DFlash weights + arch


def build_step(wrapped_model, *, target_head=None, **_options):
    """Build the DTED step function.

    Uses a dedicated ``DTEDTrainStrategy`` (not ``DFlashTrainStrategy``)
    because DTED requires ``target_last_hidden_states`` in the batch --
    the DFlash strategy does not forward that tensor to the model.
    """
    del target_head
    from specforge.training.strategies.base import DTEDTrainStrategy

    return DTEDTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved DTED architecture, sampling, and objective knobs.

    Prevents cross-strategy checkpoint mixup: if a DTED-produced ckpt
    is loaded with strategy=dflash the resume validator will refuse to
    restore because ``dted_*`` keys are absent from the target contract
    (and vice versa).
    """
    return {
        "dted_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "dted_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "dted_block_size": int(training_model.block_size),
        "dted_mask_token_id": int(training_model.mask_token_id),
        "dted_attention_backend": str(training_model.attention_backend),
        "dted_num_anchors": int(training_model.num_anchors),
        "dted_tree_budget": int(training_model.tree_budget),
        "dted_alpha": float(training_model.dted_alpha),
        "dted_eps": float(training_model.dted_eps),
        "dted_weight_type": str(training_model.weight_type),
        "dted_ce_weight": float(training_model.ce_weight),
        "dted_reinforce_weight": float(training_model.dted_weight),
    }


def build_draft(config, draft_config):
    """Reuse the DFlash draft builder (same architecture, same warm-start)."""
    from specforge.algorithms.model_providers import build_dflash_draft

    # DTED does not use Liger kernels (they only impact loss, and DTED
    # rewrites the loss entirely). Pass kernels=None.
    return build_dflash_draft(config, draft_config, kernels=None)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_dted_model

    return build_dted_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def populate_target_defaults(payload, target_config, config):
    from specforge.algorithms.model_providers import populate_dflash_generated_config

    return populate_dflash_generated_config(payload, target_config, config)


def apply_draft_overrides(config, draft_config):
    from specforge.algorithms.model_providers import apply_dflash_overrides

    return apply_dflash_overrides(config, draft_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    # DTED requires the same features as DSpark plus target_last_hidden_states.
    ready = {
        "input_ids",
        "loss_mask",
        "hidden_states",
        "target_last_hidden_states",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"num_hidden_layers", "block_size"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=DSPARK_NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                expected_auto_map_model="dflash.DFlashDraftModel",
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="qwen3",
                    num_hidden_layers=1,
                    populate=populate_target_defaults,
                ),
                apply_overrides=apply_draft_overrides,
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=DSPARK_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    # SGLang / offline capture uses the DFlash aux-layer
                    # pipeline (target_layer_ids [1,9,17,25,33] for Qwen3-4B).
                    capture_method="dflash",
                    aux_feature="hidden_states",
                    # But we ALSO ship the target last hidden so the trainer
                    # can compute p_tgt without a separate target-model forward.
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_dspark_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_dspark_offline_normalizer,
                build_collator=build_dspark_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_dspark_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
