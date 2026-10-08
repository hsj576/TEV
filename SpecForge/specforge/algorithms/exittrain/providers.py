# coding=utf-8
# Copyright 2026 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Built-in ExitTrain (DTED full-tree-verify) registration and providers.

This algorithm is a **variant** of ``dted``: same wire-format
(``target_last_hidden_states`` is still shipped, though ignored at
loss time), same warm-start (any DFlash checkpoint), same offline /
streaming data pipelines. It differs from the base ``dted`` in two
ways:

  * ``build_training_model`` returns an
    :class:`OnlineExitTrainModel` that runs a target-tree-attention
    forward on every step.
  * ``resume_contract`` persists an extra
    ``exittrain_prefix_window`` field.

Configuration is driven through the new ``exittrain_*`` fields in
:mod:`specforge.config.schema`.
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


ALGORITHM_NAME = "exittrain"
DRAFT_ARCHITECTURE = "DFlashDraftModel"


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    # Reuse DTEDTrainStrategy: it forwards ``target_last_hidden_states``
    # (which we ignore) but otherwise behaves identically.
    from specforge.training.strategies.base import DTEDTrainStrategy

    return DTEDTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    return {
        "exittrain_draft_num_hidden_layers": int(
            draft_model.config.num_hidden_layers
        ),
        "exittrain_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "exittrain_block_size": int(training_model.block_size),
        "exittrain_mask_token_id": int(training_model.mask_token_id),
        "exittrain_attention_backend": str(training_model.attention_backend),
        "exittrain_num_anchors": int(training_model.num_anchors),
        "exittrain_tree_budget": int(training_model.tree_budget),
        "exittrain_alpha": float(training_model.dted_alpha),
        "exittrain_eps": float(training_model.dted_eps),
        "exittrain_weight_type": str(training_model.weight_type),
        "exittrain_ce_weight": float(training_model.ce_weight),
        "exittrain_reinforce_weight": float(training_model.dted_weight),
        "exittrain_prefix_window": int(training_model.prefix_window),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_dflash_draft

    return build_dflash_draft(config, draft_config, kernels=None)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_exittrain_model

    return build_exittrain_model(
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
                    capture_method="dflash",
                    aux_feature="hidden_states",
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
