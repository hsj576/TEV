"""End-to-end benchmark for DDTree with the TEV verifier.

Runs the target model with a DFlash draft model over a chosen dataset and
records per-round stage timings, acceptance lengths and TPOT. When
``--skip-baseline-dflash`` is not set, also runs the AR baseline (block_size=1)
and DFlash block-parallel speculative decoding as reference points.

The DDTree branch always uses the Tree Exit Verification (TEV) verifier
described in the paper (temperature > 0). At temperature == 0 the DDTree loop
uses argmax tree-follow, which is distributionally identical to any verifier
under argmax and strictly cheaper.
"""
import argparse
import random
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset, load_target
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True,
                        help="Target model (HuggingFace repo id or local path).")
    parser.add_argument("--draft-name-or-path", type=str, required=True,
                        help="DFlash draft model (HuggingFace repo id or local path).")
    parser.add_argument("--block-size", type=int, default=None,
                        help="DFlash draft block size. Defaults to draft.block_size.")
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024",
                        help="Comma-separated list of tree budgets to sweep.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="One of gsm8k / humaneval / math500 / mt-bench / mbpp / ...")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true",
                        help="Use flash_attention_2 for the target (disables DDTree tree mask).")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true",
                        help="Disable the inline C++ tail cache-compaction extension.")
    parser.add_argument("--skip-baseline-dflash", action="store_true",
                        help="Skip AR baseline (block_size=1) and DFlash generation; only run DDTree+TEV.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save the responses .pt file.")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the DFlash draft model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"

    if not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. Forcing the target verifier to torch.sdpa for compatibility.")

    target = load_target(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
        device=device,
    )

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = [] if args.skip_baseline_dflash else ["dflash"]
    method_key_to_tree_budget = {}
    if not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})
    if not methods_to_run:
        raise RuntimeError("No methods to run: with --skip-baseline-dflash and --flash-attn combined, DDTree variants are disabled. Drop --flash-attn to enable DDTree.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    if not args.skip_baseline_dflash:
        _ = dflash_generate(
            model=draft_model,
            target=target,
            input_ids=warmup_input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=warmup_max_new_tokens,
            block_size=1,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
        )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        else:
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            if not args.skip_baseline_dflash:
                response["baseline"] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=1,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )
            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                else:
                    response[method_key] = ddtree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
