"""Gradio Web UI for interactive comparison between DDTree+TEV and AR baseline.

Loads the target model and DFlash draft model once at startup, then serves a
chat interface where the user can toggle "Use DDTree+TEV" on/off to compare
speculative decoding against the vanilla auto-regressive baseline on the same
prompt. Live tokens are streamed as they are decoded; tokens produced by the
speculative round are highlighted in orange when the toggle is on.

Usage:
    python -m application.webui \
        --target Qwen/Qwen3-4B \
        --draft  /path/to/Qwen3-4B-DFlash \
        --cuda-visible-devices 0 \
        --share

The script must be run from the repository root (``TEV-ICLR/ddtree``) so that
``ddtree``, ``dflash`` and ``model`` are importable, or launched via
``python -m application.webui``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI: parse --cuda-visible-devices BEFORE importing torch.
# ---------------------------------------------------------------------------
_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument("--cuda-visible-devices", type=str, default=None,
                        help="Comma-separated GPU indices to expose (sets CUDA_VISIBLE_DEVICES).")
_pre_args, _ = _PRE_PARSER.parse_known_args()
if _pre_args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.cuda_visible_devices

# Make the repository root importable no matter how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr
import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Compatibility shim: some (gradio_client, pydantic) version combos crash in
# ``gradio_client.utils`` when a JSON schema contains a boolean literal (e.g.
# ``additionalProperties: True``). The upstream fix ships in newer gradio;
# here we monkey-patch both entry points defensively so the demo also runs on
# older gradio 4.x builds already in the environment.
# ---------------------------------------------------------------------------
try:
    from gradio_client import utils as _gc_utils
    _orig_get_type = _gc_utils.get_type
    _orig_schema_to_py = _gc_utils._json_schema_to_python_type

    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    def _safe_schema_to_py(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        try:
            return _orig_schema_to_py(schema, defs)
        except Exception:
            return "Any"

    _gc_utils.get_type = _safe_get_type
    _gc_utils._json_schema_to_python_type = _safe_schema_to_py
except Exception:
    pass

from model import DFlashDraftModel, load_target
from dflash import dflash_generate_stream
from ddtree import ddtree_generate_stream, maybe_enable_cpp_compact


# ---------------------------------------------------------------------------
# Argparse (full).
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive DDTree+TEV vs AR baseline chat demo.",
        parents=[_PRE_PARSER],
    )
    parser.add_argument("--target", type=str, required=True,
                        help="Target model path or HuggingFace repo id.")
    parser.add_argument("--draft", type=str, required=True,
                        help="DFlash draft model path or HuggingFace repo id.")
    parser.add_argument("--tree-budget", type=int, default=64,
                        help="Default DDTree budget (adjustable in the UI).")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum new tokens generated per response.")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true",
                        help="Disable the inline C++ tail cache-compaction extension.")
    parser.add_argument("--share", action="store_true",
                        help="Enable gradio public sharing link.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------
def load_models(args: argparse.Namespace):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    target = load_target(
        args.target,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        device=device,
    )
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    return target, draft_model, tokenizer, device


def warmup(target, draft_model, tokenizer, device, max_new_tokens: int, tree_budget: int) -> None:
    """One short generation each to trigger CUDA graph / kernel compilation."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    warm_tokens = min(max_new_tokens, 8)
    stop_ids = [tokenizer.eos_token_id]

    for _ in dflash_generate_stream(
        model=draft_model, target=target, input_ids=input_ids,
        mask_token_id=draft_model.mask_token_id, max_new_tokens=warm_tokens,
        block_size=1, stop_token_ids=stop_ids, temperature=0.0,
    ):
        pass
    for _ in ddtree_generate_stream(
        model=draft_model, target=target, input_ids=input_ids,
        mask_token_id=draft_model.mask_token_id, max_new_tokens=warm_tokens,
        block_size=draft_model.block_size, tree_budget=tree_budget,
        stop_token_ids=stop_ids, temperature=0.0,
    ):
        pass


# ---------------------------------------------------------------------------
# Highlighting helpers (adapted from EAGLE's webui).
# ---------------------------------------------------------------------------
_LIST_MARKER_RE = re.compile(r"(?m)(^\d+\.\s|\n)")


def _find_list_markers(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _LIST_MARKER_RE.finditer(text)]


def _inside_marker(pointer: int, start: int, markers: list[tuple[int, int]]) -> bool:
    for b, e in markers:
        if b <= pointer < e or b <= start < e:
            return True
    return False


def highlight_text(text: str, chunk_texts: list[str], color: str = "orange") -> str:
    """Wrap each ``chunk_text`` (in order of appearance) with a coloured span.

    Consecutive list-marker regions (e.g. "\\n" or "1. ") are excluded from
    highlighting so ordered/unordered list rendering is not corrupted.
    """
    pointer = 0
    result = ""
    markers = _find_list_markers(text)
    for sub in chunk_texts:
        if not sub:
            continue
        start = text.find(sub, pointer)
        if start == -1:
            continue
        end = start + len(sub)
        gap = text[pointer:start]
        if _inside_marker(pointer, start, markers):
            result += gap
        else:
            result += f"<span style='color: {color};'>{gap}</span>" if gap else ""
        result += f"<span style='color: {color};'>{sub}</span>"
        pointer = end
    if pointer < len(text):
        result += text[pointer:]
    return result


def _truncate_at(ids: list[int], stop_ids: list[int]) -> list[int]:
    idx = len(ids)
    for sid in stop_ids:
        if sid in ids:
            idx = min(idx, ids.index(sid) + 1)
    return ids[:idx]


# ---------------------------------------------------------------------------
# Build prompt from chat history + new user message using the tokenizer's
# chat template (falls back to a naive concat if the tokenizer has none).
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, pure_history: list[list[str | None]]) -> str:
    messages = []
    for user_msg, bot_msg in pure_history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg is not None:
            messages.append({"role": "assistant", "content": bot_msg})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        pieces = [f"User: {m['content']}\n" if m["role"] == "user" else f"Assistant: {m['content']}\n" for m in messages]
        pieces.append("Assistant: ")
        return "".join(pieces)


# ---------------------------------------------------------------------------
# Main bot generator (streams to gradio).
# ---------------------------------------------------------------------------
def make_bot_fn(target, draft_model, tokenizer, device, args: argparse.Namespace):
    stop_ids = [tokenizer.eos_token_id]
    eot_token = None
    try:
        # Llama-3 style chat templates use <|eot_id|> as the assistant end marker.
        _eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(_eot, int) and _eot != tokenizer.unk_token_id:
            stop_ids.append(_eot)
            eot_token = _eot
    except Exception:
        eot_token = None

    def _decode(ids: list[int]) -> str:
        return tokenizer.decode(
            ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def bot(history, temperature, use_tev, highlight_tev, tree_budget, max_new_tokens, session_state):
        if not history:
            yield history, "0.00 tokens/s", "0.00", session_state
            return

        pure_history = session_state.get("pure_history", [])
        prompt = build_prompt(tokenizer, pure_history)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_len = int(input_ids.shape[1])

        # Choose generator based on the toggle.
        gen_kwargs = dict(
            model=draft_model, target=target, input_ids=input_ids,
            mask_token_id=draft_model.mask_token_id,
            max_new_tokens=int(max_new_tokens),
            stop_token_ids=stop_ids,
            temperature=float(temperature),
        )
        if use_tev:
            gen = ddtree_generate_stream(
                block_size=draft_model.block_size,
                tree_budget=int(tree_budget),
                **gen_kwargs,
            )
        else:
            gen = dflash_generate_stream(block_size=1, **gen_kwargs)

        chunk_texts: list[str] = []
        prev_new_tokens = 0
        total_time = 0.0
        rounds = 0
        text = ""
        start_time = time.time()

        for step in gen:
            elapsed = time.time() - start_time
            total_time += elapsed
            rounds = int(step["decode_rounds"])
            output_ids = step["output_ids"]  # (1, num_input + new)
            new_ids = output_ids[0, input_len:].tolist()
            new_ids = _truncate_at(new_ids, stop_ids)
            text = _decode(new_ids)

            # Chunk boundaries: tokens produced in *this* streaming step
            # (i.e. one speculative round for TEV, one AR step otherwise).
            curr_new_tokens = len(new_ids)
            if curr_new_tokens > prev_new_tokens:
                chunk_ids = output_ids[0, input_len + prev_new_tokens : input_len + curr_new_tokens].tolist()
                chunk_text = _decode(chunk_ids)
                chunk_texts.append(chunk_text)
                prev_new_tokens = curr_new_tokens

            if highlight_tev and use_tev:
                display_text = highlight_text(text, chunk_texts, color="orange")
            else:
                display_text = text

            history[-1][1] = display_text
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history

            speed = curr_new_tokens / total_time if total_time > 0 else 0.0
            avg_accept = (curr_new_tokens / rounds) if rounds > 0 else 0.0
            yield history, f"{speed:.2f} tokens/s", f"{avg_accept:.2f}", session_state
            start_time = time.time()

        # Reached generator end (EOS or max_new_tokens). Nothing else to yield.

    return bot


# ---------------------------------------------------------------------------
# Gradio glue.
# ---------------------------------------------------------------------------
def user(user_message, history, session_state):
    if history is None:
        history = []
    pure_history = session_state.get("pure_history", [])
    pure_history.append([user_message, None])
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]], session_state


def regenerate(history, session_state):
    if not history:
        return history, None, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])
    if pure_history:
        pure_history[-1][-1] = None
    session_state["pure_history"] = pure_history
    if len(history) >= 1:
        last_user_message = history[-1][0]
        new_history = history[:-1] + [[last_user_message, None]]
        return new_history, None, "0.00 tokens/s", "0.00", session_state
    return history, None, "0.00 tokens/s", "0.00", session_state


def clear(_history, session_state):
    session_state["pure_history"] = []
    return [], "0.00 tokens/s", "0.00", session_state


# ---------------------------------------------------------------------------
# Main entry.
# ---------------------------------------------------------------------------
def build_demo(bot_fn, default_tree_budget: int, default_max_new_tokens: int):
    custom_css = """
    #speed textarea, #accept textarea {
        color: red;
        font-size: 28px;
    }
    """
    with gr.Blocks(css=custom_css) as demo:
        gs = gr.State({"pure_history": []})
        gr.Markdown("## DDTree + TEV — Speculative Decoding Live Demo")
        gr.Markdown(
            "Toggle **Use DDTree+TEV** to switch between our tree-based speculative "
            "decoding and the plain auto-regressive baseline on the same prompt. "
            "When TEV is on and highlighting is enabled, tokens produced by each "
            "speculative round are shown in orange."
        )

        with gr.Row():
            speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
            accept_box = gr.Textbox(label="Mean tokens / round", elem_id="accept", interactive=False, value="0.00")

        with gr.Row():
            with gr.Column():
                use_tev = gr.Checkbox(label="Use DDTree+TEV", value=True)
                highlight_tev = gr.Checkbox(label="Highlight tokens produced by TEV", value=True)
            temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, label="Temperature", value=0.0)
            tree_budget = gr.Slider(minimum=8, maximum=256, step=8, label="Tree budget (TEV)", value=default_tree_budget)
            max_new = gr.Slider(minimum=64, maximum=4096, step=32, label="Max new tokens", value=default_max_new_tokens)

        gr.Markdown(
            "*Mean tokens / round* = total new tokens produced ÷ number of decode rounds. "
            "For the AR baseline this is always 1.0; for TEV it corresponds to the mean "
            "acceptance length + bonus."
        )

        chatbot = gr.Chatbot(height=560, show_label=False, type="tuples")
        msg = gr.Textbox(label="Your input")
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            stop_button = gr.Button("Stop")
            regenerate_button = gr.Button("Regenerate")
            clear_button = gr.Button("Clear")

        controls = [chatbot, temperature, use_tev, highlight_tev, tree_budget, max_new, gs]
        outputs = [chatbot, speed_box, accept_box, gs]

        enter_event = msg.submit(user, [msg, chatbot, gs], [msg, chatbot, gs], queue=True).then(
            bot_fn, controls, outputs
        )
        send_event = send_button.click(user, [msg, chatbot, gs], [msg, chatbot, gs], queue=True).then(
            bot_fn, controls, outputs
        )
        regenerate_event = regenerate_button.click(
            regenerate, [chatbot, gs], [chatbot, msg, speed_box, accept_box, gs], queue=True
        ).then(bot_fn, controls, outputs)
        clear_button.click(clear, [chatbot, gs], [chatbot, speed_box, accept_box, gs], queue=True)
        stop_button.click(fn=None, inputs=None, outputs=None,
                          cancels=[send_event, regenerate_event, enter_event])
    return demo


def main() -> None:
    args = parse_args()
    print(f"[webui] loading target={args.target!r}  draft={args.draft!r}", flush=True)
    target, draft_model, tokenizer, device = load_models(args)
    print(f"[webui] warmup ... (this compiles the C++ compaction extension once)", flush=True)
    warmup(target, draft_model, tokenizer, device, args.max_new_tokens, args.tree_budget)
    print(f"[webui] warmup done. Launching gradio ...", flush=True)

    bot_fn = make_bot_fn(target, draft_model, tokenizer, device, args)
    demo = build_demo(bot_fn, args.tree_budget, args.max_new_tokens)
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
