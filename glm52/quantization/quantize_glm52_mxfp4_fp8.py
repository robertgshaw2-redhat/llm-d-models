#!/usr/bin/env python3
"""Quantize zai-org/GLM-5.2 with MXFP4 routed experts and FP8 everywhere else.

The routed experts are ~97% of GLM-5.2's 744B parameters, so they are the only
part worth taking to 4 bits. Everything else -- MLA projections, the DSA indexer
projections, the shared expert, the leading dense MLPs -- stays at FP8 block
(DeepSeek-V3 style, 128x128 weight blocks + dynamic per-128-group activations).

Expert activation precision is selectable, because that is the part the serving
side disagrees about (see README.md):

  --expert-acts mxfp8   MXFP4 weights x MXFP8 activations. This is the format
                        you want on Blackwell, but vLLM's compressed-tensors
                        MoE path ignores the declared activation spec today.
  --expert-acts mxfp4   MXFP4 weights x MXFP4 activations (the stock
                        compressed-tensors MXFP4 preset). Runs today.
  --expert-acts none    MXFP4A16, weight-only.

Every scheme used here is round-to-nearest or dynamic, so no calibration data is
required; pass --calibration-samples > 0 only if you plan to layer GPTQ/AWQ on
top of this recipe.

Launch on a multi-GPU host, e.g.:

    torchrun --nproc-per-node 8 quantize_glm52_mxfp4_fp8.py --save-dir GLM-5.2-MXFP4-FP8
"""

import argparse

import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4, MXFP8
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# llm-compressor linearizes GLM-5.2's fused 3D expert parameters into per-expert
# Linears (gate_proj/up_proj/down_proj) for calibration, which is also the naming
# vLLM matches config_groups against when it rebuilds the fused MoE layer. So the
# targets below are written against `...mlp.experts.<i>.<proj>`.
EXPERT_TARGETS = [
    r"re:.*\.mlp\.experts\.\d+\.(gate|up|down)_proj$",
]

# Everything else that is a real Linear: MLA, the DSA indexer's wq_b/wk, the
# shared expert, and the leading dense MLPs.
FP8_TARGETS = [
    r"re:.*\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
    r"re:.*\.self_attn\.indexer\.(wq_b|wk)$",
    r"re:.*\.mlp\.shared_experts\.(gate|up|down)_proj$",
    r"re:.*\.mlp\.(gate|up|down)_proj$",  # dense layers 0-2, if not ignored
]

IGNORE = [
    # MoE router: 256-way top-8 routing decisions do not survive quantization.
    r"re:.*\.mlp\.gate$",
    # DSA indexer head-weight projection; transformers keeps this one in fp32
    # (_keep_in_fp32_modules) and its 32-row output is not block-quantizable.
    r"re:.*\.indexer\.weights_proj$",
    "lm_head",
]

# The three leading dense layers (config.first_k_dense_replace == 3) are the
# most quantization-sensitive part of the model and are only ~1.4 GB in bf16.
LEADING_DENSE_IGNORE = r"re:^model\.layers\.[0-2]\..*"


def build_expert_scheme(expert_acts: str) -> QuantizationScheme:
    """MXFP4 expert weights, with the requested activation precision.

    MXFP4 and MXFP8 share the OCP microscaling layout -- group_size 32 along the
    input dim with E8M0 (uint8) scales -- so pairing MXFP4 weights with MXFP8
    activations is just a matter of taking one half of each preset. There is no
    MXFP4A8 preset in compressed-tensors, but QuantizationScheme does not
    require the weight and activation bit widths to match.
    """
    weights = MXFP4["weights"].model_copy(deep=True)

    if expert_acts == "mxfp8":
        input_activations = MXFP8["input_activations"].model_copy(deep=True)
    elif expert_acts == "mxfp4":
        input_activations = MXFP4["input_activations"].model_copy(deep=True)
    elif expert_acts == "none":
        input_activations = None
    else:
        raise ValueError(f"unknown --expert-acts value: {expert_acts}")

    return QuantizationScheme(
        targets=EXPERT_TARGETS,
        weights=weights,
        input_activations=input_activations,
    )


def build_recipe(expert_acts: str, quantize_leading_dense: bool) -> QuantizationModifier:
    ignore = list(IGNORE)
    if not quantize_leading_dense:
        ignore.insert(0, LEADING_DENSE_IGNORE)

    return QuantizationModifier(
        config_groups={
            "experts_mxfp4": build_expert_scheme(expert_acts),
            "dense_fp8_block": QuantizationScheme(targets=FP8_TARGETS, **FP8_BLOCK),
        },
        ignore=ignore,
    )


def build_calibration_dataset(tokenizer, num_samples: int, max_seq_len: int):
    from datasets import load_dataset

    from llmcompressor.datasets.utils import get_rank_partition

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=get_rank_partition("train_sft", num_samples),
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(preprocess)
    return ds.map(tokenize, remove_columns=ds.column_names)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2")
    parser.add_argument("--save-dir", default="GLM-5.2-MXFP4-FP8")
    parser.add_argument(
        "--expert-acts",
        default="mxfp8",
        choices=("mxfp8", "mxfp4", "none"),
        help="activation precision recorded for the routed experts",
    )
    parser.add_argument(
        "--quantize-leading-dense",
        action="store_true",
        help="also FP8 the first 3 (dense) decoder layers instead of leaving them bf16",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=0,
        help="0 (default) runs data-free RTN; >0 pulls ultrachat samples",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--cpu-memory",
        default="500GiB",
        help="CPU offload budget passed to from_pretrained",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    init_dist()

    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto_offload",
            max_memory={"cpu": args.cpu_memory},
            offload_folder="offload_folder",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    recipe = build_recipe(args.expert_acts, args.quantize_leading_dense)

    oneshot_kwargs = {}
    if args.calibration_samples > 0:
        oneshot_kwargs["dataset"] = build_calibration_dataset(
            tokenizer, args.calibration_samples, args.max_seq_len
        )
        oneshot_kwargs["batch_size"] = args.batch_size
        oneshot_kwargs["shuffle_calibration_samples"] = False

    oneshot(model=model, recipe=recipe, **oneshot_kwargs)

    model.save_pretrained(args.save_dir, save_compressed=True)
    tokenizer.save_pretrained(args.save_dir)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
