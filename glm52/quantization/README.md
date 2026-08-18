# GLM-5.2: MXFP4 routed experts + FP8 everything else

`quantize_glm52_mxfp4_fp8.py` produces a compressed-tensors checkpoint where the
256 routed experts are MXFP4 and the rest of the model is FP8-block.

## Why this split

GLM-5.2 (`GlmMoeDsaForCausalLM`) is 78 layers, `first_k_dense_replace: 3`,
256 routed experts + 1 shared expert, `hidden_size` 6144, `moe_intermediate_size`
2048. That puts ~725B of the ~744B parameters in the routed experts:

| component | params | bf16 | FP8 | MXFP4 |
|---|---|---|---|---|
| routed experts (75 sparse layers x 256 x 3 x 2048 x 6144) | ~725B | ~1450 GB | ~725 GB | ~380 GB |
| everything else (MLA, DSA indexer, shared expert, dense MLPs, embeddings) | ~19B | ~38 GB | ~19 GB | — |

So 4-bit experts is the entire memory story, and FP8 vs bf16 for the remainder is
worth ~19 GB. Nothing outside the experts is worth risking accuracy on.

## Layer-by-layer coverage

| module | scheme |
|---|---|
| `...mlp.experts.<i>.{gate,up,down}_proj` | MXFP4 weights (group 32, E8M0 scales) |
| `...self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj}` | FP8_BLOCK |
| `...self_attn.indexer.{wq_b,wk}` | FP8_BLOCK |
| `...mlp.shared_experts.{gate,up,down}_proj` | FP8_BLOCK |
| `model.layers.[0-2].*` (leading dense layers) | left bf16 (`--quantize-leading-dense` to FP8 them) |
| `...mlp.gate` (256-way router) | bf16 |
| `...self_attn.indexer.weights_proj` | bf16 — transformers keeps it in fp32 and its 32-row output is not block-quantizable |
| `lm_head` | bf16 |

`kv_a_proj_with_mqa` is 576 rows (`kv_lora_rank` 512 + `qk_rope_head_dim` 64),
which is not a multiple of the 128x128 FP8 block; that ragged tail is the same
one DeepSeek-V3 FP8 checkpoints carry, and both compressed-tensors and vLLM's
block-FP8 kernels handle it.

Every scheme here is RTN or dynamic, so the script is data-free by default — a
real advantage at 744B. `--calibration-samples 512` is there if you later layer
GPTQ/AWQ on the expert group.

## The MXFP4 x MXFP8 question

There is no `MXFP4A8` preset in compressed-tensors, but you do not need one:
MXFP4 and MXFP8 are the same OCP microscaling layout (group_size 32 along the
input dim, E8M0/uint8 scales), and `QuantizationScheme` does not require the
weight and activation bit widths to match. `--expert-acts mxfp8` just takes
`MXFP4["weights"]` and `MXFP8["input_activations"]`. The checkpoint writes
cleanly.

**The gap is on the serving side.** In vLLM's compressed-tensors MoE path
(`compressed_tensors_moe/compressed_tensors_moe.py`), dispatch keys off the
*weight* args only:

```python
if quant_config._is_mxfp4(weight_quant):
    return CompressedTensorsW4A4Mxfp4MoEMethod(layer.moe_config)
```

and that method never reads `input_activations`. It picks its kernel purely from
the device:

```python
self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()
```

`CutlassExpertsMxfp4` is MXFP4 x MXFP4 (`_supports_quant_scheme` returns
`(kMxfp4Static, kMxfp4Dynamic)`); the fallback is Marlin W4A16. Two consequences:

- An MXFP4 x MXFP8 checkpoint loads, but the experts run **W4A4** on Blackwell.
  The declared MXFP8 activation spec is ignored, not honored and not rejected.
- The same is true in reverse: an `MXFP4A16` weight-only checkpoint also runs
  W4A4 on SM100, because nothing in that path consults the checkpoint's
  activation spec.

The kernels you actually want do exist — `Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8`
and `FLASHINFER_CUTLASS_MXFP4_MXFP8`, plus `mxfp4_mxfp8_moe_quant_config()` — but
they are only reachable through the standalone `mxfp4` quant method
(`quantization/mxfp4.py`), which was built for gpt-oss and DeepSeek-V4-style
checkpoints. That path leaves every `LinearBase` unquantized (bf16), so it cannot
give you FP8 attention in the same checkpoint. vLLM issue #35528, asking for the
equivalent for ModelOpt's `W4A8_MXFP4_FP8_CFG`, was closed as not planned.

### Wiring it up

The plumbing gap is small, because the two paths already agree on weight layout:
both `Mxfp4MoEMethod.create_weights` and `CompressedTensorsW4A4Mxfp4MoEMethod.create_weights`
allocate `[E, N, K//2]` uint8 packed weights and `[E, N, K//32]` uint8 E8M0
scales; only the parameter names differ (`w13_weight` vs `w13_weight_packed`).
A patch would:

1. Pass `input_quant` through `CompressedTensorsMoEMethod.get_moe_method` into
   `CompressedTensorsW4A4Mxfp4MoEMethod.__init__` (the dispatcher already has it
   in scope as `scheme_dict["input_activations"]`).
2. When the activation spec is 8-bit float / group 32, select the backend with
   `select_deepseek_v4_mxfp4_moe_backend(moe)` from
   `fused_moe/oracle/mxfp4.py` instead of hardcoding Marlin/CUTLASS.
3. In `process_weights_after_loading`, after the existing
   `w13_weight_packed -> w13_weight` rename, hand off to
   `Mxfp4MoEMethod._setup_kernel(...)`, which already does the FlashInfer scale
   swizzling and shape round-up, and return `mxfp4_mxfp8_moe_quant_config(...)`
   from `get_fused_moe_quant_config`.

Until that lands, pick from the table below.

## What to actually deploy

| goal | how | status |
|---|---|---|
| MXFP4 experts x MXFP8 acts, FP8 rest | `--expert-acts mxfp8` + the vLLM patch above | needs the patch |
| MXFP4 experts, FP8 rest, today | `--expert-acts mxfp4` | works on B200 (CUTLASS W4A4); MXFP4 activations are the accuracy risk |
| 4-bit experts + FP8 rest, today, best accuracy | NVFP4 experts instead of MXFP4 — swap `MXFP4` for the `NVFP4` preset in `build_expert_scheme`, per the upstream [GLM-5.2 NVFP4+FP8 example](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/glm-5.2/nvfp4-fp8-example/) | fully supported; `nvidia/GLM-5.2-NVFP4` already ships this shape. NVFP4's FP8 per-16 scales generally beat MXFP4's E8M0 per-32 at W4A4 |

MXFP4 (any variant) needs SM100 — B200, not the H200 pool.

## Caveats

- **MTP is dropped.** The checkpoint's layer 78 is the MTP/nextn layer
  (`eh_proj`, `enorm`, `hnorm`, `shared_head.norm`). `transformers`'
  `glm_moe_dsa` implements 78 hidden layers and no MTP module, so those 791
  tensors are not loaded and will not be written back out. If you want MTP
  speculative decoding in vLLM, copy those tensors into the output directory
  separately (bf16 is fine — they are ~1% of the model) and re-add them to the
  safetensors index.
- The MXFP4 support in llm-compressor is still labelled experimental upstream.
- Group size 32 divides every quantized expert dim cleanly here (6144 and 2048),
  so no padding is involved.

## Run

```bash
pip install llmcompressor
torchrun --nproc-per-node 8 quantize_glm52_mxfp4_fp8.py \
  --expert-acts mxfp8 \
  --save-dir GLM-5.2-MXFP4-MXFP8-FP8
```

Then serve as usual — vLLM reads the scheme out of `quantization_config`, no
`--quantization` flag needed:

```bash
vllm serve GLM-5.2-MXFP4-MXFP8-FP8 --kv-cache-dtype fp8
```
