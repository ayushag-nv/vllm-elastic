# Nemotron Elastic serving

Serve the small and full widths of a Nemotron Elastic checkpoint from one weight
allocation. Each request chooses `small`, `big`, or `auto`. Requests with different
widths can be active together; the scheduler alternates between batches of each
width. `auto` starts small and switches to big after generating the checkpoint's
`elastic_think_end_token_id` (`</think>`). A marker in the prompt does not switch it.

This is an experimental single-GPU implementation for BF16 Nemotron-H checkpoints
trained for width slicing. It does not turn an ordinary Nemotron checkpoint into
an Elastic model.

## Start the server

Install this branch using vLLM's [source installation instructions](../getting_started/installation/gpu.md).
Use an Elastic checkpoint whose `config.json` contains:

```json
{
  "elastic_use_lazy_slicing": true,
  "elastic_thinking": {
    "elastic_hidden_size": 1920,
    "elastic_moe_intermediate_size": 960
  },
  "elastic_think_end_token_id": 13
}
```

These dimensions are for the 30B-A3B checkpoint with full hidden width 2688 and
expert intermediate width 1856. Use the dimensions and marker from your checkpoint.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/elastic-checkpoint \
  --served-model-name nemotron-elastic \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.88 \
  --no-async-scheduling --moe-backend triton \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --enable-prefix-caching --enable-prompt-tokens-details
```

The BF16 weights alone occupy about 59 GiB. Leave space for the cache and runtime
buffers. For debugging, replace `--compilation-config` with `--enforce-eager`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-elastic",
    "messages": [{"role":"user","content":"What is 2 plus 3?"}],
    "elastic_phase": "auto",
    "temperature": 0,
    "max_tokens": 256
  }'
```

`elastic_phase` also works on `/v1/completions`. For the Python `LLM` API, use
`SamplingParams(extra_args={"elastic_phase": "small"})`. Omitting the setting means
`auto`. Invalid values, or using the setting on a non-Elastic model, are rejected.
Beam search and resumable input sessions are not supported. Streaming responses
are supported.

## What is shared

The small model reads narrower views of the full weights. Attention head counts
and Mamba state dimensions stay fixed, so a request can change width without
reallocating its cache. Decode CUDA graphs are captured separately for each width.
Prefill runs eagerly.

Prefix caching uses one namespace for both widths. A big request can therefore
reuse state created by a small request, and vice versa. This is deliberate mixed-
width inference: the reused state is **not numerically equivalent** to recomputing
that prefix entirely at the destination width. Validate answer quality on your
workload. Use `--no-enable-prefix-caching` to disable reuse between requests; the
state within an `auto` request is still shared across its width change.
After preemption, recomputation follows the request's original width transition:
the prefix before the generated marker runs small, and the remainder runs big.

The implementation currently requires Model Runner V2, synchronous scheduling,
Triton MoE, and one GPU. Tensor/pipeline/data parallelism, quantization, LoRA,
speculative decoding and disaggregated serving are outside its supported scope.
A full-weight checkpoint must fit even when every request chooses the small width.

## Verify the implementation

Run the CPU tests with `pytest tests/v1/core/test_elastic.py`. The small GPU test
creates a tiny hybrid checkpoint and compares logits with separately materialized
Transformers models. It covers both widths, the transition, prompt embeddings
and recovery after preemption, in eager mode and with decode graphs:

```bash
pytest tests/models/language/generation/test_nemotron_elastic.py -k logits_and_recovery -v
```

The serving smoke test needs the real checkpoint and a dedicated GPU with at
least 80 GiB:

```bash
ELASTIC_TEST_MODEL=/path/to/elastic-checkpoint \
VLLM_GPU_SYNC_CHECK=error \
pytest tests/models/language/generation/test_nemotron_elastic.py -k switching_and_shared_cache -v
```

It checks the generated-marker transition, fixed modes, concurrent requests and
cache reuse in eager mode and with decode graphs. The tiny-checkpoint test runs
in hybrid-model CI; the real-checkpoint smoke test is manual. These checks do not
establish task-level quality or a throughput improvement.
