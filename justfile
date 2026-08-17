# AIPerf AgentX-MVP benchmark against the running llm-d optimized-baseline deployment.
#
# Usage:
#   just deploy            # apply the manifest and wait for the pod
#   just check             # confirm the runner can reach the model endpoint
#   just run               # run the full AgentX-MVP benchmark (default 1800s)
#   just run 16 900        # override concurrency / duration
#   just smoke             # fast plumbing test (~60s, marks result invalid)
#   just results           # copy artifacts out to ./results
#   just logs / just shell # inspect the runner
#   just clean             # delete the runner
#   just lmeval            # run lm-eval (gsm8k) at 100 concurrency via the lm-eval pod
#   just lmeval mmlu 50    # override tasks / concurrency
#   just bench             # vllm bench serve, 10k ISL / 1 OSL, from inside the inkling pod
#   just bench 32 128      # override concurrency / prompt count
#
# The GuideLLM *Pride and Prejudice* recipes live at the bottom of this file:
#   just pride-dataset     # once per cluster: PVC + build the prompt JSONL
#   just pride-sweep       # sweep concurrency, one Job at a time
#
# Correctness rather than load, at the very bottom:
#   just bfcl              # BFCL tool-call eval (multi_turn) against the endpoint
#   just bfcl live_simple  # a cheaper category

# namespace / model / url can be overridden from the environment, e.g.
#   NAMESPACE=other-ns MODEL=Qwen/Qwen3-32B URL=http://qwen3-svc:8000 just run 16
namespace := env_var_or_default("NAMESPACE", "default")
deploy    := "aiperf-agentx"
# Must match what the vLLM pods actually serve (inkling-small/aggregated/base/) and the
# token-producer modelName in router.values.yaml.
model     := env_var_or_default("MODEL", "thinkingmachines/Inkling-NVFP4")
# MODEL=Qwen/Qwen3-4B-Thinking-2507
# MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
# MODEL=Qwen/Qwen3-32B
# inkling-small-svc from inkling-small/aggregated/base/ -- direct to the pods,
# bypassing EPP scheduling. Plain HTTP in-cluster, resolves from any pod in
# {{namespace}}. No trailing slash.
# url       := env_var_or_default("URL", "http://inkling-large-agg-svc:8000")
url := env_var_or_default("URL", "http://inkling-large-epp:80")
# URL=http://qwen3-svc:8000
# URL=http://inkling-small-epp
# Through the EPP/InferencePool instead, so the prefix-cache/token-load scorers
# apply. Swap this in to benchmark the routed path -- but the router chart isn't
# installed yet, so confirm the service name/port before trusting this
# (`helm install inkling-small ...` per router.values.yaml, which maps 80->8081).
# URL=http://inkling-small-epp:80
# duration    := "10"
duration    := "900"

# The vLLM serving Deployment itself (inkling-small/aggregated/base/), not the
# runner: `just bench` execs into this pod and drives its own localhost:8000.
# The model id has to be what that pod actually serves -- `vllm bench serve`
# sends it as the request body's "model" and loads its tokenizer by that name.
inkling_deploy := "inkling-small"
inkling_model  := "thinkingmachines/Inkling-Small"

default:
    @just --list

# Apply the manifest and wait for the runner pod to be ready.
deploy:
    kubectl apply -f benchmarks/agentx.yaml
    kubectl rollout status deploy/{{deploy}} -n {{namespace}} --timeout=300s
    just patch
    just install-lmeval

# Sanity check: list models served through the llm-d router from inside the runner.
# (The slim image has no curl, so use python's urllib.)
check:
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- \
      python -c "import urllib.request as u; print(u.urlopen('{{url}}/v1/models', timeout=10).read().decode())"

# Run the AgentX-MVP benchmark. Args: [concurrency] [duration-seconds].
run concurrency duration=duration:
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- \
      aiperf profile \
        --scenario inferencex-agentx-mvp \
        --url {{url}} \
        --model {{model}} \
        --max-context-length 256000 \
        --endpoint-type chat \
        --streaming \
        --use-server-token-count \
        --public-dataset semianalysis_cc_traces_weka_with_subagents \
        --concurrency {{concurrency}} \
        --benchmark-duration {{duration}} \
        --output-artifact-dir /workspace/artifacts \
        --ui simple

# Fast plumbing validation (~60s). Uses --unsafe-override so it runs below the
# scenario's 900s minimum; result is marked submission_valid: false.
smoke concurrency:
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- \
      aiperf profile \
        --scenario inferencex-agentx-mvp \
        --unsafe-override \
        --url {{url}} \
        --model {{model}} \
        --max-context-length 256000 \
        --endpoint-type chat \
        --streaming \
        --use-server-token-count \
        --public-dataset semianalysis_cc_traces_weka_with_subagents \
        --concurrency {{concurrency}} \
        --benchmark-duration {{duration}} \
        --output-artifact-dir /workspace/artifacts \
        --ui simple

# Sweep over a range of concurrency values using the smoke config (fast, marks
# results invalid via --unsafe-override). Args: [duration-seconds].
sweep:
    just smoke 16
    sleep 10
    just smoke 32
    sleep 10
    just smoke 64
    sleep 10
    just smoke 128
    sleep 10
    just smoke 256

# Copy benchmark artifacts out of the runner to a local directory (default ./results).
results dest="./results":
    mkdir -p {{dest}}
    kubectl cp {{namespace}}/$(kubectl get pod -n {{namespace}} -l app={{deploy}} -o jsonpath='{.items[0].metadata.name}'):/workspace/artifacts {{dest}}

# Run lm-evaluation-harness from the lm-eval pod against the deployment's
# OpenAI-compatible chat endpoint. Args: [tasks] [concurrency].
lmeval limit="1300" concurrency="100":
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- \
        lm_eval \
            --model local-completions \
            --model_args "model={{model}},base_url={{url}}/v1/completions,num_concurrent={{concurrency}},tokenized_requests=False" \
            --tasks gsm8k --limit {{limit}} --num_fewshot 20

install-lmeval:
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- pip install lm-eval[api] transformers

# vllm bench serve at a fixed 10k input / 1 output token, run from inside the
# inkling pod against its own localhost:8000 -- the vLLM image already ships the
# `vllm bench` CLI, and going through loopback keeps the Service and the EPP out
# of the numbers. 1 OSL makes this a pure prefill measurement: end-to-end
# latency is TTFT plus one decode step.
#
# --random-range-ratio 0 pins every prompt at exactly 10k tokens instead of
# sampling a range around it, and --ignore-eos keeps the 1-token output from
# being cut shorter. Prompts are random token ids, so they share no prefix with
# each other -- but a repeat run reuses the same seed, so bump --seed (or expect
# prefix-cache hits) when re-running back to back.

# 10k ISL / 1 OSL vllm bench serve. Args: [concurrency] [num-prompts].
bench concurrency="16" num_prompts="":
    #!/usr/bin/env bash
    set -euo pipefail
    num_prompts="{{num_prompts}}"
    if [[ -z "$num_prompts" ]]; then num_prompts=$(( {{concurrency}} * 10 )); fi
    echo "==> ISL=10000 OSL=1 CONCURRENCY={{concurrency}} NUM_PROMPTS=$num_prompts"
    kubectl exec -n {{namespace}} deploy/{{inkling_deploy}} -- \
      vllm bench serve \
        --backend vllm \
        --base-url http://localhost:8000 \
        --model {{inkling_model}} \
        --trust-remote-code \
        --dataset-name random \
        --random-input-len 10000 \
        --random-output-len 1 \
        --random-range-ratio 0 \
        --ignore-eos \
        --num-prompts "$num_prompts" \
        --max-concurrency {{concurrency}} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --seed 42

logs:
    kubectl logs -n {{namespace}} deploy/{{deploy}} -f

shell:
    kubectl exec -it -n {{namespace}} deploy/{{deploy}} -- bash

clean:
    kubectl delete -f benchmarks/agentx.yaml --ignore-not-found

patch:
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl exec -i -n {{namespace}} deploy/{{deploy}} -- python - <<'PYEOF'
    import pathlib
    p = pathlib.Path("/aiperf/src/aiperf/workers/session_manager.py")
    src = p.read_text()
    old = "                self.context_mode == ConversationContextMode.DELTAS_WITH_RESPONSES\n"
    new = (
        "                self.context_mode\n"
        "                in (\n"
        "                    ConversationContextMode.DELTAS_WITH_RESPONSES,\n"
        "                    ConversationContextMode.DELTAS_WITHOUT_RESPONSES,\n"
        "                )\n"
    )
    if new in src:
        print("session_manager.py: already patched")
    elif old in src:
        p.write_text(src.replace(old, new, 1))
        print("session_manager.py: patched")
    else:
        raise SystemExit("session_manager.py: no match -- aiperf source changed, re-check the patch")
    PYEOF

# ---------------------------------------------------------------------------
# GuideLLM *Pride and Prejudice* sweep (benchmarks/pride-*.yaml)
#
# A different shape of load from AgentX-MVP: fixed 15k-token passages of real
# prose decoding a fixed 900 tokens each, which is what makes a
# speculative-decoding acceptance rate meaningful (Faker word salad has nothing
# for a draft model to predict). Ported from ../wells/run-pride.sh.
#
#   just pride-dataset          # once per cluster: PVC + build the JSONL
#   just pride 8                # one concurrency point
#   just pride 8 256            # ... with an explicit request budget
#   just pride-sweep            # 1 5 10 20, strictly sequential
#   just pride-sweep 1 8 32     # explicit points
#   just pride-jobs             # what ran / what is running
#   just pride-clean            # delete the finished Jobs
#
# The manifests still carry the wells cluster's model and target as defaults;
# every recipe below overrides them from {{model}} / {{url}} at create time
# (`kubectl set env --local`) rather than editing the YAML, so the two clusters
# can share one copy of the manifests.
# ---------------------------------------------------------------------------

pride_log_dir := "./results-pride"
# Wall clock allowed for one point after its pod is running.
pride_timeout := "60m"
# Distinct passages written to the PVC. A run asking for more than this just
# replays the head of the dataset, so every point is capped here.
pride_samples := "256"

# Run this again after changing {{model}}: the windows are cut with the served
# model's tokenizer, and prompt lengths measured against any other are fiction.

# Create the dataset PVC and build the prompt JSONL on it. Once per cluster.
pride-dataset:
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl apply -n {{namespace}} -f benchmarks/dataset-pvc.yaml
    job=$(
      kubectl create -f benchmarks/pride-preprocess-job.yaml --dry-run=client -o yaml \
        | kubectl set env --local -f - -o yaml \
            "MODEL={{model}}" "SAMPLES={{pride_samples}}" \
        | kubectl create -n {{namespace}} -f - -o name
    )
    echo "==> created $job"
    # The pod does not exist for a moment, and downloading the book plus the
    # tokenizer can outlast the wait; the log follow below does the real waiting.
    kubectl wait -n {{namespace}} --for=condition=Ready pod \
      --selector="job-name=${job#job.batch/}" --timeout=10m || true
    kubectl logs -n {{namespace}} -f "$job"
    kubectl wait -n {{namespace}} --for=condition=complete --timeout=60m "$job"

# One concurrency point, teed to {{pride_log_dir}}. Args: [concurrency] [max-requests].
pride concurrency max_requests="":
    #!/usr/bin/env bash
    set -euo pipefail
    # ~10 requests per stream: enough that every stream gets several turns,
    # without making the concurrency-1 point decode 256 * 900 tokens serially.
    max_requests="{{max_requests}}"
    if [[ -z "$max_requests" ]]; then max_requests=$(( {{concurrency}} * 10 )); fi
    if (( max_requests > {{pride_samples}} )); then max_requests={{pride_samples}}; fi

    mkdir -p "{{pride_log_dir}}"
    log="{{pride_log_dir}}/pride-c{{concurrency}}.log"
    echo "==> CONCURRENCY={{concurrency}} MAX_REQUESTS=$max_requests -> $log"

    # generateName, so each create gets a fresh Job rather than colliding.
    job=$(
      kubectl create -f benchmarks/pride-benchmark-job.yaml --dry-run=client -o yaml \
        | kubectl set env --local -f - -o yaml \
            "CONCURRENCY={{concurrency}}" "MAX_REQUESTS=$max_requests" \
            "TARGET={{url}}" "MODEL={{model}}" \
        | kubectl create -n {{namespace}} -f - -o name
    )
    echo "==> created $job"

    kubectl wait -n {{namespace}} --for=condition=Ready pod \
      --selector="job-name=${job#job.batch/}" --timeout=10m || true

    # tee rather than redirect: these runs are long enough that watching matters.
    kubectl logs -n {{namespace}} -f "$job" 2>&1 | tee "$log"

    # `logs -f` returns when the stream closes, which is not the same as the Job
    # having been recorded complete.
    kubectl wait -n {{namespace}} --for=condition=complete --timeout={{pride_timeout}} "$job" \
      || { echo "!! $job did not complete -- see $log"; exit 1; }

# Strictly sequential on purpose: two load generators aimed at one deployment
# would just measure each other. Each point is its own Job, so a point that
# fails costs that point rather than the whole sweep.

# Sweep concurrencies one Job at a time. Args: [concurrency...] (def. 1 5 10 20).
pride-sweep *concurrencies:
    #!/usr/bin/env bash
    set -euo pipefail
    points=({{concurrencies}})
    if (( ${#points[@]} == 0 )); then points=(1 5 10 20); fi

    failed=()
    for c in "${points[@]}"; do
      just pride "$c" || failed+=("$c")
      echo
    done

    echo "==> logs in {{pride_log_dir}}"
    if (( ${#failed[@]} )); then
      echo "!! failed at concurrency: ${failed[*]}"
      exit 1
    fi
    # /results in the Job is an emptyDir, so benchmarks.json/csv die with the pod
    # and cannot be kubectl cp'd out afterwards. The summary table in these logs
    # is the durable artifact; point the `results` volume at a PVC if you need
    # the raw reports.
    echo "==> done"

pride-jobs:
    kubectl get jobs -n {{namespace}} -l app.kubernetes.io/name=guidellm
    kubectl get pods -n {{namespace}} -l app.kubernetes.io/name=guidellm

# Completed Jobs are kept by default -- `kubectl describe` on a failed one is
# the only postmortem left once the sweep has moved on.

# Delete every GuideLLM Job in the namespace. Local logs are untouched.
pride-clean:
    kubectl delete job -n {{namespace}} -l app.kubernetes.io/name=guidellm --ignore-not-found

# ---------------------------------------------------------------------------
# BFCL tool-call correctness (benchmarks/bfcl-job.yaml)
#
# Not a load test: this measures whether the deployment emits *correct* tool
# calls, which is the thing an AgentX-style throughput number says nothing
# about. Ported from vLLM's .buildkite/scripts/tool_call/run-bfcl-eval.sh, but
# aimed at the already-running `url` endpoint above instead of a vLLM server the
# script starts itself.
#
#   just bfcl                       # multi_turn against `url`
#   just bfcl live_simple           # one cheaper category
#   just bfcl "simple,multiple" 16  # several categories, 16 threads
#   just bfcl-jobs / just bfcl-clean
#
# The endpoint must already be served with --enable-auto-tool-choice and a
# --tool-call-parser that matches the model, and `model` must be the id
# /v1/models reports -- the Job's preflight check fails fast on both.
# ---------------------------------------------------------------------------

bfcl_log_dir  := "./results-bfcl"
# multi_turn is the full stateful suite and the slowest; the wall clock below is
# for the Job's pod once it is running, not the queue wait.
bfcl_category := env_var_or_default("BFCL_CATEGORY", "multi_turn")
bfcl_threads  := env_var_or_default("BFCL_THREADS", "8")
bfcl_timeout  := "240m"

# BFCL tool-call eval as a Job. Args: [test-category(,...)] [threads].
bfcl test_category=bfcl_category threads=bfcl_threads:
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p "{{bfcl_log_dir}}"
    # Categories can be a comma-separated list; keep it out of the filename.
    slug=$(echo "{{test_category}}" | tr -c 'A-Za-z0-9_' '-')
    log="{{bfcl_log_dir}}/bfcl-${slug}-t{{threads}}.log"
    echo "==> CATEGORY={{test_category}} THREADS={{threads}} MODEL={{model}} URL={{url}}/v1 -> $log"

    # generateName, so each create gets a fresh Job rather than colliding.
    job=$(
      kubectl create -f benchmarks/bfcl-job.yaml --dry-run=client -o yaml \
        | kubectl set env --local -f - -o yaml \
            "BASE_URL={{url}}/v1" "MODEL={{model}}" \
            "TEST_CATEGORY={{test_category}}" "NUM_THREADS={{threads}}" \
        | kubectl create -n {{namespace}} -f - -o name
    )
    echo "==> created $job"

    kubectl wait -n {{namespace}} --for=condition=Ready pod \
      --selector="job-name=${job#job.batch/}" --timeout=10m || true

    # tee rather than redirect: /results is an emptyDir, so the score tables at
    # the end of this log are the only copy that outlives the pod.
    kubectl logs -n {{namespace}} -f "$job" 2>&1 | tee "$log"

    # `logs -f` returns when the stream closes, which is not the same as the Job
    # having been recorded complete.
    kubectl wait -n {{namespace}} --for=condition=complete --timeout={{bfcl_timeout}} "$job" \
      || { echo "!! $job did not complete -- see $log"; exit 1; }

bfcl-jobs:
    kubectl get jobs -n {{namespace}} -l app.kubernetes.io/name=bfcl
    kubectl get pods -n {{namespace}} -l app.kubernetes.io/name=bfcl

# Delete every BFCL Job in the namespace. Local logs are untouched.
bfcl-clean:
    kubectl delete job -n {{namespace}} -l app.kubernetes.io/name=bfcl --ignore-not-found