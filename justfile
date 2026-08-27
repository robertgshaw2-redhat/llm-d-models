# AIPerf AgentX-MVP benchmark against the running llm-d optimized-baseline deployment.
#

namespace := env_var_or_default("NAMESPACE", "robshaw-dev")
# namespace := env_var_or_default("NAMESPACE", "default")
deploy    := "aiperf-agentx"
# model     := env_var_or_default("MODEL", "moonshotai/Kimi-K3")
# model     := env_var_or_default("MODEL", "thinkingmachines/Inkling-NVFP4")
# model     := env_var_or_default("MODEL", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4")
# model     := env_var_or_default("MODEL", "RedHatAI/GLM-5.2-NVFP4-FP8")
# model     := env_var_or_default("MODEL", "granite-4-0-h-small")
# model     := env_var_or_default("MODEL", "google/gemma-4-31B-it")
# model     := env_var_or_default("MODEL", "zai-org/GLM-5.2-FP8")
model     := env_var_or_default("MODEL", "zai-org/GLM-5.3-Flash")
# model     := env_var_or_default("MODEL", "RedHatAI/GLM-5.2-MXFP4xFP8_BLOCK")
# tokenizer     := env_var_or_default("TOKENIZER", "ibm-granite/granite-4.0-h-small")
# tokenizer     := env_var_or_default("TOKENIZER", "google/gemma-4-31B-it")
tokenizer     := env_var_or_default("TOKENIZER", "zai-org/GLM-5.2-FP8")

# url       := env_var_or_default("URL", "http://kimik3-epp:80")
# url       := env_var_or_default("URL", "http://glm-epp:80")
# url       := env_var_or_default("URL", "http://glm-agg-epp:80")
# url       := env_var_or_default("URL", "http://inkling-epp:80")
# url       := env_var_or_default("URL", "http://nemotron-ultra-epp:80")
# url       := env_var_or_default("URL", "http://granite-epp:80")
# url       := env_var_or_default("URL", "http://granite4-small-agg-svc:80")
# url       := env_var_or_default("URL", "http://gemma-4-31b-agg:80")
# url       := env_var_or_default("URL", "http://glm-epp:80")
url := env_var_or_default("URL", "http://glm-flash-svc:80")
# url       := env_var_or_default("URL", "https://test.rhai.ibm.com/v1/projects/e0c67991-420e-4136-9ff4-793849bf0ee7/inference")
duration  := "900"

# The vLLM serving Deployment itself (inkling-small/aggregated/base/), not the
# runner: `just bench` execs into this pod and drives its own localhost:8000.
# The model id has to be what that pod actually serves -- `vllm bench serve`
# sends it as the request body's "model" and loads its tokenizer by that name.
deployment := "gemma"

node:
    kubectl cp script.mjs rhai-gemma-repro:/tmp/repro.mjs
    kubectl exec rhai-gemma-repro -- \
        sh -c 'export RHAI_API_KEY=base RHAI_API_BASE={{url}}/v1; node /tmp/repro.mjs'

default:
    @just --list

prof_prefill:
    kubectl exec -n {{namespace}} glm-prefill-0 -- curl -X POST http://localhost:8000/start_profile

prof_decode:
    kubectl exec -n {{namespace}} glm-decode-0 -- curl -X POST http://localhost:8200/start_profile

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
        --no-server-metrics \
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
        --endpoint-type chat \
        --streaming \
        --use-server-token-count \
        --tokenizer-trust-remote-code \
        --public-dataset semianalysis_cc_traces_weka_with_subagents \
        --concurrency {{concurrency}} \
        --benchmark-duration {{duration}} \
        --output-artifact-dir /workspace/artifacts \
        --no-server-metrics \
        --max-context-length 256000 \
        --tokenizer {{tokenizer}} \
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

# Copy benchmark artifacts out of the runner to a local directory (default ./results).
results dest="./results":
    mkdir -p {{dest}}
    kubectl cp {{namespace}}/$(kubectl get pod -n {{namespace}} -l app={{deploy}} -o jsonpath='{.items[0].metadata.name}'):/workspace/artifacts {{dest}}

# Run lm-evaluation-harness from the lm-eval pod against the deployment's
# OpenAI-compatible chat endpoint. Args: [tasks] [concurrency].
lmeval limit="1300" concurrency="100" fewshot="20":
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- \
        lm_eval \
            --model local-completions \
            --model_args "model={{model}},base_url={{url}}/v1/completions,num_concurrent={{concurrency}},tokenized_requests=False,trust_remote_code=True" \
            --tasks gsm8k --limit {{limit}} --num_fewshot {{fewshot}}

install-lmeval:
    kubectl exec -n {{namespace}} deploy/{{deploy}} -- pip install lm-eval[api] transformers

# 10k ISL / 1 OSL vllm bench serve. Args: [concurrency] [num-prompts].
bench_random isl="1" osl="1000" concurrency="64" num_prompts="256":
    #!/usr/bin/env bash
    set -euo pipefail
    num_prompts="{{num_prompts}}"
    if [[ -z "$num_prompts" ]]; then num_prompts=$(( {{concurrency}} * 2 )); fi
    echo "==> ISL={{isl}} OSL={{osl}} CONCURRENCY={{concurrency}} NUM_PROMPTS=$num_prompts"
    echo "{{namespace}}"
    kubectl exec -n {{namespace}} glm-flash-agg-8cb54f968-fwxh4 -- \
      vllm bench serve \
        --backend vllm \
        --base-url {{url}} \
        --model {{model}} \
        --trust-remote-code \
        --dataset-name random \
        --random-input-len {{isl}} \
        --random-output-len {{osl}} \
        --ignore-eos \
        --num-prompts "$num_prompts" \
        --max-concurrency {{concurrency}} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --seed $(date +%s)

# 10k ISL / 1 OSL vllm bench serve. Args: [concurrency] [num-prompts].
bench_prefix num_prompts="1000" rr="20":
    kubectl exec -n {{namespace}} deploy/{{deployment}} -- \
      vllm bench serve \
        --backend vllm \
        --base-url {{url}} \
        --model {{model}} \
        --trust-remote-code \
        --num-prompts {{num_prompts}} \
        --request-rate {{rr}} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --dataset-name prefix_repetition \
        --prefix-repetition-prefix-len 10000 \
        --prefix-repetition-suffix-len 3000 \
        --prefix-repetition-num-prefixes 10 \
        --prefix-repetition-output-len 500 \
        --seed $(date +%s)

bench_shared num_prompts="1000" concurrency="100":
    kubectl exec -n {{namespace}} deploy/{{deployment}} -- \
      vllm bench serve \
        --backend vllm \
        --base-url {{url}} \
        --model {{model}} \
        --trust-remote-code \
        --num-prompts {{num_prompts}} \
        --request-rate inf \
        --max-concurrency {{concurrency}} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --dataset-name prefix_repetition \
        --prefix-repetition-prefix-len 5000 \
        --prefix-repetition-suffix-len 5000 \
        --prefix-repetition-num-prefixes 1 \
        --prefix-repetition-output-len 500 \
        --seed $(date +%s)

sweep_shared:
    just bench_shared 1000 100
    just bench_shared 2000 200
    just bench_shared 3000 300

sweep_prefix:
    just bench_prefix 1000 10
    just bench_prefix 1000 20
    just bench_prefix 1000 40
    just bench_prefix 1000 60
    just bench_prefix 1000 80
    just bench_prefix 1000 100


# vllm bench serve against the sonnet dataset that ships with vLLM
# (benchmarks/sonnet.txt), run from inside the serving pod against its own
# localhost:8000, same as `bench` above.
#
# What this measures that `bench` does not: `bench` sends random token ids,
# which are word salad -- fine for pinning token counts, useless for anything
# that depends on the text being predictable (a draft model has nothing to
# predict). The sonnet sampler instead fills each prompt with real lines of
# verse sampled with replacement until it hits ISL, so the token lengths are
# still pinned exactly but the content is English. Cheaper to set up than the
# `pride` sweep below -- no PVC, no preprocess Job -- at the cost of prompts
# that differ run to run, since the lines are resampled every time.
#
# --backend openai-chat: the sonnet sampler formats prompts through the model's
# chat template, so the request has to go to /v1/chat/completions. That is also
# the path guidellm and lm-eval drive, which keeps the numbers comparable.
#
# sonnet_prefix_len is the number of leading tokens every prompt shares. vLLM's
# default is 200 and it is kept here because the aggregated Deployments run with
# prefix caching on -- a zero-length prefix would measure a cache that never
# hits. Set it to 0 to benchmark a cold path.

sonnet_isl := "28"
sonnet_osl := "1000"
sonnet_prefix_len := "200"

# Sonnet-dataset vllm bench serve. Args: [concurrency] [num-prompts].
sonnet concurrency="16" num_prompts="":
    #!/usr/bin/env bash
    set -euo pipefail
    num_prompts="{{num_prompts}}"
    if [[ -z "$num_prompts" ]]; then num_prompts=$(( {{concurrency}} * 10 )); fi
    echo "==> ISL={{sonnet_isl}} OSL={{sonnet_osl}} PREFIX={{sonnet_prefix_len}} CONCURRENCY={{concurrency}} NUM_PROMPTS=$num_prompts"

    # -i so the heredoc reaches `bash -s` in the pod; everything just needs to
    # interpolate is passed as a positional parameter, so the remote script
    # itself is free of local expansions.
    kubectl exec -i -n {{namespace}} deploy/{{deployment}} -- vllm bench serve \
      --backend openai-chat \
      --base-url http://localhost:8000 \
      --endpoint /v1/chat/completions \
      --model {{model}} \
      --trust-remote-code \
      --dataset-name sonnet \
      --dataset-path /tmp/sonnet.txt \
      --sonnet-input-len {{sonnet_isl}} \
      --sonnet-output-len {{sonnet_osl}} \
      --sonnet-prefix-len {{sonnet_prefix_len}} \
      --num-prompts {{num_prompts}} \
      --max-concurrency {{concurrency}} \
      --request-rate inf \
      --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,95

# Sweep concurrencies through the sonnet benchmark, one at a time -- two load
# generators aimed at one deployment would just measure each other.
# Args: [concurrency...] (def. 1 4 8 16 32).
sonnet-sweep *concurrencies:
    #!/usr/bin/env bash
    set -euo pipefail
    points=({{concurrencies}})
    if (( ${#points[@]} == 0 )); then points=(1 4 8 16 32); fi
    for c in "${points[@]}"; do
      just sonnet "$c"
      echo
      sleep 10
    done

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
# Nsight Systems traces (glm5.2/disaggregated/base/{prefill,decode}.yaml)
#
# Those containers exec vllm under `nsys profile --capture-range=cudaProfilerApi`,
# so nsys sits idle until `just prof_prefill` / `just prof_decode` hits
# /start_profile. It then records max_iterations engine steps and writes
# /profiles/vllm_<host>_<pid>.nsys-rep inside the pod.
#
# /profiles is an emptyDir -- a trace not copied out dies with the pod.
#
#   just nsys-ls                    # what the pod has written so far
#   just nsys-fetch                 # copy them here
#   just nsys-fetch glm-prefill-0   # ... from the prefill pod instead
#   just nsys-clean                 # free the emptyDir between captures
# ---------------------------------------------------------------------------

nsys_pod := env_var_or_default("NSYS_POD", "glm-decode-0")
nsys_dir := env_var_or_default("NSYS_DIR", "./traces-nsys")

# List the traces a serving pod has written. Args: [pod].
nsys-ls pod=nsys_pod:
    kubectl exec -n {{namespace}} {{pod}} -- ls -lh /profiles

# Copy every finished .nsys-rep out of a serving pod. Args: [pod] [dest].
nsys-fetch pod=nsys_pod dest=nsys_dir:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{dest}}"

    # nsys writes a .qdstrm while capturing and only renames it to .nsys-rep
    # once the capture is finalized, so a still-recording profile is simply
    # absent from this list rather than copied out half-written.
    files=$(kubectl exec -n {{namespace}} {{pod}} -- \
      sh -c 'ls -1 /profiles/*.nsys-rep 2>/dev/null' || true)
    if [[ -z "$files" ]]; then
      echo "!! {{pod}}:/profiles has no .nsys-rep yet"
      echo "   run 'just prof_decode' (or prof_prefill), then give nsys a few"
      echo "   seconds to finalize -- 'just nsys-ls {{pod}}' shows the .qdstrm"
      exit 1
    fi

    for f in $files; do
      out="{{dest}}/{{pod}}-$(basename "$f")"
      if [[ -s "$out" ]]; then
        echo "==> skip $out (already local)"
        continue
      fi
      echo "==> $f -> $out"
      # `exec -- cat` rather than `kubectl cp`: cp shells out to tar inside the
      # container, and each of these is one large binary blob anyway. No -t, so
      # the stream stays binary-clean.
      kubectl exec -n {{namespace}} {{pod}} -- cat "$f" > "$out"
    done

    ls -lh "{{dest}}"

# Delete the traces in a pod's /profiles. Local copies are untouched. Args: [pod].
nsys-clean pod=nsys_pod:
    kubectl exec -n {{namespace}} {{pod}} -- \
      sh -c 'rm -f /profiles/*.nsys-rep /profiles/*.qdstrm; ls -lh /profiles'
