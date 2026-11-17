# /// script
# requires-python = "==3.12.*"
# dependencies = ["vllm==0.34.0"]
# ///
"""Generate greedily with vLLM 0.34 and compare against a reference run.

vLLM pins compressed-tensors 0.17, so uv runs this in its own environment:
    uv run tests/vllm_smoke.py run MODEL_DIR result.json [--unquantized]
    uv run tests/vllm_smoke.py compare reference.json result.json
"""

import argparse
import json

PROMPTS = ["The capital of France is", "def fibonacci(n):", "1, 2, 3, 4, 5,"]


def run(model: str, out: str, unquantized: bool):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        # vLLM strict-checks unquantized qwen4_exp for the hyper-connection padding
        # shard, which no checkpoint contains (its output is discarded)
        model_loader_extra_config={"enable_weights_track": False} if unquantized else None,
    )
    sampling = SamplingParams(temperature=0, max_tokens=32, logprobs=1)
    results = [
        {"tokens": list(c.token_ids), "logprobs": [next(iter(d.values())).logprob for d in c.logprobs]}
        for c in (o.outputs[0] for o in llm.generate(PROMPTS, sampling))
    ]
    with open(out, "w") as f:
        json.dump(results, f)


def compare(reference: str, result: str):
    with open(reference) as f, open(result) as g:
        pairs = list(zip(json.load(f), json.load(g), strict=True))
    diffs = []
    for a, b in pairs:
        # compare up to the first differing token (random weights give near-ties)
        same = next(
            (i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"], strict=False)) if x != y), len(a["tokens"])
        )
        diffs += [abs(x - y) for x, y in zip(a["logprobs"][:same], b["logprobs"][:same], strict=True)]
    print(f"{len(diffs)} shared tokens, |logprob diff| mean={sum(diffs) / len(diffs):.4f} max={max(diffs):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("model")
    run_parser.add_argument("out")
    run_parser.add_argument("--unquantized", action="store_true")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("reference")
    compare_parser.add_argument("result")
    args = parser.parse_args()
    if args.command == "run":
        run(args.model, args.out, args.unquantized)
    else:
        compare(args.reference, args.result)
