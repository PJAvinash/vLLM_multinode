#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request


def run_bench(url: str, model: str, num_prompts: int, output_len: int, request_rate: float):
    def send_one(i: int):
        payload = {
            "model": model,
            "prompt": f"Benchmark prompt {i}: Explain in one sentence.",
            "max_tokens": output_len,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status != 200:
                    return False, time.monotonic() - t0
                _ = json.loads(resp.read().decode("utf-8"))
                return True, time.monotonic() - t0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return False, time.monotonic() - t0

    completed = 0
    failed = 0
    latencies = []
    start = time.monotonic()
    max_workers = max(4, min(64, int(max(1.0, request_rate) * 4)))

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i in range(num_prompts):
            # Pace submissions to approximate target RPS.
            target_t = start + (i / request_rate if request_rate > 0 else 0.0)
            now = time.monotonic()
            if target_t > now:
                time.sleep(target_t - now)
            futures.append(pool.submit(send_one, i))

        for fut in concurrent.futures.as_completed(futures):
            ok, latency = fut.result()
            if ok:
                completed += 1
                latencies.append(latency)
            else:
                failed += 1

    elapsed = time.monotonic() - start
    return {
        "completed": completed,
        "failed": failed,
        "num_prompts": num_prompts,
        "request_rate": request_rate,
        "elapsed_sec": elapsed,
        "avg_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--request-rate", type=float, required=True)
    ap.add_argument("--result-file", required=True)
    args = ap.parse_args()

    result = run_bench(
        url=args.url,
        model=args.model,
        num_prompts=args.num_prompts,
        output_len=args.output_len,
        request_rate=args.request_rate,
    )
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"completed={result['completed']} failed={result['failed']}")


if __name__ == "__main__":
    main()
