"""Ranking eval over the golden set.

Each case pins retrieval to a fixed candidate pool (the tools are patched),
so scores measure what prompts and models change: ranking and filtering.
Retrieval quality is out of scope here.

  python -m evals.run --offline                         # no-LLM cases only (what CI runs)
  python -m evals.run                                   # every case, using LLM_PROVIDER
  python -m evals.run --offline --check evals/baseline.json
  python -m evals.run --offline --write-baseline evals/baseline.json

Cases with a query take the LLM ranking route; cases without one take
rank_deterministic or popular_fallback. --offline skips the LLM cases.
"""

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path
from statistics import mean
from unittest.mock import AsyncMock, patch

from evals.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.agent.graph import recommendation_graph
from src.agent.impressions import current_model_id
from src.agent.nodes.rank import PROMPT_VERSION
from src.agent.state import AgentState
from src.config import config

CASES_DIR = Path(__file__).parent / "cases"
K = 10
# Mean scores may dip by this much before the gate fails (float noise, not slack).
TOLERANCE = 0.005
# A single case may not lose more NDCG than this, even if the mean improves:
# a mean-only gate hides a fix for one case that breaks another. An
# intentional tradeoff means rewriting the baseline.
PER_CASE_TOLERANCE = 0.05
METRICS = (f"ndcg@{K}", f"precision@{K}", f"recall@{K}")


def load_cases(cases_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    ids = [c["id"] for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case ids: {sorted(dupes)}")
    return cases


def needs_llm(case: dict) -> bool:
    return bool(case.get("query"))


def _patched_retrieval(case: dict):
    r = case.get("retrieval", {})
    history = case.get("watch_history", [])
    stack = contextlib.ExitStack()
    for target, value in {
        "src.agent.nodes.retrieve.get_user_history": [h["video_id"] for h in history],
        "src.agent.nodes.retrieve.get_video_titles": {h["video_id"]: h["title"] for h in history},
        "src.agent.nodes.retrieve.search_videos": [
            {"id": v["video_id"], **{k: x for k, x in v.items() if k != "video_id"}} for v in r.get("search", [])
        ],
        "src.agent.nodes.retrieve.semantic_search": r.get("semantic", []),
        "src.agent.nodes.retrieve.get_similar_videos": r.get("similar", []),
        "src.agent.nodes.retrieve.get_trending_videos": r.get("trending", []),
        "src.agent.nodes.popular_fallback.get_trending_videos": r.get("popular", []),
    }.items():
        stack.enter_context(patch(target, AsyncMock(return_value=value)))
    return stack


async def run_case(case: dict) -> dict:
    limit = case.get("limit", K)
    with _patched_retrieval(case):
        # ainvoke directly (not get_recommendations) to read the route and
        # to keep eval runs out of the impression log.
        final = await recommendation_graph.ainvoke(
            AgentState(user_id=case.get("user_id", f"eval-{case['id']}"), query=case.get("query"), limit=limit)
        )
    ranked = [r["video_id"] for r in final["ranked_results"][:limit]]
    relevance = case["relevance"]
    return {
        "id": case["id"],
        "route": final.get("route"),
        "rank_fallback": final.get("rank_fallback"),
        "served": ranked,
        f"ndcg@{K}": ndcg_at_k(ranked, relevance, K),
        f"precision@{K}": precision_at_k(ranked, relevance, K),
        f"recall@{K}": recall_at_k(ranked, relevance, K),
    }


def summarize(results: list[dict]) -> dict:
    summary = {"cases": len(results)}
    for m in METRICS:
        summary[m] = round(mean(r[m] for r in results), 4) if results else 0.0
    summary["per_case_ndcg"] = {r["id"]: round(r[f"ndcg@{K}"], 4) for r in results}
    return summary


def check(summary: dict, baseline: dict) -> list[str]:
    failures = []
    if summary["cases"] != baseline["cases"]:
        failures.append(
            f"case count {summary['cases']} != baseline {baseline['cases']} — rerun with --write-baseline"
        )
    for m in METRICS:
        if summary[m] < baseline[m] - TOLERANCE:
            failures.append(f"{m} regressed: {summary[m]:.4f} < baseline {baseline[m]:.4f}")
    for case_id, before in baseline.get("per_case_ndcg", {}).items():
        after = summary["per_case_ndcg"].get(case_id)
        if after is None:
            failures.append(f"case {case_id} is in the baseline but was not run")
        elif after < before - PER_CASE_TOLERANCE:
            failures.append(f"case {case_id} ndcg@{K} regressed: {after:.4f} < baseline {before:.4f}")
    return failures


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=CASES_DIR)
    parser.add_argument("--offline", action="store_true", help="skip cases that need an LLM")
    parser.add_argument("--check", type=Path, help="fail if scores regress below this baseline")
    parser.add_argument("--write-baseline", type=Path, help="write the current scores as the baseline")
    parser.add_argument("--report", type=Path, help="write per-case results as JSON")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    if args.offline:
        cases = [c for c in cases if not needs_llm(c)]
    subset = "offline" if args.offline else f"full:{config.llm_provider}:{current_model_id()}"

    results = [await run_case(c) for c in cases]
    summary = summarize(results)

    for r in results:
        flag = f" FALLBACK={r['rank_fallback']}" if r["rank_fallback"] else ""
        print(f"{r['id']:<46} {r['route'] or '-':<18} ndcg={r[f'ndcg@{K}']:.3f}{flag}")
    totals = " ".join(f"{k}={v}" for k, v in summary.items() if k != "per_case_ndcg")
    print(f"\n[{subset}] prompt_version={PROMPT_VERSION} {totals}")

    if args.report:
        args.report.write_text(json.dumps(
            {"subset": subset, "prompt_version": PROMPT_VERSION, "summary": summary, "results": results}, indent=2,
        ) + "\n")

    fallbacks = [r["id"] for r in results if r["rank_fallback"]]
    if fallbacks:
        # The LLM never ranked these, so their scores say nothing about the prompt or model.
        print(f"FAIL: {len(fallbacks)} case(s) fell back from LLM ranking: {fallbacks}", file=sys.stderr)
        return 1

    if args.write_baseline:
        existing = json.loads(args.write_baseline.read_text()) if args.write_baseline.exists() else {}
        existing[subset] = summary
        args.write_baseline.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        print(f"wrote baseline [{subset}] to {args.write_baseline}")

    if args.check:
        baseline = json.loads(args.check.read_text()).get(subset)
        if baseline is None:
            print(f"no baseline for [{subset}] in {args.check}", file=sys.stderr)
            return 1
        failures = check(summary, baseline)
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        if failures:
            return 1
        print("eval gate passed")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
