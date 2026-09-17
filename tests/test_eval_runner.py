"""Tests for the eval runner: case loading, the regression gate, and the seed set."""

import json

import pytest

from evals.run import CASES_DIR, check, load_cases, main, run_case


def _summary(**overrides):
    s = {"cases": 2, "ndcg@10": 0.8, "precision@10": 0.3, "recall@10": 0.9,
         "per_case_ndcg": {"a": 1.0, "b": 0.6}}
    s.update(overrides)
    return s


def test_check_passes_on_equal_scores():
    assert check(_summary(), _summary()) == []


def test_check_flags_mean_regression():
    failures = check(_summary(**{"ndcg@10": 0.7}), _summary())
    assert any("ndcg@10 regressed" in f for f in failures)


def test_check_flags_single_case_regression_even_if_mean_improves():
    current = _summary(**{"ndcg@10": 0.9, "per_case_ndcg": {"a": 0.5, "b": 1.0}})
    failures = check(current, _summary())
    assert failures == ["case a ndcg@10 regressed: 0.5000 < baseline 1.0000"]


def test_check_flags_case_count_and_missing_case():
    failures = check(_summary(cases=1, per_case_ndcg={"a": 1.0}), _summary())
    assert any("case count" in f for f in failures)
    assert any("case b is in the baseline but was not run" in f for f in failures)


def test_load_cases_rejects_duplicate_ids(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps([{"id": "x", "relevance": {}}]))
    (tmp_path / "two.json").write_text(json.dumps([{"id": "x", "relevance": {}}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path)


def test_seed_cases_are_well_formed():
    for case in load_cases(CASES_DIR):
        retrieved = {v["video_id"] for pool in case.get("retrieval", {}).values() for v in pool}
        assert set(case["relevance"]) <= retrieved, f"{case['id']}: labels for videos never retrieved"
        assert all(0 <= g <= 3 for g in case["relevance"].values()), case["id"]


@pytest.mark.asyncio
async def test_run_case_uses_pinned_retrieval():
    case = {
        "id": "t", "watch_history": [{"video_id": "w", "title": "W"}],
        "retrieval": {"similar": [{"video_id": "w", "title": "W"}, {"video_id": "s", "title": "S"}],
                      "trending": [{"video_id": "t1", "title": "T"}]},
        "relevance": {"s": 3},
    }
    result = await run_case(case)
    assert result["route"] == "rank_deterministic"
    assert result["served"] == ["s", "t1"]  # watched "w" filtered on the feed path
    assert result["ndcg@10"] == 1.0


@pytest.mark.asyncio
async def test_committed_offline_baseline_passes():
    assert await main(["--offline", "--check", str(CASES_DIR.parent / "baseline.json")]) == 0
