"""Tests for BaseReranker: scoring, sorting, time decay, unknown reranker."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from zotero_arxiv_daily.protocol import CorpusPaper
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray):
        self.config = None
        self._sim = sim_matrix

    def get_similarity_score(self, s1, s2):
        return self._sim


class LexicalProfileReranker(BaseReranker):
    def get_similarity_score(self, s1, s2):
        return np.array([
            [_lexical_similarity(left, right) for right in s2]
            for left in s1
        ])


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(left.lower().replace("-", " ").split())
    right_tokens = set(right.lower().replace("-", " ").split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _profile_config(**overrides):
    profile = {
        "enabled": True,
        "max_profiles": 4,
        "representative_papers": 2,
        "top_profiles_per_candidate": 1,
        "cluster_threshold": 0.12,
        "profile_similarity_weight": 0.75,
        "representative_similarity_weight": 0.25,
        "collection_weights": None,
        "cache_enabled": False,
        "cache_path": ".cache/test-interest-profiles.json",
    }
    profile.update(overrides)
    return OmegaConf.create({"reranker": {"profile": profile}})


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_time_decay_weighting():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="P")]

    # Only similar to the oldest paper (index 2 after reverse-sort by date)
    sim = np.array([[0.0, 0.0, 1.0]])
    reranker = StubReranker(sim)
    ranked_old = reranker.rerank(papers, corpus)
    score_old = ranked_old[0].score

    # Only similar to the newest paper (index 0 after reverse-sort by date)
    papers2 = [make_sample_paper(title="P")]
    sim2 = np.array([[1.0, 0.0, 0.0]])
    reranker2 = StubReranker(sim2)
    ranked_new = reranker2.rerank(papers2, corpus)
    score_new = ranked_new[0].score

    # Newest corpus paper gets higher time-decay weight, so score should be higher
    assert score_new > score_old


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")


def test_profile_rerank_scores_against_interest_profiles():
    corpus = [
        CorpusPaper(
            title="Agent Memory Planning",
            abstract="language agent memory planning tool use",
            added_date=make_sample_corpus(1)[0].added_date,
            paths=["2026/current/agents"],
        ),
        CorpusPaper(
            title="Agent Reflection",
            abstract="reflection planning language agents",
            added_date=make_sample_corpus(2)[1].added_date,
            paths=["2026/current/agents"],
        ),
        CorpusPaper(
            title="Protein Folding",
            abstract="protein structure folding biology",
            added_date=make_sample_corpus(3)[2].added_date,
            paths=["2026/biology"],
        ),
    ]
    candidates = [
        make_sample_paper(title="New Agent Memory Method", abstract="language agent memory planning"),
        make_sample_paper(title="New Protein Structure Method", abstract="protein folding structure"),
    ]

    reranker = LexicalProfileReranker(_profile_config())
    ranked = reranker.rerank(candidates, corpus)

    assert ranked[0].title == "New Agent Memory Method"
    assert ranked[0].score is not None
    assert ranked[0].matched_profile is not None
    assert "agent" in ranked[0].matched_keywords
    assert ranked[0].matched_corpus
    assert ranked[0].matched_corpus[0].title in {"Agent Memory Planning", "Agent Reflection"}


def test_profile_rerank_can_be_disabled():
    corpus = make_sample_corpus(2)
    candidates = [make_sample_paper(title="A"), make_sample_paper(title="B")]
    sim = np.array([
        [0.1, 0.1],
        [0.9, 0.9],
    ])

    reranker = StubReranker(sim)
    reranker.config = _profile_config(enabled=False)
    ranked = reranker.rerank(candidates, corpus)

    assert ranked[0].title == "B"
    assert ranked[0].matched_profile is None
