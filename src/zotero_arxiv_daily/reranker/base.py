from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from omegaconf import DictConfig, OmegaConf
from ..protocol import Paper, CorpusPaper, MatchedCorpusPaper
from ..utils import glob_match
import numpy as np
from typing import Type


_STOPWORDS = {
    "about", "after", "again", "against", "all", "also", "and", "are", "based",
    "between", "both", "can", "data", "for", "from", "has", "have", "how",
    "into", "its", "model", "models", "more", "new", "not", "our", "paper",
    "present", "propose", "results", "show", "study", "than", "that", "the",
    "their", "this", "through", "title", "abstract", "using", "was", "we", "with",
}


@dataclass
class InterestProfile:
    name: str
    member_indices: list[int]
    representative_indices: list[int]
    text: str
    keywords: list[str]
    importance: float


class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        if not candidates:
            return candidates
        if self._profile_rerank_enabled():
            return self._rerank_with_interest_profiles(candidates, corpus)
        return self._rerank_against_full_corpus(candidates, corpus)

    def _rerank_against_full_corpus(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        if not corpus:
            for candidate in candidates:
                candidate.score = 0.0
            return candidates
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score([self._candidate_text(c) for c in candidates], [self._corpus_text(c) for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    def _profile_rerank_enabled(self) -> bool:
        if self.config is None:
            return False
        profile_config = getattr(self.config.reranker, "profile", None)
        if profile_config is None:
            return False
        return bool(profile_config.get("enabled", True))

    def _rerank_with_interest_profiles(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        if not corpus:
            return candidates

        profiles = self._build_interest_profiles(corpus)
        if not profiles:
            return self._rerank_against_full_corpus(candidates, corpus)

        profile_config = self.config.reranker.profile
        top_profiles = int(profile_config.get("top_profiles_per_candidate", 2))
        profile_weight = float(profile_config.get("profile_similarity_weight", 0.75))
        representative_weight = float(profile_config.get("representative_similarity_weight", 0.25))

        candidate_texts = [self._candidate_text(c) for c in candidates]
        profile_texts = [p.text for p in profiles]
        profile_sim = self.get_similarity_score(candidate_texts, profile_texts)
        assert profile_sim.shape == (len(candidates), len(profiles))

        representative_indices = sorted({idx for p in profiles for idx in p.representative_indices})
        representative_texts = [self._corpus_text(corpus[idx]) for idx in representative_indices]
        representative_sim = self.get_similarity_score(candidate_texts, representative_texts)
        representative_index_lookup = {corpus_idx: pos for pos, corpus_idx in enumerate(representative_indices)}

        for candidate_idx, candidate in enumerate(candidates):
            weighted_profile_scores = np.array([
                profile_sim[candidate_idx, profile_idx] * profile.importance
                for profile_idx, profile in enumerate(profiles)
            ])
            selected_profile_indices = np.argsort(weighted_profile_scores)[::-1][:max(top_profiles, 1)]
            best_profile_idx = int(selected_profile_indices[0])
            best_profile = profiles[best_profile_idx]

            profile_score = float(np.mean(weighted_profile_scores[selected_profile_indices]))
            matched_representatives = []
            representative_scores = []
            for corpus_idx in best_profile.representative_indices:
                rep_pos = representative_index_lookup[corpus_idx]
                similarity = float(representative_sim[candidate_idx, rep_pos])
                representative_scores.append(similarity)
                matched_representatives.append(
                    MatchedCorpusPaper(
                        title=corpus[corpus_idx].title,
                        similarity=similarity,
                        paths=corpus[corpus_idx].paths,
                        added_date=corpus[corpus_idx].added_date,
                    )
                )
            matched_representatives.sort(key=lambda p: p.similarity, reverse=True)
            representative_score = float(np.mean(sorted(representative_scores, reverse=True)[:3])) if representative_scores else 0.0

            candidate.score = (profile_weight * profile_score + representative_weight * representative_score) * 10
            candidate.matched_profile = best_profile.name
            candidate.matched_keywords = best_profile.keywords
            candidate.matched_corpus = matched_representatives

        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    def _build_interest_profiles(self, corpus:list[CorpusPaper]) -> list[InterestProfile]:
        cached_profiles = self._load_cached_interest_profiles(corpus)
        if cached_profiles is not None:
            return cached_profiles

        profile_config = self.config.reranker.profile
        max_profiles = int(profile_config.get("max_profiles", 12))
        representative_count = int(profile_config.get("representative_papers", 3))
        cluster_threshold = float(profile_config.get("cluster_threshold", 0.72))

        corpus_texts = [self._corpus_text(c) for c in corpus]
        corpus_similarity = self.get_similarity_score(corpus_texts, corpus_texts)
        assert corpus_similarity.shape == (len(corpus), len(corpus))

        corpus_weights = self._corpus_weights(corpus)
        assignment_order = np.argsort(corpus_weights)[::-1]
        profile_members: list[list[int]] = []

        for corpus_idx in assignment_order:
            best_profile_idx = None
            best_similarity = -1.0
            for profile_idx, member_indices in enumerate(profile_members):
                similarity = float(np.mean(corpus_similarity[corpus_idx, member_indices]))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_profile_idx = profile_idx

            if best_profile_idx is None or (best_similarity < cluster_threshold and len(profile_members) < max_profiles):
                profile_members.append([int(corpus_idx)])
            else:
                profile_members[best_profile_idx].append(int(corpus_idx))

        profile_masses = [float(corpus_weights[members].sum()) for members in profile_members]
        max_mass = max(profile_masses) if profile_masses else 1.0
        profiles = []
        for members, mass in zip(profile_members, profile_masses):
            members = sorted(members, key=lambda idx: corpus_weights[idx], reverse=True)
            representative_indices = members[:max(representative_count, 1)]
            representative_titles = [corpus[idx].title for idx in representative_indices]
            keywords = self._extract_keywords([corpus_texts[idx] for idx in members])
            name = self._profile_name(representative_titles, keywords)
            text = "\n\n".join(corpus_texts[idx] for idx in representative_indices)
            importance = 0.7 + 0.3 * (mass / max_mass if max_mass else 1.0)
            profiles.append(
                InterestProfile(
                    name=name,
                    member_indices=members,
                    representative_indices=representative_indices,
                    text=text,
                    keywords=keywords,
                    importance=importance,
                )
            )
        self._save_cached_interest_profiles(corpus, profiles)
        return profiles

    def _profile_cache_enabled(self) -> bool:
        profile_config = self.config.reranker.profile
        return bool(profile_config.get("cache_enabled", True))

    def _profile_cache_path(self) -> str:
        profile_config = self.config.reranker.profile
        cache_path = str(profile_config.get("cache_path", ".cache/zotero_arxiv_daily/interest_profiles.json"))
        if os.path.isabs(cache_path):
            return cache_path
        try:
            from hydra.utils import get_original_cwd

            base_dir = get_original_cwd()
        except ValueError:
            base_dir = os.getcwd()
        return os.path.join(base_dir, cache_path)

    def _corpus_fingerprint(self, corpus:list[CorpusPaper]) -> str:
        profile_config = self.config.reranker.profile
        collection_weights = profile_config.get("collection_weights")
        if OmegaConf.is_config(collection_weights):
            collection_weights = OmegaConf.to_container(collection_weights, resolve=True)
        payload = {
            "corpus": [
                {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "added_date": paper.added_date.isoformat(),
                    "paths": sorted(paper.paths),
                }
                for paper in corpus
            ],
            "profile_config": {
                "max_profiles": profile_config.get("max_profiles"),
                "representative_papers": profile_config.get("representative_papers"),
                "cluster_threshold": profile_config.get("cluster_threshold"),
                "collection_weights": collection_weights,
            },
            "reranker_config": {
                "local_model": getattr(getattr(self.config.reranker, "local", None), "model", None),
                "api_model": getattr(getattr(self.config.reranker, "api", None), "model", None),
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_cached_interest_profiles(self, corpus:list[CorpusPaper]) -> list[InterestProfile] | None:
        if not self._profile_cache_enabled():
            return None
        cache_path = self._profile_cache_path()
        try:
            with open(cache_path, encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        if cached.get("fingerprint") != self._corpus_fingerprint(corpus):
            return None
        try:
            return [
                InterestProfile(
                    name=item["name"],
                    member_indices=[int(idx) for idx in item["member_indices"]],
                    representative_indices=[int(idx) for idx in item["representative_indices"]],
                    text=item["text"],
                    keywords=[str(keyword) for keyword in item["keywords"]],
                    importance=float(item["importance"]),
                )
                for item in cached["profiles"]
            ]
        except (KeyError, TypeError, ValueError):
            return None

    def _save_cached_interest_profiles(self, corpus:list[CorpusPaper], profiles:list[InterestProfile]) -> None:
        if not self._profile_cache_enabled():
            return
        cache_path = self._profile_cache_path()
        payload = {
            "fingerprint": self._corpus_fingerprint(corpus),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "profiles": [asdict(profile) for profile in profiles],
        }
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file, ensure_ascii=False, indent=2)
        except OSError:
            return

    def _corpus_weights(self, corpus:list[CorpusPaper]) -> np.ndarray:
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        collection_weights = np.array([self._collection_weight(c) for c in corpus], dtype=float)
        weights = time_decay_weight * collection_weights
        total_weight = weights.sum()
        if total_weight <= 0:
            return np.ones(len(corpus)) / len(corpus)
        return weights / total_weight

    def _collection_weight(self, paper:CorpusPaper) -> float:
        profile_config = self.config.reranker.profile
        configured_weights = profile_config.get("collection_weights") or {}
        if not configured_weights:
            return 1.0
        matched_weights = [
            float(weight)
            for pattern, weight in configured_weights.items()
            if any(glob_match(path, pattern) for path in paper.paths)
        ]
        return max(matched_weights) if matched_weights else 1.0

    def _candidate_text(self, paper:Paper) -> str:
        return f"Title: {paper.title}\nAbstract: {paper.abstract}"

    def _corpus_text(self, paper:CorpusPaper) -> str:
        return f"Title: {paper.title}\nAbstract: {paper.abstract}"

    def _extract_keywords(self, texts:list[str]) -> list[str]:
        import re
        from collections import Counter

        tokens = []
        for text in texts:
            tokens.extend(
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
                if token.lower() not in _STOPWORDS
            )
        return [token for token, _ in Counter(tokens).most_common(6)]

    def _profile_name(self, representative_titles:list[str], keywords:list[str]) -> str:
        if keywords:
            return " / ".join(keywords[:4])
        if representative_titles:
            title = representative_titles[0]
            return title if len(title) <= 80 else title[:77] + "..."
        return "General interest"
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
