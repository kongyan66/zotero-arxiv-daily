from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional
import hashlib
import json
import os
import re
from collections import Counter

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from paper import ArxivPaper


_STOPWORDS = {
    "about", "after", "again", "against", "all", "also", "and", "are", "based",
    "between", "both", "can", "data", "for", "from", "has", "have", "how",
    "into", "its", "model", "models", "more", "new", "not", "our", "paper",
    "present", "propose", "results", "show", "study", "than", "that", "the",
    "their", "this", "through", "title", "abstract", "using", "was", "we", "with",
    "across", "approach", "benchmark", "benchmarks", "categories", "collection",
    "collections", "comparative", "dataset", "datasets", "diverse", "evaluation",
    "experiments", "framework", "method", "methods", "novel", "performance",
    "report", "task", "tasks", "technical", "training", "understanding",
    "unlimited", "via", "work", "works",
}

_PHRASE_KEYWORDS = [
    (r"\bvision[- ]language\b|\bvlm\b|\bmultimodal\b", "vision-language models"),
    (r"\blarge language model\b|\bllm\b|\blanguage models\b", "LLM"),
    (r"\bretrieval[- ]augmented\b|\brag\b", "retrieval-augmented generation"),
    (r"\bocr\b|\boptical character recognition\b", "OCR"),
    (r"\bpdf\b|\bdocument parsing\b|\bdocling\b", "document parsing"),
    (r"\bscene text\b|\btext recognition\b", "scene text recognition"),
    (r"\bgaze\b|\bgaze estimation\b", "gaze estimation"),
    (r"\bdepth\b|\bdepth estimation\b", "depth estimation"),
    (r"\blong[- ]tail\b|\blong tail\b", "long-tail learning"),
    (r"\bnoisy label\b|\bnoisy labels\b|\blabel noise\b", "noisy labels"),
    (r"\bvisual recognition\b|\bimage recognition\b", "visual recognition"),
    (r"\bobject detection\b", "object detection"),
    (r"\bsemantic segmentation\b|\bsegmentation\b", "segmentation"),
    (r"\bdiffusion\b|\bgenerative\b", "generative models"),
    (r"\bvideo\b|\bopen video\b", "video understanding"),
    (r"\bfinancial\b|\bfinance\b", "financial research"),
]

_THEME_RULES = [
    (
        "Document AI / PDF Parsing / OCR",
        {"document parsing", "pdf", "docling", "scene text recognition"},
        {"ocr", "document", "parsing"},
    ),
    (
        "Vision-Language Models / Multimodal OCR",
        {"vision-language models", "multimodal", "vlm", "qwen"},
        {"ocr", "visual recognition", "visual", "reasoning"},
    ),
    (
        "Long-tail Recognition / Noisy Labels",
        {"long-tail learning", "noisy labels", "long-tail", "noisy"},
        {"label", "rarity", "calibration", "recognition"},
    ),
    (
        "Gaze Estimation / Visual Perception",
        {"gaze estimation", "gaze"},
        {"visual", "estimation", "perception"},
    ),
    (
        "Depth Estimation / 3D Vision",
        {"depth estimation", "depth"},
        {"3d", "visual", "estimation"},
    ),
    (
        "Video Understanding / Multimodal Pre-training",
        {"video understanding", "video"},
        {"multimodal", "pre-training", "pretraining"},
    ),
    (
        "LLM / Retrieval-Augmented Workflows",
        {"llm", "retrieval-augmented generation", "rag", "retrieval"},
        {"workflow", "workflows", "language"},
    ),
    (
        "Financial AI / Research Workflows",
        {"financial research", "financial", "finance"},
        {"workflow", "workflows", "research"},
    ),
]


@dataclass
class MatchedCorpusPaper:
    title: str
    similarity: float
    paths: list[str]
    added_date: str


@dataclass
class InterestProfile:
    name: str
    member_indices: list[int]
    representative_indices: list[int]
    text: str
    keywords: list[str]
    importance: float
    representative_titles: list[str]
    member_count: int
    confidence: float


def _corpus_title(paper:dict) -> str:
    return paper.get("data", {}).get("title") or "Untitled"


def _corpus_abstract(paper:dict) -> str:
    return paper.get("data", {}).get("abstractNote") or ""


def _corpus_paths(paper:dict) -> list[str]:
    return [str(path) for path in paper.get("paths", [])]


def _corpus_added_date(paper:dict) -> datetime:
    value = paper.get("data", {}).get("dateAdded")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return datetime.min


def _corpus_text(paper:dict) -> str:
    paths = ", ".join(_corpus_paths(paper))
    return f"Title: {_corpus_title(paper)}\nAbstract: {_corpus_abstract(paper)}\nCollections: {paths}"


def _corpus_semantic_text(paper:dict) -> str:
    return f"Title: {_corpus_title(paper)}\nAbstract: {_corpus_abstract(paper)}"


def _candidate_text(paper:ArxivPaper) -> str:
    return f"Title: {paper.title}\nAbstract: {paper.summary}"


def _as_numpy(matrix) -> np.ndarray:
    if hasattr(matrix, "detach"):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _document_tokens(text:str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
        if token.lower() not in _STOPWORDS
    ]


def _phrase_document_counts(texts:list[str]) -> dict[str, int]:
    counts = {}
    for _, label in _PHRASE_KEYWORDS:
        counts[label] = 0
    for text in texts:
        lower_text = text.lower()
        for pattern, label in _PHRASE_KEYWORDS:
            if re.search(pattern, lower_text):
                counts[label] += 1
    return counts


def _token_document_counts(texts:list[str]) -> Counter:
    counts = Counter()
    for text in texts:
        counts.update(set(_document_tokens(text)))
    return counts


def _extract_keywords(texts:list[str], background_texts:Optional[list[str]]=None) -> list[str]:
    background_texts = background_texts or texts
    background_size = max(len(background_texts), 1)

    joined = "\n".join(texts).lower()
    profile_phrase_df = _phrase_document_counts(texts)
    background_phrase_df = _phrase_document_counts(background_texts)
    phrase_scores = []
    for pattern, label in _PHRASE_KEYWORDS:
        if not re.search(pattern, joined):
            continue
        profile_hits = profile_phrase_df[label]
        profile_coverage = profile_hits / max(len(texts), 1)
        if profile_coverage < 0.06:
            continue
        idf = np.log((background_size + 1) / (background_phrase_df[label] + 1)) + 1
        phrase_scores.append((label, profile_coverage * float(idf)))
    phrase_keywords = [label for label, _ in sorted(phrase_scores, key=lambda item: item[1], reverse=True)[:4]]

    tokens = []
    for text in texts:
        tokens.extend(_document_tokens(text))
    background_df = _token_document_counts(background_texts)
    token_counts = Counter(tokens)
    token_keywords = []
    phrase_parts = set()
    for phrase in phrase_keywords:
        phrase_parts.update(re.findall(r"[a-z0-9]+", phrase.lower()))
    token_scores = []
    for token, count in token_counts.items():
        idf = np.log((background_size + 1) / (background_df[token] + 1)) + 1
        token_scores.append((token, count * float(idf)))
    for token, _ in sorted(token_scores, key=lambda item: item[1], reverse=True):
        normalized = token.replace("-", "")
        token_parts = re.findall(r"[a-z0-9]+", token)
        if token in phrase_parts or normalized in phrase_parts or all(part in phrase_parts for part in token_parts):
            continue
        token_keywords.append(token)
        if len(token_keywords) >= 6:
            break
    return (phrase_keywords + token_keywords)[:8]


def _profile_name(representative_titles:list[str], keywords:list[str]) -> str:
    title_blob = " ".join(representative_titles).lower()
    keyword_blob = " ".join(keywords).lower()
    if re.search(r"\bqwen\b|\bvlm\b|vision[- ]language|multimodal", title_blob):
        if re.search(r"ocr|visual|multimodal|vision", keyword_blob + " " + title_blob):
            return "Vision-Language Models / Multimodal OCR"
    if re.search(r"docling|pdf parsing|document parsing|scene text", title_blob):
        return "Document AI / PDF Parsing / OCR"
    if re.search(r"long[- ]tail|noisy label|label rarity|calibration", title_blob):
        return "Long-tail Recognition / Noisy Labels"
    if re.search(r"gaze", title_blob):
        return "Gaze Estimation / Visual Perception"
    if re.search(r"depth", title_blob):
        return "Depth Estimation / 3D Vision"

    evidence = {keyword.lower() for keyword in keywords}
    for keyword in keywords:
        evidence.update(re.findall(r"[a-z0-9]+", keyword.lower()))
    for title in representative_titles:
        evidence.update(re.findall(r"[a-z0-9]+", title.lower()))

    best_name = None
    best_score = 0
    for name, required_terms, support_terms in _THEME_RULES:
        required_hits = len(evidence.intersection({term.lower() for term in required_terms}))
        if required_hits == 0:
            continue
        support_hits = len(evidence.intersection({term.lower() for term in support_terms}))
        score = required_hits * 2 + support_hits
        if score > best_score:
            best_name = name
            best_score = score
    if best_name and best_score >= 2:
        return best_name

    if keywords:
        return " / ".join(keywords[:3])
    if representative_titles:
        title = representative_titles[0]
        return title if len(title) <= 80 else title[:77] + "..."
    return "General interest"


def _profile_confidence(members:list[int], corpus_similarity:np.ndarray) -> float:
    if len(members) <= 1:
        return 0.48
    sub_matrix = corpus_similarity[np.ix_(members, members)]
    upper = sub_matrix[np.triu_indices(len(members), k=1)]
    mean_similarity = float(np.mean(upper)) if upper.size else 0.5
    size_penalty = min(0.18, max(0, len(members) - 40) * 0.002)
    return max(0.35, min(0.93, 0.35 + mean_similarity * 0.55 - size_penalty))


def _split_members_by_seeds(
    members:list[int],
    corpus_similarity:np.ndarray,
    corpus_weights:np.ndarray,
    target_count:int,
) -> list[list[int]]:
    if target_count <= 1 or len(members) <= 2:
        return [members]

    seed_indices = [max(members, key=lambda idx: corpus_weights[idx])]
    while len(seed_indices) < target_count:
        remaining = [idx for idx in members if idx not in seed_indices]
        if not remaining:
            break
        next_seed = max(
            remaining,
            key=lambda idx: (1 - max(float(corpus_similarity[idx, seed]) for seed in seed_indices)) * corpus_weights[idx],
        )
        seed_indices.append(next_seed)

    clusters = {seed: [seed] for seed in seed_indices}
    for member in members:
        if member in clusters:
            continue
        best_seed = max(seed_indices, key=lambda seed: float(corpus_similarity[member, seed]))
        clusters[best_seed].append(member)
    return [cluster for cluster in clusters.values() if cluster]


def _split_large_profiles(
    profile_members:list[list[int]],
    corpus_similarity:np.ndarray,
    corpus_weights:np.ndarray,
    max_profiles:int,
    max_profile_members:int,
) -> list[list[int]]:
    if max_profile_members <= 0:
        return profile_members

    queue = sorted(profile_members, key=len, reverse=True)
    final_profiles = []
    while queue:
        members = queue.pop(0)
        remaining_slots = max_profiles - len(final_profiles) - len(queue)
        if len(members) <= max_profile_members or remaining_slots <= 1:
            final_profiles.append(members)
            continue

        target_count = min(int(np.ceil(len(members) / max_profile_members)), remaining_slots)
        split_profiles = _split_members_by_seeds(members, corpus_similarity, corpus_weights, target_count)
        if len(split_profiles) <= 1:
            final_profiles.append(members)
            continue
        queue.extend(split_profiles)
        queue.sort(key=len, reverse=True)

    return final_profiles[:max_profiles]


def _deduplicate_profile_names(profiles:list[InterestProfile]) -> None:
    seen = Counter()
    for profile in profiles:
        base_name = profile.name
        seen[base_name] += 1
        if seen[base_name] == 1:
            continue
        name_terms = set(re.findall(r"[a-z0-9]+", base_name.lower()))
        suffix = next(
            (
                keyword
                for keyword in profile.keywords
                if not set(re.findall(r"[a-z0-9]+", keyword.lower())).issubset(name_terms)
            ),
            profile.representative_titles[0] if profile.representative_titles else f"Variant {seen[base_name]}",
        )
        profile.name = f"{base_name} ({suffix})"


def _corpus_weights(corpus:list[dict]) -> np.ndarray:
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    total_weight = time_decay_weight.sum()
    if total_weight <= 0:
        return np.ones(len(corpus)) / len(corpus)
    return time_decay_weight / total_weight


def _cache_path() -> str:
    return os.environ.get(
        "INTEREST_PROFILE_CACHE_PATH",
        ".cache/zotero_arxiv_daily/interest_profiles.json",
    )


def _cache_enabled() -> bool:
    return os.environ.get("INTEREST_PROFILE_CACHE", "1").lower() not in {"0", "false", "no"}


def _corpus_fingerprint(
    corpus:list[dict],
    model:str,
    max_profiles:int,
    representative_count:int,
    cluster_threshold:float,
    max_profile_members:int,
) -> str:
    payload = {
        "profile_version": 3,
        "model": model,
        "max_profiles": max_profiles,
        "representative_count": representative_count,
        "cluster_threshold": cluster_threshold,
        "max_profile_members": max_profile_members,
        "corpus": [
            {
                "title": _corpus_title(paper),
                "abstract": _corpus_abstract(paper),
                "paths": sorted(_corpus_paths(paper)),
                "dateAdded": paper.get("data", {}).get("dateAdded"),
            }
            for paper in corpus
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_profiles(fingerprint:str) -> Optional[list[InterestProfile]]:
    if not _cache_enabled():
        return None
    try:
        with open(_cache_path(), encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    try:
        profiles = [
            InterestProfile(
                name=item["name"],
                member_indices=[int(idx) for idx in item["member_indices"]],
                representative_indices=[int(idx) for idx in item["representative_indices"]],
                text=str(item["text"]),
                keywords=[str(keyword) for keyword in item["keywords"]],
                importance=float(item["importance"]),
                representative_titles=[str(title) for title in item.get("representative_titles", [])],
                member_count=int(item.get("member_count", len(item["member_indices"]))),
                confidence=float(item.get("confidence", 0.6)),
            )
            for item in payload["profiles"]
        ]
        logger.info(f"Loaded {len(profiles)} interest profiles from cache.")
        return profiles
    except (KeyError, TypeError, ValueError):
        return None


def _save_cached_profiles(fingerprint:str, profiles:list[InterestProfile]) -> None:
    if not _cache_enabled():
        return
    cache_path = _cache_path()
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "profiles": [asdict(profile) for profile in profiles],
                },
                cache_file,
                ensure_ascii=False,
                indent=2,
            )
    except OSError as error:
        logger.warning(f"Failed to save interest profile cache: {error}")


def _build_interest_profiles(
    corpus:list[dict],
    corpus_similarity:np.ndarray,
    corpus_weights:np.ndarray,
    max_profiles:int,
    representative_count:int,
    cluster_threshold:float,
    max_profile_members:int,
) -> list[InterestProfile]:
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

    profile_members = _split_large_profiles(
        profile_members,
        corpus_similarity,
        corpus_weights,
        max_profiles,
        max_profile_members,
    )

    corpus_texts = [_corpus_semantic_text(paper) for paper in corpus]
    profile_masses = [float(corpus_weights[members].sum()) for members in profile_members]
    max_mass = max(profile_masses) if profile_masses else 1.0
    profiles = []
    for members, mass in zip(profile_members, profile_masses):
        members = sorted(members, key=lambda idx: corpus_weights[idx], reverse=True)
        representative_indices = members[:max(representative_count, 1)]
        representative_titles = [_corpus_title(corpus[idx]) for idx in representative_indices]
        keywords = _extract_keywords([corpus_texts[idx] for idx in members], corpus_texts)
        confidence = _profile_confidence(members, corpus_similarity)
        profiles.append(
            InterestProfile(
                name=_profile_name(representative_titles, keywords),
                member_indices=members,
                representative_indices=representative_indices,
                text="\n\n".join(corpus_texts[idx] for idx in representative_indices),
                keywords=keywords,
                importance=0.7 + 0.3 * (mass / max_mass if max_mass else 1.0),
                representative_titles=representative_titles,
                member_count=len(members),
                confidence=confidence,
            )
        )
    profiles.sort(key=lambda profile: profile.importance, reverse=True)
    _deduplicate_profile_names(profiles)
    logger.info("Built interest profiles:")
    for idx, profile in enumerate(profiles[:5], start=1):
        logger.info(
            f"Profile {idx}: {profile.name} | keywords={', '.join(profile.keywords[:6])} "
            f"| representatives={'; '.join(profile.representative_titles[:3])}"
        )
    return profiles


def rerank_paper(candidate:list[ArxivPaper], corpus:list[dict], model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    if not candidate:
        return candidate
    if not corpus:
        for paper in candidate:
            paper.score = 0.0
        return candidate

    max_profiles = int(os.environ.get("INTEREST_PROFILE_MAX", "10"))
    representative_count = int(os.environ.get("INTEREST_PROFILE_REPRESENTATIVES", "3"))
    cluster_threshold = float(os.environ.get("INTEREST_PROFILE_CLUSTER_THRESHOLD", "0.72"))
    max_profile_members = int(os.environ.get("INTEREST_PROFILE_MAX_MEMBERS", "45"))
    top_profiles_per_candidate = int(os.environ.get("INTEREST_PROFILE_TOP_MATCHES", "2"))
    profile_weight = float(os.environ.get("INTEREST_PROFILE_SCORE_WEIGHT", "0.75"))
    representative_weight = float(os.environ.get("INTEREST_REPRESENTATIVE_SCORE_WEIGHT", "0.25"))
    diversity_penalty = float(os.environ.get("INTEREST_PROFILE_DIVERSITY_PENALTY", "0.35"))

    encoder = SentenceTransformer(model)
    corpus = sorted(corpus,key=_corpus_added_date,reverse=True)
    corpus_texts = [_corpus_semantic_text(paper) for paper in corpus]
    candidate_texts = [_candidate_text(paper) for paper in candidate]
    corpus_weights = _corpus_weights(corpus)

    fingerprint = _corpus_fingerprint(
        corpus,
        model,
        max_profiles,
        representative_count,
        cluster_threshold,
        max_profile_members,
    )
    profiles = _load_cached_profiles(fingerprint)
    if profiles is None:
        logger.info("Building interest profiles from Zotero corpus...")
        corpus_features = encoder.encode(corpus_texts)
        corpus_similarity = _as_numpy(encoder.similarity(corpus_features, corpus_features))
        profiles = _build_interest_profiles(
            corpus,
            corpus_similarity,
            corpus_weights,
            max_profiles,
            representative_count,
            cluster_threshold,
            max_profile_members,
        )
        _save_cached_profiles(fingerprint, profiles)

    profile_texts = [profile.text for profile in profiles]
    representative_indices = sorted({idx for profile in profiles for idx in profile.representative_indices})
    representative_texts = [corpus_texts[idx] for idx in representative_indices]
    representative_index_lookup = {corpus_idx: pos for pos, corpus_idx in enumerate(representative_indices)}

    candidate_features = encoder.encode(candidate_texts)
    profile_sim = _as_numpy(encoder.similarity(candidate_features, encoder.encode(profile_texts)))
    representative_sim = _as_numpy(encoder.similarity(candidate_features, encoder.encode(representative_texts)))

    for candidate_idx, paper in enumerate(candidate):
        weighted_profile_scores = np.array([
            float(profile_sim[candidate_idx, profile_idx]) * profile.importance
            for profile_idx, profile in enumerate(profiles)
        ])
        selected_profile_indices = np.argsort(weighted_profile_scores)[::-1][:max(top_profiles_per_candidate, 1)]
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
                    title=_corpus_title(corpus[corpus_idx]),
                    similarity=similarity,
                    paths=_corpus_paths(corpus[corpus_idx]),
                    added_date=corpus[corpus_idx].get("data", {}).get("dateAdded") or "",
                )
            )
        matched_representatives.sort(key=lambda item: item.similarity, reverse=True)
        representative_score = float(np.mean(sorted(representative_scores, reverse=True)[:3])) if representative_scores else 0.0

        paper.score = (profile_weight * profile_score + representative_weight * representative_score) * 10
        paper.matched_profile = best_profile.name
        paper.matched_profile_confidence = best_profile.confidence
        paper.matched_profile_member_count = best_profile.member_count
        paper.matched_keywords = best_profile.keywords
        paper.matched_corpus = matched_representatives
        paper.interest_profiles = profiles[:5]

    ranked = sorted(candidate,key=lambda x: x.score,reverse=True)
    profile_counts = {}
    for paper in ranked:
        profile = paper.matched_profile or "Unknown profile"
        repeat_count = profile_counts.get(profile, 0)
        if repeat_count:
            paper.score = max(0.0, paper.score - min(1.4, repeat_count * diversity_penalty))
        profile_counts[profile] = repeat_count + 1

    return sorted(ranked,key=lambda x: x.score,reverse=True)
