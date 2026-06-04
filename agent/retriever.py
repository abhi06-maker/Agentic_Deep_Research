"""
Hybrid retriever: BM25 (lexical) + FAISS (semantic) with RRF fusion.
"""
import json
import pickle
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"

# Shared cache to avoid loading large models/indices multiple times
_SHARED_CHUNKS = None
_SHARED_BM25 = None
_SHARED_TOKENIZED_CORPUS = None
_SHARED_MODEL = None
_SHARED_INDEX = None

class HybridRetriever:
    def __init__(self, use_bm25=True, use_dense=True):
        self.use_bm25 = use_bm25
        self.use_dense = use_dense

        # Load chunks
        global _SHARED_CHUNKS, _SHARED_BM25, _SHARED_TOKENIZED_CORPUS, _SHARED_MODEL, _SHARED_INDEX
        if _SHARED_CHUNKS is None:
            self.chunks = []
            with open(os.path.join(DATA_DIR, "chunks.jsonl")) as f:
                for line in f:
                    self.chunks.append(json.loads(line))
            _SHARED_CHUNKS = self.chunks
        else:
            self.chunks = _SHARED_CHUNKS

        if use_bm25:
            if _SHARED_BM25 is None:
                with open(os.path.join(DATA_DIR, "bm25.pkl"), "rb") as f:
                    _SHARED_BM25 = pickle.load(f)
            self.bm25 = _SHARED_BM25

            if _SHARED_TOKENIZED_CORPUS is None:
                _SHARED_TOKENIZED_CORPUS = [c["text"].lower().split() for c in self.chunks]
            self.tokenized_corpus = _SHARED_TOKENIZED_CORPUS

        if use_dense:
            if _SHARED_MODEL is None:
                _SHARED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            self.model = _SHARED_MODEL

            if _SHARED_INDEX is None:
                _SHARED_INDEX = faiss.read_index(
                    os.path.join(DATA_DIR, "faiss_index", "index.faiss")
                )
            self.faiss_index = _SHARED_INDEX

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Returns top-k chunks using hybrid retrieval (RRF fusion)."""
        if self.use_bm25 and self.use_dense:
            return self._hybrid_retrieve(query, top_k)
        elif self.use_bm25:
            return self._bm25_retrieve(query, top_k)
        elif self.use_dense:
            return self._dense_retrieve(query, top_k)
        else:
            raise ValueError("At least one retrieval method must be enabled")

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i]), "retrieval_method": "bm25"}
            for i in top_indices
        ]

    def _dense_retrieve(self, query: str, top_k: int) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [
            {**self.chunks[indices[0][i]], "score": float(scores[0][i]), "retrieval_method": "dense"}
            for i in range(len(indices[0]))
        ]

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) of BM25 + dense rankings."""
        k_rrf = 60  # RRF constant

        bm25_results = self._bm25_retrieve(query, top_k * 2)
        dense_results = self._dense_retrieve(query, top_k * 2)

        rrf_scores = {}

        for rank, chunk in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k_rrf + rank + 1)

        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k_rrf + rank + 1)

        # Build unified chunk map
        chunk_map = {c["chunk_id"]: c for c in bm25_results + dense_results}

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            {**chunk_map[cid], "score": rrf_scores[cid], "retrieval_method": "hybrid"}
            for cid in sorted_ids
        ]

    def global_rerank(self, main_question: str, passages: list[dict], threshold: float = 0.20) -> list[dict]:
        """
        [Innovative Novelty: Global Context-Aware Semantic Reranking (GCASR)]
        Reranks retrieved passages using local cosine similarity to the main question,
        filtering out irrelevant noise or query drift chunks.
        """
        if not passages or not self.use_dense:
            return passages

        # Deduplicate passages based on chunk_id
        seen_ids = set()
        unique_passages = []
        for p in passages:
            if p["chunk_id"] not in seen_ids:
                seen_ids.add(p["chunk_id"])
                unique_passages.append(p)

        # Get passage texts
        texts = [p["text"] for p in unique_passages]

        # Compute embeddings locally using pre-loaded sentence-transformer
        question_emb = self.model.encode([main_question], normalize_embeddings=True)[0]
        passage_embs = self.model.encode(texts, normalize_embeddings=True)

        # Calculate cosine similarities
        similarities = np.dot(passage_embs, question_emb)

        # Rerank and filter based on threshold
        reranked = []
        for i, similarity in enumerate(similarities):
            p = dict(unique_passages[i])
            p["global_relevance_score"] = float(similarity)
            # Retain chunk if it meets relevance threshold
            if similarity >= threshold:
                reranked.append(p)

        # Sort by global relevance score descending
        reranked.sort(key=lambda x: x["global_relevance_score"], reverse=True)
        return reranked