"""
Main agentic loop: plan → retrieve → reflect → synthesize → verify.
Configurable so each component can be toggled for ablations.
"""
import time
import numpy as np
from agent.planner import plan
from agent.retriever import HybridRetriever
from agent.reflector import reflect
from agent.synthesizer import synthesize
from agent.citation_verifier import verify_citations

class AgentConfig:
    def __init__(
        self,
        name: str = "full_agent",
        use_planner: bool = True,
        use_hybrid: bool = True,       # False = dense only
        use_reflector: bool = True,
        use_citation_verifier: bool = True,
        max_rounds: int = 3,
        top_k: int = 10,
    ):
        self.name = name
        self.use_planner = use_planner
        self.use_hybrid = use_hybrid
        self.use_reflector = use_reflector
        self.use_citation_verifier = use_citation_verifier
        self.max_rounds = max_rounds
        self.top_k = top_k

# Pre-defined configs for ablations
CONFIGS = {
    "full_agent": AgentConfig("full_agent"),
    "no_planner": AgentConfig("no_planner", use_planner=False),
    "no_hybrid": AgentConfig("no_hybrid", use_hybrid=False),
    "no_reflector": AgentConfig("no_reflector", use_reflector=False),
    "no_citation_verifier": AgentConfig("no_citation_verifier", use_citation_verifier=False),
}


class DeepResearchAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.retriever = HybridRetriever(
            use_bm25=config.use_hybrid,
            use_dense=True
        )

    def run(self, question: str) -> dict:
        start_time = time.time()
        trace = {
            "question": question,
            "plan": [],
            "retrievals": [],
            "reflector_decisions": [],
            "final_synthesis": "",
            "cited_ids": [],
            "tool_calls": 0,
            "latency_sec": 0,
        }

        # Step 1: Plan
        sub_questions = plan(question, enabled=self.config.use_planner)
        if self.config.use_planner and len(sub_questions) > 0 and question not in sub_questions:
            aligned_questions = []
            for sq in sub_questions:
                q_emb = self.retriever.model.encode([question], normalize_embeddings=True)[0]
                sq_emb = self.retriever.model.encode([sq], normalize_embeddings=True)[0]
                sim = float(np.dot(q_emb, sq_emb))
                if sim >= 0.35:
                    aligned_questions.append(sq)
                else:
                    aligned_questions.append(f"{sq} (in context of: {question})")
            sub_questions = aligned_questions
        trace["plan"] = sub_questions

        all_passages = []
        round_num = 0

        # Step 2: Retrieve + Reflect loop
        queries_to_run = list(sub_questions)

        while queries_to_run and round_num < self.config.max_rounds:
            round_passages = []

            for query in queries_to_run:
                chunks = self.retriever.retrieve(query, top_k=self.config.top_k)
                trace["tool_calls"] += 1
                round_passages.extend(chunks)
                trace["retrievals"].append({
                    "round": round_num,
                    "query": query,
                    "num_results": len(chunks),
                    "top_titles": [c["title"][:50] for c in chunks[:3]],
                })

            # Deduplicate
            seen_chunk_ids = {p["chunk_id"] for p in all_passages}
            new_passages = [p for p in round_passages if p["chunk_id"] not in seen_chunk_ids]

            # RAAES: Redundancy-Aware Adaptive Early Stopping
            if round_num > 0 and new_passages and all_passages:
                old_texts = [p["text"] for p in all_passages]
                new_texts = [p["text"] for p in new_passages]
                old_embs = self.retriever.model.encode(old_texts, normalize_embeddings=True)
                new_embs = self.retriever.model.encode(new_texts, normalize_embeddings=True)
                similarities = np.dot(new_embs, old_embs.T)
                max_similarities = np.max(similarities, axis=1)
                avg_max_sim = float(np.mean(max_similarities))
                if avg_max_sim > 0.85:
                    trace["reflector_decisions"].append({
                        "round": round_num,
                        "sufficient": True,
                        "reason": f"RAAES: Adaptive early stopping triggered (redundancy score {avg_max_sim:.3f} > 0.85).",
                        "missing": ""
                    })
                    break

            all_passages.extend(new_passages)

            # Step 3: Reflect
            reflection = reflect(
                question=question,
                passages=all_passages,
                round_num=round_num,
                max_rounds=self.config.max_rounds,
                enabled=self.config.use_reflector
            )
            trace["reflector_decisions"].append({
                "round": round_num,
                **reflection
            })

            if reflection["sufficient"]:
                break

            # Generate follow-up query from missing info
            queries_to_run = [reflection.get("missing", "")] if reflection.get("missing") else []
            round_num += 1

        # Global Context-Aware Semantic Reranking (GCASR Novelty)
        # Rerank and filter out irrelevant/drifted chunks using direct semantic similarity to the main question
        if hasattr(self.retriever, "global_rerank") and self.retriever.use_dense:
            all_passages = self.retriever.global_rerank(question, all_passages)
        else:
            all_passages.sort(key=lambda x: x.get("score", 0), reverse=True)


        # Step 4: Synthesize
        synthesis = synthesize(question, all_passages)
        trace["tool_calls"] += 1

        # Step 5: Verify Citations
        verification = verify_citations(
            answer=synthesis["answer"],
            passages=all_passages,
            enabled=self.config.use_citation_verifier
        )

        trace["final_synthesis"] = verification["verified_answer"]
        trace["cited_ids"] = verification["valid_ids"]
        trace["latency_sec"] = round(time.time() - start_time, 2)

        return trace