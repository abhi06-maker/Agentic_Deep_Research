"""
Non-agentic baseline: single retrieval + one LLM call. No planning/reflection loop.
"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.retriever import HybridRetriever
from agent.synthesizer import synthesize
from agent.citation_verifier import verify_citations

def run_baseline(questions_path: str, output_path: str):
    retriever = HybridRetriever(use_bm25=True, use_dense=True)

    questions = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    for q in questions:
        print(f"  Baseline Q{q['id']}: {q['question'][:60]}...")
        start = time.time()

        # Single retrieval, no loop
        passages = retriever.retrieve(q["question"], top_k=10)
        synthesis = synthesize(q["question"], passages)
        verification = verify_citations(synthesis["answer"], passages, enabled=True)

        results.append({
            "id": q["id"],
            "answer": verification["verified_answer"],
            "cited_ids": verification["valid_ids"],
            "latency_sec": round(time.time() - start, 2),
            "tool_calls": 2,  # 1 retrieval + 1 LLM
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[Done] Baseline saved to {output_path}")

if __name__ == "__main__":
    run_baseline("eval/questions.jsonl", "predictions/baseline.jsonl")