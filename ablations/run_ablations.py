"""
Runs all agent configurations (full + ablations) and saves predictions/.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.agent import DeepResearchAgent, CONFIGS

def run_config(config_name: str, questions_path: str):
    config = CONFIGS[config_name]
    agent = DeepResearchAgent(config)

    questions = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line))

    os.makedirs("predictions", exist_ok=True)
    output_path = f"predictions/{config_name}.jsonl"
    results = []

    print(f"\n[Running] Running config: {config_name}")
    for q in questions:
        print(f"  Q{q['id']}: {q['question'][:60]}...")
        trace = agent.run(q["question"])
        results.append({
            "id": q["id"],
            "answer": trace["final_synthesis"],
            "cited_ids": trace["cited_ids"],
            "latency_sec": trace["latency_sec"],
            "tool_calls": trace["tool_calls"],
            "trace": trace,  # full trace for debugging
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"  [Done] Saved to {output_path}")

if __name__ == "__main__":
    configs_to_run = list(CONFIGS.keys())
    for c in configs_to_run:
        run_config(c, "eval/questions.jsonl")