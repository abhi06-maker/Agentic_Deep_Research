"""
LLM-as-judge evaluator. Scores predictions on:
- Answer accuracy (0-5)
- Faithfulness (0-5)
- Citation precision & recall (exact set overlap)
- Latency & tool call count
Outputs a summary table.
"""
import json
import os
import re
import time
from agent.llm_client import get_chat_completion
MODEL = "llama-3.3-70b-versatile"

JUDGE_SYSTEM = """You are a strict academic research evaluator.
Score the answer on two dimensions (0-5 each):
- accuracy: Does the answer correctly address the question? Is the information factually plausible?
- faithfulness: Does the answer stay grounded in evidence? Are claims supported rather than hallucinated?

Return ONLY JSON: {"accuracy": N, "faithfulness": N, "reason": "brief"}"""

def judge_answer(question: str, answer: str) -> dict:
    try:
        text = get_chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer[:1000]}"}
            ],
            temperature=0.1,
            max_tokens=150,
        )
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return {"accuracy": 0, "faithfulness": 0, "reason": "judge error"}

def citation_metrics(predicted_ids: list, ground_truth_ids: list) -> dict:
    """Exact set overlap precision and recall."""
    if not predicted_ids and not ground_truth_ids:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    pred_set = set(predicted_ids)
    gt_set = set(ground_truth_ids)
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}  # no ground truth = skip
    precision = len(pred_set & gt_set) / len(pred_set)
    recall = len(pred_set & gt_set) / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

def evaluate_config(config_name: str, questions_path: str, gt_path: str = None):
    preds_path = f"predictions/{config_name}.jsonl"
    if not os.path.exists(preds_path):
        print(f"  [Warning] Missing: {preds_path}")
        return None

    questions = {}
    with open(questions_path) as f:
        for line in f:
            q = json.loads(line)
            questions[q["id"]] = q

    # Load ground truth if available
    ground_truth = {}
    if gt_path and os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                gt = json.loads(line)
                ground_truth[gt["id"]] = gt

    preds = []
    with open(preds_path) as f:
        for line in f:
            preds.append(json.loads(line))

    scores = []
    for pred in preds:
        qid = pred["id"]
        question = questions[qid]["question"]
        gt = ground_truth.get(qid, {})

        # LLM judge
        judge = judge_answer(question, pred.get("answer", ""))
        time.sleep(0.5)  # Rate limit

        # Citation metrics
        gt_ids = gt.get("must_cite", [])
        cite_metrics = citation_metrics(pred.get("cited_ids", []), gt_ids)

        scores.append({
            "id": qid,
            "type": questions[qid]["type"],
            "accuracy": judge["accuracy"],
            "faithfulness": judge["faithfulness"],
            "cite_precision": cite_metrics["precision"],
            "cite_recall": cite_metrics["recall"],
            "cite_f1": cite_metrics["f1"],
            "latency_sec": pred.get("latency_sec", 0),
            "tool_calls": pred.get("tool_calls", 0),
        })

    # Aggregate
    def avg(key):
        return round(sum(s[key] for s in scores) / len(scores), 3)

    summary = {
        "config": config_name,
        "n": len(scores),
        "accuracy": avg("accuracy"),
        "faithfulness": avg("faithfulness"),
        "cite_precision": avg("cite_precision"),
        "cite_recall": avg("cite_recall"),
        "cite_f1": avg("cite_f1"),
        "latency_sec": avg("latency_sec"),
        "tool_calls": avg("tool_calls"),
    }

    return summary

def run_full_evaluation():
    configs = ["full_agent", "baseline", "no_planner", "no_hybrid", "no_reflector", "no_citation_verifier"]
    all_summaries = []

    for config in configs:
        print(f"\n[Evaluating] Evaluating: {config}")
        summary = evaluate_config(config, "eval/questions.jsonl")
        if summary:
            all_summaries.append(summary)
            print(f"  Accuracy: {summary['accuracy']}/5 | Faith: {summary['faithfulness']}/5 | "
                  f"Cite-F1: {summary['cite_f1']} | Latency: {summary['latency_sec']}s | "
                  f"Calls: {summary['tool_calls']}")

    # Print ablation table
    print("\n\n" + "="*90)
    print("ABLATION TABLE")
    print("="*90)
    header = f"{'Config':<25} {'Acc':>5} {'Faith':>6} {'Cite-P':>7} {'Cite-R':>7} {'Cite-F1':>8} {'Lat(s)':>7} {'Calls':>6}"
    print(header)
    print("-"*90)
    for s in all_summaries:
        print(f"{s['config']:<25} {s['accuracy']:>5.2f} {s['faithfulness']:>6.2f} "
              f"{s['cite_precision']:>7.3f} {s['cite_recall']:>7.3f} {s['cite_f1']:>8.3f} "
              f"{s['latency_sec']:>7.1f} {s['tool_calls']:>6.1f}")
    print("="*90)

    # Save
    with open("eval/results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\n[Done] Saved to eval/results.json")

if __name__ == "__main__":
    run_full_evaluation()
