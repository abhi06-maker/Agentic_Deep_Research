"""
Decides whether retrieved evidence is sufficient to answer the question,
or whether another retrieval round is needed.
"""
import json
from agent.llm_client import get_chat_completion
MODEL = "llama-3.3-70b-versatile"

REFLECTOR_SYSTEM = """You are a research quality assessor.
Given a question and retrieved passages, decide if the evidence is sufficient to write a complete answer.
Return ONLY a JSON object: {"sufficient": true/false, "reason": "brief reason", "missing": "what is still needed if not sufficient"}
Be strict: if citations to specific papers are needed but not found, mark as insufficient."""

def reflect(question: str, passages: list[dict], round_num: int,
            max_rounds: int = 3, enabled: bool = True) -> dict:
    """
    Returns {"sufficient": bool, "reason": str, "missing": str}
    If disabled (ablation), always returns sufficient=True after round 1.
    """
    if not enabled or round_num >= max_rounds:
        return {"sufficient": True, "reason": "Reflector disabled or max rounds reached", "missing": ""}

    # Summarize passages for the LLM
    passage_summary = "\n\n".join([
        f"[{p['arxiv_id']}] {p['title'][:60]}: {p['text'][:300]}..."
        for p in passages[:8]
    ])

    try:
        text = get_chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": f"""Question: {question}

Retrieved passages:
{passage_summary}

Is the evidence sufficient?"""}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"  [Reflector Error] failed: {e}")
        return {"sufficient": True, "reason": "Reflector error, proceeding", "missing": ""}
