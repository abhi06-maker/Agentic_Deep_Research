"""
Decomposes a complex research question into sub-queries.
"""
from agent.llm_client import get_chat_completion
MODEL = "llama-3.3-70b-versatile"  # Free on Groq

PLANNER_SYSTEM = """You are a research planning assistant.
Given a research question, decompose it into 2-4 specific sub-questions
that together would fully answer the main question.
Each sub-question should be independently searchable.
Return ONLY a JSON array of strings. No explanation, no markdown.
Example: ["What methods does paper X use?", "How does X compare to Y?"]"""

def plan(question: str, enabled: bool = True) -> list[str]:
    """
    Returns a list of sub-questions.
    If disabled, returns the original question as-is (ablation: no planner).
    """
    if not enabled:
        return [question]

    try:
        text = get_chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"Research question: {question}"}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        import json
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        sub_questions = json.loads(text)
        if isinstance(sub_questions, list) and len(sub_questions) > 0:
            return sub_questions
    except Exception as e:
        print(f"  [Planner Error] failed: {e}")

    return [question]  # fallback
