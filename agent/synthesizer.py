"""
Synthesizes a final answer from retrieved passages with inline citations.
"""
from agent.llm_client import get_chat_completion
MODEL = "llama-3.3-70b-versatile"

SYNTHESIZER_SYSTEM = """You are a research synthesis assistant.
Using ONLY the provided passages, write a clear, accurate answer to the question.
- Use inline citations in the format [arxiv:PAPER_ID] after each claim
- Only cite papers that actually appear in the provided passages
- Be factual and grounded - don't add information not in the passages
- For comparative questions, explicitly compare the relevant papers
- For survey questions, mention all relevant papers found"""

def synthesize(question: str, passages: list[dict]) -> dict:
    """Returns {"answer": str, "cited_ids": list[str]}"""

    # Format passages for context
    context = ""
    arxiv_ids_seen = set()
    for p in passages[:15]:  # Top 15 passages
        context += f"\n---\nPaper ID: {p['arxiv_id']}\nTitle: {p['title']}\nPassage: {p['text'][:500]}\n"
        arxiv_ids_seen.add(p['arxiv_id'])

    try:
        answer = get_chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM},
                {"role": "user", "content": f"""Question: {question}

Passages:
{context}

Write a comprehensive answer with citations:"""}
            ],
            temperature=0.2,
            max_tokens=800,
        )

        # Extract cited arXiv IDs from answer
        import re
        cited = re.findall(r'\[arxiv:([\d.v]+)\]', answer)
        cited = list(set(cited))

        return {"answer": answer, "cited_ids": cited}

    except Exception as e:
        print(f"  [Synthesizer Error] failed: {e}")
        return {"answer": f"Error generating answer: {e}", "cited_ids": []}
