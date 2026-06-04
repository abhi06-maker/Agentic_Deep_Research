"""
Verifies that each cited arXiv ID is actually supported by retrieved passages.
Uses structural validation followed by semantic entailment check (EVCSC) using the LLM.
"""
import re
from agent.llm_client import get_chat_completion

def verify_citations(answer: str, passages: list[dict], enabled: bool = True) -> dict:
    """
    Returns {"verified_answer": str, "valid_ids": list, "removed_ids": list}
    """
    cited_ids = set(re.findall(r'\[arxiv:([\d.v]+)\]', answer))
    retrieved_ids = {p["arxiv_id"] for p in passages}

    if not enabled:
        return {
            "verified_answer": answer,
            "valid_ids": list(cited_ids),
            "removed_ids": [],
        }

    # Step 1: Structural verification (must exist in retrieved set)
    structurally_valid = cited_ids & retrieved_ids
    removed_ids = cited_ids - retrieved_ids

    verified_answer = answer
    # Remove structurally invalid citations
    for bad_id in removed_ids:
        verified_answer = verified_answer.replace(f"[arxiv:{bad_id}]", "")

    # Step 2: Semantic entailment check (EVCSC)
    # Split the answer into sentences to check individual citation sentences
    sentences = re.split(r'(?<=[.!?])\s+', verified_answer)
    new_sentences = []
    
    semantically_valid = set()
    semantically_invalid = set()

    for sent in sentences:
        s_citations = re.findall(r'\[arxiv:([\d.v]+)\]', sent)
        if not s_citations:
            new_sentences.append(sent)
            continue

        # Strip citations from sentence to check clean claim
        clean_sent = sent
        for cid in s_citations:
            clean_sent = clean_sent.replace(f"[arxiv:{cid}]", "")
        clean_sent = clean_sent.strip()

        retained_citations = []
        for cid in s_citations:
            # Find retrieved passages for this citation
            matching_chunks = [p for p in passages if p["arxiv_id"] == cid]
            if not matching_chunks:
                continue

            # Combine top 2 passage contents to use as context
            context = "\n\n".join([c["text"] for c in matching_chunks[:2]])

            # Call verifier
            prompt = f"Claim: \"{clean_sent}\"\n\nContext:\n{context}"
            messages = [
                {
                    "role": "system",
                    "content": "You are a fact-checking assistant. Determine if the context logically supports the claim. Respond ONLY with YES or NO."
                },
                {"role": "user", "content": prompt}
            ]

            try:
                res = get_chat_completion(
                    model="gemini-3.1-flash-lite",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10
                )
                if "YES" in res.upper():
                    retained_citations.append(cid)
                    semantically_valid.add(cid)
                else:
                    semantically_invalid.add(cid)
            except Exception as e:
                # Fallback: keep if LLM call fails
                retained_citations.append(cid)
                semantically_valid.add(cid)

        # Reconstruct sentence with verified citations only
        temp_sent = sent
        for cid in s_citations:
            temp_sent = temp_sent.replace(f"[arxiv:{cid}]", "")
        temp_sent = temp_sent.strip()

        if retained_citations:
            cit_string = " " + " ".join([f"[arxiv:{cid}]" for cid in retained_citations])
            if temp_sent and temp_sent[-1] in ".!?":
                temp_sent = temp_sent[:-1] + cit_string + temp_sent[-1]
            else:
                temp_sent = temp_sent + cit_string
        new_sentences.append(temp_sent)

    verified_answer = " ".join(new_sentences)
    verified_answer = re.sub(r"  +", " ", verified_answer).strip()

    return {
        "verified_answer": verified_answer,
        "valid_ids": list(semantically_valid),
        "removed_ids": list(removed_ids | semantically_invalid),
    }