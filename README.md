# Agentic_Deep_Research

An implementation of an Agentic Deep Research System specialized in synthesizing recent literature on Large Language Model (LLM) agents (arXiv 2024–2026). This repository contains the complete codebase to scrape papers, construct hybrid indices, run ablation studies across six configurations, evaluate them via LLM-as-a-judge, and launch an interactive Gradio trace demo.


# Project Structure
To satisfy the deliverables, the repository contains the following structure:
```
├── scraper/
│   └── fetch_arxiv.py       # Ingests PDF research papers from arXiv
├── indexer/
│   └── build_index.py       # Chunks text and builds BM25 & FAISS indices
├── agent/
│   ├── agent.py             # Main agent loop (Planning, Reflection, Synthesis)
│   ├── retriever.py         # Hybrid search retriever & GCASR reranking layer
│   ├── planner.py           # Sub-query planning & decomposition
│   ├── reflector.py         # Info sufficiency reflection assessment
│   ├── citation_verifier.py # EVCSC semantic entailment check verifier
│   └── llm_client.py        # Centralized caching client with OpenRouter fallback
├── baseline/
│   └── baseline.py          # Naive single-turn retrieval RAG baseline
├── ablations/
│   └── run_ablations.py     # Batch executor for modular agent configurations
├── judge/
│   └── evaluate.py          # LLM-as-a-judge quantitative scorer
├── demo/
│   └── app.py               # Gradio web interface with trace views
├── report/
│   └── report.md            # Technical academic evaluation report (4-6 pages)
├── eval/
│   ├── questions.jsonl      # Test set of 30 research questions
│   └── results.json         # Evaluation table outputs (F1, Accuracy, Faithfulness)
├── predictions/             # Generated answer outputs for all 6 configurations
│   ├── full_agent.jsonl
│   ├── baseline.jsonl
│   ├── no_planner.jsonl
│   ├── no_hybrid.jsonl
│   ├── no_reflector.jsonl
│   └── no_citation_verifier.jsonl
├── run_all.py               # Single-command replication script
├── requirements.txt         # Package dependencies
└── .gitignore               # Excludes large PDF binary blocks and keys


3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. *Set Up API Credentials*
   Create a .env file in the root directory:
   env
   # Set either direct Gemini Key or OpenRouter Key (recommended fallback)
   GEMINI_API_KEY=your_gemini_key_here
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```


# Replication (Single Command)

To reproduce the prediction files and evaluation scores reported in the technical report from scratch, execute:
```bash
python run_all.py

This script runs the scraper, builds the index, generates predictions for all configurations, and executes the LLM judge. The aggregated scores will be saved in `eval/results.json` and printed in the terminal.


# Web Browser Demo

To launch the interactive Gradio web application with plans, retrievals, reflector loops, and verified citation trace views, execute:
```bash
python demo/app.py
```
By default, this will host the demo locally at `http://127.0.0.1:7860` and generate a public `.gradio.live` share link.

# Key Contributions & Novelties

Our agent introduces four lightweight guardrail layers designed to mitigate error propagation:
* Global Context-Aware Semantic Reranking (GCASR)**: Reranks all retrieved chunks against the original main query using local embeddings to prune query drift noise.
* Dynamic Semantic-Drift Prevention (DSDP)**: Rewrites drifted planning sub-queries by automatically appending the global question context.
* Redundancy-Aware Adaptive Early Stopping (RAAES)**: Evaluates information gain between retrieval rounds, halting reflection loops if new chunks are redundant ($> 0.85$ similarity).
* Entailment-Verified Citation Self-Correction (EVCSC)**: Executes post-hoc LLM checks to verify if referenced passages logically support claims before attributing citation tags.
