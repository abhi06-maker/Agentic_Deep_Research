import os
import sys
import subprocess

def run_step(cmd, desc):
    print("\n" + "="*50)
    print(f"🚀 Running: {desc}")
    print("="*50)
    
    # Use the current virtual environment's python if active
    python_exe = sys.executable
    full_cmd = [python_exe] + cmd
    
    # Run process and stream output
    process = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    if process.returncode != 0:
        print(f"\n❌ Error: '{desc}' failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    print(f"✅ Completed: {desc}")

def main():
    print("==========================================")
    print("  Agentic Deep Research - Full Pipeline")
    print("==========================================")

    # Load and check env
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable or .env entry not found!")
        print("   Please create a .env file containing: GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    print(f"🔑 API Key verified: {api_key[:10]}...")

    # Phase 1: Corpus Collection & Indexing
    run_step(["scraper/fetch_arxiv.py"], "Phase 1a: Fetching arXiv Corpus")
    run_step(["indexer/build_index.py"], "Phase 1b: Building Retrieval Index")

    # Phase 2: Run all agent configurations
    run_step(["baseline/baseline.py"], "Phase 2a: Running Baseline Agent")
    run_step(["ablations/run_ablations.py"], "Phase 2b: Running Ablation Configs")

    # Phase 3: Evaluate
    run_step(["-m", "judge.evaluate"], "Phase 3: Evaluating All Configurations (LLM-as-a-judge)")

    print("\n==========================================")
    print("  🎉 Done! All pipeline steps completed.")
    print("  📊 Results saved in eval/results.json")
    print("  📁 Predictions stored in predictions/")
    print("==========================================")

if __name__ == "__main__":
    main()
