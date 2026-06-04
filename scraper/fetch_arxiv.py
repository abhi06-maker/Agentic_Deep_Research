"""
Second pass - different queries to expand corpus
"""
import json
import os
import time
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

# DIFFERENT queries from first run
SEARCH_QUERIES = [
    "tool+calling+language+model",
    "retrieval+augmented+generation+2024",
    "language+model+planning+reasoning",
    "LLM+workflow+automation",
    "agent+environment+reward+LLM",
    "self+RAG+adaptive+retrieval",
    "chain+of+thought+agent",
    "web+agent+browser+LLM",
]

BASE_URL = "https://export.arxiv.org/api/query"

def fetch_query(query, max_results=80):
    params = f"?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    url = BASE_URL + params
    for attempt in range(6):
        try:
            print(f"    Attempt {attempt+1}: fetching...")
            resp = requests.get(url, timeout=40)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code in (429, 503):
                wait = 30 * (attempt + 1)
                print(f"    Rate limited ({resp.status_code}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {resp.status_code}, skipping.")
                return None
        except Exception as e:
            print(f"    Error: {e}, retrying in 20s...")
            time.sleep(20)
    return None

def parse_entries(xml_text):
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        try:
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1].strip()
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            published = entry.find("atom:published", ns).text.strip()
            if int(published[:4]) < 2024:
                continue
            cats = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
            if not any(c.startswith(("cs.CL", "cs.AI", "cs.LG")) for c in cats):
                continue
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)][:5]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            papers.append({"id": arxiv_id, "title": title, "abstract": abstract,
                          "authors": authors, "published": published,
                          "categories": cats, "pdf_url": pdf_url})
        except:
            continue
    return papers

def fetch_papers():
    # Load existing papers to avoid duplicates
    existing_ids = set()
    existing_papers = []
    papers_path = os.path.join(DATA_DIR, "papers.jsonl")
    if os.path.exists(papers_path):
        with open(papers_path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                existing_ids.add(p["id"])
                existing_papers.append(p)
        print(f"📚 Loaded {len(existing_papers)} existing papers")

    new_papers = []

    for query in SEARCH_QUERIES:
        print(f"\n🔍 Searching: {query.replace('+', ' ')}")
        print(f"  ⏳ Waiting 20s...")
        time.sleep(20)

        xml_text = fetch_query(query)
        if not xml_text:
            print(f"  ⚠️ Skipping")
            continue

        papers = parse_entries(xml_text)
        added = 0
        for p in papers:
            if p["id"] not in existing_ids:
                existing_ids.add(p["id"])
                new_papers.append(p)
                added += 1

        print(f"  ✅ Added {added} new papers (total new this run: {len(new_papers)})")

    all_papers = existing_papers + new_papers
    print(f"\n✅ Total corpus: {len(all_papers)} papers ({len(new_papers)} new)")

    # Save combined
    with open(papers_path, "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p) + "\n")
    print(f"📄 Saved to {papers_path}")

    # Download only new PDFs
    print(f"\n📥 Downloading {len(new_papers)} new PDFs...")
    for paper in tqdm(new_papers):
        safe_id = paper["id"].replace("/", "_").replace(".", "_")
        pdf_path = os.path.join(PDF_DIR, f"{safe_id}.pdf")
        if os.path.exists(pdf_path):
            continue
        try:
            r = requests.get(paper["pdf_url"], timeout=30,
                           headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 1000:
                with open(pdf_path, "wb") as f:
                    f.write(r.content)
            time.sleep(2)
        except:
            pass

    print(f"\n✅ Done! Total corpus: {len(all_papers)} papers")
    return all_papers

if __name__ == "__main__":
    fetch_papers()