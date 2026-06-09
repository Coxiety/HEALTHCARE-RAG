import sys
import os
import re
import json
import uuid
import urllib.parse
import yaml
import requests
from bs4 import BeautifulSoup
from Bio import Entrez

# Add the root directory to sys.path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.vector_store import VectorStore
from src.data_pipeline.chunker import Chunker

CONFIG_PATH = "configs/config.yaml"
CORPUS_OUT = "data/en/corpus.jsonl"

Entrez.email = "healthcare.rag.bot@example.com"

def get_existing_pmids() -> set:
    """Read corpus.jsonl and extract all PMIDs that are already downloaded."""
    existing = set()
    if not os.path.exists(CORPUS_OUT):
        return existing
        
    with open(CORPUS_OUT, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                src = data.get("source", "")
                match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", src)
                if match:
                    existing.add(match.group(1))
            except:
                continue
    return existing

def fetch_pubmed_batch(pmid_list: list) -> list:
    """Fetch multiple PubMed articles in one API call."""
    results = []
    print(f"Fetching {len(pmid_list)} articles from PubMed via Biopython Entrez...")
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), retmode="xml")
        records = Entrez.read(handle)
        
        if not records or "PubmedArticle" not in records:
            return results
            
        for article_wrapper in records["PubmedArticle"]:
            article = article_wrapper["MedlineCitation"]["Article"]
            pmid = str(article_wrapper["MedlineCitation"]["PMID"])
            title = article.get("ArticleTitle", "Unknown Title")
            
            abstract_text = ""
            if "Abstract" in article and "AbstractText" in article["Abstract"]:
                abstract_parts = article["Abstract"]["AbstractText"]
                if isinstance(abstract_parts, list):
                    parts = []
                    for part in abstract_parts:
                        label = getattr(part, "attributes", {}).get("Label", "")
                        if label:
                            parts.append(f"{label}: {part}")
                        else:
                            parts.append(str(part))
                    abstract_text = "\n".join(parts)
                else:
                    abstract_text = str(abstract_parts)
                    
            if abstract_text:
                results.append({
                    "pmid": pmid,
                    "id": f"MED-{pmid}",
                    "text": f"{title}\n\n{abstract_text}",
                    "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
    except Exception as e:
        print(f"Failed to fetch batch via Biopython: {e}")
    return results

def process_targets(url_or_query: str):
    existing_pmids = get_existing_pmids()
    print(f"Found {len(existing_pmids)} articles already in the database.")
    
    pmids_to_fetch = []
    generic_urls = []
    
    # Format lại url để phòng hờ PowerShell làm mất dấu =
    url_or_query = url_or_query.strip().strip('"').strip("'")
    
    # Check if the input is a single PubMed link, a PubMed Search Link, or a general query
    match_single = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url_or_query)
    match_search = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/\?term=(.*)", url_or_query)
    
    if match_single:
        pmids_to_fetch.append(match_single.group(1))
    elif match_search:
        term = urllib.parse.unquote_plus(match_search.group(1))
        term = term.split('&')[0] 
        print(f"Detected PubMed search query: '{term}'")
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=15) 
            record = Entrez.read(handle)
            found_ids = record.get("IdList", [])
            print(f"Found {len(found_ids)} articles matching the query.")
            pmids_to_fetch.extend(found_ids)
        except Exception as e:
            print(f"Search failed: {e}")
    elif "http" not in url_or_query:
        print(f"Treating input as raw PubMed search query: '{url_or_query}'")
        try:
            handle = Entrez.esearch(db="pubmed", term=url_or_query, retmax=15)
            record = Entrez.read(handle)
            found_ids = record.get("IdList", [])
            print(f"Found {len(found_ids)} articles matching the query.")
            pmids_to_fetch.extend(found_ids)
        except Exception as e:
            print(f"Search failed: {e}")
    else:
        generic_urls.append(url_or_query)

    # LỌC TRÙNG LẶP: Bỏ qua những bài đã có trong corpus.jsonl
    new_pmids = []
    for p in pmids_to_fetch:
        if p in existing_pmids:
            print(f"SKIP: PMID {p} is already in the database.")
        else:
            new_pmids.append(p)
            
    docs_to_chunk = []
    
    if new_pmids:
        print(f"Proceeding to fetch {len(new_pmids)} NEW articles...")
        for i in range(0, len(new_pmids), 10):
            batch = new_pmids[i:i+10]
            fetched = fetch_pubmed_batch(batch)
            docs_to_chunk.extend(fetched)
            
    for g_url in generic_urls:
        print(f"Using generic HTML scraper for: {g_url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(g_url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(chunk for chunk in lines if chunk)
            if clean_text and len(clean_text) > 50:
                docs_to_chunk.append({
                    "id": str(uuid.uuid4()),
                    "text": clean_text,
                    "source": g_url,
                    "pmid": None
                })
        except Exception as e:
            print(f"Failed to fetch generic URL {g_url}: {e}")

    if not docs_to_chunk:
        print("No new documents to add to the RAG database.")
        sys.exit(0)

    try:
        cfg = yaml.safe_load(open(CONFIG_PATH))
    except FileNotFoundError:
        print("config.yaml not found.")
        sys.exit(1)
        
    vs = VectorStore(cfg["chroma_persist_dir"], cfg["chroma_collection"], cfg["embedding_model"])
    chunker = Chunker(chunk_size=cfg.get("chunk_size", 500), chunk_overlap=cfg.get("chunk_overlap", 100))
    
    all_chunks = []
    for doc in docs_to_chunk:
        # Không tách chunk nữa, lấy nguyên văn toàn bộ bài (full text/abstract)
        chunk = {
            "id": doc["id"],
            "text": doc["text"],
            "source": doc["source"]
        }
        all_chunks.append(chunk)

    if not all_chunks:
        print("No chunks generated.")
        sys.exit(0)

    print(f"Generated {len(all_chunks)} total chunks. Updating ChromaDB...")
    vs.add(all_chunks)
    
    print(f"Appending to {CORPUS_OUT} for BM25...")
    os.makedirs(os.path.dirname(CORPUS_OUT), exist_ok=True)
    with open(CORPUS_OUT, "a", encoding="utf-8") as f:
        for chunk in all_chunks:
            record = {
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"]
            }
            f.write(json.dumps(record) + "\n")
            
    print("Knowledge base updated successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_url_to_rag.py <URL_or_SearchTerm>")
        sys.exit(1)
    
    # Gộp tất cả arg lại để chống PowerShell tự cắt chuỗi tại dấu cách/bằng
    query = " ".join(sys.argv[1:])
    process_targets(query)
