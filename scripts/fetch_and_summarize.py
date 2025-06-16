# scripts/fetch_and_summarize_ss.py
import re, requests, json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# --- 検索設定 ---
BASE_QUERY   = '"midinfrared" OR "vibrational control" OR "rotational excitation"'
KEY_PHRASES  = [r"vibrational.+control", r"rotational.+control"]
MAX_KEEP     = 5
CANDIDATE_N  = 200        # 1次検索ヒット件数

PDF_DIR      = Path("pdf"); PDF_DIR.mkdir(exist_ok=True)
SUM_DIR      = Path("summaries"); SUM_DIR.mkdir(exist_ok=True)

# --- LLM 要約設定 ---
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-4o")
PROMPT = """あなたは、赤外レーザーによる分子の振動回転状態制御を研究する物理学者です。
以下の要約を読んで、次の5点を明確に示してください：
1. 研究背景と着眼点
2. アプローチや用いた技術・理論
3. 主要な実験結果や理論的知見
4. 私の研究（赤外パルスによる状態制御）に関連する示唆
5. 本研究の限界と将来の展望
"""

def safe(name: str) -> str:
    return re.sub(r"[\\/*?\"<>|:]", "_", name)[:90] + ".pdf"

def matches_keywords(text: str) -> bool:
    return all(re.search(p, text, re.I) for p in KEY_PHRASES)

def fetch_candidates():
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": BASE_QUERY,
        "limit": CANDIDATE_N,
        "fields": "title,abstract,openAccessPdf"
    }
    return requests.get(url, params=params, timeout=30).json().get("data", [])

def download_and_filter():
    kept = 0
    for paper in fetch_candidates():
        if kept >= MAX_KEEP:
            break
        abs_txt = paper.get("abstract") or ""
        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "")

        if not pdf_url.lower().endswith(".pdf"):
            continue
        # if not matches_keywords(abs_txt):
        #     continue

        fname = safe(paper["title"])
        path  = PDF_DIR / fname
        try:
            pdf = requests.get(pdf_url, timeout=30).content
            path.write_bytes(pdf)
            kept += 1
            print(f"⬇️  {fname}")
        except Exception as e:
            print(f"[DL error] {fname}: {e}")
    print(f"✅ {kept} PDF downloaded → {PDF_DIR}")

# ---------- 2. 要約 ----------
def summarize_pdf(pdf_path: Path):
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        chain = load_summarize_chain(llm, chain_type="map_reduce")
        raw_summary = chain.run(split_docs)

        final = llm.predict(
            f"{PROMPT}\n\n--- 論文要約 ---\n{raw_summary}\n\n--- 出力 ---"
        )

        SUMMARY_DIR.mkdir(exist_ok=True)
        out_file = SUMMARY_DIR / f"{pdf_path.stem}_summary.txt"
        out_file.write_text(final, encoding="utf-8")
        print(f"\n=== {pdf_path.name} ===\n{final}\n[Saved] {out_file}")
    except Exception as e:
        print(f"[Summarize error] {pdf_path.name}: {e}")

def main():
    download_and_filter()
    for pdf in PDF_DIR.glob("*.pdf"):
        summarize_pdf(pdf)

if __name__ == "__main__":
    main()
