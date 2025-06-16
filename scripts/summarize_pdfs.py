import sys
import os
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# .env から OPENAI_API_KEY を取得
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4o")
PDF_DIR = Path("pdf")

custom_prompt = """
あなたは、赤外レーザーによる分子の振動回転状態制御を研究する物理学者です。
以下の要約を読んで、次の5点を明確に日本語で示してください：

1. 研究背景と着眼点
2. アプローチや用いた技術・理論
3. 主要な実験結果や理論的知見
4. 私の研究（赤外パルスによる状態制御）に関連する示唆
5. 本研究の限界と将来の展望
"""


def summarize_pdf(pdf_path: Path):
    print(f"\n=== {pdf_path.name} ===")
    try:
        # --- 1. PDF 読み込みと分割 ---
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # --- 2. LangChain 標準要約（map_reduce）---
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        raw_summary = chain.run(split_docs)

        # --- 3. 研究文脈プロンプトで再要約 ---
        final_summary = llm.predict(f"{custom_prompt}\n\n--- 論文要約 ---\n{raw_summary}\n\n--- 出力 ---")

        # --- 4. 画面出力 & 保存 ---
        print(final_summary)
        out_file = Path("summaries") / f"{pdf_path.stem}_summary.txt"
        out_file.parent.mkdir(exist_ok=True)
        out_file.write_text(final_summary, encoding="utf-8")
        print(f"[Saved] {out_file}")

    except Exception as e:
        print(f"[Error] {pdf_path.name}: {e}")

def main():
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        summarize_pdf(pdf_file)

if __name__ == "__main__":
    main()
