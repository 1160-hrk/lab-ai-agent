"""対話型 CLI - Ctrl+C で終了"""
import sys
import os
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

from src.rag_pipeline import RagPipeline

def main():
    rag = RagPipeline()
    print("RAG agent ready. Type your question:")
    while True:
        try:
            q = input("> ")
            print(rag.ask(q))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
