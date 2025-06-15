"""対話型 CLI - Ctrl+C で終了"""
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
