"""簡易 RAG パイプライン"""
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQA
from .config import OPENAI_API_KEY, CHROMA_PERSIST_PATH

class RagPipeline:
    def __init__(self, model="gpt-4o-mini", k=4):
        self.embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.store = Chroma(persist_directory=str(CHROMA_PERSIST_PATH), embedding_function=self.embed)
        self.llm   = ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY)
        prompt = PromptTemplate(
            input_variables=["context","question"],
            template="あなたは研究支援 AI です。\n### 参考文書\n{context}\n### 質問\n{question}\n### 回答（日本語で簡潔に）"
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.store.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, q: str) -> str:
        return self.chain.run(q)
