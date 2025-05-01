from typing import List
from langchain_huggingface import HuggingFaceEmbeddings


class LangChainEmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)
