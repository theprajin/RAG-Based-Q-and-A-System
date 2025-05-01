from typing import List
from langchain_core.documents import Document


class LangChainRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        relevant_docs = self.vector_store.similarity_search(query, k=top_k)

        return relevant_docs
