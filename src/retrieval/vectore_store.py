from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class LangChainVectorStore:
    def __init__(
        self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None

    def create_from_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> Chroma:
        self.vector_store = Chroma.from_texts(
            texts, self.embeddings, metadatas=metadatas
        )
        return self.vector_store

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        self.vector_store = Chroma.from_documents(documents, self.embeddings)
        return self.vector_store

    def add_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        if self.vector_store is None:
            return self.create_from_texts(texts, metadatas)

        ids = self.vector_store.add_texts(texts, metadatas=metadatas)
        return ids

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("Vector store not created yet.")

        docs = self.vector_store.similarity_search(query, k=k)
        return docs

    def save_local(self, folder_path: str) -> None:
        if self.vector_store is None:
            raise ValueError("Vector store not created yet.")

        self.vector_store.save(folder_path)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: Embeddings = None
    ) -> "LangChainVectorStore":
        instance = cls()
        if embeddings is not None:
            instance.embeddings = embeddings

        instance.vector_store = Chroma.load(folder_path, instance.embeddings)
        return instance
