from typing import Any, Dict, List, Optional
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain.document_loaders.base import BaseLoader

class LangChainDatasetManager:
    def __init__(self, file_path: str, loader_type: str = "json", jq_schema: str = '.[]'):
        self.file_path = file_path
        self.loader_type = loader_type
        self.jq_schema = jq_schema
        self.loader = self._create_loader()
        self.documents = None
    
    def _create_loader(self) -> BaseLoader:
        if self.loader_type.lower() == 'json':
            return JSONLoader(file_path=self.file_path, jq_schema=self.jq_schema, text_content=False)
        elif self.loader_type.lower() == 'text':
            return TextLoader(file_path=self.file_path)
        else:
            raise ValueError(f"Unsupported loader type: {self.loader_type}")
    
    def load_documents(self) -> List[Document]:
        if self.documents is None:
            self.documents = self.loader.load()
        return self.documents
    
    def get_documents(self) -> List[Document]:
        return self.load_documents() if self.documents is None else self.documents
    
    def get_document(self, index: int) -> Optional[Document]:
        documents = self.get_documents()
        return documents[index] if 0 <= index < len(documents) else None
    
    def count_documents(self) -> int:
        return len(self.get_documents())
    
    def get_raw_data(self) -> List[Dict[str, Any]]:
        if self.loader_type.lower() != 'json':
            raise ValueError("Raw data can only be retrieved for JSON files.")
        
        with open(self.file_path, 'r') as file:
            return json.load(file)