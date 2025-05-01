from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    text = text.lower()

    return text


def chunk_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)


def load_text_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path)
        documents.extend(loader.load())

    return documents


def load_json_documents(file_path: str, jq_schema: str = ".[]") -> List[Document]:
    loader = JSONLoader(file_path=file_path, jq_schema=jq_schema, text_content=False)

    return loader.load()


def preprocess_dataset(
    file_path: str,
    is_json: bool = True,
    jq_schema: str = ".[]",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    if is_json:
        documents = load_json_documents(file_path, jq_schema)
    else:
        documents = load_text_documents([file_path])

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    chunked_documents = chunk_documents(documents, chunk_size, chunk_overlap)

    return chunked_documents
