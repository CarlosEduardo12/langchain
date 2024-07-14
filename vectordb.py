from typing import List
from xml.dom.minidom import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_file(markdown_path: str) -> List[Document]:
    """
    Carrega um arquivo markdown e retorna seu conteúdo como uma lista de documentos.

    Args:
        markdown_path (str): O caminho para o arquivo markdown.

    Returns:
        List[Document]: Uma lista de documentos carregados do arquivo markdown.
    """
    loader = UnstructuredMarkdownLoader(markdown_path)
    document = loader.load()
    return document


def create_faiss_retriever(document: List[Document]) -> VectorStoreRetriever:  # type: ignore
    """
    Cria um retriever FAISS a partir de uma lista de documentos.

    Args:
        document (List[Dict[str, Any]]): Uma lista de documentos.

    Returns:
        VectorStoreRetriever: Um objeto retriever que pode ser usado para buscar informações nos documentos.
    """
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(document)

    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return retriever
