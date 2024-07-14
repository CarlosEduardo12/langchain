from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAI
from langchain_core.vectorstores import VectorStoreRetriever


def answer_question_with_rag(pergunta: str, retriever: VectorStoreRetriever, llm: OpenAI) -> str:  # type: ignore
    """
    Usa a técnica de Recuperação-Augmentação-Geração (RAG) para responder a uma pergunta com base em documentos recuperados.

    Args:
        pergunta (str): A pergunta a ser respondida.

    Returns:
        str: A resposta gerada com base nos documentos recuperados.
    """
    prompt = ChatPromptTemplate.from_template(
        """Responda a pergunta apenas com base no contexto:{context} Pergunta: {input} """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    response = retriever_chain.invoke({"input": pergunta})
    return response["answer"]
