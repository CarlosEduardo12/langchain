from langchain_openai import OpenAI
from dotenv import load_dotenv

from rag import answer_question_with_rag
from vectordb import create_faiss_retriever, load_file

load_dotenv()
llm = OpenAI()

if __name__ == "__main__":
    # Define a pergunta que será respondida
    pergunta = "como conseguiu meu numero?"
    # Carrega o conteúdo do arquivo markdown "teste.md"
    document = load_file("teste.md")
    # Cria um retriever FAISS a partir do documento carregado
    retriever = create_faiss_retriever(document)
    # Usa a técnica de RAG para responder à pergunta com base nos documentos recuperados
    resposta = answer_question_with_rag(pergunta, retriever, llm)
    print(resposta)
