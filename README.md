# Projeto de Recuperação-Augmentação-Geração (RAG) com LangChain

Este projeto demonstra como usar a técnica de Recuperação-Augmentação-Geração (RAG) para responder a perguntas com base em documentos recuperados. Utiliza a biblioteca LangChain para carregar documentos, criar embeddings, e usar o modelo de linguagem OpenAI para gerar respostas.

## Estrutura do Projeto

- `load_file`: Carrega um arquivo markdown e retorna seu conteúdo como uma lista de documentos.
- `create_faiss_retriever`: Cria um retriever FAISS a partir de uma lista de documentos.
- `answer_question_with_rag`: Usa a técnica de RAG para responder a uma pergunta com base em documentos recuperados.

## Requisitos

- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2. Crie e ative um ambiente virtual:

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

4. Crie um arquivo `.env` na raiz do projeto e adicione suas credenciais da OpenAI:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Uso

1. Coloque seu arquivo markdown no diretório do projeto. Por exemplo, `teste.md`.

2. Execute o script principal:

    ```bash
    python main.py
    ```

3. O script irá carregar o documento, criar um retriever FAISS, e usar a técnica de RAG para responder à pergunta definida no código.
