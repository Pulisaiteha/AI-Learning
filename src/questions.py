from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from logger import logger

def build_rag_chain(retriever, model_name: str = "llama3.2"):
    """Constructs the RAG pipeline combining retriever, prompt, and local Ollama model."""
    logger.info(f"Building RAG chain with Ollama LLM model: '{model_name}'")
    prompt_template = """
    Answer the following question based ONLY on the provided context:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model=model_name, temperature=0)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain pipeline successfully constructed.")
    return rag_chain