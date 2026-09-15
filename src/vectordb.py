import os
from langchain_community.vectorstores import Chroma
from logger import logger

def create_or_load_vectordb(chunks, embedding_fn, persist_directory: str = "./chroma_db"):
    """Stores chunks in local Chroma vector database or loads existing db."""
    abs_persist_dir = os.path.abspath(persist_directory)
    logger.info(f"Accessing Vector DB at path: {abs_persist_dir}")
    if chunks:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            persist_directory=persist_directory
        )
        logger.info("Successfully persisted document embeddings into Chroma DB.")
    else:
        logger.info("No new chunks provided. Loading existing Chroma Vector DB...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_fn
        )
    return vectorstore.as_retriever(search_kwargs={"k": 2})