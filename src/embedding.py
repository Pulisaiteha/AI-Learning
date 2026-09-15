from langchain_ollama import OllamaEmbeddings
from logger import logger

def get_embedding_function(model_name: str = "nomic-embed-text"):
    """Returns the Ollama embeddings generator."""
    logger.info(f"Initializing OllamaEmbeddings with model: '{model_name}'")
    return OllamaEmbeddings(model=model_name)