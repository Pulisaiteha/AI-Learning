import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger


def load_and_chunk_docs(data_dir: str):
    """Loads text files from data_dir and splits them into chunks with detailed logging."""
    logger.info(f"Loading documents from directory: {data_dir}")

    if not os.path.exists(data_dir):
        logger.error(f"Directory non-existent: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    raw_documents = loader.load()
    logger.info(f"Successfully loaded {len(raw_documents)} raw document(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=15
    )

    all_chunks = []

    # Process each document individually for file-level logging
    for doc_idx, doc in enumerate(raw_documents):
        source_file = os.path.basename(doc.metadata.get("source", f"doc_{doc_idx + 1}"))
        file_char_count = len(doc.page_content)

        # Split the current document
        file_chunks = text_splitter.split_documents([doc])

        logger.info(
            f"--- [File {doc_idx + 1}/{len(raw_documents)}] Source: '{source_file}' | Total Characters: {file_char_count} | Chunks Created: {len(file_chunks)} ---")

        # Assign explicit Chunk IDs and log detailed chunk properties
        for chunk_idx, chunk in enumerate(file_chunks):
            chunk_id = f"{source_file}_chunk_{chunk_idx + 1}"
            chunk.metadata["chunk_id"] = chunk_id
            chunk_char_count = len(chunk.page_content)

            # Clean single-line snippet for logging readability
            snippet = chunk.page_content.replace("\n", " ")[:60]

            logger.info(
                f" -> [ID: {chunk_id}] | Chars: {chunk_char_count} | Snippet: \"{snippet}...\""
            )

        all_chunks.extend(file_chunks)

    logger.info(f"Total chunks created across all files: {len(all_chunks)}")
    return all_chunks