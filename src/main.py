import os
from dotenv import load_dotenv
from logger import logger
from chunks import load_and_chunk_docs
from embedding import get_embedding_function
from vectordb import create_or_load_vectordb
from questions import build_rag_chain


def main():
    logger.info("=== Starting Ollama RAG Interactive Application ===")

    load_dotenv()
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    try:
        # Step 1: Load & Chunk Documents
        chunks = load_and_chunk_docs(data_folder)

        # Step 2: Initialize Vector DB & Embeddings
        embeddings = get_embedding_function(model_name="nomic-embed-text")
        chroma_db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
        retriever = create_or_load_vectordb(chunks, embeddings, persist_directory=chroma_db_dir)

        # Step 3: Build RAG Chain
        rag_chain = build_rag_chain(retriever, model_name=model_name)

        logger.info("=== Interactive Session Ready ===")
        print("\n" + "=" * 50)
        print(" RAG System Ready! Type your question below.")
        print(" Type 'exit' or 'quit' to stop.")
        print("=" * 50 + "\n")

        # Step 4: Dynamic Terminal Input Loop
        while True:
            try:
                user_query = input("Ask a question: ").strip()

                if not user_query:
                    continue

                if user_query.lower() in ["exit", "quit", "q"]:
                    print("\nExiting application. Goodbye!")
                    logger.info("User ended interactive session.")
                    break

                logger.info(f"User Query Received: '{user_query}'")

                # Retrieve relevant chunk IDs for logging
                relevant_docs = retriever.invoke(user_query)
                retrieved_ids = [doc.metadata.get("chunk_id", "N/A") for doc in relevant_docs]
                logger.info(f"Retrieved Chunk IDs: {retrieved_ids}")

                # Execute RAG pipeline
                response = rag_chain.invoke(user_query)
                logger.info(f"Response Generated for Query: '{user_query}'")

                # Display Results
                print("\n" + "-" * 40)
                print(f"[Retrieved Chunks]: {', '.join(retrieved_ids)}")
                print(f"[Answer]: {response.strip()}")
                print("-" * 40 + "\n")

            except KeyboardInterrupt:
                print("\nSession interrupted. Exiting...")
                logger.info("Session interrupted via KeyboardInterrupt.")
                break

    except Exception as e:
        logger.error(f"Execution failed due to error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()