import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma


DB_PATH = "db/chroma_db"
DOCS_PATH = "data/raw"


def load_documents(docs_path: str = DOCS_PATH):
    """Load all PDF files from data/raw using LangChain loaders."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,   # 1 Document / page
        show_progress=True,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}.")

    print(f"Loaded {len(documents)} documents (pages).")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Preview: {doc.page_content[:200]!r}")

    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks with overlap."""
    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(chunk.page_content[:300])
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory: str = DB_PATH):
    """Create and persist Chroma vector store using VoyageAI embeddings."""
    print("\nCreating embeddings and storing in ChromaDB...")

    # VoyageAIEmbeddings sẽ tự dùng VOYAGE_API_KEY từ environment
    embedding_model = VoyageAIEmbeddings(
        model="voyage-3-lite",  # model nhẹ, hợp lý cho free tier
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main(docs_path: str = DOCS_PATH, persist_directory: str = DB_PATH):
    """Main ingestion pipeline."""
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Nếu DB đã tồn tại thì load luôn, không ingest lại
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Vector store already exists. Loading existing store...")

        embedding_model = VoyageAIEmbeddings(
            model="voyage-3-lite",
        )

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("Loaded existing vector store.")
        return vectorstore

    print("No existing vector store. Building from raw PDFs...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persist_directory)

    print("\n Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG ingestion pipeline (LangChain-style).")
    parser.add_argument("--docs_path", type=str, default=DOCS_PATH)
    parser.add_argument("--persist_dir", type=str, default=DB_PATH)
    args = parser.parse_args()

    main(docs_path=args.docs_path, persist_directory=args.persist_dir)
