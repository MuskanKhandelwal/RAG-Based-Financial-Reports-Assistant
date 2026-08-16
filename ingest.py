import argparse
import os
import re
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from embeddings import get_embedding_function
from metadata import metadata_for_documents
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def normalize_whitespace(documents: list[Document]):
    """Collapse the tabs and runs of spaces PyPDF emits between words.

    Extracted pages arrive as "we\tproduced\t1,845,985\tvehicles", which is hard
    for the LLM to read. Newlines are preserved so the splitter can still break
    on paragraph boundaries.
    """
    for doc in documents:
        text = doc.page_content.replace("\t", " ")
        text = re.sub(r"[  ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        doc.page_content = text.strip()
    return documents


def split_documents(documents: list[Document]):
    documents = normalize_whitespace(documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return add_filing_metadata(chunks, metadata_for_documents(documents))


def add_filing_metadata(chunks: list[Document], metadata_by_source: dict):
    """Stamp company / ticker / form type / fiscal year onto every chunk.

    Without this, chunks from different issuers and years are indistinguishable
    at query time and a question about one company can be answered from
    another's filing.

    :param chunks: Chunks produced by the text splitter.
    :param metadata_by_source: Filing metadata keyed by source path.
    :return: The same chunks, with filing metadata merged in.
    """
    for chunk in chunks:
        filing_metadata = metadata_by_source.get(chunk.metadata.get("source"))
        if filing_metadata:
            chunk.metadata.update(filing_metadata)
    return chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()