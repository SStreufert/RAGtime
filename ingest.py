import time
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import settings
import tracker

DATA_DIR = settings.data_dir
CHROMA_DIR = settings.chroma_dir
EMBED_MODEL = settings.embedding_model
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap


def load_documents(folder=DATA_DIR):
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"No such folder: {folder_path}")

    pdfs = sorted(folder_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder_path}")

    docs = []
    for pdf_path in pdfs:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.replace("\n", "")
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": page_number},
                )
            )
        print(f"{pdf_path.name}: loaded")
    return docs


def main():
    docs = load_documents()

    from healthcheck import ensure_or_exit

    ensure_or_exit(("embeddings",))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    print("embedding and indexing...")
    start = time.perf_counter()
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model=EMBED_MODEL,
            base_url=settings.embeddings_base_url,
            api_key=settings.embeddings_api_key,
            check_embedding_ctx_length=False,
        ),
        persist_directory=CHROMA_DIR,
    )
    elapsed = time.perf_counter() - start
    print(f"indexed into {Path(CHROMA_DIR).resolve()}")

    if tracker.is_available():
        with tracker.start_run(run_name="ingestion") as run:
            if run:
                tracker.log_ingestion(
                    docs=docs,
                    chunks=chunks,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    embedding_model=EMBED_MODEL,
                    ingestion_time=elapsed,
                )
                print(f"MLflow run: {run.info.run_id}")
    else:
        print("MLflow not available — skipping tracking")


if __name__ == "__main__":
    main()
