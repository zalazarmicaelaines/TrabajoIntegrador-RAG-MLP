import os
from pathlib import Path
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =========================
# Configuración
# =========================

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()


def run_ingest(docs_folder: str = None, index_folder: str = None) -> dict:
    """
    Procesa los PDFs de docs_folder, genera embeddings y guarda el índice FAISS.
    Retorna un dict con el resultado: {"ok": bool, "message": str, "chunks": int}
    """
    docs_path = Path(docs_folder) if docs_folder else ROOT_DIR / os.getenv("DOCUMENTS_PATH", "docs")
    index_path = Path(index_folder) if index_folder else ROOT_DIR / os.getenv("VECTOR_INDEX_PATH", "promtior_index")

    # --------------------------
    # Carga de PDFs
    # --------------------------
    pdf_files = list(docs_path.glob("*.pdf"))

    if not pdf_files:
        msg = f"No se encontraron PDFs en {docs_path}"
        logger.warning(msg)
        return {"ok": False, "message": msg, "chunks": 0}

    all_docs = []
    for pdf in pdf_files:
        try:
            logger.info(f"Cargando: {pdf.name}")
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_name"] = pdf.name
                doc.metadata["tema"] = pdf.stem  # nombre del archivo como tema
            logger.info(f"  → {len(docs)} páginas")
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error cargando {pdf.name}: {e}")

    if not all_docs:
        msg = "No se pudo cargar ningún documento."
        return {"ok": False, "message": msg, "chunks": 0}

    logger.info(f"Total páginas cargadas: {len(all_docs)}")

    # --------------------------
    # Chunking
    # --------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)
    logger.info(f"Chunks generados: {len(chunks)}")

    # --------------------------
    # Embeddings e índice FAISS
    # --------------------------
    logger.info("Generando embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --------------------------
    # Guardar índice
    # --------------------------
    index_path.mkdir(exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info(f"Índice guardado en: {index_path}")

    temas = sorted({doc.metadata.get("tema", "desconocido") for doc in all_docs})
    msg = f"Índice generado con {len(chunks)} chunks desde {len(pdf_files)} archivo(s): {', '.join(temas)}"
    return {"ok": True, "message": msg, "chunks": len(chunks), "temas": temas}


# Permite correrlo también directamente desde consola
if __name__ == "__main__":
    result = run_ingest()
    print(result["message"])
