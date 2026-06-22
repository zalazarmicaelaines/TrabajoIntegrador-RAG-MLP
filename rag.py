import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq


# ======================
# Configuración
# ======================

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
INDEX_PATH = ROOT_DIR / os.getenv("VECTOR_INDEX_PATH", "promtior_index")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ======================
# Embeddings y LLM
# (se inicializan al importar, no dependen del índice)
# ======================

logger.info("Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key="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"
)
logger.info("LLM cargado correctamente.")


# ======================
# Verificación del índice
# ======================

def index_exists() -> bool:
    """Verifica si el índice FAISS está disponible."""
    return (INDEX_PATH / "index.faiss").exists() and (INDEX_PATH / "index.pkl").exists()


# ======================
# Prompt del tutor
# ======================

prompt_template = """Eres un tutor universitario especializado en Inteligencia Artificial.
Tu objetivo es que el estudiante comprenda los conceptos con claridad y pueda aplicarlos.

ESTILO DE ENSEÑANZA:
- Respondé siempre en español, con lenguaje claro y accesible.
- Comenzá con una explicación directa del concepto consultado.
- Siempre incluí al menos un ejemplo concreto o analogía en tu respuesta, tomándolo del contexto si existe o construyendo uno simple si no.
- Usá analogías simples cuando el tema sea abstracto.
- Respondé con entre 5 y 8 oraciones, siempre incluyendo un ejemplo o analogía.
- Si el estudiante pide profundizar, extendé la explicación con más detalle y ejemplos.
- Si el estudiante no entendió, reformulá el concepto con palabras más simples y una analogía diferente.

LÍMITES:
- Usá únicamente la información del contexto provisto.
- Si el tema no está en el contexto, respondé exactamente:
  "Ese tema no está en el material disponible. ¿Querés que busquemos algo relacionado?"
- No inventes definiciones, fórmulas ni autores.

CONTEXTO DEL MATERIAL:
{context}

HISTORIAL DE LA CONVERSACIÓN:
{chat_history}

PREGUNTA DEL ESTUDIANTE:
{question}

RESPUESTA DEL TUTOR:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)


# ======================
# Creación de la cadena RAG
# (carga lazy: solo cuando se llama esta función)
# ======================

def create_rag_chain():
    """
    Carga el índice FAISS y crea la cadena conversacional.
    Lanza ValueError si el índice no existe todavía.
    """
    if not index_exists():
        raise ValueError(
            "El índice FAISS no existe todavía. "
            "Andá a ⚙️ Configuración, cargá tus PDFs y ejecutá la ingestión."
        )

    logger.info("Cargando índice FAISS...")
    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("Índice cargado correctamente.")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )

    return chain
