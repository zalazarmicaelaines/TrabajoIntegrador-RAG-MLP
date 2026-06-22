import json
import re
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from rag import embeddings, INDEX_PATH, llm

logger = logging.getLogger(__name__)


# ======================
# Parseo del plan de estudios
# ======================

def parsear_plan(txt_path: str) -> list:
    """
    Parsea el archivo de plan de estudios y retorna lista de temas.
    Formato esperado: "Tema N Nombre del tema"
    """
    temas = []
    current_tema = None
    current_subtitulos = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            # Detectar línea de tema: "Tema 1 Nombre" o "Tema 1. Nombre"
            match = re.match(r"^Tema\s+(\d+)[\.\s]+(.+)$", linea, re.IGNORECASE)
            if match:
                if current_tema:
                    temas.append({
                        "numero": current_tema["numero"],
                        "nombre": current_tema["nombre"],
                        "subtitulos": current_subtitulos
                    })
                current_tema = {
                    "numero": int(match.group(1)),
                    "nombre": match.group(2).strip()
                }
                current_subtitulos = []
            else:
                if current_tema:
                    partes = [p.strip() for p in re.split(r"[.,]", linea) if p.strip()]
                    current_subtitulos.extend(partes)

    # Agregar el último tema
    if current_tema:
        temas.append({
            "numero": current_tema["numero"],
            "nombre": current_tema["nombre"],
            "subtitulos": current_subtitulos
        })

    return temas


def cargar_plan(plan_path: str) -> list:
    """Carga el plan de estudios desde el archivo txt."""
    path = Path(plan_path)
    if not path.exists():
        return []
    return parsear_plan(str(path))


# ======================
# Generador de preguntas
# ======================

def generar_preguntas(tema_nombre: str, subtema: str, cantidad_mc: int, cantidad_vf: int) -> list:
    """
    Recupera contexto del índice FAISS y genera preguntas con el LLM.
    Retorna lista de preguntas en formato dict.
    """

    # Cargar vectorstore
    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Construir query combinando tema y subtema
    query = f"{tema_nombre}: {subtema}" if subtema else tema_nombre
    docs = vectorstore.similarity_search(query, k=6)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    # Prompt de generación
    prompt = f"""Eres un profesor universitario de Inteligencia Artificial.
Basándote ÚNICAMENTE en el siguiente contexto, generá preguntas de evaluación académica.

TEMA: {tema_nombre}
SUBTEMA / FOCO: {subtema if subtema else "contenido general del tema"}

INSTRUCCIONES:
- Generá exactamente {cantidad_mc} preguntas de opción múltiple y {cantidad_vf} preguntas de verdadero/falso.
- Las preguntas deben evaluar comprensión real, no memorización literal.
- Cada opción múltiple debe tener exactamente 4 opciones (a, b, c, d).
- Indicá claramente la respuesta correcta.
- No uses información fuera del contexto provisto.
- Respondé ÚNICAMENTE con un JSON válido, sin texto adicional, sin markdown, sin explicaciones.

FORMATO JSON REQUERIDO:
{{
  "preguntas": [
    {{
      "tipo": "multiple_choice",
      "pregunta": "texto de la pregunta",
      "opciones": {{
        "a": "opción a",
        "b": "opción b",
        "c": "opción c",
        "d": "opción d"
      }},
      "respuesta_correcta": "a"
    }},
    {{
      "tipo": "verdadero_falso",
      "pregunta": "texto de la pregunta",
      "respuesta_correcta": "verdadero"
    }}
  ]
}}

CONTEXTO:
{contexto}

JSON:"""

    logger.info(f"Generando preguntas para: {query}")
    response = llm.invoke(prompt)

    # Extraer texto de la respuesta
    texto = response.content if hasattr(response, "content") else str(response)

    # Limpiar posibles markdown fences
    texto = re.sub(r"```json|```", "", texto).strip()

    # Parsear JSON
    data = json.loads(texto)
    preguntas = data.get("preguntas", [])

    logger.info(f"Preguntas generadas: {len(preguntas)}")
    return preguntas
