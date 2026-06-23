import os
import json
from pathlib import Path
import streamlit as st
from rag import create_rag_chain, index_exists

# ======================
# CONFIGURACIÓN GENERAL
# ======================

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)
PLAN_PATH = ROOT_DIR / "plan_estudios.txt"
PREGUNTAS_PATH = ROOT_DIR / "preguntas_generadas.json"
RESULTADOS_PATH = ROOT_DIR / "resultados.json"

st.set_page_config(
    page_title="Tutor Inteligente IA",
    page_icon="📚",
    layout="wide"
)

# ======================
# ESTILOS
# ======================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@600&display=swap');

    html, body, .stApp {
        background-color: #f0f4f9;
        color: #1c2b3a;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #1c2b3a;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #ccd6e0 !important;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }

    h1 { font-family: 'Lora', serif; color: #1c2b3a; }
    h2, h3 { color: #2e4057; }

    .stButton > button {
        background-color: #2e4057;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #3d5a80;
    }

    .action-btn > button {
        background-color: #ffffff !important;
        color: #2e4057 !important;
        border: 1.5px solid #2e4057 !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        padding: 0.3rem 0.9rem !important;
    }
    .action-btn > button:hover {
        background-color: #e8eef5 !important;
    }

    .welcome-box {
        background: linear-gradient(135deg, #2e4057 0%, #3d5a80 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .welcome-box strong { font-size: 1.05rem; }

    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #3d5a80;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    [data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .stAlert { border-radius: 8px; }
    .stChatMessage { border-radius: 10px; margin-bottom: 0.4rem; }

    .pregunta-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-left: 4px solid #3d5a80;
    }
</style>
""", unsafe_allow_html=True)


# ======================
# HELPERS
# ======================

def plan_cargado() -> bool:
    return PLAN_PATH.exists()


def cargar_temas() -> list:
    return [
        {
            "numero": 1,
            "nombre": "Introducción a la Inteligencia Artificial",
            "subtitulos": ["Definición de IA", "Evolución histórica y perspectivas de la IA",
                          "Aplicaciones de la IA", "Sistemas Inteligentes", "Ingeniería del Conocimiento",
                          "Representación", "Búsqueda", "Planificación", "Aprendizaje",
                          "Disciplinas y subdisciplinas en IA", "Ética e IA"]
        },
        {
            "numero": 2,
            "nombre": "Agentes inteligentes",
            "subtitulos": ["Agentes inteligentes", "Definición", "Autonomía", "Representación",
                          "Tipos de agentes inteligentes", "Estructura de los agentes inteligentes",
                          "Arquitecturas", "Sistemas multiagentes"]
        },
        {
            "numero": 3,
            "nombre": "Inteligencia artificial simbólica",
            "subtitulos": ["Sistemas expertos", "Sistemas basados en reglas",
                          "Mecanismos de inferencia", "Métodos", "Herramientas", "Aplicaciones"]
        },
        {
            "numero": 4,
            "nombre": "Inteligencia artificial no simbólica",
            "subtitulos": ["Algoritmos de aprendizaje automático", "Redes neuronales artificiales",
                          "Métodos", "Herramientas", "Aplicaciones"]
        },
        {
            "numero": 5,
            "nombre": "Tecnologías emergentes en IA simbólica y no simbólica",
            "subtitulos": ["Tecnologías emergentes", "Métodos", "Herramientas",
                          "Aplicaciones", "Estudios de casos"]
        },
    ]

def guardar_preguntas(tema: str, subtema: str, preguntas: list):
    """Guarda las preguntas generadas en preguntas_generadas.json"""
    import datetime
    registro = {
        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tema": tema,
        "subtema": subtema,
        "preguntas": preguntas
    }

    historial = []
    if PREGUNTAS_PATH.exists():
        with open(PREGUNTAS_PATH, "r", encoding="utf-8") as f:
            try:
                historial = json.load(f)
            except Exception:
                historial = []

    historial.append(registro)

    with open(PREGUNTAS_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)


def guardar_resultado(tema: str, subtema: str, total: int, correctas: int):
    """Guarda el resultado del examen en resultados.json"""
    import datetime
    resultado = {
        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tema": tema,
        "subtema": subtema,
        "total": total,
        "correctas": correctas,
        "incorrectas": total - correctas,
        "porcentaje": round((correctas / total) * 100, 1) if total > 0 else 0
    }

    resultados = []
    if RESULTADOS_PATH.exists():
        with open(RESULTADOS_PATH, "r", encoding="utf-8") as f:
            try:
                resultados = json.load(f)
            except Exception:
                resultados = []

    resultados.append(resultado)

    with open(RESULTADOS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)


# ======================
# MENÚ LATERAL
# ======================

with st.sidebar:
    st.markdown("## 📚 Tutor IA")
    menu = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "💬 Tutor (RAG)",
            "🧠 Generador de Preguntas",
            "📝 Examen",
            "📊 Resultados",
            "⚙️ Configuración"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("TRABAJO INTEGRADOR - INTELIGENCIA ARTIFICIAL 2026 UNNE")


# ======================
# INICIO
# ======================

if menu == "🏠 Inicio":
    st.title("📚 Tutor Inteligente de Inteligencia Artificial")
    st.markdown("Sistema educativo híbrido que combina recuperación de información y redes neuronales para personalizar tu aprendizaje.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="feature-card">
            <strong>💬 Tutor con RAG</strong><br>
            Conversá con el material de estudio. El tutor recupera información relevante y te explica cada concepto de forma clara.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="feature-card">
            <strong>📝 Examen adaptativo</strong><br>
            Respondé preguntas generadas automáticamente desde el material. El sistema evalúa tu nivel por tema.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="feature-card">
            <strong>🧠 Generador de preguntas</strong><br>
            El LLM analiza el documento y genera preguntas de opción múltiple y verdadero/falso por tema.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="feature-card">
            <strong>📊 Diagnóstico de aprendizaje</strong><br>
            Una red neuronal (MLP) analiza tus resultados y te indica qué temas reforzar.
        </div>""", unsafe_allow_html=True)


# ======================
# TUTOR (RAG)
# ======================

elif menu == "💬 Tutor (RAG)":
    st.title("💬 Tutor Inteligente")

    pdfs_actuales = list(DOCS_DIR.glob("*.pdf"))
    if not index_exists() or not pdfs_actuales:
        st.error("⚠️ No hay documentos cargados.")
        st.info("Andá a **⚙️ Configuración**, subí tus PDFs y ejecutá la ingestión.")
        st.stop()

    if "rag_chain" not in st.session_state:
        with st.spinner("Cargando base de conocimiento..."):
            try:
                st.session_state.rag_chain = create_rag_chain()
            except ValueError as e:
                st.error(str(e))
                st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_question" not in st.session_state:
        st.session_state.last_question = None

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-box">
            👋 <strong>¡Hola! Soy tu tutor de Inteligencia Artificial.</strong><br>
            Podés preguntarme sobre cualquier tema del material cargado: redes neuronales,
            algoritmos de búsqueda, lógica, aprendizaje automático y más.<br><br>
            💡 <em>Tip: si querés que profundice en algo o que te lo explique de otra manera,
            usá los botones que aparecen después de cada respuesta.</em>
        </div>
        """, unsafe_allow_html=True)

    col_title, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🗑️ Limpiar", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.last_question = None
            st.session_state.rag_chain = create_rag_chain()
            st.rerun()

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.write(item["a"])

    if st.session_state.chat_history:
        st.markdown("**¿Qué querés hacer?**")
        col1, col2, col3 = st.columns([2, 2, 6])

        with col1:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            profundizar = st.button("🔍 Profundizar", key="btn_profundizar")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            no_entendi = st.button("🔄 No entendí", key="btn_no_entendi")
            st.markdown('</div>', unsafe_allow_html=True)

        if profundizar and st.session_state.last_question:
            pregunta_accion = f"Profundizá más en lo que explicaste sobre: {st.session_state.last_question}"
            with st.spinner("Ampliando la explicación..."):
                response = st.session_state.rag_chain.invoke({"question": pregunta_accion})
                answer = response["answer"]
            st.session_state.chat_history.append({"q": "🔍 Quiero profundizar en ese tema.", "a": answer})
            st.rerun()

        if no_entendi and st.session_state.last_question:
            pregunta_accion = f"No entendí la explicación anterior sobre: {st.session_state.last_question}. Explicalo de otra manera, con una analogía simple y diferente."
            with st.spinner("Reformulando la explicación..."):
                response = st.session_state.rag_chain.invoke({"question": pregunta_accion})
                answer = response["answer"]
            st.session_state.chat_history.append({"q": "🔄 No entendí, explicalo de otra manera.", "a": answer})
            st.rerun()

    question = st.chat_input("Escribí tu pregunta sobre el material...")

    if question:
        with st.spinner("Consultando el material..."):
            response = st.session_state.rag_chain.invoke({"question": question})
            answer = response["answer"]

        st.session_state.last_question = question
        st.session_state.chat_history.append({"q": question, "a": answer})
        st.rerun()


# ======================
# GENERADOR DE PREGUNTAS
# ======================

elif menu == "🧠 Generador de Preguntas":
    st.title("🧠 Generador de Preguntas")
    st.markdown("Elegí el tema, especificá el subtema y generá tu evaluación personalizada.")
    st.markdown("---")

    if not index_exists():
        st.error("⚠️ No hay base de conocimiento cargada.")
        st.info("Andá a **⚙️ Configuración** y ejecutá la ingestión primero.")
        st.stop()

    temas = cargar_temas()
    opciones_temas = [f"Tema {t['numero']} - {t['nombre']}" for t in temas]

    tema_seleccionado = st.selectbox("📚 Seleccioná el tema del programa", opciones_temas)
    subtema = st.text_input(
        "🔍 ¿Sobre qué querés que te pregunten?",
        placeholder="ej: backpropagation, redes convolucionales, algoritmo A*..."
    )

    col1, col2 = st.columns(2)
    with col1:
        cantidad_mc = st.number_input("Preguntas opción múltiple", min_value=1, max_value=10, value=3)
    with col2:
        cantidad_vf = st.number_input("Preguntas verdadero/falso", min_value=1, max_value=10, value=2)

    if st.button("⚡ Generar y comenzar examen"):
        if not subtema.strip():
            st.warning("Escribí un subtema o concepto específico para enfocar las preguntas.")
        else:
            with st.spinner("Generando preguntas desde el material... ⏳"):
                try:
                    from generador import generar_preguntas

                    # Extraer nombre del tema seleccionado
                    idx = opciones_temas.index(tema_seleccionado)
                    tema_nombre = temas[idx]["nombre"]

                    preguntas = generar_preguntas(
                        tema_nombre=tema_nombre,
                        subtema=subtema,
                        cantidad_mc=int(cantidad_mc),
                        cantidad_vf=int(cantidad_vf)
                    )

                    guardar_preguntas(tema_seleccionado, subtema, preguntas)

                    if preguntas:
                        # Guardar en session_state y redirigir al examen
                        st.session_state.preguntas_examen = preguntas
                        st.session_state.tema_examen = tema_seleccionado
                        st.session_state.subtema_examen = subtema
                        st.session_state.examen_activo = True
                        st.session_state.respuestas_alumno = {}
                        st.success(f"✅ {len(preguntas)} preguntas generadas. ¡Andá a 📝 Examen!")
                    else:
                        st.error("No se pudieron generar preguntas. Intentá con un subtema diferente.")

                except Exception as e:
                    st.error(f"Error al generar preguntas: {e}")


# ======================
# EXAMEN
# ======================

elif menu == "📝 Examen":
    st.title("📝 Evaluación")

    if "preguntas_examen" not in st.session_state or not st.session_state.get("examen_activo"):
        st.info("Todavía no generaste preguntas. Andá a **🧠 Generador de Preguntas** primero.")
        st.stop()

    preguntas = st.session_state.preguntas_examen
    tema = st.session_state.get("tema_examen", "")
    subtema = st.session_state.get("subtema_examen", "")

    st.markdown(f"**Tema:** {tema}  |  **Subtema:** {subtema}")
    st.markdown(f"**Total de preguntas:** {len(preguntas)}")
    st.markdown("---")

    if "respuestas_alumno" not in st.session_state:
        st.session_state.respuestas_alumno = {}

    # Mostrar preguntas
    for i, pregunta in enumerate(preguntas):
        st.markdown(f"**{i+1}. {pregunta['pregunta']}**")

        if pregunta["tipo"] == "multiple_choice":
            opciones = pregunta.get("opciones", {})
            opciones_lista = [f"{k}) {v}" for k, v in opciones.items()]
            respuesta = st.radio(
                "Seleccioná una opción:",
                opciones_lista,
                key=f"pregunta_{i}",
                label_visibility="collapsed"
            )
            if respuesta:
                # Guardar solo la letra (a, b, c, d)
                st.session_state.respuestas_alumno[i] = respuesta[0]

        elif pregunta["tipo"] == "verdadero_falso":
            respuesta = st.radio(
                "Seleccioná:",
                ["Verdadero", "Falso"],
                key=f"pregunta_{i}",
                label_visibility="collapsed"
            )
            if respuesta:
                st.session_state.respuestas_alumno[i] = respuesta.lower()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("✅ Enviar respuestas"):
        if len(st.session_state.respuestas_alumno) < len(preguntas):
            st.warning("Respondé todas las preguntas antes de enviar.")
        else:
            # Corregir
            correctas = 0
            resultados_detalle = []

            for i, pregunta in enumerate(preguntas):
                respuesta_alumno = st.session_state.respuestas_alumno.get(i, "")
                respuesta_correcta = pregunta.get("respuesta_correcta", "").lower().strip()
                es_correcta = respuesta_alumno.lower().strip() == respuesta_correcta

                if es_correcta:
                    correctas += 1

                resultados_detalle.append({
                    "pregunta": pregunta["pregunta"],
                    "respuesta_alumno": respuesta_alumno,
                    "respuesta_correcta": respuesta_correcta,
                    "es_correcta": es_correcta
                })

            porcentaje = round((correctas / len(preguntas)) * 100, 1)

            # Guardar resultado
            guardar_resultado(tema, subtema, len(preguntas), correctas)

            # Guardar detalle en session_state para mostrar
            st.session_state.resultado_examen = {
                "correctas": correctas,
                "total": len(preguntas),
                "porcentaje": porcentaje,
                "detalle": resultados_detalle
            }
            st.session_state.examen_activo = False
            st.rerun()

    # Mostrar resultado si ya fue enviado
    if "resultado_examen" in st.session_state and not st.session_state.get("examen_activo"):
        resultado = st.session_state.resultado_examen
        st.markdown("---")
        st.subheader("📊 Resultado")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correctas", f"{resultado['correctas']} / {resultado['total']}")
        with col2:
            st.metric("Porcentaje", f"{resultado['porcentaje']}%")
        with col3:
            if resultado['porcentaje'] >= 70:
                st.metric("Nivel", "✅ Aprobado")
            else:
                st.metric("Nivel", "⚠️ A reforzar")

        st.markdown("---")
        st.subheader("Detalle por pregunta")
        for i, d in enumerate(resultado["detalle"]):
            icono = "✅" if d["es_correcta"] else "❌"
            with st.expander(f"{icono} Pregunta {i+1}"):
                st.write(f"**{d['pregunta']}**")
                st.write(f"Tu respuesta: `{d['respuesta_alumno']}`")
                if not d["es_correcta"]:
                    st.write(f"Respuesta correcta: `{d['respuesta_correcta']}`")

        if st.button("🔄 Nuevo examen"):
            del st.session_state.resultado_examen
            del st.session_state.preguntas_examen
            st.rerun()


# ======================
# RESULTADOS (MLP)
# ======================

elif menu == "📊 Resultados":
    st.title("📊 Diagnóstico de Aprendizaje")
    st.markdown("Análisis de tu rendimiento basado en la red neuronal MLP.")
    st.markdown("---")

    from mlp import (
        modelo_disponible, obtener_scores_por_tema,
        predecir_nivel_general, TEMAS, EXPERIMENTOS_PATH
    )

    if not RESULTADOS_PATH.exists():
        st.info("Todavía no hay resultados guardados. Completá al menos un examen primero.")
        st.stop()

    with open(RESULTADOS_PATH, "r", encoding="utf-8") as f:
        resultados = json.load(f)

    if not resultados:
        st.info("No hay resultados registrados aún.")
        st.stop()

    # Métricas generales
    total_examenes = len(resultados)
    promedio = round(sum(r["porcentaje"] for r in resultados) / total_examenes, 1)
    ultimo = resultados[-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Exámenes realizados", total_examenes)
    with col2:
        st.metric("Promedio general", f"{promedio}%")
    with col3:
        st.metric("Último examen", f"{ultimo['porcentaje']}%")

    st.markdown("---")

    st.subheader("📋 Historial de exámenes")
    with st.container(height=300):
        for r in reversed(resultados):
            icono = "✅" if r["porcentaje"] >= 70 else "⚠️"
            st.markdown(f"{icono} **{r['fecha']}** — {r['tema']} › {r['subtema']} — {r['correctas']}/{r['total']} ({r['porcentaje']}%)")

    st.markdown("---")

    # Verificar si hay scores de todos los temas
    scores_por_tema = obtener_scores_por_tema()
    temas_faltantes = [t for t in TEMAS if t not in scores_por_tema]

    if temas_faltantes:
        st.warning(f"Para generar el diagnóstico completo necesitás rendir exámenes de todos los temas. Te faltan: **{', '.join(temas_faltantes)}**")
    else:
        # Diagnóstico MLP
        if not modelo_disponible():
            st.info("El modelo MLP no está entrenado todavía. Andá a **⚙️ Configuración** y entrenalo.")
        else:
            st.subheader("🧠 Diagnóstico del MLP")

            try:
                diagnostico = predecir_nivel_general(scores_por_tema)

                col1, col2 = st.columns(2)
                with col1:
                    MAPA_INV = {0: "bajo", 1: "medio", 2: "alto"}

                    nivel = diagnostico["nivel_general"]

                    if isinstance(nivel, str):
                        nivel_nombre = nivel
                    else:
                        nivel_nombre = MAPA_INV.get(int(nivel), "desconocido")
                    icono_nivel = "🟢" if nivel_nombre == "alto" else "🟡" if nivel_nombre == "medio" else "🔴"
                    st.metric("Nivel general", f"{icono_nivel} {nivel_nombre.capitalize()}")
                with col2:
                    st.metric("Probabilidad de aprobar", f"{diagnostico['prob_aprobar']}%")

                st.markdown("---")
                st.subheader("📌 Diagnóstico por subtema")

                with st.container(height=250):
                    for item in diagnostico["diagnostico_subtemas"]:
                        nivel_sub = item["nivel"]
                        if nivel_sub == "alto":
                            icono = "✔"
                            color = "🟢"
                            texto = "dominio alto"
                        elif nivel_sub == "medio":
                            icono = "✔"
                            color = "🟡"
                            texto = "dominio medio"
                        else:
                            icono = "✘"
                            color = "🔴"
                            texto = "reforzar"

                        st.markdown(f"{color} **{item['subtema'].capitalize()}**: {texto} ({item['promedio']}%)")

                # Recomendación
                a_reforzar = [i["subtema"] for i in diagnostico["diagnostico_subtemas"] if i["nivel"] == "bajo"]
                if a_reforzar:
                    st.markdown("---")
                    st.info(f"📚 **Recomendación:** Repasar {', '.join(a_reforzar)} antes de rendir.")

            except Exception as e:
                st.error(f"Error al generar diagnóstico: {e}")

    # Experimentos
    if EXPERIMENTOS_PATH.exists():
        st.markdown("---")
        st.subheader("🔬 Resultados de experimentos")

        with open(EXPERIMENTOS_PATH, "r", encoding="utf-8") as f:
            experimentos = json.load(f)

        with st.expander("Experimento 1 — Arquitectura de la red"):
            import pandas as pd
            df1 = pd.DataFrame(experimentos["exp1_arquitectura"])[["nombre", "capas", "acc_test", "f1_test", "iteraciones"]]
            df1.columns = ["Modelo", "Capas", "Accuracy Test (%)", "F1-Score Test", "Iteraciones"]
            st.dataframe(df1, use_container_width=True)

        with st.expander("Experimento 2 — Función de activación"):
            df2 = pd.DataFrame(experimentos["exp2_activacion"])[["nombre", "activacion", "acc_test", "f1_test", "iteraciones"]]
            df2.columns = ["Modelo", "Activación", "Accuracy Test (%)", "F1-Score Test", "Iteraciones"]
            st.dataframe(df2, use_container_width=True)

        with st.expander("Experimento 3 — Solver de optimización"):
            df3 = pd.DataFrame(experimentos["exp3_solver"])[["nombre", "solver", "acc_test", "f1_test", "iteraciones"]]
            df3.columns = ["Modelo", "Solver", "Accuracy Test (%)", "F1-Score Test", "Iteraciones"]
            st.dataframe(df3, use_container_width=True)

        mejor = experimentos.get("mejor_modelo", {})
        balance_original = experimentos.get("balance_original", {})
        balance_smote = experimentos.get("balance_smote", {})

        with st.expander("📊 Balance del dataset y entrenamiento"):

            st.markdown("### Distribución original")

            perfiles = {0: "Bajo", 1: "Medio", 2: "Alto"}
            for clase, datos in balance_original.get("distribucion", {}).items():
                nombre = perfiles.get(int(clase), str(clase))
                st.markdown(
                    f"- **{nombre.capitalize()}**: {datos['count']} registros "
                    f"({datos['porcentaje']}%)"
                )

            estado = balance_original.get("estado_balance")

            if estado == "balanceado":
                st.success("✅ Dataset balanceado")
            elif estado == "moderadamente desbalanceado":
                st.warning("⚠️ Dataset moderadamente desbalanceado")
            else:
                st.error("❌ Dataset fuertemente desbalanceado")

            st.markdown("### Distribución luego de SMOTE")

            perfiles = {0: "Bajo", 1: "Medio", 2: "Alto"}
            for clase, datos in balance_smote.get("distribucion", {}).items():
                nombre = perfiles.get(int(clase), str(clase))
                st.markdown(
                    f"- **{nombre.capitalize()}**: {datos['count']} registros "
                    f"({datos['porcentaje']}%)"
                )

            st.success(
                "✅ El conjunto de entrenamiento fue balanceado mediante SMOTE "
                "antes del entrenamiento del MLP."
            )
            
            st.divider()

            st.markdown(f"**Mejor modelo:** capas={mejor.get('capas')}, activación={mejor.get('activacion')}, solver={mejor.get('solver')}")
            st.markdown(f"**Accuracy final:** {mejor.get('acc_test')}%  |  **F1-Score:** {mejor.get('f1_test')}")

            st.markdown("**Matriz de confusión (bajo / medio / alto):**")
            st.caption("Filas: valor real — Columnas: valor predicho. Los valores en la diagonal son aciertos.")
            cm = mejor.get("confusion_matrix", [])
            if cm:
                df_cm = pd.DataFrame(cm, index=["Real bajo", "Real medio", "Real alto"], columns=["Predicho bajo", "Predicho medio", "Predicho alto"])
                st.dataframe(df_cm, use_container_width=True)


# ======================
# CONFIGURACIÓN
# ======================

elif menu == "⚙️ Configuración":
    st.title("⚙️ Configuración del sistema")
    st.markdown("---")

    # Estado
    st.subheader("📋 Estado del sistema")
    col1, col2 = st.columns(2)
    with col1:
        if index_exists():
            st.success("✅ Base de conocimiento lista")
        else:
            st.warning("⚠️ Sin base de conocimiento")
    with col2:
        from mlp import modelo_disponible
        if modelo_disponible():
            st.success("✅ Modelo MLP entrenado")
        else:
            st.warning("⚠️ Modelo MLP sin entrenar")

    st.markdown("---")

    # Carga de PDFs
    st.subheader("📂 Carga de documentos")
    st.caption("Subí los PDFs de cada tema y los libros de referencia.")

    uploaded_files = st.file_uploader(
        "Seleccioná uno o más PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    archivos_guardados = []
    if uploaded_files:
        for file in uploaded_files:
            save_path = DOCS_DIR / file.name
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            archivos_guardados.append(file.name)
        st.success(f"✅ {len(archivos_guardados)} archivo(s) guardado(s): {', '.join(archivos_guardados)}")

    pdfs_actuales = list(DOCS_DIR.glob("*.pdf"))
    if pdfs_actuales:
        st.markdown(f"📁 **Documentos en /docs:** {', '.join(p.name for p in pdfs_actuales)}")

    st.markdown("---")

    # Ingestión
    st.subheader("🔄 Generar base de conocimiento")
    st.caption("Procesá los PDFs para crear el índice FAISS.")

    if st.button("🚀 Ejecutar ingestión"):
        if not pdfs_actuales and not uploaded_files:
            st.error("No hay PDFs en /docs. Subí al menos un documento primero.")
        else:
            with st.spinner("Procesando documentos y generando embeddings..."):
                try:
                    from ingest import run_ingest
                    result = run_ingest()
                    if result["ok"]:
                        st.success(f"✅ {result['message']}")
                        if "rag_chain" in st.session_state:
                            del st.session_state["rag_chain"]
                        if "chat_history" in st.session_state:
                            del st.session_state["chat_history"]
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"Error durante la ingestión: {e}")

    st.markdown("---")

    # MLP
    st.subheader("🧠 Entrenamiento del modelo MLP")
    st.caption("Entrena la red neuronal con el dataset sintético y los resultados reales acumulados.")

    n_sintetico = st.slider("Alumnos sintéticos en el dataset", 100, 2000, 1000, step=100)

    if st.button("🎯 Entrenar modelo MLP"):
        with st.spinner("Ejecutando experimentos y entrenando el modelo... ⏳"):
            try:
                from mlp import entrenar_y_guardar
                experimentos = entrenar_y_guardar(n_sintetico=n_sintetico)

                balance = experimentos.get("balance", {})
                mejor = experimentos.get("mejor_modelo", {})

                st.success("✅ Modelo entrenado y guardado correctamente.")
                st.info("Andá a 📊 Resultados para ver el diagnóstico completo y los experimentos.")

            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")
