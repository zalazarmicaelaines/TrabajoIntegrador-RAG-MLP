import json
import logging
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
RESULTADOS_PATH = ROOT_DIR / "resultados.json"
DATASET_PATH = ROOT_DIR / "dataset_mlp.json"
MODELO_PATH = ROOT_DIR / "modelo_mlp.pkl"
SCALER_PATH = ROOT_DIR / "scaler_mlp.pkl"
EXPERIMENTOS_PATH = ROOT_DIR / "experimentos_mlp.json"

TEMAS = ["Tema 1", "Tema 2", "Tema 3", "Tema 4", "Tema 5"]


# ======================
# Serialización segura
# ======================

def convertir_serializable(obj):
    """Convierte tipos numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convertir_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_serializable(i) for i in obj]
    return obj


# ======================
# Etiquetado
# ======================

def clasificar_nivel(porcentaje: float) -> str:
    if porcentaje < 50:
        return "bajo"
    elif porcentaje < 75:
        return "medio"
    else:
        return "alto"


# ======================
# Dataset sintético
# ======================

def generar_dataset_sintetico(n: int = 500, seed: int = 42) -> list:
    np.random.seed(seed)
    dataset = []

    perfiles = {
        "avanzado":  {"media": 85, "std": 8,  "peso": 0.25},
        "medio":     {"media": 62, "std": 10, "peso": 0.35},
        "debil":     {"media": 35, "std": 10, "peso": 0.25},
        "irregular": {"media": 60, "std": 25, "peso": 0.15},
    }

    for _ in range(n):
        perfil = np.random.choice(
            list(perfiles.keys()),
            p=[p["peso"] for p in perfiles.values()]
        )
        config = perfiles[perfil]
        scores = np.clip(
            np.random.normal(config["media"], config["std"], 5), 0, 100
        ).tolist()
        promedio = np.mean(scores)
        dataset.append({
            "scores": [round(s, 1) for s in scores],
            "nivel_general": clasificar_nivel(promedio)
        })

    return dataset


def cargar_resultados_reales() -> list:
    if not RESULTADOS_PATH.exists():
        return []
    with open(RESULTADOS_PATH, "r", encoding="utf-8") as f:
        try:
            resultados = json.load(f)
        except Exception:
            return []

    scores_por_tema = {t: [] for t in TEMAS}
    for r in resultados:
        tema = r.get("tema", "")
        for t in TEMAS:
            if t in tema:
                scores_por_tema[t].append(r.get("porcentaje", 0))
                break

    if not all(scores_por_tema[t] for t in TEMAS):
        return []

    scores = [np.mean(scores_por_tema[t]) for t in TEMAS]
    promedio = np.mean(scores)
    return [{
        "scores": [round(s, 1) for s in scores],
        "nivel_general": clasificar_nivel(promedio)
    }]


def construir_dataset(n_sintetico: int = 500) -> tuple:
    sintetico = generar_dataset_sintetico(n_sintetico)
    reales = cargar_resultados_reales()
    dataset = sintetico + reales

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(convertir_serializable(dataset), f, ensure_ascii=False, indent=2)

    X = np.array([d["scores"] for d in dataset])
    y = np.array([d["nivel_general"] for d in dataset])
    return X, y


def analizar_balance(y: np.ndarray) -> dict:
    clases, conteos = np.unique(y, return_counts=True)
    total = len(y)
    distribucion = {
        str(c): {
            "count": int(n),
            "porcentaje": round(float(n / total * 100), 1)
        }
        for c, n in zip(clases, conteos)
    }
    porcentajes = [v["porcentaje"] for v in distribucion.values()]
    balanceado = bool(max(porcentajes) - min(porcentajes) < 20)
    return {
        "distribucion": distribucion,
        "balanceado": balanceado,
        "total": int(total)
    }


# ======================
# Preparación de datos
# ======================

def preparar_datos(X: np.ndarray, y: np.ndarray):
    # Split 80% train+val / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Split 85% train / 15% val (del 80% anterior)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Escalar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # SMOTE solo sobre entrenamiento
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def entrenar_modelo(hidden_layers: tuple, activation: str,
                    X_train, y_train, X_val, y_val) -> tuple:
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15
    )
    mlp.fit(X_train, y_train)
    y_pred_val = mlp.predict(X_val)
    acc = float(round(accuracy_score(y_val, y_pred_val) * 100, 2))
    f1 = float(round(f1_score(y_val, y_pred_val, average="weighted"), 4))
    return mlp, acc, f1


# ======================
# Experimentos
# ======================

def experimento_arquitectura(X_train, X_val, X_test, y_train, y_val, y_test) -> tuple:
    """
    Experimento 1: compara 3 arquitecturas con ReLU.
    Retorna (resultados, mejor_capas).
    """
    arquitecturas = [
        ("MLP-A (1 capa, 8 neuronas)", (8,)),
        ("MLP-B (2 capas, 16-8 neuronas)", (16, 8)),
        ("MLP-C (3 capas, 32-16-8 neuronas)", (32, 16, 8)),
    ]

    resultados = []
    for nombre, capas in arquitecturas:
        modelo, acc_val, f1_val = entrenar_modelo(capas, "relu", X_train, y_train, X_val, y_val)
        y_pred_test = modelo.predict(X_test)
        acc_test = float(round(accuracy_score(y_test, y_pred_test) * 100, 2))
        f1_test = float(round(f1_score(y_test, y_pred_test, average="weighted"), 4))

        resultados.append({
            "nombre": nombre,
            "capas": str(capas),
            "activacion": "relu",
            "acc_val": acc_val,
            "f1_val": f1_val,
            "acc_test": acc_test,
            "f1_test": f1_test
        })
        logger.info(f"{nombre}: acc_test={acc_test}%, f1_test={f1_test}")

    # Ganador por F1-Score en test
    ganador = max(resultados, key=lambda x: x["f1_test"])
    mejor_capas = eval(ganador["capas"])
    logger.info(f"Ganador exp 1: {ganador['nombre']} → capas={mejor_capas}")

    return resultados, mejor_capas


def experimento_activacion(X_train, X_val, X_test, y_train, y_val, y_test, capas) -> tuple:
    """
    Experimento 2: compara activaciones usando la mejor arquitectura del exp 1.
    Retorna (resultados, mejor_activacion).
    """
    activaciones = ["relu", "tanh", "logistic"]

    resultados = []
    for act in activaciones:
        modelo, acc_val, f1_val = entrenar_modelo(capas, act, X_train, y_train, X_val, y_val)
        y_pred_test = modelo.predict(X_test)
        acc_test = float(round(accuracy_score(y_test, y_pred_test) * 100, 2))
        f1_test = float(round(f1_score(y_test, y_pred_test, average="weighted"), 4))

        resultados.append({
            "nombre": f"Activación {act}",
            "capas": str(capas),
            "activacion": act,
            "acc_val": acc_val,
            "f1_val": f1_val,
            "acc_test": acc_test,
            "f1_test": f1_test
        })
        logger.info(f"Activación {act}: acc_test={acc_test}%, f1_test={f1_test}")

    ganador = max(resultados, key=lambda x: x["f1_test"])
    mejor_activacion = ganador["activacion"]
    logger.info(f"Ganador exp 2: activación={mejor_activacion}")

    return resultados, mejor_activacion


def experimento_solver(X_train, X_val, X_test, y_train, y_val, y_test, capas, activacion) -> tuple:
    """
    Experimento 3: compara solvers usando la mejor arquitectura
    y activación de los experimentos anteriores.
    Retorna (resultados, mejor_solver).
    """
    solvers = ["adam", "sgd", "lbfgs"]

    resultados = []
    for solver in solvers:
        mlp = MLPClassifier(
            hidden_layer_sizes=capas,
            activation=activacion,
            solver=solver,
            max_iter=500,
            random_state=42,
            early_stopping=True if solver != "lbfgs" else False,
            validation_fraction=0.1 if solver != "lbfgs" else 0.0,
            n_iter_no_change=15 if solver != "lbfgs" else 10
        )
        mlp.fit(X_train, y_train)
        y_pred_val = mlp.predict(X_val)
        acc_val = float(round(accuracy_score(y_val, y_pred_val) * 100, 2))
        f1_val = float(round(f1_score(y_val, y_pred_val, average="weighted"), 4))

        y_pred_test = mlp.predict(X_test)
        acc_test = float(round(accuracy_score(y_test, y_pred_test) * 100, 2))
        f1_test = float(round(f1_score(y_test, y_pred_test, average="weighted"), 4))

        resultados.append({
            "nombre": f"Solver {solver}",
            "capas": str(capas),
            "activacion": activacion,
            "solver": solver,
            "acc_val": acc_val,
            "f1_val": f1_val,
            "acc_test": acc_test,
            "f1_test": f1_test
        })
        logger.info(f"Solver {solver}: acc_test={acc_test}%, f1_test={f1_test}")

    ganador = max(resultados, key=lambda x: x["f1_test"])
    mejor_solver = ganador["solver"]
    logger.info(f"Ganador exp 3: solver={mejor_solver}")

    return resultados, mejor_solver


# ======================
# Entrenamiento final
# ======================

def entrenar_y_guardar(n_sintetico: int = 500) -> dict:
    logger.info("Construyendo dataset...")
    X, y = construir_dataset(n_sintetico)
    balance = analizar_balance(y)
    logger.info(f"Balance: {balance}")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preparar_datos(X, y)

    # Experimento 1 → encuentra mejor arquitectura
    logger.info("Experimento 1: Arquitectura...")
    exp1, mejor_capas = experimento_arquitectura(X_train, X_val, X_test, y_train, y_val, y_test)

    # Experimento 2 → usa mejor arquitectura, encuentra mejor activación
    logger.info("Experimento 2: Activación...")
    exp2, mejor_activacion = experimento_activacion(X_train, X_val, X_test, y_train, y_val, y_test, capas=mejor_capas)

    # Experimento 3 → usa ganadores exp 1 y 2, varía solver
    logger.info("Experimento 3: Solver...")
    exp3, mejor_solver = experimento_solver(X_train, X_val, X_test, y_train, y_val, y_test,capas=mejor_capas, activacion=mejor_activacion)

    # Modelo final con los mejores parámetros
    mejor_modelo = MLPClassifier(
        hidden_layer_sizes=mejor_capas,
        activation=mejor_activacion,
        solver=mejor_solver,
        max_iter=500,
        random_state=42,
        early_stopping=True if mejor_solver != "lbfgs" else False,
        validation_fraction=0.1 if mejor_solver != "lbfgs" else 0.0,
        n_iter_no_change=15 if mejor_solver != "lbfgs" else 10
    )
    mejor_modelo.fit(X_train, y_train)

    y_pred_final = mejor_modelo.predict(X_test)
    acc_final = float(round(accuracy_score(y_test, y_pred_final) * 100, 2))
    f1_final = float(round(f1_score(y_test, y_pred_final, average="weighted"), 4))
    reporte = {k: convertir_serializable(v) for k, v in
               classification_report(y_test, y_pred_final, output_dict=True).items()}
    cm = confusion_matrix(y_test, y_pred_final, labels=["bajo", "medio", "alto"]).tolist()

    # Guardar modelo y scaler
    joblib.dump(mejor_modelo, MODELO_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Modelo guardado.")

    experimentos = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_sintetico_base": n_sintetico,
        "balance": balance,
        "exp1_arquitectura": exp1,
        "exp2_activacion": exp2,
        "exp3_solver": exp3,
        "mejor_modelo": {
            "capas": str(mejor_capas),
            "activacion": mejor_activacion,
            "solver": mejor_solver,
            "acc_test": acc_final,
            "f1_test": f1_final,
            "reporte": reporte,
            "confusion_matrix": cm
        }
    }

    with open(EXPERIMENTOS_PATH, "w", encoding="utf-8") as f:
        json.dump(convertir_serializable(experimentos), f, ensure_ascii=False, indent=2)

    return experimentos


# ======================
# Predicción y diagnóstico
# ======================

def modelo_disponible() -> bool:
    return MODELO_PATH.exists() and SCALER_PATH.exists()


def predecir_nivel_general(scores_por_tema: dict) -> dict:
    if not modelo_disponible():
        raise ValueError("El modelo no está entrenado todavía.")

    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)

    vector = [scores_por_tema.get(t, 0) for t in TEMAS]
    X = scaler.transform([vector])
    nivel_general = modelo.predict(X)[0]
    proba = modelo.predict_proba(X)[0]
    clases = modelo.classes_

    clases_list = list(clases)
    prob_aprobar = 0
    if "medio" in clases_list:
        prob_aprobar += proba[clases_list.index("medio")]
    if "alto" in clases_list:
        prob_aprobar += proba[clases_list.index("alto")]

    diagnostico_subtemas = generar_diagnostico_subtemas()

    return {
        "nivel_general": nivel_general,
        "prob_aprobar": round(float(prob_aprobar) * 100, 1),
        "scores_por_tema": scores_por_tema,
        "diagnostico_subtemas": diagnostico_subtemas
    }


def generar_diagnostico_subtemas() -> list:
    if not RESULTADOS_PATH.exists():
        return []
    with open(RESULTADOS_PATH, "r", encoding="utf-8") as f:
        try:
            resultados = json.load(f)
        except Exception:
            return []

    subtemas = {}
    for r in resultados:
        subtema = r.get("subtema", "").strip()
        porcentaje = r.get("porcentaje", 0)
        if subtema:
            if subtema not in subtemas:
                subtemas[subtema] = []
            subtemas[subtema].append(porcentaje)

    diagnostico = []
    for subtema, porcentajes in subtemas.items():
        promedio = round(float(np.mean(porcentajes)), 1)
        diagnostico.append({
            "subtema": subtema,
            "promedio": promedio,
            "nivel": clasificar_nivel(promedio)
        })

    diagnostico.sort(key=lambda x: x["promedio"])
    return diagnostico


def obtener_scores_por_tema() -> dict:
    if not RESULTADOS_PATH.exists():
        return {}
    with open(RESULTADOS_PATH, "r", encoding="utf-8") as f:
        try:
            resultados = json.load(f)
        except Exception:
            return {}

    scores = {t: [] for t in TEMAS}
    for r in resultados:
        tema_resultado = r.get("tema", "")
        for t in TEMAS:
            if t in tema_resultado:
                scores[t].append(r.get("porcentaje", 0))
                break

    return {t: round(float(np.mean(v)), 1) for t, v in scores.items() if v}
