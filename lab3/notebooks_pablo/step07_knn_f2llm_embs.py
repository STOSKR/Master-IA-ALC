"""
Clasificación para EXIST 2026 usando embeddings F2LLM precomputados.

El script:
  1. Carga train.json y test.json para obtener los id_EXIST de cada split
  2. Carga el NPZ con TODOS los embeddings (indexado por id_EXIST)
  3. Filtra los embeddings que corresponden a train/val/test
  4. Filtra train/val por idioma (--lang es | en | es_en)
  5. Clasifica con KNN por similaridad coseno
  6. Busca el threshold óptimo en val maximizando F1-macro
  7. Genera predicciones sobre TODO el test (sin filtro de idioma)
  8. Guarda predicciones en formato PyEvALL

Uso:
    python embed_exist2026.py --lang es
    python embed_exist2026.py --lang en
    python embed_exist2026.py --lang es_en
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
MAIN_PATH       = ".."
DATA_TRAIN_PATH = os.path.join(MAIN_PATH, "preprocessed_data", "training.json")
DATA_TEST_PATH  = os.path.join(MAIN_PATH, "preprocessed_data", "test.json")
PREDICTIONS_DIR = os.path.join(MAIN_PATH, "predictions")

GROUP_ID   = "BeingChillingWeWillWin"
MODEL_BASE = "f2llm4BEmbed"

LABEL_MAP     = {"NO": 0, "YES": 1}
LABEL_MAP_INV = {0: "NO", 1: "YES"}

EMBEDDINGS_PATH = os.path.join(MAIN_PATH, "embeddings", "f2llm_embeddings.npz")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificación KNN con embeddings F2LLM precomputados para EXIST 2026"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["es", "en", "es_en"],
        help="Idioma(s) de entrenamiento: es | en | es_en",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de vecinos para KNN (default: 5)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fracción de train usada como validación (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_json_dataset(path):
    logger.info(f"Cargando: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.values())
    logger.info(f"  → {len(df)} ejemplos")
    return df


def filter_by_lang(df, lang):
    if lang == "es_en":
        logger.info("  → Sin filtro de idioma (es + en)")
        return df.reset_index(drop=True)
    filtered = df[df["lang"] == lang].reset_index(drop=True)
    logger.info(f"  → {len(filtered)} ejemplos tras filtrar por lang='{lang}'")
    return filtered


def split_train_val(df, val_fraction, seed, label_col="label_int"):
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df[label_col],
        random_state=seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    logger.info(
        f"Split → train: {len(train_df)}, val: {len(val_df)} "
        f"(val_split={val_fraction})"
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Carga y filtrado de embeddings
# ---------------------------------------------------------------------------
def load_all_embeddings(path):
    """
    Carga el NPZ con todos los embeddings.
    Espera dos arrays: 'ids' (id_EXIST como strings) y 'embeddings' (N, D).
    Devuelve un diccionario {id_EXIST: embedding_vector}.
    """
    logger.info(f"Cargando embeddings desde: {path}")
    data = np.load(path, allow_pickle=True)
    id_to_emb = {key: data[key] for key in data.files}
    logger.info(f"  → {len(id_to_emb)} embeddings cargados, dim={next(iter(id_to_emb.values())).shape[0]}")
    return id_to_emb


def select_embeddings(df, id_to_emb, split_name="split"):
    """
    Dada una lista de id_EXIST en el DataFrame, extrae los embeddings
    correspondientes del diccionario global. Avisa si falta algún id.
    """
    ids = df["id_EXIST"].astype(str).tolist()
    missing = [id_ for id_ in ids if id_ not in id_to_emb]
    if missing:
        logger.warning(
            f"[{split_name}] {len(missing)} ids sin embedding: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    embeddings = np.stack([id_to_emb[id_] for id_ in ids if id_ in id_to_emb])
    ids_found  = [id_ for id_ in ids if id_ in id_to_emb]
    logger.info(f"  [{split_name}] {len(ids_found)} embeddings seleccionados, shape={embeddings.shape}")
    return ids_found, embeddings


# ---------------------------------------------------------------------------
# Clasificación KNN por similaridad coseno
# ---------------------------------------------------------------------------
def knn_predict_proba(train_embeddings, train_labels, query_embeddings, k):
    """
    KNN ponderado por similaridad coseno.
    Los embeddings deben estar normalizados L2 (producto punto = coseno).
    Devuelve probabilidades de clase YES en [0, 1].
    """
    sim       = query_embeddings @ train_embeddings.T   # (N_query, N_train)
    top_k_idx = np.argsort(-sim, axis=1)[:, :k]        # (N_query, k)

    probs = []
    for i in range(len(query_embeddings)):
        neighbors = train_labels[top_k_idx[i]]
        weights   = np.clip(sim[i, top_k_idx[i]], 0, None)
        if weights.sum() == 0:
            prob = neighbors.mean()
        else:
            prob = np.dot(weights, neighbors) / weights.sum()
        probs.append(prob)

    return np.array(probs)


# ---------------------------------------------------------------------------
# Threshold óptimo (F1-macro)
# ---------------------------------------------------------------------------
def find_optimal_threshold(y_true, y_probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_th, best_score = 0.5, 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_th    = th

    logger.info(
        f"Threshold óptimo (max F1-macro en val): {best_th:.4f} "
        f"→ F1-macro={best_score:.4f}"
    )
    return best_th, best_score


# ---------------------------------------------------------------------------
# Evaluación completa
# ---------------------------------------------------------------------------
def evaluate_full(y_true, y_probs, threshold, split_name="val"):
    preds    = (y_probs >= threshold).astype(int)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_bin   = f1_score(y_true, preds, average="binary", zero_division=0)
    acc      = accuracy_score(y_true, preds)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = float("nan")

    logger.info(f"\n{'='*55}")
    logger.info(f"  Métricas en {split_name.upper()} (threshold={threshold:.4f})")
    logger.info(f"{'='*55}")
    logger.info(f"  F1-macro   : {f1_macro:.4f}  ← métrica principal")
    logger.info(f"  F1-binary  : {f1_bin:.4f}")
    logger.info(f"  Accuracy   : {acc:.4f}")
    logger.info(f"  Precision  : {prec:.4f}")
    logger.info(f"  Recall     : {rec:.4f}")
    logger.info(f"  ROC-AUC    : {auc:.4f}")
    logger.info(f"{'='*55}\n")

    return {
        "f1_macro": f1_macro, "f1_binary": f1_bin, "accuracy": acc,
        "precision": prec, "recall": rec, "roc_auc": auc,
    }


# ---------------------------------------------------------------------------
# Guardado de predicciones
# ---------------------------------------------------------------------------
def save_predictions_pyevall(ids, preds_int, path, test_case="EXIST2026"):
    records = [
        {"test_case": test_case, "id": str(id_), "value": LABEL_MAP_INV[int(p)]}
        for id_, p in zip(ids, preds_int)
    ]
    df = pd.DataFrame(records)
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_json(orient="records", force_ascii=False))
    logger.info(f"Predicciones PyEvALL guardadas en: {path}")


def save_probs_json(ids, probs, path, labels=None):
    records = []
    for i, (id_, prob) in enumerate(zip(ids, probs)):
        rec = {"id": str(id_), "prob_YES": round(float(prob), 6)}
        if labels is not None:
            rec["label"] = LABEL_MAP_INV[int(labels[i])]
        records.append(rec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Probabilidades guardadas en: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    lang     = args.lang
    model_id = f"{MODEL_BASE}_{lang}"

    logger.info("=" * 60)
    logger.info(f"  EXIST 2026 — KNN con embeddings F2LLM precomputados")
    logger.info(f"  Idioma de entrenamiento : {lang}")
    logger.info(f"  Model ID               : {model_id}")
    logger.info(f"  K (KNN)                : {args.k}")
    logger.info("=" * 60)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Cargar JSONs para obtener los ids y metadatos de cada split
    # ------------------------------------------------------------------
    raw_train = load_json_dataset(DATA_TRAIN_PATH)
    raw_test  = load_json_dataset(DATA_TEST_PATH)

    # Filtrar train por idioma; test siempre completo
    train_full = filter_by_lang(raw_train, lang)
    train_full["label_int"] = train_full["label"].map(LABEL_MAP)
    train_full = train_full.dropna(subset=["label_int"]).reset_index(drop=True)

    test_df = raw_test.reset_index(drop=True)
    logger.info(f"Test (completo): {len(test_df)} ejemplos")

    # ------------------------------------------------------------------
    # 2. Split train / val estratificado
    # ------------------------------------------------------------------
    train_df, val_df = split_train_val(
        train_full, val_fraction=args.val_split, seed=args.seed
    )
    logger.info(f"Distribución TRAIN: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"Distribución VAL  : {val_df['label'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 3. Cargar el NPZ global y extraer embeddings por split
    # ------------------------------------------------------------------
    id_to_emb = load_all_embeddings(EMBEDDINGS_PATH)

    logger.info("\nSeleccionando embeddings por split...")
    train_ids, train_embeddings = select_embeddings(train_df, id_to_emb, "train")
    val_ids,   val_embeddings   = select_embeddings(val_df,   id_to_emb, "val")
    test_ids,  test_embeddings  = select_embeddings(test_df,  id_to_emb, "test")

    train_labels = train_df.set_index(
        train_df["id_EXIST"].astype(str)
    ).loc[train_ids, "label_int"].values.astype(int)

    val_labels = val_df.set_index(
        val_df["id_EXIST"].astype(str)
    ).loc[val_ids, "label_int"].values.astype(int)

    # ------------------------------------------------------------------
    # 4. KNN en VAL → threshold óptimo por F1-macro
    # ------------------------------------------------------------------
    logger.info(f"\n>>> KNN (k={args.k}) en VAL <<<")
    val_probs = knn_predict_proba(train_embeddings, train_labels, val_embeddings, k=args.k)

    optimal_threshold, _ = find_optimal_threshold(val_labels, val_probs)
    val_metrics = evaluate_full(val_labels, val_probs, optimal_threshold, split_name="val")

    val_preds_int = (val_probs >= optimal_threshold).astype(int)

    save_probs_json(
        val_ids, val_probs,
        os.path.join(PREDICTIONS_DIR, f"{GROUP_ID}_{model_id}_probs_val.json"),
        labels=val_labels,
    )
    save_predictions_pyevall(
        val_ids, val_preds_int,
        os.path.join(PREDICTIONS_DIR, f"{GROUP_ID}_{model_id}_val.json"),
    )

    # ------------------------------------------------------------------
    # 5. KNN en TEST
    # ------------------------------------------------------------------
    logger.info(f"\n>>> KNN (k={args.k}) en TEST <<<")
    test_probs    = knn_predict_proba(train_embeddings, train_labels, test_embeddings, k=args.k)
    test_preds_int = (test_probs >= optimal_threshold).astype(int)

    logger.info(
        f"Predicciones TEST (threshold={optimal_threshold:.4f}): "
        f"YES={int(np.sum(test_preds_int == 1))} ({100 * np.mean(test_preds_int == 1):.2f}%) | "
        f"NO={int(np.sum(test_preds_int == 0))} ({100 * np.mean(test_preds_int == 0):.2f}%)"
    )

    save_probs_json(
        test_ids, test_probs,
        os.path.join(PREDICTIONS_DIR, f"{GROUP_ID}_{model_id}_probs_test.json"),
    )
    save_predictions_pyevall(
        test_ids, test_preds_int,
        os.path.join(PREDICTIONS_DIR, f"{GROUP_ID}_{model_id}.json"),
    )

    # ------------------------------------------------------------------
    # 6. Resumen final
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"  Lang usado para train  : {lang}")
    logger.info(f"  Model ID               : {model_id}")
    logger.info(f"  K (KNN)                : {args.k}")
    logger.info(f"  Threshold óptimo       : {optimal_threshold:.4f}")
    logger.info(f"  F1-macro (val)         : {val_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-binary (val)        : {val_metrics['f1_binary']:.4f}")
    logger.info(f"  Accuracy (val)         : {val_metrics['accuracy']:.4f}")
    logger.info(f"\n  Predicciones en        : {PREDICTIONS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()