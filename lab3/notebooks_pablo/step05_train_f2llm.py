"""
Fine-tuning con LoRA para detección de sexismo (EXIST 2026)
Tarea: clasificación binaria YES/NO sobre texto de TikTok

Uso:
    python train_exist2026.py --lang es
    python train_exist2026.py --lang en
    python train_exist2026.py --lang es_en

El parámetro --lang controla:
  - qué subconjunto de train/val se usa para entrenar
  - el nombre del modelo guardado
  - el nombre del archivo de predicciones final
  - el test siempre es el conjunto completo en idioma 'es'
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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
# Argparse
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA para EXIST 2026 — detección de sexismo"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["es", "en", "es_en"],
        help=(
            "Idioma(s) de entrenamiento:\n"
            "  es    → solo ejemplos en español\n"
            "  en    → solo ejemplos en inglés\n"
            "  es_en → todos los ejemplos (español + inglés)"
        ),
    )
    parser.add_argument(
        "--main_path",
        type=str,
        default="..",
        help="Ruta raíz del proyecto (default: ..)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="codefuse-ai/F2LLM-4B",
        help="Nombre del modelo base en HuggingFace Hub",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="BeingChillingWeWillWin",
        help="Identificador de grupo para los ficheros de predicciones",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Longitud máxima de tokenización (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas de entrenamiento (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--train_batch",
        type=int,
        default=64,
        help="Batch size de entrenamiento por dispositivo (default: 8)",
    )
    parser.add_argument(
        "--eval_batch",
        type=int,
        default=64,
        help="Batch size de evaluación por dispositivo (default: 16)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rango LoRA (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha LoRA (default: 32)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help=(
            "Fracción del train original usada como validación si no existe "
            "fichero dev separado (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Token de HuggingFace",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constantes derivadas de lang
# ---------------------------------------------------------------------------
def build_config(args):
    """Devuelve un diccionario con todas las rutas y nombres derivados de --lang."""
    lang = args.lang
    main = args.main_path

    model_id = f"f2llm4BText_{lang}"

    config = dict(
        lang=lang,
        model_id=model_id,
        model_name=args.model_name,
        group_id=args.group_id,
        text_column="text",
        label_column="label",
        # Datos
        data_train_path=os.path.join(main, "preprocessed_data", "training.json"),
        data_test_path=os.path.join(main, "preprocessed_data", "test.json"),
        # Salidas
        output_dir=os.path.join(main, "weights", f"F2LLM-4B_text_{lang}_lora"),
        save_path=os.path.join(main, "weights", f"F2LLM-4B_text_{lang}_final"),
        predictions_dir=os.path.join(main, "predictions"),
        # Hiperparámetros
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        train_batch=args.train_batch,
        eval_batch=args.eval_batch,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        val_split=args.val_split,
        seed=args.seed,
    )
    return config


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_json_dataset(path):
    """Carga el JSON orientado a diccionario {id: {...}} y devuelve un DataFrame."""
    logger.info(f"Cargando datos desde: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.values())
    logger.info(f"  → {len(df)} ejemplos cargados")
    return df


def filter_by_lang(df, lang):
    """
    Filtra el DataFrame según el parámetro de idioma.
      es    → solo filas con lang == 'es'
      en    → solo filas con lang == 'en'
      es_en → todas las filas
    """
    if lang == "es_en":
        logger.info("  → Sin filtro de idioma (es + en)")
        return df
    filtered = df[df["lang"] == lang].reset_index(drop=True)
    logger.info(f"  → {len(filtered)} ejemplos después de filtrar por lang='{lang}'")
    return filtered


def split_train_val(df, val_fraction, seed, label_col="label_int"):
    """
    Divide un DataFrame en train y val estratificadamente.
    Se usa cuando no hay fichero de dev separado.
    """
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df[label_col],
        random_state=seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info(
        f"Split train/val: {len(train_df)} train, {len(val_df)} val "
        f"(fracción val={val_fraction})"
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Tokenización
# ---------------------------------------------------------------------------
def build_tokenize_fn(tokenizer, text_column, max_length):
    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return tokenize


def prepare_hf_dataset(df, text_column, tokenize_fn, has_label=True):
    """Convierte un DataFrame a Dataset de HuggingFace y aplica la tokenización."""
    cols = ["id_EXIST", text_column]
    if has_label:
        cols.append("label_int")
    rename = {"label_int": "label"} if has_label else {}

    ds = Dataset.from_pandas(df[cols].rename(columns=rename))
    ds = ds.map(tokenize_fn, batched=True)
    return ds


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
LABEL_MAP = {"NO": 0, "YES": 1}
LABEL_MAP_INV = {0: "NO", 1: "YES"}


def compute_metrics(eval_pred):
    """
    Métrica principal: F1-macro (binario, por lo tanto equivale al promedio
    no ponderado de F1-NO y F1-YES).
    """
    logits, labels = eval_pred
    logits_tensor = torch.tensor(logits).squeeze(-1)
    probs = torch.sigmoid(logits_tensor).numpy()
    preds = (probs >= 0.5).astype(int)

    if labels is None or any(l < 0 for l in labels):
        return {"f1_macro": 0.0, "f1_binary": 0.0, "accuracy": 0.0}

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_bin = f1_score(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "f1_macro": f1_macro,
        "f1_binary": f1_bin,
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# Modelo y entrenador
# ---------------------------------------------------------------------------
class BinaryTrainer(Trainer):
    """Trainer personalizado con BCEWithLogitsLoss (num_labels=1)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_model(model_name, tokenizer, lora_r, lora_alpha):
    logger.info(f"Cargando modelo base: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="single_label_classification",
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Inferencia y guardado
# ---------------------------------------------------------------------------
def run_inference(trainer, dataset):
    """Ejecuta inferencia y devuelve probabilidades numpy."""
    predictions = trainer.predict(dataset)
    logits = torch.tensor(predictions.predictions).squeeze(-1)
    probs = torch.sigmoid(logits).numpy()
    labels = predictions.label_ids
    return probs, labels


def find_optimal_threshold(y_true, y_probs, metric="f1_macro"):
    """
    Busca el threshold que maximiza F1-macro en el conjunto de validación.
    Prueba un grid de thresholds para garantizar que se evalúa con F1-macro.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_th, best_score = 0.5, 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_th = th

    logger.info(
        f"Threshold óptimo (max F1-macro en val): {best_th:.4f}  "
        f"→ F1-macro={best_score:.4f}"
    )
    return best_th, best_score


def evaluate_full(y_true, y_probs, threshold, split_name="val"):
    """Calcula y loguea un conjunto completo de métricas."""
    preds = (y_probs >= threshold).astype(int)

    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_bin = f1_score(y_true, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_true, preds)
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
        "f1_macro": f1_macro,
        "f1_binary": f1_bin,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
    }


def save_predictions_pyevall(ids, preds_int, path, test_case="EXIST2026"):
    """Guarda predicciones en formato PyEvALL."""
    records = [
        {"test_case": test_case, "id": str(id_), "value": LABEL_MAP_INV[int(p)]}
        for id_, p in zip(ids, preds_int)
    ]
    df = pd.DataFrame(records)
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_json(orient="records", force_ascii=False))
    logger.info(f"Predicciones PyEvALL guardadas en: {path}")


def save_probs_json(ids, probs, path, labels=None):
    """Guarda probabilidades crudas con metadatos opcionales."""
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
    cfg = build_config(args)

    # HF token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    logger.info("=" * 60)
    logger.info(f"  EXIST 2026 — Fine-tuning LoRA")
    logger.info(f"  Idioma de entrenamiento : {cfg['lang']}")
    logger.info(f"  Model ID               : {cfg['model_id']}")
    logger.info(f"  Modelo base            : {cfg['model_name']}")
    logger.info("=" * 60)

    # Directorios de salida
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["save_path"], exist_ok=True)
    os.makedirs(cfg["predictions_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Carga y filtrado de datos
    # ------------------------------------------------------------------
    raw_train = load_json_dataset(cfg["data_train_path"])
    raw_test = load_json_dataset(cfg["data_test_path"])

    # Filtrar train/val por idioma
    train_full = filter_by_lang(raw_train, cfg["lang"])

    # El test SIEMPRE es el conjunto completo, independientemente del lang de entrenamiento
    test_df = raw_test.reset_index(drop=True)
    logger.info(f"Test (completo): {len(test_df)} ejemplos")

    # Mapeo de etiquetas
    train_full["label_int"] = train_full[cfg["label_column"]].map(LABEL_MAP)

    # Comprobación: etiquetas válidas
    if train_full["label_int"].isna().any():
        logger.warning(
            "Algunos ejemplos de train tienen etiqueta desconocida; se eliminarán."
        )
        train_full = train_full.dropna(subset=["label_int"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Split train / validación (estratificado)
    # ------------------------------------------------------------------
    train_df, val_df = split_train_val(
        train_full,
        val_fraction=cfg["val_split"],
        seed=cfg["seed"],
        label_col="label_int",
    )

    logger.info(
        f"\nDistribución etiquetas TRAIN: "
        f"{train_df[cfg['label_column']].value_counts().to_dict()}"
    )
    logger.info(
        f"Distribución etiquetas VAL  : "
        f"{val_df[cfg['label_column']].value_counts().to_dict()}"
    )

    # ------------------------------------------------------------------
    # 3. Tokenización
    # ------------------------------------------------------------------
    logger.info(f"\nCargando tokenizer: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = build_tokenize_fn(tokenizer, cfg["text_column"], cfg["max_length"])

    train_dataset = prepare_hf_dataset(train_df, cfg["text_column"], tokenize_fn)
    eval_dataset = prepare_hf_dataset(val_df, cfg["text_column"], tokenize_fn)

    # Test no tiene label_int → has_label=False
    test_df["label_int"] = -1
    test_dataset = prepare_hf_dataset(
        test_df, cfg["text_column"], tokenize_fn, has_label=True
    )

    logger.info(
        f"\nTamaños datasets: "
        f"train={len(train_dataset)}, "
        f"val={len(eval_dataset)}, "
        f"test={len(test_dataset)}"
    )

    # ------------------------------------------------------------------
    # 4. Modelo + LoRA
    # ------------------------------------------------------------------
    model = build_model(
        cfg["model_name"], tokenizer, cfg["lora_r"], cfg["lora_alpha"]
    )

    # ------------------------------------------------------------------
    # 5. Entrenamiento
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["train_batch"],
        per_device_eval_batch_size=cfg["eval_batch"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1_macro",      # ← F1-macro como criterio principal
        greater_is_better=True,
        load_best_model_at_end=True,
        bf16=True,
        lr_scheduler_type="cosine",
        seed=cfg["seed"],
        logging_steps=50,
        report_to="none",
    )

    trainer = BinaryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("\n>>> Iniciando entrenamiento <<<")
    trainer.train()

    logger.info(f"\nGuardando modelo en: {cfg['save_path']}")
    model.save_pretrained(cfg["save_path"])
    tokenizer.save_pretrained(cfg["save_path"])

    # ------------------------------------------------------------------
    # 6. Evaluación en VAL — threshold óptimo por F1-macro
    # ------------------------------------------------------------------
    logger.info("\n>>> Inferencia en VAL <<<")
    y_probs_val, y_true_val = run_inference(trainer, eval_dataset)

    optimal_threshold, best_f1_macro_val = find_optimal_threshold(
        y_true_val, y_probs_val, metric="f1_macro"
    )

    val_metrics = evaluate_full(
        y_true_val, y_probs_val, optimal_threshold, split_name="val"
    )

    # Guardar probabilidades val
    val_probs_path = os.path.join(
        cfg["predictions_dir"],
        f"{cfg['group_id']}_{cfg['model_id']}_probs_val.json",
    )
    save_probs_json(
        val_df["id_EXIST"].values,
        y_probs_val,
        val_probs_path,
        labels=val_df["label_int"].values,
    )

    # Guardar predicciones val en formato PyEvALL
    val_preds_int = (y_probs_val >= optimal_threshold).astype(int)
    val_pyevall_path = os.path.join(
        cfg["predictions_dir"],
        f"{cfg['group_id']}_{cfg['model_id']}_val.json",
    )
    save_predictions_pyevall(val_df["id_EXIST"].values, val_preds_int, val_pyevall_path)

    # ------------------------------------------------------------------
    # 7. Inferencia en TEST
    # ------------------------------------------------------------------
    logger.info("\n>>> Inferencia en TEST <<<")
    y_probs_test, _ = run_inference(trainer, test_dataset)
    test_preds_int = (y_probs_test >= optimal_threshold).astype(int)

    logger.info(
        f"Predicciones TEST (threshold={optimal_threshold:.4f}): "
        f"YES={int(np.sum(test_preds_int == 1))} "
        f"({100 * np.mean(test_preds_int == 1):.2f}%) | "
        f"NO={int(np.sum(test_preds_int == 0))} "
        f"({100 * np.mean(test_preds_int == 0):.2f}%)"
    )

    # Guardar probabilidades test
    test_probs_path = os.path.join(
        cfg["predictions_dir"],
        f"{cfg['group_id']}_{cfg['model_id']}_probs_test.json",
    )
    save_probs_json(test_df["id_EXIST"].values, y_probs_test, test_probs_path)

    # Guardar predicciones test en formato PyEvALL (fichero de entrega)
    test_pyevall_path = os.path.join(
        cfg["predictions_dir"],
        f"{cfg['group_id']}_{cfg['model_id']}.json",
    )
    save_predictions_pyevall(
        test_df["id_EXIST"].values, test_preds_int, test_pyevall_path
    )

    # ------------------------------------------------------------------
    # 8. Resumen final
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"  Lang usado para train  : {cfg['lang']}")
    logger.info(f"  Model ID               : {cfg['model_id']}")
    logger.info(f"  Threshold óptimo       : {optimal_threshold:.4f}")
    logger.info(f"  F1-macro (val)         : {val_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-binary (val)        : {val_metrics['f1_binary']:.4f}")
    logger.info(f"  Accuracy (val)         : {val_metrics['accuracy']:.4f}")
    logger.info(f"\n  Archivos generados:")
    logger.info(f"    Modelo              : {cfg['save_path']}")
    logger.info(f"    Probs val           : {val_probs_path}")
    logger.info(f"    Preds val PyEvALL   : {val_pyevall_path}")
    logger.info(f"    Probs test          : {test_probs_path}")
    logger.info(f"    Preds test PyEvALL  : {test_pyevall_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()