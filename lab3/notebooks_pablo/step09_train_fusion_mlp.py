"""
Entrenamiento de Modelo de Fusión Multimodal (Texto + Video) - EXIST 2026
Lee los embeddings generados previamente (.npz) de F2LLM y de VideoTransformer.
Se entrenan secuencialmente tres variaciones de un clasificador MLP (concat, add, gated),
se evalúan en validación, y automáticamente se guardan sólo las predicciones del mejor.
"""

import argparse
import json
import logging
import os
import sys
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
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
        description="Auto-búsqueda de Fusión Multimodal MLP para Texto y Video"
    )
    # Único parámetro a elegir en flujo normal
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["es", "en", "es_en"],
        help="Idioma de entrenamiento.",
    )
    # Rutas por defecto
    parser.add_argument(
        "--main_path",
        type=str,
        default="..",
        help="Ruta raíz del proyecto",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="BeingChillingWeWillWin",
        help="Identificador de grupo",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carga de Datos y Metadatos
# ---------------------------------------------------------------------------
def load_json_dataset(path):
    logger.info(f"Cargando metadatos: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data.values())

def filter_by_lang(df, lang):
    if lang == "es_en":
        return df
    filtered = df[df["lang"] == lang].reset_index(drop=True)
    return filtered

def split_train_val(df, val_fraction, seed, label_col="label_int"):
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df[label_col], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def load_npz_dict(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Archivo NPZ no encontrado en: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    obj_dict = {key: data[key] for key in data.keys()}
    first_key = list(obj_dict.keys())[0]
    dim = obj_dict[first_key].shape[-1]
    logger.info(f"Cargados {len(obj_dict)} embeddings desde {npz_path} (Dimensión: {dim})")
    return obj_dict, dim


# ---------------------------------------------------------------------------
# Dataset de Fusión Pytorch
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, df, text_dict, video_dict, has_label=True):
        self.df = df
        self.valid_data = []
        not_found = 0
        
        for _, row in df.iterrows():
            exist_id = str(row['id_EXIST'])
            if exist_id in text_dict and exist_id in video_dict:
                item = {
                    'id': exist_id,
                    'text_emb': torch.tensor(text_dict[exist_id], dtype=torch.float32),
                    'video_emb': torch.tensor(video_dict[exist_id], dtype=torch.float32)
                }
                if has_label and 'label_int' in row:
                    item['labels'] = torch.tensor(row['label_int'], dtype=torch.float32)
                self.valid_data.append(item)
            else:
                not_found += 1
                
        if not_found > 0:
            logger.warning(
                f"  Dataset omitió {not_found}/{len(df)} muestras por no estar presentes en uno de los NPZ."
            )
            
    def __len__(self):
        return len(self.valid_data)
        
    def __getitem__(self, idx):
        return self.valid_data[idx]

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    text_emb = torch.stack([item['text_emb'] for item in batch])
    video_emb = torch.stack([item['video_emb'] for item in batch])
    
    res = {
        'ids': ids,
        'text_emb': text_emb,
        'video_emb': video_emb
    }
    
    if 'labels' in batch[0]:
        res['labels'] = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)
        
    return res


# ---------------------------------------------------------------------------
# Arquitecturas Multimodales (Fusión)
# ---------------------------------------------------------------------------
class FusionClassifier(nn.Module):
    def __init__(self, text_dim, video_dim, hidden_dim=512, dropout=0.3, fusion_mode="concat"):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.dropout_layer = nn.Dropout(dropout)
        
        if fusion_mode == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(text_dim + video_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            
        elif fusion_mode == "add":
            self.proj_text = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU())
            self.proj_video = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.ReLU())
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        elif fusion_mode == "gated":
            self.proj_text = nn.Linear(text_dim, hidden_dim)
            self.proj_video = nn.Linear(video_dim, hidden_dim)
            self.gate = nn.Linear(text_dim + video_dim, hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            raise ValueError(f"Técnica {fusion_mode} no soportada.")

    def forward(self, text_emb, video_emb, labels=None, ids=None, **kwargs):
        t_drop = self.dropout_layer(text_emb)
        v_drop = self.dropout_layer(video_emb)
        
        if self.fusion_mode == "concat":
            fused = torch.cat([t_drop, v_drop], dim=1)
            logits = self.mlp(fused)
            
        elif self.fusion_mode == "add":
            t = self.proj_text(t_drop)
            v = self.proj_video(v_drop)
            logits = self.mlp(t + v)
            
        elif self.fusion_mode == "gated":
            t = torch.tanh(self.proj_text(t_drop))
            v = torch.tanh(self.proj_video(v_drop))
            z = torch.sigmoid(self.gate(torch.cat([text_emb, video_emb], dim=1)))
            fused = z * t + (1 - z) * v
            logits = self.mlp(fused)
            
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), labels.float())
            
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Trainer Auxiliar
# ---------------------------------------------------------------------------
class BinaryTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        _ = inputs.pop("ids", None)
        labels = inputs.pop("labels").float()
        
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Métricas y Utilería
# ---------------------------------------------------------------------------
LABEL_MAP_INV = {0: "NO", 1: "YES"}
LABEL_MAP = {"NO": 0, "YES": 1}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.tensor(logits).squeeze(-1)
    probs = torch.sigmoid(logits_tensor).numpy()
    preds = (probs >= 0.5).astype(int)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_macro": f1_macro}

def find_optimal_threshold(y_true, y_probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_th, best_score = 0.5, 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_th = th
    return best_th, best_score

def save_predictions_pyevall(ids, preds_int, path, test_case="EXIST2026"):
    records = [
        {"test_case": test_case, "id": str(id_), "value": LABEL_MAP_INV[int(p)]}
        for id_, p in zip(ids, preds_int)
    ]
    pd.DataFrame(records).to_json(path, orient="records", force_ascii=False)

def save_probs_json(ids, probs, path, labels=None):
    records = []
    for i, (id_, prob) in enumerate(zip(ids, probs)):
        rec = {"id": str(id_), "prob_YES": round(float(prob), 6)}
        if labels is not None:
            rec["label"] = LABEL_MAP_INV[int(labels[i])]
        records.append(rec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Bucle Principal Automatizado
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    main_dir = args.main_path
    
    # Manejo dinámico de rutas NPZ por defecto
    text_npz = os.path.join(main_dir, "embeddings", "f2llm_embeddings.npz")
    video_npz = os.path.join(main_dir, "embeddings", f"{args.group_id}_videoTrans_{args.lang}_generated_embeddings.npz")
    
    pred_dir = os.path.join(main_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"  FUSIÓN MULTIMODAL AUTO-TEST (Video + Texto)")
    logger.info(f"  Idioma de ejecución: {args.lang}")
    logger.info("=" * 60)

    # 1. Cargar metadatos JSON
    raw_train = load_json_dataset(os.path.join(main_dir, "preprocessed_data", "training.json"))
    raw_test = load_json_dataset(os.path.join(main_dir, "preprocessed_data", "test.json"))

    train_full = filter_by_lang(raw_train, args.lang)
    train_full["label_int"] = train_full["label"].map(LABEL_MAP)
    train_full = train_full.dropna(subset=["label_int"]).reset_index(drop=True)

    test_df = raw_test.reset_index(drop=True)
    test_df["label_int"] = -1

    # 2. Split train/val
    train_df, val_df = split_train_val(train_full, val_fraction=0.2, seed=1337, label_col="label_int")

    # 3. Cargar ambos diccionarios de Embeddings
    text_dict, dim_t = load_npz_dict(text_npz)
    video_dict, dim_v = load_npz_dict(video_npz)

    # 4. Crear Datasets Unificados
    train_dataset = FusionDataset(train_df, text_dict, video_dict, has_label=True)
    eval_dataset  = FusionDataset(val_df, text_dict, video_dict, has_label=True)
    test_dataset  = FusionDataset(test_df, text_dict, video_dict, has_label=False)

    logger.info(f"Ejemplos hábiles: train={len(train_dataset)}, val={len(eval_dataset)}, test={len(test_dataset)}")

    # -------------------------------------------------------------------
    # Ciclo Interactivo sobre Modelos de Fusión
    # -------------------------------------------------------------------
    modos_fusion = ["concat", "gated", "add"]
    mejores_métricas = {
        "f1_val": -1.0,
        "modo": None,
        "preds_test": None,
        "probs_test": None,
        "preds_val": None,
        "probs_val": None,
        "labels_val": None,
        "th": 0.5
    }

    # Hiperparámetros constantes
    epochs = 15
    lr = 2e-4
    train_batch = 64
    eval_batch = 128
    hidden_dim = 512
    dropout = 0.3

    val_ids = [item['id'] for item in eval_dataset]
    test_ids = [item['id'] for item in test_dataset]

    for mode in modos_fusion:
        logger.info(f"\n" + "-"*50)
        logger.info(f" Evaluando Arquitectura: {mode.upper()}")
        logger.info("-" * 50)
        
        output_dir = os.path.join(main_dir, "weights", f"{args.group_id}_fusion_{mode}_{args.lang}_train")
        os.makedirs(output_dir, exist_ok=True)
        
        # Iniciar modelo fresco
        model = FusionClassifier(
            text_dim=dim_t,
            video_dim=dim_v,
            hidden_dim=hidden_dim,
            dropout=dropout,
            fusion_mode=mode
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            load_best_model_at_end=True,
            bf16=torch.cuda.is_available(),
            seed=1337,
            logging_steps=50,
            report_to="none",
            remove_unused_columns=False, 
        )

        trainer = BinaryTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Inferencia Validacion
        val_pred_raw = trainer.predict(eval_dataset)
        y_probs_val = torch.sigmoid(torch.tensor(val_pred_raw.predictions).squeeze(-1)).numpy()
        y_true_val = val_pred_raw.label_ids

        opt_th, val_f1 = find_optimal_threshold(y_true_val, y_probs_val)
        val_preds_int = (y_probs_val >= opt_th).astype(int)

        logger.info(f"[*] Resultado para {mode.upper()} -> F1-Macro Validacion: {val_f1:.4f} (Threshold: {opt_th:.4f})")

        # Inferencia Test temporal en memoria
        test_pred_raw = trainer.predict(test_dataset)
        y_probs_test = torch.sigmoid(torch.tensor(test_pred_raw.predictions).squeeze(-1)).numpy()
        test_preds_int = (y_probs_test >= opt_th).astype(int)

        # Si supera al mejor modelo actual, guardamos estado en RAM
        if val_f1 > mejores_métricas["f1_val"]:
            mejores_métricas["f1_val"] = val_f1
            mejores_métricas["modo"] = mode
            mejores_métricas["preds_test"] = test_preds_int
            mejores_métricas["probs_test"] = y_probs_test
            mejores_métricas["preds_val"] = val_preds_int
            mejores_métricas["probs_val"] = y_probs_val
            mejores_métricas["labels_val"] = y_true_val
            mejores_métricas["th"] = opt_th
        
        # Limpieza de VRAM
        del model
        del trainer
        del training_args
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------
    # Escritura en Disco del Ganador Definitivo
    # -------------------------------------------------------------------
    best_mode = mejores_métricas["modo"]
    logger.info("\n" + "=" * 60)
    logger.info(f"  🌟 GANADOR GLOBAL: '{best_mode.upper()}' ")
    logger.info(f"  F1-Macro (Validation): {mejores_métricas['f1_val']:.4f}")
    logger.info("=" * 60)

    model_id = f"fusion_{best_mode}_{args.lang}"
    val_pyevall_path = os.path.join(pred_dir, f"{args.group_id}_{model_id}_val.json")
    test_pyevall_path = os.path.join(pred_dir, f"{args.group_id}_{model_id}.json")

    logger.info(f"Guardando predicciones FINALES del mejor modelo en {pred_dir}...")
    save_predictions_pyevall(val_ids, mejores_métricas["preds_val"], val_pyevall_path)
    save_predictions_pyevall(test_ids, mejores_métricas["preds_test"], test_pyevall_path)
    
    logger.info("¡Script de Auto-Búsqueda de Fusión completado con éxito!")


if __name__ == "__main__":
    main()
