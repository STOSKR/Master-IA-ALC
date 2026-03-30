"""
Entrenamiento de modelo Transformer propio para secuencias de embeddings de video (EXIST 2026).
Acepta embeddings `.npz` y entrena un encoder transformer, inyectando un [CLS] 
en la primera posición para clasificar, además de soportar videos con 1 solo frame.
"""

import argparse
import json
import logging
import math
import os
import sys

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
        description="Entrenamiento Video Transformer para clasificación binaria"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="es_en",
        choices=["es", "en", "es_en"],
        help="Filtro de idioma sobre los datos JSON para train.",
    )
    parser.add_argument(
        "--main_path",
        type=str,
        default="..",
        help="Ruta raíz del proyecto (default: ..)",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="BeingChillingWeWillWin",
        help="Identificador de grupo para los ficheros de predicciones",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--train_batch",
        type=int,
        default=32,
        help="Batch size de entrenamiento por dispositivo (default: 32)",
    )
    parser.add_argument(
        "--eval_batch",
        type=int,
        default=64,
        help="Batch size de evaluación por dispositivo (default: 64)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fracción del train original usada como validación (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    # Hiperparámetros del Transformer
    parser.add_argument("--num_heads", type=int, default=8, help="Número de cabezas de atención")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de capas del Transformer encoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout del Transformer")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constantes derivadas de lang
# ---------------------------------------------------------------------------
def build_config(args):
    lang = args.lang
    main = args.main_path
    model_id = f"videoTrans_{lang}"

    config = dict(
        lang=lang,
        model_id=model_id,
        group_id=args.group_id,
        label_column="label",
        
        # Rutas de datos
        data_train_path=os.path.join(main, "preprocessed_data", "training.json"),
        data_test_path=os.path.join(main, "preprocessed_data", "test.json"),
        embeddings_npz_path=os.path.join(main, "embeddings", "dinov3_embeddings.npz"),
        
        # Salidas
        output_dir=os.path.join(main, "weights", f"VideoTransformer_{lang}_train"),
        save_path=os.path.join(main, "weights", f"VideoTransformer_{lang}_final"),
        predictions_dir=os.path.join(main, "predictions"),
        embeddings_out_npz=os.path.join(main, "embeddings", f"{args.group_id}_{model_id}_generated_embeddings.npz"),
        
        # Hiperparámetros
        epochs=args.epochs,
        lr=args.lr,
        train_batch=args.train_batch,
        eval_batch=args.eval_batch,
        val_split=args.val_split,
        seed=args.seed,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    return config


# ---------------------------------------------------------------------------
# Carga de JSON (Filosofía de step05)
# ---------------------------------------------------------------------------
def load_json_dataset(path):
    logger.info(f"Cargando datos JSON desde: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.values())
    logger.info(f"  → {len(df)} ejemplos cargados")
    return df

def filter_by_lang(df, lang):
    if lang == "es_en":
        return df
    filtered = df[df["lang"] == lang].reset_index(drop=True)
    logger.info(f"  → {len(filtered)} ejemplos después de filtrar por lang='{lang}'")
    return filtered

def split_train_val(df, val_fraction, seed, label_col="label_int"):
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df[label_col],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga de Embeddings de Video (.npz)
# ---------------------------------------------------------------------------
def load_video_embeddings(npz_path):
    logger.info(f"Cargando embeddings de video desde: {npz_path}")
    data = np.load(npz_path)
    video_dict = {}
    
    for key in data.keys():
        # Formato esperado: video_{momento} o cualquier_cosa
        clean_key = key.replace(".jpg", "").replace(".png", "")
        
        if "_" in clean_key:
            parts = clean_key.rsplit("_", 1)
            vid = parts[0]
            try:
                momento = float(parts[1])
            except ValueError:
                momento = parts[1] # fallback string
        else:
            vid = clean_key
            momento = 0
            
        if vid not in video_dict:
            video_dict[vid] = []
        video_dict[vid].append((momento, data[key]))
        
    embed_dim = None
    for vid in video_dict:
        # Ordenar por momento (de menor a mayor)
        video_dict[vid].sort(key=lambda x: x[0])
        # Apilar frames en un array [Seq_len, Dim]
        stacked = np.stack([x[1] for x in video_dict[vid]])
        video_dict[vid] = stacked
        
        if embed_dim is None:
            embed_dim = stacked.shape[1]
            
    logger.info(f"  → {len(video_dict)} secuencias de video procesadas (Dimensión: {embed_dim})")
    return video_dict, embed_dim


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class VideoDataset(Dataset):
    def __init__(self, df, video_dict, has_label=True):
        self.df = df
        self.valid_data = []

        not_found = 0
        for _, row in df.iterrows():
            exist_id = str(row['id_EXIST'])
            tiktok_id = str(row['id_Tiktok'])  # clave real en video_dict

            if tiktok_id in video_dict:
                item = {
                    'id': exist_id,
                    'sequences': torch.tensor(video_dict[tiktok_id], dtype=torch.float32)
                }
                if has_label and 'label_int' in row:
                    item['labels'] = torch.tensor(row['label_int'], dtype=torch.float32)
                self.valid_data.append(item)
            else:
                not_found += 1

        if not_found > 0:
            logger.warning(
                f"  [VideoDataset] {not_found}/{len(df)} ejemplos sin embedding en video_dict."
            )
                
    def __len__(self):
        return len(self.valid_data)
        
    def __getitem__(self, idx):
        return self.valid_data[idx]

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    sequences = [item['sequences'] for item in batch]
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    
    # Pad secuencias: [Batch, MaxSeqLen, Dim]
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    res = {
        'ids': ids,
        'sequences': padded_seqs,
        'lengths': lengths
    }
    
    if 'labels' in batch[0]:
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)
        res['labels'] = labels
        
    return res


# ---------------------------------------------------------------------------
# Arquitectura del Modelo
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class VideoClassificationModel(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer para longitud > 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP para longitud == 1
        self.mlp_single = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Capa final compartida
        self.classifier = nn.Linear(embed_dim, 1)

    def get_embeddings(self, sequences, lengths):
        B = sequences.size(0)
        D = sequences.size(2)
        device = sequences.device
        
        # Almacenar la representación latente de cada video (Dim: D)
        output_emb = torch.zeros(B, D, device=device, dtype=sequences.dtype)
        
        mask_gt1 = lengths > 1
        mask_eq1 = lengths == 1
        
        # Lógica videos largos
        if mask_gt1.any():
            x_gt1 = sequences[mask_gt1]
            l_gt1 = lengths[mask_gt1]
            max_l = l_gt1.max().item()
            x_gt1 = x_gt1[:, :max_l, :] # recorta hasta max length real en el sub-batch
            
            # [B, 1, D] expandido y concatenado
            cls_tokens = self.cls_token.expand(x_gt1.size(0), -1, -1)
            x_gt1 = torch.cat((cls_tokens, x_gt1), dim=1)
            
            x_gt1 = self.pos_encoder(x_gt1)
            
            # Mask para posiciones de padding de Transformer: (True=ignorar)
            # Longitud incluye la posición 0 del CLS
            pad_mask = torch.arange(max_l + 1, device=device).unsqueeze(0) >= (l_gt1 + 1).unsqueeze(1)
            
            trans_out = self.transformer(x_gt1, src_key_padding_mask=pad_mask)
            
            # Recoger output del CLS (posición 0)
            output_emb[mask_gt1] = trans_out[:, 0, :]
            
        # Lógica videos estáticos de 1 frame
        if mask_eq1.any():
            x_eq1 = sequences[mask_eq1, 0, :]
            feat_eq1 = self.mlp_single(x_eq1)
            output_emb[mask_eq1] = feat_eq1.to(output_emb.dtype)
            
        return output_emb

    def forward(self, sequences, lengths, labels=None, ids=None, **kwargs):
        # Obtener embeddings generados
        output_emb = self.get_embeddings(sequences, lengths)
        
        # Clasificar
        logits = self.classifier(output_emb)
        
        loss = None
        if labels is not None:
            # logits forma [B, 1], labels [B]
            loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), labels.float())
            
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class BinaryTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Trainer por defecto puede dar problemas si model.forward usa kwargs diferentes
        # Por seguridad calcularemos aquí también. HF Trainer delega el loss si compute_loss está definida.
        _ = inputs.pop("ids", None)
        labels = inputs.pop("labels").float()
        
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Métricas y Guardado
# ---------------------------------------------------------------------------
LABEL_MAP = {"NO": 0, "YES": 1}
LABEL_MAP_INV = {0: "NO", 1: "YES"}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.tensor(logits).squeeze(-1)
    probs = torch.sigmoid(logits_tensor).numpy()
    preds = (probs >= 0.5).astype(int)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_bin = f1_score(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)

    return {"f1_macro": f1_macro, "f1_binary": f1_bin, "accuracy": acc}


def find_optimal_threshold(y_true, y_probs, metric="f1_macro"):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_th, best_score = 0.5, 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_th = th

    logger.info(f"Threshold óptimo (max F1-macro en val): {best_th:.4f} → F1-macro={best_score:.4f}")
    return best_th, best_score

def evaluate_full(y_true, y_probs, threshold, split_name="val"):
    preds = (y_probs >= threshold).astype(int)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_bin = f1_score(y_true, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_true, preds)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
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
    return {"f1_macro": f1_macro, "f1_binary": f1_bin, "accuracy": acc}

def save_predictions_pyevall(ids, preds_int, path, test_case="EXIST2026"):
    records = [
        {"test_case": test_case, "id": str(id_), "value": LABEL_MAP_INV[int(p)]}
        for id_, p in zip(ids, preds_int)
    ]
    pd.DataFrame(records).to_json(path, orient="records", force_ascii=False)
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


def extract_and_save_embeddings(model, dataset_train, dataset_test, path_out):
    """
    Genera el archivo .npz de embeddings latentes pasados por el encoder
    """
    logger.info("Extrayendo representaciones latentes [CLS] / MLP para nuevo .npz...")
    model.eval()
    device = next(model.parameters()).device
    
    all_embs = {}
    
    for ds in [dataset_train, dataset_test]:
        dataloader = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate_fn, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                seqs = batch['sequences'].to(device)
                lens = batch['lengths'].to(device)
                ids = batch['ids']
                
                out_emb = model.get_embeddings(seqs, lens)
                
                for id_, emb in zip(ids, out_emb.cpu().numpy()):
                    all_embs[str(id_)] = emb

    np.savez_compressed(path_out, **all_embs)
    logger.info(f"→ Embeddings guardados en: {path_out} (Total videos procesados: {len(all_embs)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = build_config(args)

    logger.info("=" * 60)
    logger.info(f"  Entrenamiento Video Transformer")
    logger.info(f"  Lang filtro   : {cfg['lang']}")
    logger.info("=" * 60)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["save_path"], exist_ok=True)
    os.makedirs(cfg["predictions_dir"], exist_ok=True)

    # 1. Carga de metadatos
    raw_train = load_json_dataset(cfg["data_train_path"])
    raw_test = load_json_dataset(cfg["data_test_path"])

    train_full = filter_by_lang(raw_train, cfg["lang"])
    test_df = raw_test.reset_index(drop=True)

    train_full["label_int"] = train_full[cfg["label_column"]].map(LABEL_MAP)
    train_full = train_full.dropna(subset=["label_int"]).reset_index(drop=True)

    # 2. Split train / eval
    train_df, val_df = split_train_val(train_full, cfg["val_split"], cfg["seed"], "label_int")

    # 3. Cargar Embeddings reales
    if not os.path.exists(cfg["embeddings_npz_path"]):
        # Default placeholder si no está donde creemos, en fallback lo probamos sin path final
        logger.error(f"Embeddings npz no encontrados en: {cfg['embeddings_npz_path']}")
        return
        
    video_dict, embed_dim = load_video_embeddings(cfg["embeddings_npz_path"])

    # 4. Preparar Datasets PyTorch
    train_dataset = VideoDataset(train_df, video_dict, has_label=True)
    eval_dataset = VideoDataset(val_df, video_dict, has_label=True)
    
    # Dataset Test sin etiqueta para evaluar e inferir
    test_df["label_int"] = -1
    test_dataset = VideoDataset(test_df, video_dict, has_label=False)
    
    # Dataset para re-extracción embeddings al final (todo el train origin)
    full_train_dataset = VideoDataset(train_full, video_dict, has_label=False)

    logger.info(
        f"Tamaños dataloading: train={len(train_dataset)}, "
        f"val={len(eval_dataset)}, test={len(test_dataset)}"
    )

    # 5. Modelo
    model = VideoClassificationModel(
        embed_dim=embed_dim,
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    )

    # 6. Entrenamiento via HF Trainer
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["train_batch"],
        per_device_eval_batch_size=cfg["eval_batch"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),
        lr_scheduler_type="cosine",
        seed=cfg["seed"],
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False, # CRÍTICO: permite a trainer mandar param "sequences", "lengths" e "ids"
    )

    trainer = BinaryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    logger.info(">>> Iniciando entrenamiento de Video Transformer <<<")
    trainer.train()

    logger.info(f"Guardando modelo base Transformer en {cfg['save_path']}")
    torch.save(model.state_dict(), os.path.join(cfg["save_path"], "model_weights.pth"))

    # 7. Evaluación y Predicciones
    logger.info("\n>>> Inferencia en VAL <<<")
    predictions_val = trainer.predict(eval_dataset)
    y_probs_val = torch.sigmoid(torch.tensor(predictions_val.predictions).squeeze(-1)).numpy()
    y_true_val = predictions_val.label_ids

    optimal_threshold, _ = find_optimal_threshold(y_true_val, y_probs_val, metric="f1_macro")
    val_metrics = evaluate_full(y_true_val, y_probs_val, optimal_threshold, split_name="val")

    # Inferencia TEST
    logger.info("\n>>> Inferencia en TEST <<<")
    predictions_test = trainer.predict(test_dataset)
    y_probs_test = torch.sigmoid(torch.tensor(predictions_test.predictions).squeeze(-1)).numpy()
    test_preds_int = (y_probs_test >= optimal_threshold).astype(int)

    # Ficheros de salida
    val_pyevall_path = os.path.join(cfg["predictions_dir"], f"{cfg['group_id']}_{cfg['model_id']}_val.json")
    test_pyevall_path = os.path.join(cfg["predictions_dir"], f"{cfg['group_id']}_{cfg['model_id']}.json")
    
    val_ids = [item['id'] for item in eval_dataset]
    test_ids = [item['id'] for item in test_dataset]
    
    save_predictions_pyevall(val_ids, (y_probs_val >= optimal_threshold).astype(int), val_pyevall_path)
    save_predictions_pyevall(test_ids, test_preds_int, test_pyevall_path)

    # 8. Extraer embeddings generados en npz
    extract_and_save_embeddings(
        model=model, 
        dataset_train=full_train_dataset, 
        dataset_test=test_dataset, 
        path_out=cfg["embeddings_out_npz"]
    )
    
    logger.info("Fin del proceso.")

if __name__ == "__main__":
    main()