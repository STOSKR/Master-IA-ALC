import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# =========================
# Rutas de entrada/salida
# =========================
TRAIN_JSON = "/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/training.json"
TEST_JSON = "/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/test.json"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embeddings")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "f2llm_embeddings.npz")

# =========================
# Configuración del modelo
# =========================
MODEL_NAME = "codefuse-ai/F2LLM-4B"

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Si es dict → convertir a lista
    if isinstance(data, dict):
        # Caso 1: dict con items
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        
        # Caso 2: tiene key tipo "data"
        elif "data" in data:
            return data["data"]

    return data

def encode(sentences, tokenizer, model):
    """Genera embeddings normalizados"""
    batch_size = len(sentences)
    
    tokenized_inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(model.device)

    with torch.inference_mode():
        last_hidden_state = model(**tokenized_inputs).last_hidden_state

    eos_positions = tokenized_inputs.attention_mask.sum(dim=1) - 1
    embeddings = last_hidden_state[
        torch.arange(batch_size, device=model.device),
        eos_positions
    ]

    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando datos...")
    train_data = load_data(TRAIN_JSON)
    test_data = load_data(TEST_JSON)

    all_data = train_data + test_data
    print(f"Total textos: {len(all_data)}")

    print("Cargando modelo F2LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Modelo cargado. Generando embeddings...")

    embeddings_dict = {}

    BATCH_SIZE = 16  # ajusta según VRAM

    for i in tqdm(range(0, len(all_data), BATCH_SIZE), desc="Procesando textos"):
        batch = all_data[i:i + BATCH_SIZE]

        ids = [item["id_EXIST"] for item in batch]
        texts = [item["text"] for item in batch]

        try:
            emb = encode(texts, tokenizer, model)

            emb_np = emb.cpu().numpy().astype(np.float32)

            for j, id_ in enumerate(ids):
                embeddings_dict[id_] = emb_np[j]

        except Exception as e:
            print(f"\nError en batch {i}: {e}")

    print(f"\nGuardando {len(embeddings_dict)} embeddings en {OUTPUT_FILE}...")
    np.savez_compressed(OUTPUT_FILE, **embeddings_dict)

if __name__ == "__main__":
    main()