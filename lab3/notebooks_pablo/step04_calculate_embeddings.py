import os
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Rutas de entrada y salida
SCENES_DIR = "/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/scenes"

os.environ["HF_TOKEN"] = ""

# Directorio donde se guardarán los embeddings: "../embeddings" (relativo a la ubicación de este script)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embeddings")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dinov3_embeddings.npz")

def main():
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Buscando imágenes en {SCENES_DIR}...")
    image_paths = []
    # Buscar varios formatos por si acaso, incluyendo subcarpetas
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(SCENES_DIR, "**", ext), recursive=True))
    
    if not image_paths:
        print("No se encontraron imágenes en el directorio.")
        return
        
    print(f"Se encontraron {len(image_paths)} imágenes. Cargando modelo DINOv3...")
    
    # Cargar procesador y modelo DINOv3
    pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
    )
    # Asegurar que el modelo está en evaluación (aunque no afecte a dinov3, es buena práctica)
    model.eval()
    
    print("Modelo cargado. Comenzando cálculo de embeddings...")
    embeddings_dict = {}
    
    # Procesar imagen por imagen
    for img_path in tqdm(image_paths, desc="Procesando imágenes"):
        try:
            # Extraer el ID: nombre del archivo sin extensión
            img_id = Path(img_path).stem
            
            # Cargar imagen con PIL y asegurar que sea RGB para evitar problemas con imágenes en blanco y negro o con canal alfa
            image = Image.open(img_path).convert("RGB")
            
            # Preparar inputs y enviarlos al dispositivo donde esté el modelo
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            
            # Hacer inferencia
            with torch.inference_mode():
                outputs = model(**inputs)
                
            # Extraer el vector del embedding (pooler_output)
            pooled_output = outputs.pooler_output
            
            # Quitar la dimensión de batch: [1, dim] -> [dim] y convertir a float de numpy
            # (Convertimos también a float32 o float16 para que la serialización funcione bien y los arrastre a la CPU primero)
            embedding_np = pooled_output.squeeze(0).cpu().numpy().astype(np.float32)
            
            # Guardar el numpy array en el diccionario usando el ID como clave
            embeddings_dict[img_id] = embedding_np
            
        except Exception as e:
            print(f"\nError al procesar la imagen {img_path}: {e}")
            
    print(f"\nGuardando {len(embeddings_dict)} embeddings en {OUTPUT_FILE}...")
    
    # Guardar todos los embeddings en un archivo comprimido .npz
    # Esto creará un diccionario donde cada clave es el ID de la imagen, y el valor su embedding
    np.savez_compressed(OUTPUT_FILE, **embeddings_dict)

if __name__ == "__main__":
    main()
