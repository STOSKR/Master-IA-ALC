#!/usr/bin/env python3
"""
Script para limpiar las carpetas de results_v2, eliminando checkpoints y pesos.
Solo mantiene los archivos de predicciones (validación y test).
"""

import os
import shutil
from pathlib import Path

# Directorio raíz de results_v2
RESULTS_DIR = Path("../results_v2")

# Carpetas y archivos a MANTENER
KEEP_PATTERNS = [
    "predictions/",  # Carpeta de predicciones
    "*.json",  # Archivos JSON de predicciones
    "*.csv",  # Archivos CSV de comparación
]

# Carpetas y archivos a ELIMINAR
DELETE_PATTERNS = [
    "tweet/",
    "text_clean/",
    "tweet_lora/",
    "text_clean_lora/",
    "lora_weights/",
    "checkpoint-*/",
    "logs/",
    "*.safetensors",
    "*.bin",
    "*.pth",
    "*.ckpt",
    "adapter_model.*",
    "pytorch_model.*",
    "model.safetensors",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "adapter_config.json",
    "README.md",
    "config.json",
    "generation_config.json",
]


def get_dir_size(path):
    """Calcula el tamaño total de un directorio en MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating size for {path}: {e}")
    return total_size / (1024 * 1024)  # MB


def should_delete(path, base_dir):
    """Determina si un archivo o carpeta debe eliminarse"""
    rel_path = os.path.relpath(path, base_dir)

    # Si es una carpeta de predicciones o model_comparison.csv, no eliminar
    if "predictions" in rel_path:
        return False
    if path.endswith("model_comparison.csv"):
        return False
    if path.endswith(".json") and "BeingChillingWeWillWin" in path:
        return False

    # Verificar patrones de eliminación
    for pattern in DELETE_PATTERNS:
        if pattern.endswith("/"):
            # Es una carpeta
            if pattern.rstrip("/") in rel_path.split(os.sep):
                return True
        else:
            # Es un archivo
            if pattern.startswith("*"):
                # Patrón wildcard
                ext = pattern[1:]
                if path.endswith(ext):
                    return True
            else:
                # Nombre exacto
                if os.path.basename(path) == pattern:
                    return True

    return False


def clean_directory(base_dir):
    """Limpia el directorio eliminando checkpoints y pesos"""
    print(f"\n{'='*80}")
    print(f"Limpiando: {base_dir}")
    print(f"{'='*80}\n")

    # Calcular tamaño inicial
    initial_size = get_dir_size(base_dir)
    print(f"Tamaño inicial: {initial_size:.2f} MB\n")

    deleted_items = []
    deleted_size = 0

    # Recorrer todos los subdirectorios
    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Eliminar archivos
        for filename in files:
            filepath = os.path.join(root, filename)
            if should_delete(filepath, base_dir):
                try:
                    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    os.remove(filepath)
                    deleted_items.append(("file", filepath, file_size))
                    deleted_size += file_size
                    print(
                        f"✗ Eliminado archivo: {os.path.relpath(filepath, base_dir)} ({file_size:.2f} MB)"
                    )
                except Exception as e:
                    print(f"Error eliminando {filepath}: {e}")

        # Eliminar directorios vacíos o específicos
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            if should_delete(dirpath, base_dir):
                try:
                    dir_size = get_dir_size(dirpath)
                    shutil.rmtree(dirpath)
                    deleted_items.append(("dir", dirpath, dir_size))
                    deleted_size += dir_size
                    print(
                        f"✗ Eliminado directorio: {os.path.relpath(dirpath, base_dir)} ({dir_size:.2f} MB)"
                    )
                except Exception as e:
                    print(f"Error eliminando {dirpath}: {e}")

    # Eliminar directorios vacíos restantes
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            if "predictions" not in dirpath:  # No tocar predictions
                try:
                    if not os.listdir(dirpath):  # Si está vacío
                        os.rmdir(dirpath)
                        print(
                            f"✗ Eliminado directorio vacío: {os.path.relpath(dirpath, base_dir)}"
                        )
                except Exception as e:
                    pass

    # Calcular tamaño final
    final_size = get_dir_size(base_dir)

    print(f"\n{'='*80}")
    print(f"RESUMEN DE LIMPIEZA")
    print(f"{'='*80}")
    print(f"Tamaño inicial:  {initial_size:.2f} MB")
    print(f"Tamaño final:    {final_size:.2f} MB")
    print(
        f"Espacio liberado: {deleted_size:.2f} MB ({(deleted_size/initial_size*100):.1f}%)"
    )
    print(f"Items eliminados: {len(deleted_items)}")
    print(f"  Archivos:       {len([x for x in deleted_items if x[0] == 'file'])}")
    print(f"  Directorios:    {len([x for x in deleted_items if x[0] == 'dir'])}")
    print(f"{'='*80}\n")


def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("LIMPIEZA DE RESULTS_V2")
    print("=" * 80)
    print("\nEste script eliminará:")
    print("  - Carpetas de checkpoints (tweet/, text_clean/, *_lora/)")
    print("  - Pesos de modelos (*.safetensors, *.bin, *.pth)")
    print("  - Archivos de configuración de modelos")
    print("\nSe mantendrán:")
    print("  - Carpetas 'predictions/'")
    print("  - Archivos JSON de predicciones")
    print("  - Archivos CSV de comparación")
    print("=" * 80)

    response = input("\n¿Desea continuar? (sí/no): ").strip().lower()
    if response not in ["sí", "si", "s", "yes", "y"]:
        print("Operación cancelada.")
        return

    # Obtener todos los subdirectorios de models
    base_path = Path(RESULTS_DIR)
    if not base_path.exists():
        print(f"\nError: No se encontró el directorio {RESULTS_DIR}")
        return

    # Procesar cada subdirectorio de modelo
    model_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    total_initial_size = 0
    total_final_size = 0

    for model_dir in model_dirs:
        initial_size = get_dir_size(model_dir)
        total_initial_size += initial_size
        clean_directory(model_dir)
        final_size = get_dir_size(model_dir)
        total_final_size += final_size

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN TOTAL")
    print("=" * 80)
    print(f"Modelos procesados:  {len(model_dirs)}")
    print(f"Tamaño inicial:      {total_initial_size:.2f} MB")
    print(f"Tamaño final:        {total_final_size:.2f} MB")
    print(f"Espacio liberado:    {(total_initial_size - total_final_size):.2f} MB")
    print(
        f"Reducción:           {((total_initial_size - total_final_size)/total_initial_size*100):.1f}%"
    )
    print("=" * 80)
    print("\n✓ Limpieza completada exitosamente!\n")


if __name__ == "__main__":
    main()
