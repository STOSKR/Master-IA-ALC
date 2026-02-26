#!/usr/bin/env python3
"""
Script para corregir imports incorrectos en los notebooks
"""
import json
import sys

def fix_ministral_imports(notebook_path):
    """Corrige imports de Ministral3 que no existen en transformers"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell.get('source', []))
        
        # Fix 1: Imports incorrectos
        if 'MistralCommonBackend' in source or 'Mistral3ForConditionalGeneration' in source:
            new_source = source
            
            # Reemplazar los imports incorrectos
            if 'from transformers import' in source and ('Mistral3ForConditionalGeneration' in source or 'MistralCommonBackend' in source):
                # Reemplazar con los imports correctos
                new_source = new_source.replace(
                    'Mistral3ForConditionalGeneration',
                    'AutoModelForCausalLM'
                ).replace(
                    'MistralCommonBackend',
                    'AutoTokenizer'
                )
                
            # Reemplazar uso del tokenizer
            new_source = new_source.replace(
                'MistralCommonBackend.from_pretrained',
                'AutoTokenizer.from_pretrained'
            )
            
            # Reemplazar uso del modelo
            new_source = new_source.replace(
                'Mistral3ForConditionalGeneration.from_pretrained',
                'AutoModelForCausalLM.from_pretrained'
            )
            
            if new_source != source:
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                  for i, line in enumerate(cell['source'])]
                modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Fixed Ministral imports in: {notebook_path}")
        return True
    return False

def fix_kalm_model(notebook_path):
    """Corrige el nombre del modelo KaLM que no existe"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell.get('source', []))
        
        # Fix: Modelo KaLM incorrecto
        if 'FernandoLpz/KaLM-ft-cov19en-hs' in source:
            new_source = source.replace(
                'FernandoLpz/KaLM-ft-cov19en-hs',
                'FernandoLpz/KaLM-Embedding'
            )
            
            if new_source != source:
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                  for i, line in enumerate(cell['source'])]
                modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"✓ Fixed KaLM model name in: {notebook_path}")
        return True
    return False

if __name__ == "__main__":
    print("Fixing notebook imports and model names...")
    print("=" * 60)
    
    # Notebooks con imports de Ministral incorrectos
    ministral_notebooks = [
        "07_Ministral3_8B_inference_tweet.ipynb",
        "08_Ministral3_8B_only_ft.ipynb",
        "09_Ministral3_8B_inference_ft.ipynb"
    ]
    
    # Notebooks con modelo KaLM incorrecto
    kalm_notebooks = [
        "05_KaLM_ft_tweet.ipynb",
        "06_KaLM_ft_text_clean.ipynb"
    ]
    
    print("\n📝 Fixing Ministral3 imports...")
    for nb in ministral_notebooks:
        try:
            fix_ministral_imports(nb)
        except FileNotFoundError:
            print(f"⚠ Not found: {nb}")
        except Exception as e:
            print(f"✗ Error in {nb}: {e}")
    
    print("\n📝 Fixing KaLM model names...")
    for nb in kalm_notebooks:
        try:
            fix_kalm_model(nb)
        except FileNotFoundError:
            print(f"⚠ Not found: {nb}")
        except Exception as e:
            print(f"✗ Error in {nb}: {e}")
    
    print("\n" + "=" * 60)
    print("Done! Changes:")
    print("  - Mistral3ForConditionalGeneration → AutoModelForCausalLM")
    print("  - MistralCommonBackend → AutoTokenizer")
    print("  - FernandoLpz/KaLM-ft-cov19en-hs → FernandoLpz/KaLM-Embedding")
