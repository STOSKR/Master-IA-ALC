# Ensemble y Limpieza de Results V2

## 📋 Contenido

### 1. Notebook de Ensemble: `11_ensemble_top5.ipynb`

**Ubicación:** `notebooks_v2/11_ensemble_top5.ipynb`

Este notebook crea un modelo ensemble usando los **Top 5 mejores modelos** de la versión V2:

1. **F2LLM-4B Tweet** (F1: 0.8532)
2. **F2LLM-4B Text Clean** (F1: 0.8317)
3. **KaLM Tweet** (F1: 0.8254)
4. **Ministral 3B** (F1: 0.8073)
5. **KaLM Text Clean** (F1: 0.8000)

**Método:** Votación mayoritaria (Majority Voting)

**Salidas:**
- `results_v2/ensemble/BeingChillingWeWillWin_ensemble_top5_val.json` - Predicciones de validación
- `results_v2/ensemble/BeingChillingWeWillWin_ensemble_top5.json` - Predicciones de test (para competición)

**Cómo ejecutar:**
```bash
cd notebooks_v2
jupyter notebook 11_ensemble_top5.ipynb
# O ejecutar todas las celdas
```

### 2. Script de Limpieza: `clean_results_v2.py`

**Ubicación:** `scripts/clean_results_v2.py`

Este script limpia las carpetas de `results_v2`, eliminando:
- ✗ Carpetas de checkpoints (`tweet/`, `text_clean/`, `*_lora/`)
- ✗ Pesos de modelos (`.safetensors`, `.bin`, `.pth`)
- ✗ Archivos de configuración de modelos
- ✗ Directorios vacíos

**Mantiene:**
- ✓ Carpetas `predictions/`
- ✓ Archivos JSON de predicciones
- ✓ Archivos CSV de comparación

**Cómo ejecutar:**
```bash
cd lab1
python scripts/clean_results_v2.py
```

El script pedirá confirmación antes de eliminar archivos.

### 3. Archivo de Ensemble Antiguo: `10_ensemble.ipynb`

**Ubicación:** `notebooks_v2/10_ensemble.ipynb`

Este notebook es más genérico y permite probar diferentes configuraciones de ensemble. El nuevo `11_ensemble_top5.ipynb` está optimizado específicamente para los 5 mejores modelos.

## 🎯 Recomendación para Competición

Basado en el análisis de resultados V1 vs V2:

**Modelo recomendado:** `F2LLM-4B Tweet (V2)` o `Ensemble Top 5 (V2)`

### Opción 1: Modelo Individual
**Archivo:** `results_v2/F2LLM-4B/predictions/BeingChillingWeWillWin_f2llm4B.json`

**Métricas (validación):**
- F1 Score: **0.8532**
- Accuracy: 0.8593
- Precision: 0.7966
- Recall: 0.9185

### Opción 2: Ensemble Top 5
**Archivo:** `results_v2/ensemble/BeingChillingWeWillWin_ensemble_top5.json` (después de ejecutar el notebook)

**Ventaja:** Típicamente mejora 1-2% sobre el mejor modelo individual

## 📊 Comparación de Versiones

Los resultados de V2 son superiores a V1 en general:

| Modelo | V1 F1 | V2 F1 | Mejor |
|--------|-------|-------|-------|
| F2LLM-4B Tweet | 0.8464 | **0.8532** | ✓ V2 |
| KaLM Tweet | 0.8167 | **0.8254** | ✓ V2 |
| Modelos Clásicos | 0.7345 | **0.7415** | ✓ V2 |

## 🚀 Pasos Sugeridos

1. **Ejecutar el notebook de ensemble:**
   ```bash
   cd notebooks_v2
   jupyter notebook 11_ensemble_top5.ipynb
   ```

2. **Evaluar si el ensemble mejora sobre F2LLM-4B individual**
   - Si mejora: Usar `BeingChillingWeWillWin_ensemble_top5.json`
   - Si no mejora: Usar `BeingChillingWeWillWin_f2llm4B.json`

3. **Limpiar archivos pesados (opcional):**
   ```bash
   cd lab1
   python scripts/clean_results_v2.py
   ```
   - Esto liberará varios GB de espacio
   - Solo mantiene las predicciones necesarias

## 📝 Notas

- El ensemble usa votación mayoritaria simple
- Todos los modelos tienen el mismo peso
- Los archivos de predicción siguen el formato EXIST 2025
- Los archivos están listos para subir a la competición
