# Análisis de Archivos JSON y Ensemble Conservador

## 📁 ¿Por qué hay tantos archivos JSON?

Cada modelo genera varios archivos con propósitos específicos:

### **Archivos Estándar (la mayoría de modelos):**
```
📂 results_v2/[MODELO]/predictions/
  ├── dev_predictions_temp.json      ← Predicciones en validación
  ├── dev_gold_temp.json             ← Ground truth de validación
  └── BeingChillingWeWillWin_*.json  ← Predicciones de TEST (para competición)
```

### **F2LLM-4B (caso especial - DOBLE):**
```
📂 results_v2/F2LLM-4B/predictions/
  ├── dev_predictions_temp.json            ← Validación con "tweet" (texto original)
  ├── dev_predictions_temp_clean.json      ← Validación con "text_clean" (limpio)
  ├── dev_gold_temp.json                   ← Ground truth para "tweet"
  ├── dev_gold_temp_clean.json             ← Ground truth para "text_clean"
  ├── BeingChillingWeWillWin_f2llm4B.json       ← TEST con "tweet"
  └── BeingChillingWeWillWin_f2llm4Bclean.json  ← TEST con "text_clean"
```
**Razón:** Se entrenaron DOS versiones con diferentes columnas de texto.

### **KaLM (caso problemático):**
```
📂 results_v2/KaLM/predictions/
  ├── dev_predictions_temp.json      ← COMPARTIDO por ambas versiones ⚠️
  ├── dev_gold_temp.json             ← Ground truth compartido
  ├── BeingChillingWeWillWin_KaLM.json       ← TEST con "tweet"
  └── BeingChillingWeWillWin_KaLMclean.json  ← TEST con "text_clean"
```
**Problema detectado:** Ambos KaLM (Tweet y Text Clean) usan el MISMO archivo de validación, 
lo que significa que en validación generan predicciones idénticas. Solo difieren en TEST.

---

## 🚫 Modelos EXCLUIDOS y por qué

### **3Ministral8B_LoRA (con fine-tuning) - DESASTRE TOTAL**
```
Métricas de validación:
  Accuracy:  0.5725  ❌
  Precision: 0.9000  
  Recall:    0.0444  ❌❌❌ (prácticamente no detecta la clase YES)
  F1-Score:  0.0847  ❌❌❌ (PEOR QUE TIRAR UNA MONEDA)
```
**Veredicto:** El fine-tuning EMPEORÓ el modelo base. No aporta nada al ensemble.

### **KaLM Text Clean - Redundante**
```
F1 Score: 0.8000
```
**Razón:** Comparte archivo de validación con KaLM Tweet (F1: 0.8254), por lo que sus 
predicciones son idénticas en validación. Incluir ambos inflaría artificialmente el ensemble.

---

## ✅ TOP 5 CONSERVADOR (Ensemble Final)

| # | Modelo | F1 Score | Tipo | Comentario |
|---|--------|----------|------|-----------|
| 1 | **F2LLM-4B Tweet** | 0.8532 | LLM | 🥇 Mejor modelo individual |
| 2 | **F2LLM-4B Text Clean** | 0.8317 | LLM | 🥈 Segunda mejor variante |
| 3 | **KaLM Tweet** | 0.8254 | LLM | 🥉 Tercer mejor LLM |
| 4 | **Ministral 3B** | 0.8073 | LLM | Sin fine-tuning, sólido |
| 5 | **LogisticRegression (TF-IDF)** | 0.7251 | ML Clásico | Diversidad de enfoque |

### **Ventajas de este ensemble:**
- ✅ Solo modelos con validación independiente
- ✅ Todos tienen F1 > 0.70
- ✅ Diversidad: 4 LLMs + 1 modelo clásico
- ✅ Ningún modelo redundante o problemático

---

## 📊 Comparación de Archivos por Modelo

| Modelo | Val Preds | Val Gold | Test Preds | F1 Score |
|--------|-----------|----------|------------|----------|
| F2LLM-4B Tweet | `dev_predictions_temp.json` | `dev_gold_temp.json` | `BeingChillingWeWillWin_f2llm4B.json` | 0.8532 |
| F2LLM-4B Clean | `dev_predictions_temp_clean.json` | `dev_gold_temp_clean.json` | `BeingChillingWeWillWin_f2llm4Bclean.json` | 0.8317 |
| KaLM Tweet | `dev_predictions_temp.json` | `dev_gold_temp.json` | `BeingChillingWeWillWin_KaLM.json` | 0.8254 |
| ~~KaLM Clean~~ | ~~`dev_predictions_temp.json` (compartido)~~ | ~~`dev_gold_temp.json`~~ | ~~`BeingChillingWeWillWin_KaLMclean.json`~~ | ~~0.8000~~ |
| Ministral3B | `dev_predictions_temp.json` | `dev_gold_temp.json` | `BeingChillingWeWillWin_Mistral3B.json` | 0.8073 |
| ~~3Ministral8B_LoRA~~ | ~~`dev_predictions_temp.json`~~ | ~~`dev_gold_temp.json`~~ | ~~`BeingChillingWeWillWin_3Ministral8B_ft.json`~~ | ~~0.0847~~ |
| LogReg TF-IDF | `val_predictions_temp.json` | `val_gold_temp.json` | `BeingChillingWeWillWin_LogisticRegression_TFIDF.json` | 0.7251 |

---

## 📝 Archivos Generados por el Ensemble

```
📂 results_v2/ensemble/
  ├── BeingChillingWeWillWin_ensemble_top5_conservador.json     ← TEST (para competición)
  └── BeingChillingWeWillWin_ensemble_top5_conservador_val.json ← Validación (para evaluación)
```

---

## 🎯 Recomendación Final

**Para la competición, considera estos dos archivos:**

1. **Si quieres el modelo MÁS SEGURO:**
   ```
   results_v2/F2LLM-4B/predictions/BeingChillingWeWillWin_f2llm4B.json
   ```
   - F1: **0.8532** (probado en validación)
   - Modelo individual más robusto

2. **Si quieres probar el ENSEMBLE:**
   ```
   results_v2/ensemble/BeingChillingWeWillWin_ensemble_top5_conservador.json
   ```
   - Combina los 5 mejores modelos por votación mayoritaria
   - Podría mejorar 1-2% sobre el mejor individual
   - Reduce riesgo de errores individuales

**Mi recomendación personal:** Evalúa primero el ensemble en validación. Si mejora F1 en ≥0.01 
sobre F2LLM-4B Tweet, usa el ensemble. Si no, quédate con F2LLM-4B Tweet.

---

## 🔧 Limpieza de results_v2

El script `clean_results_v2.py` eliminará:
- ❌ Checkpoints (archivos `.pth`, `.bin`, `.safetensors`)
- ❌ Pesos del modelo (`lora_weights/`, `model/`)
- ❌ Configuraciones intermedias
- ✅ **Mantiene:** Solo los archivos JSON de predicciones

**Ahorro de espacio estimado:** 10-50 GB (dependiendo de los checkpoints)
