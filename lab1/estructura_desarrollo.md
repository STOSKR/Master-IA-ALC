## Estructura Actual del Proyecto

### Datos
- **lab1_materials/**: Datasets originales (train/test)
- **preprocessed_data/**: Datos preprocesados (2 versiones)
- **results/**: Predicciones y comparaciones de modelos

### Notebooks
1. **01_data_exploration.ipynb** - Análisis exploratorio inicial
2. **02_preprocessing.ipynb** - Limpieza y preprocesamiento
3. **03_model_comparison.ipynb** - Comparación de modelos clásicos
4. **03-09_*.ipynb** - Fine-tuning de LLMs:
   - f2llm-4B (tweet + text_clean)
   - KaLM (tweet + text_clean)
   - Ministral3-8B (inference + fine-tuning)

### Resultados
- Predicciones JSON por modelo
- Comparaciones CSV
- Modelos clásicos en `results/clasicos/`

---

## Experimentos Pendientes

### 🔥 Prioridad Alta (Rápido + Impacto)
- [ ] **BETO fine-tuned**: Transformer español ligero y efectivo
- [ ] **Ensemble**: Votación/stacking de mejores modelos actuales
- [ ] **Análisis de errores**: Matriz confusión + tweets mal clasificados
- [ ] **Threshold tuning**: Optimizar umbral de decisión por modelo

### 📊 Análisis y Mejoras
- [ ] **Sample weights**: Usar task1_agreement del gold
- [ ] **Validación cruzada**: Para LLMs (actualmente solo train/test)
- [ ] **Calibración**: Temperatura del softmax

### 🧪 Modelos Adicionales
- [ ] **RoBERTa-es / mBERT / XLM-RoBERTa**
- [ ] **Data Augmentation**: Back-translation, parafraseo
- [ ] **Prompt Engineering**: Diferentes prompts para LLMs
- [ ] **Few-shot learning**: Ejemplos en el prompt