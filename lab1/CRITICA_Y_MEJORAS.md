# Crítica y Mejoras del Pipeline

## 1. Exploración de Datos

### Problemas Identificados
- Notebook excesivamente largo (900 líneas) dificulta mantenimiento
- Visualizaciones redundantes sin insights accionables
- No analiza vocabulario ni distribución de palabras (OOV)
- Falta análisis temporal/contextual de tweets
- No explora N-gramas ni patrones textuales
- Correlaciones calculadas pero no se usan para feature selection automática

### Mejoras Implementadas
- Análisis de vocabulario y cobertura
- Distribución de N-gramas discriminativos
- Análisis de palabras clave por clase
- Detección de términos sexistas específicos
- Análisis de coocurrencia de patrones

## 2. Preprocesamiento

### Problemas Identificados
- No usa stopwords en español (pérdida de información)
- Sin lematización/stemming (inflexión verbal no tratada)
- Limpieza agresiva puede perder contexto negativo (no, nunca)
- No normaliza tildes ni caracteres especiales españoles
- No trata elongaciones (ej: "siiiiii", "noooo")
- No considera contexto de emojis (polaridad)

### Mejoras Implementadas
- Stopwords personalizadas (mantiene negaciones)
- Lematización con spaCy (modelo es_core_news_sm)
- Normalización de elongaciones
- Feature: sentiment de emojis
- Preservación de negaciones
- N-gramas de caracteres para robustez ortográfica

## 3. Modelado

### Problemas Identificados
- No usa sample weights (task1_agreement ignorado)
- Falta validación cruzada estratificada
- No aplica técnicas para desbalance de clases (SMOTE, class_weight)
- No hay análisis de errores ni matriz de confusión
- Hiperparámetros fijos sin optimización
- Métricas no consideran costos de FP vs FN

### Mejoras Implementadas
- Sample weights basados en agreement
- Validación cruzada 5-fold estratificada
- Class weights automáticos
- Optimización de threshold para F1
- Análisis detallado de errores
- Métricas: F1, Precision, Recall, ROC-AUC, PR-AUC
- Hyperparameter tuning con RandomizedSearchCV

## 4. Arquitectura General

### Problemas Identificados
- Notebooks no parametrizables (dificulta experimentación)
- Sin logging estructurado
- No guarda checkpoints de modelos
- Falta tracking de experimentos
- No reproducible (seeds parciales)

### Mejoras Implementadas
- Script modular para cluster con argparse
- Logging con nivel configurable
- Guardado de modelos y métricas
- Seeds fijos en todos los componentes
- Configuración via JSON/YAML
- Paralelización con joblib
