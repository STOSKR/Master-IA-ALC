# Resumen de modelos Lab3 (EXIST 2026 Videos)

## Pipeline comun
- Entrada: `text` (transcripcion Whisper), `sensorial`, y `path_video` desde `/home/alumno.upv.es/scheng1/EXIST 2026 Videos Dataset/training/EXIST2026_training.json`.
- Texto: embeddings de `xlm-roberta-base`.
- Video: extraccion de fotogramas clave con PySceneDetect (`ContentDetector`) y estadisticas visuales ligeras por fotograma.
- Sensorial: aplanado y agregacion por modalidad/usuario.
- Fusion temprana: concatenacion `[texto | video | sensorial]`.
- Clasificador: `LogisticRegression` con `StandardScaler`.

## Modelos que se van a entrenar
1. ES-only
- Notebook: `02_train_classifier_es.ipynb`
- Datos de entrenamiento: solo muestras con `lang == 'es'`.
- Salida JSON: `entregables/BeingChillingWeWillWin_XLMRSceneSensor_LR_ES.json`

2. EN-only
- Notebook: `03_train_classifier_en.ipynb`
- Datos de entrenamiento: solo muestras con `lang == 'en'`.
- Salida JSON: `entregables/BeingChillingWeWillWin_XLMRSceneSensor_LR_EN.json`

3. ES+EN
- Notebook: `04_train_classifier_es_en.ipynb`
- Datos de entrenamiento: muestras conjuntas `lang in {'es','en'}`.
- Salida JSON: `entregables/BeingChillingWeWillWin_XLMRSceneSensor_LR_ES_EN.json`

## Orden recomendado de ejecucion con sbatch
1. Ejecutar `01_build_multimodal_features.ipynb` una sola vez para generar artefactos.
2. Ejecutar notebooks 02, 03 y 04 para entrenar y exportar los tres JSON.

## Configuracion de cluster incorporada en notebooks
- Particion: `long`
- CPU por tarea: `8`
- Memoria: `32G`
- GRES usado en scripts: `shard:6` (secuencial) y `shard:4` (array)
- Entorno conda: `RFA2526pt`
- Variable CUDA: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
