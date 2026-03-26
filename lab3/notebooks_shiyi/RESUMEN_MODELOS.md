# Resumen de modelos Lab3 (EXIST 2026 Videos)

## Ruta de datos obligatoria
- Entrada principal: /home/alumno.upv.es/scheng1/EXIST 2026 Videos Dataset/training/EXIST2026_training.json

## Configuracion de cluster usada en todos los notebooks
- Particion: long
- CPU por tarea: 8
- Memoria: 32G
- GRES: shard:4 (experimentacion) y shard:6 (extraccion masiva)
- Entorno conda: RFA2526pt
- Variable CUDA: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## Notebooks avanzados creados
1. 05_exp_audio_vad.ipynb
- Limpieza de audio con VAD (WebRTCVAD con fallback a energia de Librosa).
- Embeddings acusticos con Wav2Vec2.
- Clasificador LR por ES, EN y ES+EN.
- JSON exportados:
  - entregables/BeingChillingWeWillWin_05AudioVAD_W2V2_ES.json
  - entregables/BeingChillingWeWillWin_05AudioVAD_W2V2_EN.json
  - entregables/BeingChillingWeWillWin_05AudioVAD_W2V2_ES_EN.json

2. 06_exp_video_clean.ipynb
- Limpieza temporal de frames (negros, transiciones bruscas y estaticos).
- Embeddings visuales con CLIP.
- Clasificador LR por ES, EN y ES+EN.
- JSON exportados:
  - entregables/BeingChillingWeWillWin_06VideoClean_CLIP_ES.json
  - entregables/BeingChillingWeWillWin_06VideoClean_CLIP_EN.json
  - entregables/BeingChillingWeWillWin_06VideoClean_CLIP_ES_EN.json

3. 07_exp_sensorial_filter.ipynb
- Filtrado de outliers con IQR en ET/EEG/HR.
- Features robustas por ventana (inicio/medio/final) alineadas a segmentos de transcripcion Whisper.
- Clasificador LR por ES, EN y ES+EN.
- JSON exportados:
  - entregables/BeingChillingWeWillWin_07SensorialFilter_IQRWin_ES.json
  - entregables/BeingChillingWeWillWin_07SensorialFilter_IQRWin_EN.json
  - entregables/BeingChillingWeWillWin_07SensorialFilter_IQRWin_ES_EN.json

4. 08_exp_fusion_avanzada.ipynb
- Fusion tardia multimodal (audio + video + sensorial + texto XLM-R).
- Clasificador independiente por modalidad y promedio de probabilidades.
- JSON exportados:
  - entregables/BeingChillingWeWillWin_08FusionLate_ES.json
  - entregables/BeingChillingWeWillWin_08FusionLate_EN.json
  - entregables/BeingChillingWeWillWin_08FusionLate_ES_EN.json

## Orden recomendado de ejecucion con sbatch
1. Ejecutar 05, 06 y 07 para generar artefactos de modalidad.
2. Ejecutar 08 para fusion tardia y export final.
