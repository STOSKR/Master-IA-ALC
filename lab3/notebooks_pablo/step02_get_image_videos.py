import os
import subprocess

VIDEOS_DIR = "/data/psegmar1@alumno.upv.es/ALC/lab3/materials/dataset_task3_exist2026/videos"
OUTPUT_DIR = "/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/scenes"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 5  # 5 imágenes por segundo

for filename in os.listdir(VIDEOS_DIR):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEOS_DIR, filename)
    video_id = os.path.splitext(filename)[0]

    print(f"Procesando: {filename}")

    output_pattern = os.path.join(OUTPUT_DIR, f"{video_id}_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-err_detect", "ignore_err",   # 🔥 clave para vídeos corruptos
        "-i", video_path,
        "-vf", f"fps={FPS}",
        "-q:v", "2",                   # calidad alta
        output_pattern
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Error con {filename}, saltando...")

print("✅ Terminado")