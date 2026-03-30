import os
from PIL import Image
import imagehash

SCENES_DIR = "/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/scenes"

HASH_THRESHOLD = 7

videos = {}

for filename in os.listdir(SCENES_DIR):
    if not filename.endswith(".jpg"):
        continue

    video_id = filename.split("_")[0]
    videos.setdefault(video_id, []).append(filename)

for video_id, images in videos.items():
    print(f"Procesando video: {video_id}")

    prev_hash = None

    for img_name in sorted(images):
        path = os.path.join(SCENES_DIR, img_name)

        try:
            img = Image.open(path)
            current_hash = imagehash.phash(img)
        except:
            print(f"  ⚠️ Error leyendo {img_name}, eliminando")
            os.remove(path)
            continue

        if prev_hash is not None:
            diff = abs(current_hash - prev_hash)

            if diff <= HASH_THRESHOLD:
                os.remove(path)
                print(f"  ❌ Eliminada (similar): {img_name}")
                continue

        prev_hash = current_hash

print("✅ Limpieza secuencial completada.")