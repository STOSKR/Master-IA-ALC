import json
from collections import Counter

TRAIN_INPUT_FILE_PATH = '/data/psegmar1@alumno.upv.es/ALC/lab3/materials/dataset_task3_exist2026/training.json'   
TRAIN_OUTPUT_FILE_PATH = '/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/training.json'

TEST_INPUT_FILE_PATH = '/data/psegmar1@alumno.upv.es/ALC/lab3/materials/dataset_task3_exist2026/test.json'   
TEST_OUTPUT_FILE_PATH = '/data/psegmar1@alumno.upv.es/ALC/lab3/preprocessed_data/test.json'

def process_json(input_path, output_path, is_train_file):
    print(f"Cargando archivo: {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_path}")
        return

    processed_data = {}

    for key, item in data.items():
        filtered_item = {
            "id_Tiktok": item.get("id_Tiktok"),
            "id_EXIST": item.get("id_EXIST"),
            "lang": item.get("lang"),
            "text": item.get("text")
        }

        if is_train_file:
            labels = item.get("labels_task3_1", [])
            
            if not labels:
                final_label = "YES"
            else:
                conteo = Counter(labels)
                mas_comunes = conteo.most_common()

                if len(mas_comunes) > 1 and mas_comunes[0][1] == mas_comunes[1][1]:
                    print(f"-> Empate detectado en el id_EXIST {item.get('id_EXIST', key)} (Etiquetas: {labels}). Se asigna 'YES'.")
                    final_label = "YES"
                else:
                    final_label = mas_comunes[0][0]
            
            filtered_item["label"] = final_label

        processed_data[key] = filtered_item

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nArchivo procesado con éxito. Guardado en: {output_path}")

if __name__ == '__main__':
    process_json(TRAIN_INPUT_FILE_PATH, TRAIN_OUTPUT_FILE_PATH, True)
    process_json(TEST_INPUT_FILE_PATH, TEST_OUTPUT_FILE_PATH, False)
