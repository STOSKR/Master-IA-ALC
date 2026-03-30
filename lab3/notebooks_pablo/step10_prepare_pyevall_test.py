import json
import os
import pandas as pd
from collections import Counter

INPUT_JSON_PATH = "../materials/dataset_task3_exist2026/EXIST2026_training.json"
TEST_JSON_PATH = "../materials/dataset_task3_exist2026/test.json"
OUTPUT_FORMATTED_JSON_PATH = "./predictions/test_gold_pyevall.json"

def process_and_format_for_pyevall(input_file, test_file, output_file):
    print(f"Cargando archivo de test para extraer las keys: {test_file}...")
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            test_keys = set(test_data.keys())
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de test {test_file}")
        return

    print(f"Cargando archivo con estructura de training: {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_file}")
        return

    records_for_pyevall = []

    # Filtrar solo los registros cuya key esté en el test_json
    print("Filtrando registros y aplicando majority voting...")
    for key, item in input_data.items():
        if key in test_keys:
            labels = item.get("labels_task3_1", [])
            
            # Majority voting
            if not labels:
                final_label = "YES"
            else:
                conteo = Counter(labels)
                mas_comunes = conteo.most_common()

                if len(mas_comunes) > 1 and mas_comunes[0][1] == mas_comunes[1][1]:
                    # Empate
                    final_label = "YES"
                else:
                    final_label = mas_comunes[0][0]
            
            # En el código proporcionado el test_case usado es EXIST2025, 
            # y el id usado para evaluar es el id_EXIST convertido en string.
            id_exist = item.get("id_EXIST", key)
            
            record = {
                'test_case': 'EXIST2026',
                'id': str(id_exist),
                'value': final_label
            }
            records_for_pyevall.append(record)

    # Convertimos a DataFrame para guardarlo con to_json() y orient='records',
    # igual que en el código de ejemplo.
    if records_for_pyevall:
        df = pd.DataFrame(records_for_pyevall)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(df.to_json(orient='records'))
        
        print(f"✅ Archivo guardado con éxito en formato PyEvALL: {output_file}")
        print(f"Registros procesados y guardados: {len(df)}")
    else:
        print("⚠️ No se encontraron registros que coincidan con las keys del test.")

if __name__ == '__main__':
    process_and_format_for_pyevall(INPUT_JSON_PATH, TEST_JSON_PATH, OUTPUT_FORMATTED_JSON_PATH)
