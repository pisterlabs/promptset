import os
import json
import jsonlines
import csv
from dotenv import load_dotenv
from openai import OpenAI
from io import StringIO

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmotionHandler:
    def __init__(self, emotion_analysis):
        self.emotion_labels_path = emotion_analysis.emotion_labels_path
        
    def refactor_id_jsonl(self):
        print("\033[0m" + "Starting refactor_id_jsonl...")
        try:
            id_count = {}
            updated_lines = []

            with open(self.emotion_labels_path, 'r') as reader:
                lines = reader.readlines()

            for line in lines:
                data = json.loads(line)
                current_id = data.get('id')

                if current_id in id_count:
                    id_count[current_id] += 1
                    data['id'] = f"{current_id}_{id_count[current_id]}"
                else:
                    id_count[current_id] = 0

                updated_lines.append(json.dumps(data))

            with open(self.emotion_labels_path, 'w') as writer:
                for updated_line in updated_lines:
                    writer.write(f"{updated_line}\n")

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        print("\033[92m\u2714 " + f'IDs processed in {self.emotion_labels_path}')

    def add_emotion_and_id_to_csv(self, csv_file):
        print("\033[0m" + "Starting add_emotion_and_id_to_csv...")
        try:
            emotions = {}
            ids = []

            with open(csv_file.name, 'r+', newline='', encoding='utf-8') as file:
                csv_data = StringIO(file.read())
                csv_data.seek(0)

                csv_reader = csv.reader(csv_data)
                csv_list = list(csv_reader)

                header = csv_list[0]
                header.insert(0, 'ID')
                header.append('Emotion label')

                csv_data.seek(0)

                with jsonlines.open(self.emotion_labels_path, 'r') as reader:
                    for line in reader:
                        ids.append(line["id"])
                        emotions[line["id"]] = line["emotion"]

                for i, row in enumerate(csv_list[1:], start=1):
                    row.insert(0, ids[i - 1])
                    emotion = emotions.get(ids[i - 1], '')
                    row.append(emotion)

                csv_data.seek(0)

                writer = csv.writer(csv_data)
                writer.writerows(csv_list)

                file.seek(0)
                file.write(csv_data.getvalue())
                file.truncate()

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        print("\033[92m\u2714 " + "IDs and emotions added")
        return csv_file