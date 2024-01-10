import os
import jsonlines
import time
import csv
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class FileHandler:
    def __init__(self, emotion_analysis):
        self.system_prompt = "Can you tell me what emotion is expressing the message above? It can be one of the following: happiness, sadness, anger, fear, surprise, disgust or not-relevant (if it doesn't express any relevant emotion)."
        self.training_dataset_with_prompt = emotion_analysis.training_dataset_with_prompt
        self.results_path = emotion_analysis.results_path
        self.training_metrics_file = emotion_analysis.training_metrics_file
        self.chat_completion_result_file = "chat_completion_result_"
        
    def add_prompt_to_jsonl_file(self, input_file_path, output_file_path):
        print("\033[0m" + "Starting add_prompt_to_jsonl_file...")
        try:
            with jsonlines.open(input_file_path, 'r') as reader, jsonlines.open(output_file_path, 'w') as writer:
                for line in reader:
                    messages = line.get('messages', [])
                    if messages:
                        messages.insert(0, {'role': 'system', 'content': self.system_prompt})
                    line['messages'] = messages
                    writer.write(line)
        
        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        print("\033[92m\u2714 " + f"Jsonl dataset with prompt saved in {output_file_path}")
        return output_file_path

    def upload_file(self):
        print("\033[0m" + "Starting upload_file...")
        try:
            file_uploaded = client.files.create(file=open(self.training_dataset_with_prompt, "rb"), purpose='fine-tune')

            while True:
                print("Waiting for file to process...")
                file_handle = client.files.retrieve(file_uploaded.id)
                if file_handle and file_handle.status == "processed":
                    print("\033[92m\u2714 " + "File processed")
                    break
                time.sleep(120)

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        print("\033[92m\u2714 " + "File uploaded")
        return file_uploaded