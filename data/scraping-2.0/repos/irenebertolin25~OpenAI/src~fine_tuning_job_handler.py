import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class FineTuningJobHandler:
    def __init__(self, emotion_analysis):
        self.model = emotion_analysis.model
        self.path = "results/"
        self.file_name = "fine_tuning_job_result_"

    def generate_fine_tuning_results(self, content):
        print("\033[0m" + "Starting generate_fine_tuning_results...")
        try:
            directory = os.path.join(self.path, os.path.dirname(self.file_name))
            if not os.path.exists(directory):
                os.makedirs(directory)

            existing_files = [item_file for item_file in os.listdir(self.path) if item_file.startswith(self.file_name) and item_file.endswith('.json')]
            next_number = 0 if not existing_files else len(existing_files) + 1
            self.file_name = f'{self.file_name}{next_number}.json'

            with open(os.path.join(self.path, self.file_name), 'w') as result_file:
                json.dump(content, result_file, indent=4)

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        print("\033[92m\u2714 " + f"Results saved in {self.path}{self.file_name}")
        return result_file

    def create_fine_tuning_job(self, file_uploaded):
        print("\033[0m" + "Starting create_fine_tuning_job...")
        try:
            fine_tuning_job = client.fine_tuning.jobs.create(training_file=file_uploaded.id, model=self.model)

            status = fine_tuning_job.status
            if status not in ["succeeded", "failed"]:
                print("Waiting for fine-tuning to complete...")
                while status not in ["succeeded", "failed"]:
                    time.sleep(60)
                    fine_tuning_job = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
                    status = fine_tuning_job.status
                    print("Status: ", status)
            
            print("\033[92m\u2714 " + f"Fine-tune job {fine_tuning_job.id} finished with status: {status}")

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        return fine_tuning_job