import os
import time
import openai
from openai.error import *
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")


class FineTuning:
    def __init__(self, data_file="data.jsonl", fine_tuning_suffix="meraki-wlan-1"):
        self.fine_tuned_model_id = None
        self._job_id = None
        self.training_file_id = None
        self.data_file = data_file
        self.fine_tuning_suffix = fine_tuning_suffix

    @property
    def job_id(self):
        return self._job_id

    @job_id.setter
    def job_id(self, value):
        self._job_id = value

    def create_file(self):
        try:
            training_response = openai.File.create(
                file=open(self.data_file, "rb"),
                purpose='fine-tune'
            )
            self.training_file_id = training_response.get("id")

        except FileNotFoundError as e:
            print(f"File not found. {e}")
            return None

        except AuthenticationError as e:
            print(f"Authentication failed. {e}")
            return None

        except APIError as e:
            print(f"API error occurred.{e}")
            return None

        else:
            print("Training file ID:", self.training_file_id)
            return self.training_file_id

    def create_job(self):
        try:
            response = openai.FineTuningJob.create(
                training_file=self.training_file_id,
                model="gpt-3.5-turbo",
                suffix=self.fine_tuning_suffix,
            )
            self._job_id = response["id"]

            print("Job ID:", response["id"])
            print("Status:", response["status"])
            print(response)

            # Fine tuning monitor
            start_time = time.time()
            while True:
                response = openai.FineTuningJob.retrieve(self.job_id)
                print("Job ID:", response["id"])
                print("Status:", response["status"])
                print("Trained Tokens:", response["trained_tokens"])

                # If the job is complete, break out of the loop
                if response["status"] == "succeeded":
                    break

                # If the job has been running for more than 20 minutes, break out of the loop
                if time.time() - start_time > 1200:
                    raise TimeoutError("Fine tuning job has timed out after 20 minutes.")

                # Wait for a while before checking again
                time.sleep(30)
        except APIError as e:
            print(f"API Error: {e}")
        except AuthenticationError as e:
            print(f"Authentication Error: {e}")
        except InvalidRequestError as e:
            print(f"Invalid Request Error: {e}")
        except RateLimitError as e:
            print(f"Rate Limit Error: {e}")
        except OpenAIError as e:
            print(f"OpenAI Error: {e}")

    def get_model_id(self):
        response = openai.FineTuningJob.retrieve(self.job_id)
        self.fine_tuned_model_id = response["fine_tuned_model"]

        if self.fine_tuned_model_id is None:
            raise RuntimeError("Fine-tuned model ID not found. Your job has likely not been completed yet.")

        print("Fine-tuned model ID:", self.fine_tuned_model_id)
        return self.fine_tuned_model_id
