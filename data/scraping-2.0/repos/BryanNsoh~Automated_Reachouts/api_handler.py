import openai
from openai import OpenAI  # Import for the updated client
import google.generativeai as genai
from retry import retry
import json
import logging
import os
import time

# Logger setup
logging.basicConfig(level=logging.INFO)


class LLM_APIHandler:
    def __init__(self, key_path):
        self.load_api_keys(key_path)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        genai.configure(api_key=self.gemini_api_key)
        self.request_timestamps = []  # List to keep track of request timestamps

    def load_api_keys(self, key_path):
        with open(key_path, "r") as file:
            api_keys = json.load(file)
            self.gemini_api_key = api_keys["GEMINI_API_KEY"]
            self.openai_api_key = api_keys["OPENAI_API_KEY"]

    def generate_openai_content(self, prompt, model="gpt-3.5-turbo"):
        self.check_rate_limit()  # Check and handle the rate limit
        try:
            completion = self.openai_client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=150,
            )
            self.request_timestamps.append(
                time.time()
            )  # Log the timestamp of the request
            return completion.choices[0].text
        except Exception as e:
            logging.error(f"Error in OpenAI API call: {e}")
            raise

    @retry(tries=5, delay=1, backoff=2)
    def generate_gemini_content(self, prompt):
        try:
            model = genai.GenerativeModel("gemini-pro")
            return model.generate_content(prompt)
        except Exception as e:
            logging.error(f"Error in Gemini API call: {e}")
            raise

    def generate_content(self, prompt, model_choice="gemini-pro"):
        if model_choice == "gpt-3.5-turbo":
            return self.generate_openai_content(prompt)
        elif model_choice == "gemini-pro":
            return self.generate_gemini_content(prompt)
        else:
            raise ValueError(
                "Invalid model choice. Choose 'gpt-3.5-turbo' or 'gemini-pro'."
            )

    def check_rate_limit(self):
        while len(self.request_timestamps) >= 3:
            if time.time() - self.request_timestamps[0] > 60:
                self.request_timestamps.pop(0)  # Remove the oldest timestamp
            else:
                time.sleep(
                    60 - (time.time() - self.request_timestamps[0])
                )  # Wait if rate limit is reached


# Example usage
if __name__ == "__main__":
    key_path = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\keys\api_keys.json"  # Replace with your key file path
    handler = LLM_APIHandler(key_path)

    # Example prompt
    prompt = "Write a short story about a sad, defeated wizard in a word perpetually drowned in cloying red mists that tasted of beetroots and saltwater."
    model_choice = "gemini-pro"  # Can be 'openai' or 'gemini'
    try:
        response = handler.generate_content(prompt, model_choice)
        print(response.text)
    except Exception as e:
        logging.error(f"Failed to generate content: {e}")
