import os
from typing import Tuple
import openai
import json
import time
import requests
from sinlingua.src.grammar_rule_resources import grammar_rule_llm_config


class LLMConfig:
    def __init__(self, api_key: str = None, org_key: str = None):
        self.json_data = grammar_rule_llm_config
        if api_key is not None and org_key is not None:
            self.json_data["api_key"] = api_key
            self.json_data["org_key"] = org_key

    # @staticmethod
    # def __read_json_config(file_path: str) -> dict:
    #     try:
    #         # Read JSON configuration file and return the data as dictionary
    #         with open(os.path.join(RESOURCE_PATH, file_path), 'r', encoding='utf-8') as json_file:
    #             json_data_c = json.load(json_file)
    #         return json_data_c
    #     except Exception as e:
    #         # Handle exceptions while reading JSON configuration
    #         print(f"Error while reading JSON configuration file '{file_path}': {str(e)}")
    #         return {}

    def __get_llm_response(self, text: str, level: int) -> str:
        completion = None
        try:
            # Set up API key and organization for OpenAI
            openai.api_key = self.json_data["api_key"]
            openai.organization = self.json_data["org_key"]

            # Create user prompt using provided text and level
            user_prompt = self.json_data["Prompts"][level]["content"].replace("{{word}}", text)

            # Check if the provided text is empty
            if not text.strip():
                raise ValueError("Text is empty. Please provide a valid text string.")

            success = False
            while not success:
                try:
                    # Create a ChatCompletion request to GPT-3
                    completion = openai.ChatCompletion.create(
                        model=self.json_data["model"],
                        messages=[
                            {
                                "role": self.json_data["Prompts"][level]['role'],
                                "content": user_prompt
                            }
                        ],
                        n=1,
                        temperature=self.json_data['temperature'],
                        max_tokens=self.json_data['max_tokens'],
                        top_p=self.json_data['Top_P'],
                        frequency_penalty=self.json_data['Frequency_penalty'],
                        presence_penalty=self.json_data['Presence_penalty']
                    )
                    success = True
                except Exception as e:
                    # Handle exceptions during GPT-3 request
                    sleep_time = 2
                    time.sleep(sleep_time)
                    print("Error:", e)
                    print("Retrying...")

            result = completion.choices[0].message.content
            sleep_time = 2
            time.sleep(sleep_time)  # To avoid rate limit
            return result
        except Exception as e:
            # Handle exceptions during GPT response processing
            print(f"Error in GPT response for text '{text}': {str(e)}")
            return ""

    def llm_check(self, word: str, level: int) -> dict:
        response = self.__get_llm_response(text=word, level=level)
        dictionary = json.loads(response)
        return dictionary

    @staticmethod
    # function for predict words
    def berto_predict(sentence: str) -> Tuple[str, str]:
        API_TOKEN = "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"
        API_URL = "https://api-inference.huggingface.co/models/keshan/SinhalaBERTo"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        # Use the API

        # Getting the similar words
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({
            "inputs": sentence,
        })

        # Pass the similar words
        lines = output
        line = lines[0]
        return line["sequence"], line["token_str"]

    @staticmethod
    # function for predict most similar words
    def berto_predict_top(sentence: str) -> list:
        API_TOKEN = "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"
        API_URL = "https://api-inference.huggingface.co/models/keshan/SinhalaBERTo"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        # Use the API

        # Getting the most similar word
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({
            "inputs": sentence,
        })

        # Pass the most similar word
        list_out = []
        lines = output
        for line in lines:
            list_out.append([line["sequence"], line["token_str"]])

        return list_out
