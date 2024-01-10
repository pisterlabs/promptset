import json
import re
import openai
import requests
from gpt4all import GPT4All


class GPTApi:
    def __init__(self, api_key, model="text-davinci-003"):
        openai.api_key = api_key
        self.model = model

    def generate_prompt(self, restaurant_type):
        # Your implementation for generating a prompt based on restaurant type
        # prompt = f"Generate a json list of ingredients for a {restaurant_type} restaurant."
        prompt = f"Generate a Python list object of ingredients for a {restaurant_type} restaurant. You should not include any unnecessary items or presentation words. Just give me the pure python list I ask you."
        return prompt

    def call_gpt_api(self, prompt):
        response = openai.chat.completions.create(
            messages=[
                {
                    "role": "system", "content": "You are a helpful assistant designed to output JSON",
                    "role": "user", "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def process_gpt_response(self, gpt_response, as_type):
        try:
            # Assuming gpt_response is a JSON object
            parsed_response = json.loads(gpt_response)
            if as_type == list:
                # Extracting relevant information as a list (customize this based on your JSON structure)
                ingredients_list = parsed_response.get("ingredients", [])
                return ingredients_list
            elif as_type == dict:
                # Extracting relevant information as a dictionary (customize this based on your JSON structure)
                ingredients_dict = parsed_response.get("ingredients", {})
                return ingredients_dict
            else:
                # Handle other data types as needed
                return None

        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            print(f"Error decoding JSON: {e}")
            return None

    def offline_call_gpt_028_api(self, prompt):
        openai.api_base = "http://localhost:4891/v1"

        model = "ggml-mpt-7b-chat.bin"
        specific_cmd = "You are a helpful assistant designed to output JSON. "
        prompt = specific_cmd + prompt
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.28,
            top_p=0.95,
            n=1,
            echo=True,
            stream=False
        )
        print(response)
        return response['choices'][0]['text']

    def reqest_offline_gpt_028_api(self, prompt):
        openai.api_base = "http://localhost:4891/v1"

        model = "ggml-mpt-7b-chat.bin"
        specific_cmd = "You are a helpful assistant designed to output JSON. "
        prompt = specific_cmd + prompt

        try:
            response = requests.post(
                f"{openai.api_base}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.28,
                    "top_p": 0.95,
                    "n": 1,
                    "echo": True,
                    "stream": False
                },
                timeout=3600  # Set your desired timeout value in seconds
            )

            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
            print(response.json())
            result = response.json()['choices'][0]['text']
        except requests.exceptions.Timeout:
            print("The request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            result = None

    def gpt4all(self, prompt, max_attempts=10):
        mistral_instruct = "mistral-7b-instruct-v0.1.Q4_0.gguf"
        model_api = GPT4All(mistral_instruct)

        attempts = 0
        response = None

        while attempts < max_attempts:
            api_response = self.return_cmd(model_api, prompt)
            response = self.test_api_response(api_response)

            if isinstance(response, list):
                return response
            else:
                attempts += 1
        # If the desired response is not obtained after max_attempts, handle it accordingly
        print(f"Error: Maximum {max_attempts} attempts reached. Unable to get the desired response.")
        return None


    def return_cmd(self, model: GPT4All, question: str):
        tokens = []
        for token in model.generate(question, max_tokens=100, streaming=True):
            tokens.append(token)
        json_string = ''.join(tokens)
        return json_string

    def test_api_response(self, api_response):
        pattern = re.compile(r'```python\n(.+?)\n```', re.DOTALL)
        match = pattern.search(api_response)
        if match:
            python_code = match.group(1)
            try:
                exec(python_code)
                variable_match = re.search(r'\b(\w+)\s*=', python_code)
                if variable_match:
                    variable_name = variable_match.group(1)
                    evaluated_code = eval(variable_name)
                    if isinstance(evaluated_code, list):
                        return evaluated_code
                    else:
                        return False
                else:
                    return False
            except Exception as e:
                print(f"Error: Unable to evaluate the extracted code. {e}")
                return False
        else:
            print("Error: API response doesn't follow the expected pattern.")
            return False
