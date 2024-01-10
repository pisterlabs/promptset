import os
from openai import OpenAI
import pandas as pd
import json

class LLMDataset:
    def __init__(self, api_key=None, dataset_path=None):        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided. Set the OPENAI_API_KEY environment variable or pass an api_key to the LLMDatasetMgr object.")
        self.dataset_path = dataset_path
        self.dataset = None
        self.dataset_columns = None
        self.dataset_description = None
        self.client = OpenAI(api_key=self.api_key)
        self.load_from_file(self.dataset_path)

    def load_from_file(self, file_path):
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        self.dataset = pd.read_csv(file_path)
        self.dataset_columns = ', '.join(self.dataset.columns.tolist())

    def add_data(self, context, model="gpt-4-1106-preview", num_samples=1, max_tokens=4096):
        self.dataset_description = f"I have a dataset with columns: {self.dataset_columns}. Your task is to add {num_samples} more row(s) of data to this dataset. Provide the data as a json object, which should match the structure of the dataset. Wrap the entire response in a json object with the key 'data'."
        print(f"\nDataset Description: {self.dataset_description}")
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant used to generate data for datasets."},
                {"role": "user", "content": self.dataset_description},
                {"role": "user", "content": context},
            ]
        )
        response_text = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            raise ValueError("Generation exceeded max_tokens or conversation exceeded token limit. Response will not parse into a valid JSON object.")
        print(f"\nResponse: {response_text}")
        self.save_data(response_text)

    def save_data(self, json_response):
        if not self.dataset_path:
            raise ValueError("Dataset path is not provided.")
        if isinstance(json_response, str):
            json_response = json.loads(json_response)

        if 'data' not in json_response or not isinstance(json_response['data'], list):
            raise ValueError("Invalid JSON response format. Expected a 'data' key with a list of entries.")

        new_rows = []
        next_id = 1
        if 'ID' in self.dataset.columns and not self.dataset.empty:
            next_id = self.dataset['ID'].max() + 1

        for entry in json_response['data']:
            if 'ID' in self.dataset.columns:
                entry['ID'] = next_id
                next_id += 1
            new_row = pd.DataFrame({**{key: [value] for key, value in entry.items()}})
            new_rows.append(new_row)

        self.dataset = pd.concat([self.dataset] + new_rows, ignore_index=True)
        self.dataset.to_csv(self.dataset_path, index=False)

    def print_attributes(self):
        print(f"\nDatset Columns: {self.dataset_columns}")