import requests
import json
import requests
from IPython.display import Image
from openai import OpenAI

import requests

import constants

class RequestsWrapper:
    def __init__(self, base_url):
        self.base_url = base_url

    def post(self, endpoint, payload):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error: {response.status_code} - {response.text}"

    def get(self, endpoint):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'accept': 'application/json'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error: {response.status_code} - {response.text}"

    def put(self, endpoint, payload):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        
        response = requests.put(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error: {response.status_code} - {response.text}"

    def delete(self, endpoint):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'accept': 'application/json'}
        
        response = requests.delete(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error: {response.status_code} - {response.text}"

