import time
import boto3
import json
import os
import openai
from datetime import datetime

def get_current_datetime():
    return datetime.now()

import unicodedata

def remove_spanish_special_characters(text):
    """
    Removes Spanish special characters from a string.
    """
    # Normalize the string by converting it to Unicode NFD form
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove combining characters
    stripped_text = ''.join(c for c in normalized_text if not unicodedata.combining(c))
    # Remove specific Spanish special characters
    removed_special_characters = stripped_text.replace('ñ', 'n').replace('Ñ', 'N').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    return removed_special_characters

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    
    openai.api_key = os.environ['OPENAI_API_KEY']

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]


def read_json_from_s3(bucket_name, file_name):
    s3 = boto3.resource('s3')
    try:
        obj = s3.Object(bucket_name, file_name)
        data = obj.get()['Body'].read().decode('utf-8')
        json_data = json.loads(data)
        return json_data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        
        return None


class Cache:
    def __init__(self, timeout=120):
        self.cache_data = {}
        self.timeout = timeout

    def get(self, key, default_value):
        value, timestamp = self.cache_data.get(key, (default_value, None))
        if self.timeout > 0:
            if timestamp and time.time() - timestamp > self.timeout:
                self.delete(key)
                return default_value
        return value

    def set(self, key, value):
        timestamp = time.time()
        self.cache_data[key] = (value, timestamp)

    def delete(self, key):
        if key in self.cache_data:
            del self.cache_data[key]
