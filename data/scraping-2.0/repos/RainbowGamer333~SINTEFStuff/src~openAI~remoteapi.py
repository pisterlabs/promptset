import os

import openai
import yaml


def load_credential():
    # change credential filepath to match your own
    current_dir = os.path.dirname(os.path.abspath(__file__))
    credential_filepath = os.path.join(os.path.dirname(current_dir), "openai.credential")

    with open(credential_filepath, 'r') as stream:
        credential_data = yaml.safe_load(stream)
    openai_config = credential_data['openai']
    openai.api_type = "azure"
    openai.api_base = openai_config['endpoint']
    openai.api_version = "2023-03-15-preview"
    openai.api_key = openai_config["key"]
