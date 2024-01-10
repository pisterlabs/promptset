
from openai import OpenAI
import google.generativeai as genai
from rich.console import Console


GOOGLE_AI_API_KEY_PATH = 'e:\\dev\\api_keys\\GOOGLEAI_API_KEY'
OPENAI_API_KEY_PATH = 'e:\\dev\\api_keys\\OPENAI_API_KEY'

def open_api_key_file(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

def create_gpt_client():
    return  OpenAI(api_key=open_api_key_file(OPENAI_API_KEY_PATH))

def create_gemini_client():
    genai.configure(api_key=open_api_key_file(GOOGLE_AI_API_KEY_PATH))

    # Set up the model
    generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 200,
}

    safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
    console = Console()

    model = genai.GenerativeModel(
    model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings
)
    return model