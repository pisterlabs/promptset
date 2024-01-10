import json
import re
from dotenv import load_dotenv
import openai
import os
import json

load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key

def main():
    json_file_path = 'backend/data/pin_data.json'

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    sorted_data = sorted(data, key=lambda x: x["aggregated_pin_data"]["saves"], reverse=True)
    top_2_grid_descriptions = [item["grid_description"] for item in sorted_data[:2]]

    top_descriptions_paragraph = "\n".join(top_2_grid_descriptions)
    openai_data = {
        "model": "text-curie-001",
        "prompt": f"List 3-4 fashion outfit items from this para :\n\n{top_descriptions_paragraph}",
        "max_tokens": 50        
    }

    response = openai.Completion.create(**openai_data)
    generated_text = response.choices[0].text.strip()
    return generated_text

if __name__ == "__main__":
    top_descriptions = main()
