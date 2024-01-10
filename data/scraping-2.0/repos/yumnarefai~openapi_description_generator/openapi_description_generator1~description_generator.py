import openai
from typing import Dict, Optional


class DescriptionGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    def generate_description(self, resource: Dict[str, str]) -> Optional[str]:
        """
        Generate a description for the provided resource using the GPT-3 model.
        """
        prompt = f"Describe the following API endpoint:\nPath: {resource['path']}\nMethod: {resource['method']}\nSummary: {resource['summary']}"
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60)

        if response.choices:
            return response.choices[0].text.strip()

        return None
