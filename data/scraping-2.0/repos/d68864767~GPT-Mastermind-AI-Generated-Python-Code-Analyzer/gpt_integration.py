```python
import openai
from openai.api_resources import Completion

class GPTIntegration:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_code_suggestions(self, code):
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=code,
                temperature=0.5,
                max_tokens=100
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"An error occurred while generating code suggestions: {str(e)}")
            return None

    def analyze_code(self, code):
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=f"{code}\n# How can I improve this code?",
                temperature=0.5,
                max_tokens=100
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"An error occurred while analyzing code: {str(e)}")
            return None
```
