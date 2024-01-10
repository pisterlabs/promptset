```python
import openai
from openai_api import OpenAI_API

class AutoGPT:
    def __init__(self, openai_api: OpenAI_API):
        self.openai_api = openai_api
        self.model = 'text-davinci-003'  # GPT-3.5 model

    def generate_text(self, prompt: str, max_tokens: int = 100):
        """
        Generate text using the GPT-3.5 model.
        """
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()

    def run_task(self, task):
        """
        Run a task using the GPT-3.5 model.
        """
        prompt = task.get_prompt()
        return self.generate_text(prompt)
```
