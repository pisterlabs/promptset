```python
import openai
from openai.api_resources.completion import Completion

class CodeGenerator:
    def __init__(self):
        self.openai_api_key = 'your_openai_api_key'
        openai.api_key = self.openai_api_key

    def generate(self, data):
        try:
            prompt = data['prompt']
            language = data['language']
            max_tokens = data.get('max_tokens', 100)

            response = Completion.create(
                engine="text-davinci-002",
                prompt=f"{prompt}\n{language}:",
                temperature=0.5,
                max_tokens=max_tokens
            )

            generated_code = response.choices[0].text.strip()

            return {
                'status': 'success',
                'code': generated_code
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
```
