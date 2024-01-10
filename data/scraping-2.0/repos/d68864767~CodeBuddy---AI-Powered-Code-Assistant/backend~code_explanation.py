```python
import openai
from openai.api_resources.completion import Completion

class CodeExplain:
    def __init__(self):
        self.openai_api_key = 'your_openai_api_key'
        openai.api_key = self.openai_api_key

    def explain(self, data):
        try:
            code = data['code']
            prompt = f"Explain the following code in simple terms:\n{code}\n"
            response = Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100)
            explanation = response.choices[0].text.strip()
            return {'status': 'success', 'explanation': explanation}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
```
