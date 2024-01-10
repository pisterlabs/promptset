```python
import openai

class ChatGPTIntegration:
    def __init__(self):
        self.api_key = "YOUR_OPENAI_API_KEY"
        self.model_name = "gpt-3.5-turbo"

    def setup_openai_api(self):
        openai.api_key = self.api_key

    def generate_response(self, prompt):
        self.setup_openai_api()
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

chat_gpt = ChatGPTIntegration()

def integrate_chat_gpt(prompt):
    response = chat_gpt.generate_response(prompt)
    return response
```