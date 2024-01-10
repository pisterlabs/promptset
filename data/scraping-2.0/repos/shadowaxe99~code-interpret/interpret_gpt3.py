import openai

class GPT3Integration:
    def __init__(self):
        self.api_key = 'your-api-key'

    def init_api(self):
        openai.api_key = self.api_key

    def get_response(self, prompt):
        response = openai.Completion.create(engine='text-davinci-002', prompt=prompt, max_tokens=150)
        return response.choices[0].text.strip()