import openai

class GPTAssistant:
    def __init__(self, config):
        self.config = config
        openai.api_key = self.config.get_openai_api_key()
        openai.api_base = self.config.get_openai_api_base()

    def generate_reply(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
