import openai


class ChatGpt:
    def __init__(self, key, temperature, initial_prompt):
        openai.api_key = key
        self.temperature = temperature
        self.prefix_messages = initial_prompt

    model = "gpt-3.5-turbo"

    def predict(self, prompt):
        res = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": self.prefix_messages},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature
        )
        return res.choices[0].message.content
