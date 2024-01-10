import os
from openai import OpenAI

class GptAgent:
    def __init__(self):
        # Get the API key from the environment variable
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        # Initialize message log
        self.message_log = [{"role": "system", "content": "You are a helpful assistant."}]

    def send_message(self):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.message_log,
            max_tokens=3800
        )
        for choice in response.choices:
            if "text" in choice:
                return choice.text
        return response.choices[0].message.content

    def process_prompt(self, prompt):
        self.message_log.append({"role": "user", "content": prompt})
        response = self.send_message()
        self.message_log.append({"role": "assistant", "content": response})
        return response


