import openai
from config import OPENAPI_ID, STORY_PROMPT, STORY_SAMPLE

class GPT:
    def __init__(self):
        openai.api_key = OPENAPI_ID
        self.model = "gpt-4"
        self.messages = []
    
    def add_message(self, message):
        self.messages.append(message)

    def set_message(self, messages):
        self.messages = messages

    def call_gpt(self):
        res = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )
        return res.choices[0].message.content
    

if __name__ == "__main__":
    gpt = GPT()
    gpt.add_message({"role": "user", "content": STORY_PROMPT})
    gpt.add_message({"role": "assistant", "content": STORY_SAMPLE})
    gpt.add_message({"role": "user", "content": "不错，再来一个"})
    print(gpt.call_gpt())
