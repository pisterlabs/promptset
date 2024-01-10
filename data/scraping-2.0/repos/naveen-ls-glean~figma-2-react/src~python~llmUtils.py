import time
from openai import OpenAI

OPEN_AI_KEY = "enter-open-ai-key-here"
MODEL_NAME = "gpt-4-vision-preview"

class LLMClient:
    def __init__(self) -> None:
        self.last_message_time = 0
        self.llm_client = OpenAI(api_key=OPEN_AI_KEY)

    def get_llm_response(self, conversation: list):
        remaining_time = 90 - (time.time() - self.last_message_time)
        if remaining_time > 0:
            time.sleep(remaining_time + 5)
        self.last_message_time = time.time()
        assistant_response = self.llm_client.chat.completions.create(
            messages=conversation,
            model=MODEL_NAME,
            max_tokens=4096,
        )
        assistant_response_content = assistant_response.choices[0].message.content
        return assistant_response_content
