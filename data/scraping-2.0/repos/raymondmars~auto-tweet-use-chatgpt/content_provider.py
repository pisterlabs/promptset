from openai import OpenAI
import os

class ContentProvider:
    def __init__(self):
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise("Please set OPENAI_API_KEY environment variable")

        base_url = os.environ.get('OPENAI_BASE_URL')

        self.client = OpenAI(
            api_key  = key,
            base_url = base_url if base_url != '' else None
        )

    def __get_persona_system_prompt(self, persona: str) -> str:
        print(persona)
        return f"You are a good assistant, but each reply to the user's question cannot exceed 140 characters. If the reply content is too long, please summarize the answer first, and then control the result within 140 characters. Add the [FromBot] at the end of each reply"

    def get_content(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.__get_persona_system_prompt(""),
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.9,
            model="gpt-4-1106-preview"
        )
        return chat_completion.choices[0].message.content