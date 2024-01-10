import os
from typing import Optional

from openai import OpenAI


class ConversationalModel:
    def __init__(self, sys_prompt: Optional[str] = None):
        self.sys_prompt = sys_prompt
        self.messages = []
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def __call__(self, txt: str) -> str:
        self.messages.append({"role": "user", "content": txt})
        msgs = []
        if self.sys_prompt is not None:
            msgs.append({"role": "system", "content": self.sys_prompt})
        msgs += self.messages
        response = self.client.chat.completions.create(model="gpt-3.5-turbo",
                                                       messages=msgs)
        txt = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": txt})
        return txt

