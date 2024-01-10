from collections import deque

from openai import OpenAI

client = OpenAI()


def gpt_chat(messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )
    return response.choices[0].message.content.strip()


class ChatMem:
    def __init__(self, system_msg=None, mem_size=50):
        self.mem = deque([], mem_size)
        if system_msg:
            self.mem.appendleft({"role": "system", "content": system_msg})

    def add(self, role, content):
        self.mem.appendleft({"role": role, "content": content})

    def get(self):
        return list(reversed(self.mem))

    def clear(self):
        self.mem.clear()

    def __repr__(self):
        return f"ChatMem({len(self.mem)} messages)"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.mem)
