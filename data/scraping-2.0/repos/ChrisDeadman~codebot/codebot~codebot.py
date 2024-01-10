from collections import deque

import openai
import requests
from utils import num_tokens_from_messages
from utils.cyclic_buffer import CyclicBuffer

# define color codes
COLOR_GREEN = "\033[32m"
COLOR_ORANGE = "\033[33m"
COLOR_GRAY = "\033[90m"
COLOR_RESET = "\033[0m"


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class Codebot:
    def __init__(
        self,
        initial_prompt: str,
        api_key: str,
        buffer_capacity=15,
        max_tokens: int = 4000,
    ):
        self.messages = CyclicBuffer[Message](buffer_capacity)
        self.initial_prompt = Message("system", initial_prompt)
        self.max_tokens = max_tokens
        openai.api_key = api_key

    def chat_with_gpt(self) -> str:
        messages = deque([m.to_dict() for m in self.messages])
        while True:
            message_dicts = [self.initial_prompt.to_dict()] + list(messages)
            num_tokens = num_tokens_from_messages(message_dicts)
            if num_tokens < self.max_tokens:
                break
            if messages:
                # remove oldest message and try again
                messages.popleft()
            else:
                # no more messages
                self.messages.pop()
                return (
                    f"Too many tokens ({num_tokens}>{self.max_tokens}), "
                    f"please limit your message size!"
                )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=message_dicts
        )

        return response.choices[0].message.content

    def parse_response(self, input_str: str):
        result = []
        in_code_block = False
        ignore_code_block = False
        for line in input_str.split("\n"):
            if line.startswith("```"):
                in_code_block = not in_code_block
                if in_code_block:
                    if line.replace(" ", "").startswith("```python") or (
                        line.strip() == "```"
                        and not ("```python" in input_str or "``` python" in input_str)
                    ):
                        ignore_code_block = False
                    else:
                        ignore_code_block = True

                if ignore_code_block:
                    result.append({"code": False, "content": line})

                if not in_code_block:
                    ignore_code_block = False
            else:
                is_code = in_code_block and not ignore_code_block
                result.append({"code": is_code, "content": line})
        return result

    def execute_code(self, code: str):
        try:
            response = requests.post(
                "http://localhost:8080/execute", data=code.encode("utf-8")
            )
            result = response.content.decode("utf-8")
        except Exception as e:
            result = str(e)

        return result

    def run(self):
        user_input = input("You: ")
        user_cmd = user_input.strip().lower()

        if user_cmd == "reset" or user_cmd == "clear":
            self.messages.clear()
            return True
        elif user_cmd == "exit":
            return False

        if user_input.strip():
            self.messages.push(Message("user", user_input))

        gpt_response = self.chat_with_gpt()

        if gpt_response.strip():
            self.messages.push(Message("assistant", gpt_response))

        gpt_response_parsed = self.parse_response(gpt_response)
        gpt_code = "\n".join(
            map(
                lambda r: r["content"],
                filter(lambda r: r["code"], gpt_response_parsed),
            )
        )

        for r in gpt_response_parsed:
            if r["code"]:
                print(f"{COLOR_GREEN}{r['content']}{COLOR_RESET}")
            else:
                print(f"{COLOR_GRAY}{r['content']}{COLOR_RESET}")

        if gpt_code.strip():
            result = self.execute_code(gpt_code)

            print(f"{COLOR_ORANGE}Output: {result}{COLOR_RESET}")

            self.messages.push(Message("system", f"Output: {result}"))

        return True


if __name__ == "__main__":
    with open("data/initial_prompt.txt", "r") as f:
        initial_prompt = f.read()

    with open("data/openai_api_key.txt", "r") as f:
        api_key = f.read()

    codebot = Codebot(initial_prompt, api_key)

    while codebot.run():
        pass  # Run until complete
