import openai


class ChatBot:
    def __init__(
        self,
        system_message: str,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
        history_length: int = 15,
        engine: str = "gpt-35-turbo",
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> None:
        self.messages = messages if messages is not None else []
        self.system_message = {"role": "system", "content": system_message}
        self.history_length = history_length
        self.temperature = temperature
        self.engine = engine
        self.top_p = top_p
        self.max_tokens = max_tokens

    def __call__(self, message: str, role: str) -> str:
        self.messages.append({"role": role, "content": message})
        messages = [self.system_message] + self.messages[-self.history_length :]

        completion = openai.ChatCompletion.create(
            engine=self.engine,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        message = completion.choices[0].message.content.strip()

        self.messages.append({"role": "assistant", "content": message})
        return message

    def set_message_content(self, index: int, content: str) -> None:
        self.messages[index]["content"] = content

    def export_conversation(self) -> str:
        result = ""
        for message in self.messages:
            result += f"\n{message['role'].upper()}:\n"

            result += f"{message['content']}"

        return result
