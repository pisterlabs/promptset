import openai


class GPTAssistant:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        self.messages = [
            {"role": "system",
             "content": "You are a helpful assistant. "
                        "Unless I don't ask you to, don't write coding snippets, but if I do make sure to append the language name to the leading three ```"}
        ]

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the message history.

        Args:
            role (str): The role of the message (e.g., "system", "user", "assistant").
            content (str): The content of the message.
        """
        self.messages.append({"role": role, "content": content})

    def chat(self) -> str:
        """
        Send the message history to the GPT model, retrieve the assistant's reply, and add it to the history.

        Returns:
            str: The assistant's reply.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            stream=False
        )
        assistant_reply = response['choices'][0]['message']['content']
        self.add_message("assistant", assistant_reply)  # Add assistant's response to history
        return assistant_reply
