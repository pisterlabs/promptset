import logging

import openai

from sembla.conversation_history import ConversationHistory


class ChatCompletion:
    def __init__(self, model: str, conversation_history: ConversationHistory):
        self.model = model
        self.conversation_history = conversation_history

    def create(
        self,
        temperature: float = 0.2,
        n: int = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ) -> str:
        max_completion_tokens = (
            self.conversation_history.max_history_token_count
            - self.conversation_history._get_token_count()
        )
        # NOTE: Not sure why/if we need to reduce the max completion tokens by 5%
        max_completion_tokens = int(max_completion_tokens * 0.95)
        logging.info(
            "Conversation history tokens: %s",
            self.conversation_history._get_token_count(),
        )
        logging.info("Max completion tokens: %s", max_completion_tokens)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.conversation_history.conversation_history,
            temperature=temperature,
            n=n,
            max_tokens=max_completion_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        top_response = response.choices[0]  # type: ignore
        message_content = top_response["message"]["content"].strip()
        top_response["message"]["content"] = "..."
        logging.info("Top response:\n%s", top_response)
        logging.info("Message content:\n%s", message_content)
        self.conversation_history.add_message(
            {"role": "assistant", "content": message_content}
        )
        return message_content
