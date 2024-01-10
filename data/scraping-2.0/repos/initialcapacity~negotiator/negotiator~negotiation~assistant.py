import openai

from negotiator.negotiation.negotiation_service import Negotiation


class Assistant:
    def reply(self, negotiation: Negotiation) -> str:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.__messages_from(negotiation)
        )

        return chat_completion.choices[0].message.content

    def __messages_from(self, negotiation: Negotiation) -> list[dict[str, str]]:
        return [
            {'role': m.role, 'content': m.content}
            for m in negotiation.messages
            if m.role in ['user', 'assistant', 'system']
        ]
