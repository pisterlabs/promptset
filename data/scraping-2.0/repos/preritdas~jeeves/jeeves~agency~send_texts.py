"""Wrapper around texts to work best as an agent."""
from langchain.agents.tools import BaseTool

import json
from typing import Any, Coroutine

from jeeves import texts


def create_text_message_tool(inbound_phone: str) -> type[BaseTool]:
    """
    Create a tool to send text messages.

    Args:
        inbound_phone: The phone number to send a confirmation text to after
            sending the message.

    Returns:
        A tool to send text messages.
    """
    class TextMessageTool(BaseTool):
        """Wrapper around texts to work best as an agent."""
        name: str = "Send Text Message"
        description = (
            "Useful for when you need to send a text message. Input must be a JSON string with "
            'the keys "content" and "recipient_phone" (10-digit phone number preceded by '
            'country code, ex. "12223334455". Do not make up phone numbers - either '
            "use a phone number explicitly provided by the user, or use a phone number from a "
            "tool that provides it for you (ex. contacts, if available). Otherwise, do not use this tool. "
            'Write the content as you, Jeeves, not as me.'
        )

        def _run(self, query: str) -> str:
            """Send a text message."""
            input_parsed = json.loads(query)

            # Validate
            assert "content" in input_parsed, 'Input must have a "content" key.'
            assert isinstance(input_parsed["content"], str), "Content must be a string."
            content = input_parsed["content"]

            assert (
                "recipient_phone" in input_parsed
            ), 'Input must have a "recipient_phone" key.'
            assert (
                len(str(input_parsed["recipient_phone"]).replace("+", "")) == 11
            ), "Recipient must be a phone number preceded by country code."
            recipient = str(input_parsed["recipient_phone"])

            try:
                send_res = texts.send_message(content=content, recipient=recipient)
            except Exception as e:
                return f"Error: {str(e)}"
            else:
                texts.send_message(
                    content=(
                        "Sir, I'm informing you that I have sent the following message to "
                        f"{recipient}:\n\n{content}"
                    ),
                    recipient=inbound_phone,
                )

            if send_res:
                return "Message delivered successfully."
            else:
                return "Message failed to deliver."

        def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, str]:
            raise NotImplementedError(f"{type(self).__name__} does not support async.")

    return TextMessageTool
