#!/usr/bin/env python
import os
import sys
from typing import Dict

from ai.core.commons import ROLE, TOPIC
from ai.core.openai_core import OpenAiCore
from ai.core.telegram_core import TelegramCore

openai_instance = OpenAiCore()
telegram_instance = TelegramCore()

abs_path: str = os.path.dirname(os.path.abspath(__file__))
father_path: str = os.path.dirname(abs_path)


def start(start_params: Dict) -> None:
    openai_new_instance = OpenAiCore()
    content: str = openai_new_instance.replace_string(start_params)
    messages: list = [
        {
            "role": "system",
            "content": content
        }
    ]

    file_name: str = f"{str(start_params['topic']).lower().replace(' ', '_')}.json"
    if start_params["telegram"]:
        telegram_instance.run(params)

    if start_params["gradio"]:
        openai_new_instance.create_file(file_name)
        openai_new_instance.set_messages(messages)
        openai_new_instance.get_messages(file_name)
        openai_new_instance.set_params(start_params)
        openai_new_instance.start_gradio_chat()


if __name__ == "__main__":
    try:
        parser = openai_instance.parser()
        parser.add_argument("-t", "--topic", type=str, default=TOPIC, help="topic to be used")
        parser.add_argument("-r", "--role", type=str,
                            default=ROLE,
                            help="Role to be used")

        params: Dict = vars(parser.parse_args())
        params = openai_instance.parse_params(params)
        start(params)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Bye!")
        sys.exit(0)
