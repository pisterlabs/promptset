import argparse
import os
import sys

from typing import Dict, Any

from ai.core.openai_core import OpenAiCore

openai_instance = OpenAiCore()

abs_path: str = os.path.dirname(os.path.abspath(__file__))
father_path: str = os.path.dirname(abs_path)


def start(start_params: Dict) -> None:
    content: str = openai_instance.replace_string(params)
    messages: list = [
        {
            "role": "system",
            "content": content
        }
    ]
    openai_instance.set_messages(messages)
    openai_instance.set_params(start_params)
    openai_instance.start_gradio_image()


if __name__ == "__main__":
    try:
        parser = openai_instance.parser()
        parser.add_argument("-t", "--topic", type=str, default="Illustration", help="Topic to be used")
        parser.add_argument("--gradio", type=bool, default=True, help="Activate gradio mode")
        parser.add_argument("--telegram", type=bool, default=False, help="Activate telegram bot")
        parser.add_argument("-n", type=int, default=2, help="Number of samples to be used")
        parser.add_argument("--size", type=str, default="512x512", help="Size to be used")
        parser.add_argument("--response_format", type=str, default="b64_json", help="ResponseModel format to be used")
        parser.add_argument("-r", "--role", type=str,
                            default="You are a digital artist specializing in {{topic}}{{sub-role}}.",
                            help="Role to be used")
        parser.add_argument("--sub-role", type=str, default="who works in a Graphic Design company.",
                            help="additional role to be used")

        params: Dict = vars(parser.parse_args())
        params = openai_instance.parse_params(params)
        start(params)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Bye!")
        sys.exit(0)
