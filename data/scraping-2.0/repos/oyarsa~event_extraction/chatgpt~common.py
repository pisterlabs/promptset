import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import openai
from openai.error import OpenAIError
from ratelimit import limits, sleep_and_retry


class ExchangeLogger:
    def __init__(self) -> None:
        self.file: Path | None = None
        self.print_log = False

    def config(self, file: Path, print_log: bool = False) -> None:
        self.file = file
        self.print_log = print_log

    def log_exchange(self, params: dict[str, Any], response: dict[str, Any]) -> None:
        if self.file is None:
            raise ValueError("Must call config() before logging exchanges.")

        log = {"params": params, "response": response}

        with self.file.open("a") as f:
            json.dump(log, f)
            f.write("\n")

        if self.print_log:
            print(json.dumps(log, indent=2))
            print()


logger = ExchangeLogger()


def get_key(key_file: Path, key_name: str) -> str:
    keys = json.loads(key_file.read_text())
    return cast(str, keys[key_name])


def make_msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


CALLS_PER_MINUTE = 3500  # full plan


def make_chat_request(**kwargs: Any) -> dict[str, Any]:
    # Ignores (mypy): untyped decorator makes function untyped
    @sleep_and_retry  # type: ignore[misc]
    @limits(calls=CALLS_PER_MINUTE, period=60)  # type: ignore[misc]
    def _make_chat_request(**kwargs: Any) -> dict[str, Any]:
        attempts = 0
        while True:
            try:
                response = cast(dict[str, Any], openai.ChatCompletion.create(**kwargs))
            except Exception as e:
                ts = datetime.now().isoformat()
                print(
                    f'{ts} | Connection error - "{e}" | Kwargs | "{kwargs}"'
                    f" | Attempt {attempts + 1}"
                )
                attempts += 1

                if isinstance(e, OpenAIError) and e.http_status == 429:
                    print("Rate limit exceeded. Waiting 10 seconds.")
                    time.sleep(10)
            else:
                logger.log_exchange(kwargs, response)
                return response

    return cast(dict[str, Any], _make_chat_request(**kwargs))


def get_result(response: dict[str, Any]) -> str:
    return cast(str, response["choices"][0]["message"]["content"])


# Costs as of 2023-11-13 from https://openai.com/pricing
# model: input cost, output cost
MODEL_COSTS = {
    "gpt-3.5-turbo-1106": (  # in: $0.001 / 1K tokens, out: $0.002 / 1K tokens
        0.000001,
        0.000002,
    ),
    "gpt-4-0613": (0.00003, 0.00006),  # in: $0.03 / 1K tokens, out: $0.06 / 1K tokens
}


def calculate_cost(model: str, response: dict[str, Any]) -> float:
    input_tokens = response["usage"]["prompt_tokens"]
    output_tokens = response["usage"]["completion_tokens"]
    cost_input, cost_output = MODEL_COSTS[model]
    return input_tokens * cost_input + output_tokens * cost_output


def log_args(args: argparse.Namespace, path: Path | None) -> None:
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(args_dict))
    else:
        print(json.dumps(args_dict, indent=2))


def init_argparser(*, prompt: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("key_file", type=Path, help="Path to JSON file with API keys")
    parser.add_argument("key_name", type=str, help="Name of key to use")
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="Model to use"
    )
    parser.add_argument(
        "--print-logs", action="store_true", help="Print logs to stdout"
    )
    parser.add_argument(
        "--log-file", type=Path, default="chat_log.jsonl", help="Log file"
    )
    parser.add_argument("--input", "-i", type=Path, help="Input file")
    parser.add_argument("--output", "-o", type=Path, help="Output file for predictions")
    parser.add_argument("--metrics-path", type=Path, help="Path where to save metrics")
    parser.add_argument("--args-path", type=Path, help="Path where to save args")

    if prompt:
        parser.add_argument(
            "--prompt",
            type=int,
            default=0,
            help="User prompt index to use for the chat session",
        )
        parser.add_argument(
            "--sys-prompt",
            type=int,
            default=0,
            help="System prompt index to use for the chat session",
        )

    return parser
