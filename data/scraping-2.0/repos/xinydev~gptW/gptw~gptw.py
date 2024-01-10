import argparse
import json
import logging
import os
import sys
from os.path import expanduser

from openai import AzureOpenAI, OpenAI

import gptw


def args_init():
    parser = argparse.ArgumentParser(
        description="\n".join(
            [
                "GPT Simplifies Your Daily Workflow (gptW)",
                "",
                "Usage:",
                'ww e "今天天气怎么样"  # translate to English',
                'ww c "who are you?"  # translate to Chinese',
                "ww --list  # get all available commands",
            ]
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "cmd", help="cmd, run `ww --list` to get all available cmd", nargs="?", type=str
    )
    parser.add_argument(
        "text",
        help="text",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "-c", "--config", dest="config", help="set config key and value"
    )
    parser.add_argument("-f", "--file", dest="file", help="read from file")
    parser.add_argument(
        "-l",
        "--list",
        dest="list",
        action="store_true",
        default=False,
        help="list all available sub cmds",
    )

    parser.add_argument(
        "--voice",
        dest="voice",
        action="store_true",
        default=False,
        help="chat with gpt in voice mode",
    )

    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="enable debug output",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + gptw.__version__,
    )
    try:
        return parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)


def init_logging(debug: bool):
    level = logging.WARN
    if debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(CURRENT_FOLDER, "prompts.json")
config_file = os.path.join(expanduser("~"), ".gptw-config.txt")


def set_config(key, value):
    cfg = {}

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(cfg, f)

    with open(config_file) as f:
        cfg = json.load(f)

    cfg[key] = value

    with open(config_file, "w") as f:
        json.dump(cfg, f)


CONFIG = None


def get_config(key, default_value=None):
    if not CONFIG:
        with open(config_file) as f:
            CFG = json.load(f)
    try:
        if key not in CFG and default_value:
            return default_value
        return CFG[key]
    except Exception:
        print(f"config {key} not found, run `ww --config` to set it")
        sys.exit(1)


def get_prompts():
    with open(prompts_file) as f:
        return json.load(f)["cmds"]


def mode_multi_pass(f, prompt, user_input, repeat_cnt, auth):
    msgs = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ]
    for _ in range(repeat_cnt):
        resp = f(msgs, auth)
        msgs.append({"role": "assistant", "content": resp})
        msgs.append({"role": "user", "content": "retry, a better one please"})
        print("")
    return ""


def mode_chat(f, content, auth):
    msgs = [{"role": "user", "content": content}]

    while True:
        resp = f(msgs, auth)
        msgs.append({"role": "assistant", "content": resp})
        user_resp = input("You: ")
        if not user_resp:
            break
        msgs.append({"role": "user", "content": user_resp})
    return ""


def ask_azure(content, auth):
    logging.debug(f"!!!ask:{content}")

    client = AzureOpenAI(
        api_key=auth["token"],
        api_version="2023-10-01-preview",
        azure_endpoint=auth["endpoint"],
    )
    completion = client.chat.completions.create(
        model=auth["depname"], messages=content, temperature=1.2, n=2, timeout=10
    )
    ret = ""
    choices = completion.choices
    for i, choice in enumerate(choices):
        ret = str(choice.message.content).strip()
        print(f"choice {i}: {ret}")
        print("--------")

    logging.debug(f"!!!resp:{ret}")
    return ret


def ask_openai(content, auth):
    logging.debug(f"!!!ask:{content}")

    client = OpenAI(
        api_key=auth["token"],
    )
    completion = client.chat.completions.create(
        model=auth["model"],
        messages=[{"role": "user", "content": content}],
        temperature=0.5,
    )
    ret = str(completion.choices[0].message.content).strip()
    print(ret)
    return ret


def list_commands(prompts):
    print(f'{"cmd":<{3}} | {"meaning":<{30}} | {"example"}')
    for pmt in prompts:
        print(
            f"{pmt: <{3}} | {prompts[pmt]['_comment']:<{30}} | {prompts[pmt]['example']}"
        )


def main():
    args = args_init()
    init_logging(args.debug)
    logging.debug(f"src folder: {CURRENT_FOLDER}")

    if args.config:
        logging.debug("set config")
        k, v = args.config.split("=")
        set_config(k, v)
        exit(0)

    prompts = get_prompts()
    logging.debug(f"configs:{prompts}")

    if args.list:
        list_commands(prompts)
        exit(0)

    text = " ".join(args.text)
    if args.file:
        with open(args.file) as f:
            text = "\n".join(f.readlines())
    if not text:
        print("Please enter some content")
        exit(0)

    if not args.cmd or args.cmd not in prompts:
        print("need a command")
        exit(0)

    logging.debug(f"cmd:{args.cmd},text:{text}")

    prompt = prompts[args.cmd]["prompt"]
    user_input = text

    provider = get_config("provider")

    if provider == "openai":
        func_ask = ask_openai
        func_auth = {
            "model": get_config("openai-model"),
            "token": get_config("openai-token"),
        }
    elif provider == "azure":
        func_ask = ask_azure
        func_auth = {
            "token": get_config("azure-token"),
            "endpoint": get_config("azure-endpoint"),
            "depname": get_config("azure-depname"),
        }
    else:
        print(f"provider:{provider} not supported")
        exit(1)

    if args.cmd == "a":
        print(mode_chat(func_ask, user_input, func_auth))
    else:
        repeat_cnt = int(get_config("repeat-cnt", 1))
        print(mode_multi_pass(func_ask, prompt, user_input, repeat_cnt, func_auth))
