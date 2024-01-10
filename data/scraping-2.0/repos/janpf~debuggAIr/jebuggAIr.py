import re
import sys
from subprocess import PIPE, STDOUT, Popen

from loguru import logger
from openai import OpenAI

system_prompt = """
The user encountered a bug in their program and is stuck debugging it. They give you full access to their jdb shell. It is crucial that you only reply with a valid jdb command in all following messages, as your entire message will be automatically fed to jdb as is.
Only ever reply with a single jdb command and wait until the debugger gives you their output before issuing the next command. Before you start, the program will be run until the error occurs ("run").
Carefully and exhaustively explore the program and find the solution to the bug using e.g. list, locals, print... Do not explain anything, as all messages will be automatically thrown into jdb.
When you have found the solution begin your message with "!solution" followed by a very specific explanation containing what the issue is and how to fix it. This will then be displayed to the user.
Be sure you give precise and explicit instructions. For example, if an exception is thrown and the user has to change a variable, tell them exactly what to change it to. Your solution is final and will automatically end the debugging session.
Do not explain how to add error handling to the program. Only explain how to fix the bug.
The script already got called with `jdb <script.java> <args>` and the output looks like this:
""".strip()
client = OpenAI()


def construct_prompt(in_hist: list[str], out_hist: list[str]):
    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n" + out_hist[0].strip(),
        }
    ]
    for in_msg, out_msg in zip(in_hist, out_hist):
        messages.append(
            {
                "role": "assistant",
                "content": in_msg,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": out_msg,
            }
        )
    return messages


if __name__ == "__main__":
    start_args = ["jdb"] + sys.argv[1:]

    dbg_proc = Popen(start_args, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    dbg_proc.stdin.write(("run\n").encode("utf-8"))
    dbg_proc.stdin.flush()

    out_hist = []
    in_hist = []

    while dbg_proc.poll() is None:
        text = ""
        while True:
            text += dbg_proc.stdout.read(1).decode("utf-8")
            if re.search(r"\[\d+\]", text):
                break

        if len(in_hist) > 20:
            logger.warning("Too many messages, avoiding bankruptcy 👋🏻")
            break

        text = text.strip()
        logger.info(text)
        out_hist.append(text)

        messages = construct_prompt(in_hist, out_hist)
        # logger.debug(messages)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-1106",
        )
        # logger.debug(chat_completion)
        next_action = chat_completion.choices[0].message.content
        next_action = next_action.replace("`", "")  # it just likes to add those
        if not "!solution" in next_action:
            next_action = next_action.split("\n")[0]  # only take the first expression
        logger.info(next_action)

        if next_action.startswith("!solution"):
            logger.success("Solution found")
            break

        dbg_proc.stdin.write((next_action + "\n").encode("utf-8"))
        dbg_proc.stdin.flush()
        in_hist.append(next_action)

    dbg_proc.stdout.close()
    dbg_proc.stdin.close()
