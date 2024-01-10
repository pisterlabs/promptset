import tiktoken
from types import SimpleNamespace
import openai
import os
from brish import z, zp
from pynight.common_bells import bell_gpt
from pynight.common_dict import simple_obj
from pynight.common_str import (
    whitespace_shared_rm,
)
from pynight.common_clipboard import (
    clipboard_copy,
    clipboard_copy_multi,
)

##
openai_key = None


def setup_openai_key():
    global openai_key

    openai_key = z("var-get openai_api_key").outrs
    assert openai_key, "setup_openai_key: could not get OpenAI API key!"

    openai.api_key = openai_key


def openai_key_get():
    global openai_key

    if openai_key is None:
        setup_openai_key()

    return openai_key


###
import openai
import pyperclip
from icecream import ic
import subprocess
import time
import sys


def print_chat_streaming(
    output,
    *,
    debug_p=False,
    # debug_p=True,
    output_mode=None,
    copy_mode="chat2",
    # copy_mode="default",
    end="\n-------",
):
    """
    Process and print out chat completions from a model when the stream is set to True.

    Args:
        output (iterable): The output from the model with stream=True.
    """
    text = ""
    r = None
    for i, r in enumerate(output):
        if not isinstance(r, dict):
            #: OpenAI v1: Response objects are now pydantic models instead of dicts.
            ##
            r = dict(r)

        text_current = None
        choice = r["choices"][0]
        if "delta" in choice:
            delta = choice["delta"]
            if i >= 1:
                #: No need to start all responses with 'assistant:'.
                ##
                if "role" in delta:
                    if i >= 1:
                        print("\n", end="")

                    print(f"{delta['role']}: ", end="")

            if "content" in delta:
                text_current = f"{delta['content']}"
        elif "text" in choice:
            text_current = f"{choice['text']}"

        if text_current:
            text += text_current
            print(f"{text_current}", end="")

    print(end, end="")

    text = text.rstrip()

    if debug_p == True:
        ic(r)

    chat_result = None
    if copy_mode:
        chat_result = chatml_response_text_process(
            text,
            copy_mode=copy_mode,
        )

    if output_mode == "chat":
        return chat_result
    elif output_mode == "text":
        return text
    elif not output_mode:
        return None
    else:
        raise ValueError(f"Unsupported output_mode: {output_mode}")


def chatml_response_process(
    response,
    end="\n-------",
    **kwargs,
):
    for choice in response["choices"]:
        text = choice["message"]["content"]

        chatml_response_text_process(
            text,
            **kwargs,
        )
        print(text, end="")

        print(end, end="")


def chatml_response_text_process(
    text,
    copy_mode="chat2",
    # copy_mode="default",
):
    #: 'rawchat' is currently useless, just use 'text'.
    ##
    text_m = None
    if copy_mode in ("chat", "chat2"):
        text_m = f'''        {{"role": "assistant", "content": r"""{text}"""}},'''
    elif copy_mode in ("rawchat"):
        text_m = f"""{text}"""

    if copy_mode == "chat2":
        text_m += f'''
        {{"role": "user", "content": r"""\n        \n        """}},'''

    if copy_mode in (
        "default",
        "chat2",
    ):
        clipboard_copy_multi(text, text_m)

    elif copy_mode in (
        "chat",
        # "chat2",
        "rawchat",
    ):
        clipboard_copy(text_m)

    elif copy_mode == "text":
        clipboard_copy(text)

    return simple_obj(
        text=text,
        text_chat=text_m,
    )


def writegpt_process(messages_lst):
    out = ""
    seen = [
        "PLACEHOLDER",
    ]
    #: We can also just count the number of assistant outputs previously seen, and skip exactly that many. That way, we can edit the text more easily.

    for messages in messages_lst:
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role in ("assistant",) and content not in seen:
                seen.append(content)

                if out:
                    out += "\n\n"
                out += content

    out = subprocess.run(
        ["perl", "-CIOE", "-0777", "-pe", r"s/(*plb:\S)(\R)(*pla:\S)/\\\\$1/g"],
        text=True,
        input=out,
        errors="strict",
        encoding="utf-8",
        capture_output=True,
    ).stdout

    clipboard_copy(out)
    return out


def openai_chat_complete(
    *args,
    model="gpt-3.5-turbo",
    messages=None,
    stream=True,
    interactive=False,
    copy_last_message=None,
    trim_p=True,
    **kwargs,
):
    if model in (
        "gpt-4-turbo",
        "4t",
    ):
        model = "gpt-4-1106-preview"

    if interactive:
        if copy_last_message is None:
            copy_last_message = True

    if messages is not None:
        if trim_p:
            #: Trim the messages:
            for message in messages:
                message["content"] = whitespace_shared_rm(message["content"])
                message["content"] = message["content"].strip()

    try:
        while True:
            if copy_last_message:
                last_message = messages[-1]["content"]
                clipboard_copy(last_message)

            try:
                return openai.ChatCompletion.create(
                    *args,
                    model=model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
            except openai.error.RateLimitError:
                print(
                    "OpenAI ratelimit encountered, sleeping ...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(10)  #: in seconds
    finally:
        pass


###
def truncate_by_tokens(text, length=3500, model="gpt-3.5-turbo"):
    #: @deprecated?
    #: @alt =ttok=
    ##
    encoder = tiktoken.encoding_for_model(model)

    encoded = encoder.encode(text)

    truncate_p = len(encoded) > length
    encoded_rest = None
    if truncate_p:
        encoded_truncated = encoded[:length]
        encoded_rest = encoded[length:]
    else:
        encoded_truncated = encoded

    text_truncated = encoder.decode(encoded_truncated)
    text_rest = None
    if encoded_rest:
        text_rest = encoder.decode(encoded_rest)

    return SimpleNamespace(
        text=text_truncated,
        text_rest=text_rest,
        truncated_p=truncate_p,
    )


###
