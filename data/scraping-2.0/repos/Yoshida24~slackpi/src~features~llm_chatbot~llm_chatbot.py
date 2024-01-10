from type.type import MentionEventHandlerArgs
from modules.bolt.reply import reply
from modules.bolt.update_message import update_message
from modules.bolt.upload_file import upload_file
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import os
import json
from modules.bolt.extract_mention_and_text import extract_mention_and_text
from modules.bolt.is_message_from_bot import is_message_from_bot


from features.llm_chatbot.tools.pokefunction import (
    fetch_pokemon_data_impl,
    fetch_pokemon_data_tool,
)
from features.llm_chatbot.tools.screenshot import (
    take_screenshot_impl,
    take_screenshot_tool,
)
from features.llm_chatbot.tools.pixoo64.pixoo64_display_image_text import (
    pixoo64_display_image_text_impl,
    pixoo64_display_image_text_tool,
)
from features.llm_chatbot.tools.command_line.make import (
    make_impl,
    make_tool,
)
from typing import Callable

import logging

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm_model = "gpt-4-1106-preview"
temperature = 0.1
system_msg = "You are a Friendly and helpful AI assistant."

tools = [
    take_screenshot_tool,
    fetch_pokemon_data_tool,
    pixoo64_display_image_text_tool,
    make_tool,
]


def tools_response(
    messages: list[ChatCompletionMessageParam],
    presenter: Callable[[str], None],
) -> dict:
    logger.info(f"\033[94mMESSAGES_INPUT\033[0m:\n{messages}")
    function_response_file = None

    first_response = openai_client.chat.completions.create(
        model=llm_model,
        temperature=temperature,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls

    if tool_calls:
        """
        Parallel function calling が実行された場合
        """
        messages.append(first_message)
        # Pallarel function calling
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            for tool in tools:
                if function_name == tool["function"]["name"]:
                    logger.info(
                        f"\033[94mTOOL_CALLED\033[0m:\nfunction_name={function_name} arguments={arguments}"
                    )
                    if function_name == "take_screenshot":
                        selected_function = take_screenshot_impl
                    if function_name == "pixoo64_display_image_text":
                        selected_function = pixoo64_display_image_text_impl
                    elif function_name == "fetch_pokemon_data":
                        selected_function = fetch_pokemon_data_impl
                    elif function_name == "make":
                        selected_function = make_impl
                    else:
                        raise Exception("function not found")

                    function_response = selected_function(**arguments)
                    function_response_msg = function_response[
                        "message"
                    ]  # TODO: メッセージと他引数をうける
                    function_response_file = function_response["file"]

                    logger.info(
                        f"\033[94mTOOL_COMPLETED\033[0m:\nmessage={function_response_msg} function_response_file={function_response_file}"
                    )
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response_msg,
                        }
                    )

        # 関数実行結果を使ってもう一度質問
        second_response = openai_client.chat.completions.create(
            model=llm_model, temperature=temperature, top_p=1, messages=messages
        )

        response = second_response
    else:
        response = first_response

    logger.info(f"\033[94mMESSAGES_UPDATED\033[0m:\n{response}")
    result = {
        "content": response.choices[0].message.content,
        "file": function_response_file,
    }
    presenter(result["content"])
    return result


def slack_reply_clojure(args: MentionEventHandlerArgs, message_ts: str):
    def slack_reply(text: str | None):
        if text is None or text == "":
            return
        update_message(
            app=args.app,
            channel=args.event.event.channel,
            ts=message_ts,
            text=text,
        )

    return slack_reply


def handler(args: MentionEventHandlerArgs) -> None:
    try:
        # Notice user to accept the request
        thread_start_res = reply(
            app=args.app,
            mention_body=args.event,
            text="...",
        )

        # Init memory
        messages = []
        messages.append({"role": "system", "content": system_msg})
        conversations = sorted(args.messages, key=lambda x: x["ts"])  # type: ignore
        parsed_conversations = [
            {
                "role": ("assistant" if is_message_from_bot(conversation) else "user"),
                "content": extract_mention_and_text(conversation["text"])["text"]
                + (
                    "\nattached files: "
                    + "".join(
                        [
                            "\n- " + file["url_private_download"]
                            for file in conversation["files"]
                        ]
                    )
                    if "files" in conversation
                    else ""
                ),
            }
            for conversation in conversations
        ]
        messages.extend(parsed_conversations)
        logger.info(f"\033[94mHISTORY\033[0m:\n{conversations}")

        slack_reply = slack_reply_clojure(args, thread_start_res["ts"])
        tools_result = tools_response(messages=messages, presenter=slack_reply)
        if tools_result["file"] is not None:
            upload_file(
                app=args.app,
                mention_body=args.event,
                file=tools_result["file"],
                thread_ts=thread_start_res["ts"],
            )
    except BaseException as e:
        reply(
            app=args.app,
            mention_body=args.event,
            text=f"Error: {str(str(e))}",
        )
        raise e
