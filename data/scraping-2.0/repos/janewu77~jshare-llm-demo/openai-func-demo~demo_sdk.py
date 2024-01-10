import cfg.cfg

import json

import openai
from termcolor import colored
from datetime import date

import my_funcs

GPT_MODEL = "gpt-3.5-turbo-0613"
# gpt-3.5-turbo-16k-0613


def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):

    json_data = {"model": model, "messages": messages, "temperature": 0, "user": "dad3"}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        return openai.ChatCompletion.create(**json_data)
        # return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
            )
        )


messages = []
messages.append({"role": "system", "content": f'''
Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
'''})


def print_result(respJson):
    finish_reason = ''
    if 'finish_reason' in respJson["choices"][0]:
        finish_reason = respJson["choices"][0]["finish_reason"]
    # print(f"finish_reason:{finish_reason}")

    assistant_message = respJson["choices"][0]["message"]
    if finish_reason == 'function_call':
        func_name = assistant_message["function_call"]['name']
        print(f"func_name:{func_name}")
        arguments = assistant_message["function_call"]['arguments']
        # parsed_json = json.loads(arguments)
        print(f"arguments: {json.loads(arguments)}")
    else:
        print(f"assistant response :{assistant_message}")


def do_chat(user, msg):
    messages.append({"role": "user", "content": f"{msg}\n\n user_id: {user} Today: {date.today()}"})
    json_data = chat_completion_request(messages, functions=my_funcs.functions)

    print("======")
    print(f"user:{msg}")

    if "error" in json_data:
        print("The JSON contains an error:", json_data["error"])
        return

    # print(f"  respJson:{json_data}")
    assistant_message = json_data["choices"][0]["message"]
    print(f"assistant_message:{assistant_message}")
    messages.append(assistant_message)

    print_result(json_data)


if __name__ == '__main__':

    # 今天买了二斤桔子 30元1斤 咖啡30元 昨天收到工资3000元
    do_chat("user1", "买了二斤桔子 30元1斤 买咖啡花了20.3元 用支付宝付的 上周收到工资3000元")
    # do_chat("后天去参加小张的婚礼")
    # do_chat("你最近怎么样？")
    do_chat("user2", "下周一去超市买点水果")

    print("\n\n")
    print("======")
    pretty_print_conversation(messages)


