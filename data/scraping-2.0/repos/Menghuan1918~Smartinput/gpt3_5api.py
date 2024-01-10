import openai
import json


def write_message_to_file(file_path, role, content):
    with open(file_path, "a" , encoding='utf-8') as file:
        message = {"role": role, "content": content.replace('"', '\\"')}
        file.write(json.dumps(message) + "\n")


def read_messages_from_file(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def chat(openai_key, user_input,file_path):
    total_tokens = 0
    openai.api_key = openai_key
    print(file_path)
    user_input.replace("\n", "\t")
    if (
        user_input == "history"
        or user_input == "His"
        or user_input == "his"
        or user_input == "History"
    ):
        with open(file_path, "r") as file:
            history = file.read().strip()
        print(history)
        return history
    write_message_to_file(file_path, "user", user_input)
    messages = read_messages_from_file(file_path)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    total_tokens += response["usage"]["total_tokens"]
    finish_reason = response["choices"][0]["finish_reason"]

    if finish_reason == "stop":
        print(f"正常返回，目前总共花费{total_tokens}字节")
    elif finish_reason == "length":
        length = "字符最高上限，请重启程序\nMaximum character limit, please restart the program"
        print(length)
        return length
    elif finish_reason == "content_filter":
        content_filter = "输入内容被屏蔽了\nThe input is blocked"
        print(content_filter)
        return content_filter
    elif finish_reason == "null":
        null_re = "未知错误\nUnknown error"
        print(null_re)
        return null_re
    assistant_output = response["choices"][0]["message"]["content"]
    print(assistant_output)
    write_message_to_file(file_path, "assistant", assistant_output)
    return assistant_output
