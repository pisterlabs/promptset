import openai
import json
import plugin_networking

funcList = {
    "networking": plugin_networking.networking,
}

funcDescList = [
    plugin_networking.get_networking_desc(),
]


def run():
    # 记得修改成你的OpenAI API Key
    openai.api_key = "sk-xxx"

    messages = [
        {"role": "user", "content": "帮我阅读一下：https://github.com/Significant-Gravitas/Auto-GPT，并给我简要介绍下"}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=funcDescList,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    response_message = response["choices"][0]["message"]
    if response_message.get("function_call"):
        func_name = response_message["function_call"]["name"]
        func_args = response_message["function_call"]["arguments"]

        func = funcList.get(func_name)
        if func:
            response = func(json.loads(func_args))
        else:
            print("no such func: '" + func_name + "'")
            return

        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": func_name,
                "content": response,
            }
        )

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        print(second_response["choices"][0]["message"]["content"])

    else:
        print(response_message["content"])


if __name__ == "__main__":
    run()
