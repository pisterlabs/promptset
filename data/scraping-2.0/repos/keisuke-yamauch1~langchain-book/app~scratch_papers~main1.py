# これはサンプルの Python スクリプトです。
import json
import os
from openai import OpenAI


# ⌃R を押して実行するか、ご自身のコードに置き換えてください。
# ⇧ を2回押す を押すと、クラス/ファイル/ツールウィンドウ/アクション/設定を検索します。


def create_open_ai_client():
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )


def get_current_weather(location, unit="celsius"):
    weather_info = {
        "location": location,
        "temperture": "25",
        "unit": "celsius",
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Tokyo",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]

    client = create_open_ai_client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions
    )

    response_message = response.choices[0].message
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    function_name = response_message.function_call.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(response_message.function_call.arguments)

    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    messages.append(response_message)
    messages.append({
        "role": "function",
        "name": function_name,
        "content": function_response,
    })

    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    print(second_response)

# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
