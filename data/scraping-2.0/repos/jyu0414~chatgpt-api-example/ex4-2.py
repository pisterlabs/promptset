import openai
import json

# API Keyを設定
openai.api_key = ""

def get_temperture_function(location: str, unit: str = "cercius"):
    # 天気を取得するコード
    return "30"

messages = [{"role": "user", "content": "大阪の気温は？"}]
chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=[
        {
            "name": "get_temp",
            "description": "与えられた位置の情報から気温を返す",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "都市 e.g.東京",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            }
        }
    ]
    )

return_message = chat_completion["choices"][0]["message"]
if return_message.get("function_call"):
    function_list = { "get_temp": get_temperture_function }
    function_to_call = function_list[return_message["function_call"]["name"]]
    function_args = json.loads(return_message["function_call"]["arguments"])
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    messages.append(return_message)
    messages.append(
                {
            "role": "function",
            "name": return_message["function_call"]["name"],
            "content": function_response,
        }
    )

    second_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    print(second_response["choices"][0]["message"]["content"])
