import openai
import json

# API Keyを設定
openai.api_key = ""

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        { "role": "system", "content": "「掛け算を使った問題をつくりなさい」という問題への回答が入力されます．入力が問題への回答として正しいかどうかを評価して，正誤とフィードバック文を表示せよ"},
        { "role": "user", "content": "太郎が花を3本，次郎が花を10本もっています．合わせていくつ？"}
    ],
    functions=[
        {
            "name": "show",
            "description": "正誤とフィードバック文を表示",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "間違っている理由 e.g. これは割り算の問題です",
                    },
                    "result": {"type": "string", "enum": ["correct", "incorrect"]},
                },
                "required": ["result"],
            }
        }
    ]
    )


print(chat_completion["choices"][0]["message"]["function_call"]["arguments"])
