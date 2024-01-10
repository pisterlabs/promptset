# ref: https://zenn.dev/75py/articles/b3fa86255f7e28
import os
import json
import openai
from openai import OpenAI

client = OpenAI()

openai_api_key = os.environ.get('OPENAI_API_KEY')


def get_product_ranking_gategories(params):
    categories = [
        {"id": "1", "name": "USBメモリ"},
        {"id": "2", "name": "ハードディスク"},
    ]
    return json.dumps(categories, ensure_ascii=False)


def get_product_ranking(params):
    if params.get("categoryId") == "1":
        return json.dumps([
            {"id": "1001", "name": "USBメモリA 512GB", "isInStock": True},
            {"id": "1002", "name": "USBメモリB 256GB", "isInStock": False},
            {"id": "1003", "name": "USBメモリC 1TB 令和最新型", "isInStock": True},
        ], ensure_ascii=False)

    return json.dumps([
        {"id": "2001", "name": "HDD A", "isInStock": False},
        {"id": "2002", "name": "HDD B", "isInStock": False},
        {"id": "2003", "name": "HDD C", "isInStock": False},
    ], ensure_ascii=False)


functions = [
    {
        "name": "get_product_ranking_gategories",
        "description": "商品ランキングのカテゴリ一覧を取得する",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "descriptionn": "カテゴリ名"
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_product_ranking",
        "description": "商品のカテゴリIDを受け取り、そのカテゴリの人気商品ランキングを返す。レスポンスには、商品IDと商品名、店舗在庫有無の情報を持ったdictを返す。",
        "parameters": {
            "type": "object",
            "properties": {
                "categoryId": {
                    "type": "string",
                    "description": "カテゴリID"
                },
            },
            "required": ["categoryId"],
        },
    }
]

# 上で定義してあるグローバル変数の functions と一緒に API を実行する


def ask_gpt3_with_function(convesation):
    response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
                                              messages=convesation,
                                              functions=functions,
                                              function_call="auto")
    print(response)  # debug
    return response


def main():
    print("どんな商品をお探しですか？")
    conversation = [{"role": "system", "content": "あなたは家電量販店の販売員です。ユーザーが欲しい商品のカテゴリを聞いて、おすすめの商品を教えてあげてください。ユーザーは正確にカテゴリ名を入力しないため、カテゴリ名が一番近いであろうカテゴリを勧めてください。ただし、店舗に在庫がある商品を優先して勧めてください。"}]
    while True:
        user_input = input(" ('exit'で終了):")
        if user_input == 'exit':
            break

        # ユーザがインプットしてきた値をプロンプトに追記する
        conversation.append({"role": "user", "content": user_input})

        while True:
            response = ask_gpt3_with_function(conversation)
            message = response.choices[0].message
            if message.function_call:
                f_call = message.function_call
                function_name = f_call.name
                args = json.loads(f_call.arguments)
                function_response = globals()[function_name](args)
                print("---dubug: function_response---")
                print(function_response)  # debug
                # 選ばれた function をプロンプトに追記して再度問い合わせる
                conversation.append({"role": "function", "name": function_name, "content": function_response,
                                     })

            else:
                resMessage = message.content.strip()
                conversation.append(
                    {"role": "assistant", "content": resMessage})

                print("GPT-3の回答:", resMessage)
                break


if __name__ == "__main__":
    main()
