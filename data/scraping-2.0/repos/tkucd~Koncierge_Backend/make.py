import openai
import os
from dotenv import load_dotenv

# .envファイルのパスを指定して読み込む
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 与えられた情報から料理を作成
def make_cuisine(ingredients, kcal, time):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":  '''
                            あなたはプロの料理人です。以下に続く指示に基づいて、
                            与えられた食材リストを用いて献立を3つ考えてください。

                            #指示
                            -料理は全て一人前としてください。
                            -与えられたカロリー以下の料理を考えてください。
                            -与えられた時間以下で料理を作ることができるようにしてください。
                            -料理工程は全部で6つになるようにしてください。
                            -各工程の最初には「1.」のように工程の番号をつけてください。
                            -各工程の説明の文字数は20文字以内に収めてください。
                            -食材の切り方や火加減などの調理方法の詳しい説明もしてください。
                            -料理の名前もつけてください。
                            -出力するときは以下のフォーマットに従ってください。

                            #フォーマット
                            {
                                "menu": [
                                    {
                                        "ingredients": 食材リストにある食材から数品（各食材を要素にしたリスト）,
                                        "dish": 料理名,
                                        "recipe": 料理の作り方（各工程を要素にしたリスト）,
                                        "kcal": カロリー,
                                        "time": 時間（分）,
                                    },
                                ]
                            }

                            #注意点
                            -キーは必ず含ませてください。
                            -JSON以外の情報は削除してください。
                            -忠実に指示に従っているか20回は見直して確認してください。
                            -特に料理工程の説明の文字数は必ず20文字以内に収めてください。
                            -指示に従っていなければ、料理人としての評価は下がり恐ろしい罰が課せられます。
                            '''
            },
            {
                "role": "user",
                "content": f"材料: {ingredients}, カロリー: {kcal}, 時間: {time}分"
            }
        ]
    )
    return res["choices"][0]["message"]["content"]