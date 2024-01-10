
from openai import OpenAI


def create_auto_article(user_input_target,user_input_target_ploblem, user_input__SEO_keyword):

    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
    {"role": "system", "content": "あなたは教師です。\
     学校の課題を制作する手助けを行ってください\
     あなたの出力した問題をそのまま使います"},
    {"role": "user", "content": f"あなたは以下のデータを基に問題を作成します。\
     科目は{user_input__SEO_keyword}\
     例題は{user_input_target}\
    どのような問題を作ってほしいかは{user_input_target_ploblem}"},
    ]
    )
    return  response.choices[0].message.content