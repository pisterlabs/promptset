import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

def generate_questions(world: str, data: list[dict], question_num:int = 20):
    data_string = "\n".join([
        f'名前\n"{datum["name"]}"\n\n特徴\n"""{datum["text"]}\n"""\n'
        for datum in data
    ])
    
    system_settings = f'''
    {world}についての対話を行います。
    {world}の例は以下に示します

    {data_string}

    上記{world}の設定を参考にして次の対話に答えてください。
    では対話を開始します。'''

    messages = [
        SystemMessage(content=system_settings),
        HumanMessage(content=f'{world}達を見分けるために最も有効な質問を{question_num}個教えて下さい。"はい"/"いいえ"/"わからない"/"部分的にそう"の4択で答えられる質問にしてください。jsonのリストで出力してください。')
    ]
    chat = ChatOpenAI(temperature=0)
    response = chat(messages)
    return json.loads(response.content)