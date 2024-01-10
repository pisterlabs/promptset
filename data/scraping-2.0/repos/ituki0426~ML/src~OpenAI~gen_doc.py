import openai
import os

import json

titles = [
    'パトカー',
    'Python',
    '写真撮影',
    '正式名称',
    'パイナップル',
    '挑戦状',
    '成人',
    '焼き肉',
    '迷彩柄',
    '竜巻',
]

SYSTEM_PROMPT = '''
提供される単語を300字以内で説明してください。
'''

docs = []
for title in titles:
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": title}
        ]
    )

    docs.append({
        'title': title,
        'body': res.choices[0].message.content
    })
    print(f'タイトル: {title}')
    print(res.choices[0].message.content)

with open('./docs.json', 'w') as f:
    json.dump(docs, f,ensure_ascii=False)