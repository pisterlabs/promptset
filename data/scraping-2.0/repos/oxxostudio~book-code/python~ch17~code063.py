# Copyright © https://steam.oxxostudio.tw

import openai
openai.api_key = '你的 API Key'

messages = ''
while True:
    msg = input('me > ')
    messages = f'{messages}{msg}\n'   # 將過去的語句連接目前的對話，後方加上 \n 可以避免標點符號結尾問題
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=messages,
        max_tokens=128,
        temperature=0.5
    )

    ai_msg = response['choices'][0]['text'].replace('\n','')
    print('ai > '+ai_msg)
    messages = f'{messages}\n{ai_msg}\n\n'  # 合併 AI 回應的話

