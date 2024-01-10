# Copyright © https://steam.oxxostudio.tw

import openai
openai.api_key = '你的 API Key'

from firebase import firebase
url = 'https://XXXXXXXXX.firebaseio.com'
fdb = firebase.FirebaseApplication(url, None)    # 初始化 Firebase Realtime database
chatgpt = fdb.get('/','chatgpt')                 # 取的 chatgpt 節點的資料

if chatgpt == None:
    messages = ''        # 如果節點沒有資料，訊息內容設定為空
else:
    messages = chatgpt   # 如果節點有資料，使用該資料作為歷史聊天記錄

while True:
    msg = input('me > ')
    if msg == '!reset':
        message = ''
        fdb.delete('/','chatgpt')         # 如果輸入 !reset 就清空歷史紀錄
        print('ai > 對話歷史紀錄已經清空！')
    else:
        messages = f'{messages}{msg}\n'   # 在輸入的訊息前方加上歷史紀錄
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=messages,
            max_tokens=128,
            temperature=0.5
        )

        ai_msg = response['choices'][0]['text'].replace('\n','')  # 取得 ChatGPT 的回應
        print('ai > '+ai_msg)
        messages = f'{messages}\n{ai_msg}\n\n'   # 在訊息中加入 ChatGPT 的回應
        fdb.put('/','chatgpt',messages)          # 更新資料庫資料

