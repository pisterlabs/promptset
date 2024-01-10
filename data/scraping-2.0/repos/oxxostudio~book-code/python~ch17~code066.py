# Copyright © https://steam.oxxostudio.tw

import openai
openai.api_key = '你的 API Key'

from firebase import firebase
url = 'https://XXXXXXXXXXX.firebaseio.com'
fdb = firebase.FirebaseApplication(url, None)   # 初始化 Firebase Realtimr database
chatgpt = fdb.get('/','chatgpt')                # 讀取 chatgpt 節點中所有的資料

if chatgpt == None:
    messages = []        # 如果沒有資料，預設訊息為空串列
else:
    messages = chatgpt   # 如果有資料，訊息設定為該資料

while True:
    msg = input('me > ')
    if msg == '!reset':
        fdb.delete('/','chatgpt')   # 如果輸入 !reset 就清空 chatgpt 的節點內容
        messages = []
        print('ai > 對話歷史紀錄已經清空！')
    else:
        messages.append({"role":"user","content":msg})  # 將輸入的訊息加入歷史紀錄的串列中
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=128,
            temperature=0.5,
            messages=messages
        )
        ai_msg = response.choices[0].message.content.replace('\n','')  # 取得回應訊息
        messages.append({"role":"assistant","content":ai_msg})  # 將回應訊息加入歷史紀錄串列中
        fdb.put('/','chatgpt',messages)   # 更新 chatgpt 節點內容
        print(f'ai > {ai_msg}')

