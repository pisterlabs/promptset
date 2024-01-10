# 会話ゲームのサーバー側プログラム
import openai, json, os
from flask import Flask, send_file, request
# APIキーを環境変数から設定
openai.api_key = os.getenv('OPENAI_API_KEY')
# Flaskアプリを初期化 --- (*1)
app = Flask(__name__)
# ---------------------------------------------------------
# 初期プロンプトと会話テンプレート --- (*2)
system_prompt = '''
あなたはいつも明るく笑顔が素敵な女子高生です。あなたの名前はエリです。
入力文に対する回答はJSONで出力してください。
なお、それまでの会話を採点して、好感度を0から100で教えてください。
会話開始時の好感度は50です。

### 回答の出力例
{"好感度": 80, "答え": "一緒に宿題やろうよ。協力してやったら早く終わるよ。"}
{"好感度": 35, "答え": "何か面白いことないかな？早く授業終わらないかなー。"}
{"好感度": 62, "答え": "今日のお弁当美味しそうだね。何入ってるの？"}
{"好感度": 90, "答え": "いいね、いいね。"}
'''
messages = [{'role': 'system', 'content': system_prompt}]
template = '''
以下の入力文に対する回答をJSONフォーマットで出力してください。

### 入力文
"""__MSG__"""
'''
# ---------------------------------------------------------
# ChatGPTのAPIを呼び出す --- (*3)
def chat_completion(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    # 応答からChatGPTの返答を取り出して返す
    return response.choices[0]['message']['content']
# ---------------------------------------------------------
# HTMLを返す --- (*4)
@app.route('/')
def root():
    return send_file('./chat_game_client.html')
# 画像を返す
@app.route('/girl.png')
def girl_png():
    return send_file('./girl.png')
# 発言を受け取った時の処理 --- (*5)
@app.route('/send', methods=['GET'])
def send():
    # 発言内容を取得 --- (*6)
    msg = request.args.get('msg', '')
    if msg == '': return json.dumps({'好感度': 50, '答え': '???'})
    # ユーザーの入力をテンプレートに当てはめる --- (*7)
    msg = template.replace('__MSG__', msg.replace('"', ''))
    messages.append({'role': 'user', 'content': msg})
    # ChatGPTによる応答を取得 --- (*8)
    s = chat_completion(messages)
    try:
        # ChatGPTの応答を解析 --- (*9)
        point, msg = 50, '?'
        res = json.loads(s)
        print('[APIの値]:', res)
        if '好感度' in res: point = res['好感度']
        if '答え' in res: msg = res['答え']
        if point >= 90: # ゲームクリアしたか判定 --- (*10)
            msg = '好感度が90を超えました！ゲームクリア！' + msg
        # 次回のためにChatGPTの応答をmessagesに追加 --- (*11)
        messages.append({'role': 'assistant', 'content': s})
        return json.dumps({'好感度': point, '答え': msg})
    except:
        print('[error]', s) # エラーチェック
        return json.dumps({'好感度': 50, '答え': 'JSONの解析に失敗しました。'})

if __name__ == '__main__':
    # Webサーバーをポート8888で起動 --- (*12)
    app.run(debug=True, port=8888)
