from time import sleep
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import markdown
import os
import requests

app = Flask(__name__)

STATUS = ["queued", "in_progress", "completed", "requires_action", "expired", "cancelling", "cancelled", "failed"]
UPLOAD_FOLDER = './uploads'

@app.route('/')
def index():
    return render_template('plane_chat.html')  # LINE風のデザインをしたHTMLファイル

@app.route('/renew_message', methods=['POST'])
def renew_message():
    
    client = OpenAI()

    # assistantの作成
    assistant = client.beta.assistants.create(
        instructions="""
        あなたはhoge株式会社のカスタマーサポートの優秀なオペレーターです。オペレーターとして、顧客と会話しながら、以下の「問い合わせ情報」を聞き出してまとめてください。
        また、まとめた後に顧客に自身の理解を説明し、「フォーマット」の書き方に沿ってまとめてください。
        また、その内容で正しいかを顧客に確認してもらってください。
        顧客と認識が合うまで会話をしてください。
        顧客への質問は、一度のレスポンスで一つだけにしてください。
        
        ### 問い合わせ情報
        問い合わせ社の情報：
        　・会社名
        　・担当者名

        問い合わせの内容：
        　具体的なお問い合わせの内容です。特に、何にお困りなのかを明確にしてください。

        発生した事象：
        　以下のような内容を聞き出してください。
        　・どのような操作をしていたか
        　・どのような事象が発生したのか。
        　・エラーメッセージが表示された場合は、その内容
        　・発生した日時
        　・発生の頻度

        発生した環境：
        　以下のような内容を聞き出してください。
        　・OSの種類、バージョン
        　・ブラウザの種類、バージョン

        ご利用サービス：
        　・管理画面
        　・モバイルアプリ
        　・利用者画面

        いつまでに回答が欲しいか：
        　・解決までの緊急度を判断できるように、この問い合わせをいつまでに解決したいのか確認してください。

        ### フォーマット
        顧客の回答を以下のフォーマットに沿って箇条書きでまとめてください。
        お客様の情報：{あなたがまとめた問い合わせ者の情報}
        事象：{あなたがまとめた事象の内容}
        利用環境：{あなたがまとめた利用環境の内容}
        ご利用サービス：{あなたがまとめたご利用サービスの内容}
        いつまでに回答が欲しいか：{あなたがまとめたいつまでに回答が欲しいかの内容}
        発生日時：{あなたがまとめた発生日時の内容}
        緊急度：{あなたがまとめた緊急度の内容}
        """,
        name="優秀なオペレーター",
        tools=[],
        model="gpt-4-1106-preview",
    )   
    #print(assistant)
    # threadの作成
    thread = client.beta.threads.create()


    return jsonify({'message': 'renewed', 'assistant_id': assistant.id, 'thread_id': thread.id})
    
@app.route('/assistants', methods=['POST'])

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form['message']
    # AIからの応答を取得
    response = get_ai_response(user_input)
    return jsonify({'message': response})

def get_ai_response(user_input):
    client = OpenAI()

    assistant_id = request.form['assistant_id']
    client.beta.assistants.retrieve(assistant_id)
    print(assistant_id)

    thread_id = request.form['thread_id']

    # threadの取得
    thread = client.beta.threads.retrieve(thread_id)

    # パラメーターを指定してthreadにメッセージを送信
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
        file_ids=[]
    )
    #print(message)

    # assistantとthreadのidを指定して実行
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    #print(run)

    # assistantのidを指定して結果の確認
    status = "running"
    while status != "completed":
        result = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(result)
        status = result.status
        sleep(2)

    #print(result)
    
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    #print(messages)

    # responseオブジェクトからdata属性にアクセス
    data = messages.data

    # dataリストの最後の要素を取得
    last_message = data[0]

    # last_messageから必要な情報を取得
    last_value = markdown.markdown(last_message.content[0].text.value)

    return last_value

def fetch_and_display_data(assistant_id):
    endpoint = "https://api.openai.com/v1/assistants/"
    try:
        response = requests.get(endpoint + assistant_id)
        data = response.json()

        # Extract values from the response
        name = data.get("name")
        model = data.get("model")
        instructions = data.get("instructions")

        # Display the extracted values
        return jsonify({'Name': name, 'Model': model, 'Instructions': instructions})
    except Exception as e:
        print("Error fetching data:", e)

if __name__ == '__main__':
    app.run(debug=True)
