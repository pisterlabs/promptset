import openai

# APIキーを設定してください。例: 'your-api-key'
api_key = 'XXX'
openai.api_key = api_key

def generate_text(prompt, role, conversation_history):
    # ユーザーの質問を会話履歴に追加
    conversation_history.append({"role": "user", "content": prompt})
    
    # GPT-4モデルを使用してテキストを生成
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"You are a {role}."}] + conversation_history,
        max_tokens=50,
        n=1,
        temperature=0.8,
    )
    message = response.choices[0].message['content'].strip()
    
    # アシスタントの回答を会話履歴に追加
    conversation_history.append({"role": "assistant", "content": message})
    
    return message

if __name__ == "__main__":
    # ロールプレイのモデルをユーザーに入力させる
    role = input("ロールプレイのモデルを指定してください（例: helpful assistant）: ")
    
    # 会話履歴を格納するためのリストを初期化
    conversation_history = []
    
    while True:
        # ユーザーに質問を入力させる
        input_prompt = input("質問を入力してください（終了するには'q'を入力）: ")
        
        # 終了条件の確認
        if input_prompt.lower() == 'q':
            break
        
        # GPT-4からの回答を生成
        generated_text = generate_text(input_prompt, role, conversation_history)
        
        # 回答を表示
        print("GPT-4からの回答:", generated_text)