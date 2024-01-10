# ユーザーが入力した特徴を元にペットの名前を3つ考えて表示
import openai, os

# APIキーを環境変数から設定 --- (*1)
openai.api_key = os.environ["OPENAI_API_KEY"]

# ChatGPTのAPIを呼び出す --- (*2)
def call_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0]['message']['content']

# ユーザにペットの特徴を尋ねる --- (*3)
features = input('ペットの特徴を入力してください: ')
if features == '': quit()
# ユーザの入力を元にペットの名前を生成するプロンプトを組む --- (*4)
prompt = f"""
ペットの名前を3つ考えてください。
特徴: '''{features}'''
"""
# ペットの名前を生成して表示 --- (*5)
pet_names = call_chatgpt(prompt)
print(pet_names)
