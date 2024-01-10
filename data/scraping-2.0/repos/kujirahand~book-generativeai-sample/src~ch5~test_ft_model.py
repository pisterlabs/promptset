import openai

# ↓以下のモデル名を書き換えてください
MODEL_NAME = 'davinci:ft-kujirahand-2023-06-19-07-05-03'

# カスタムモデルを指定してプロンプトを入力 --- (*1)
def ninja_completion(prompt):
    prompt += '->'
    res = openai.Completion.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=0.7,
        max_tokens=300,
        stop='\n')
    return res['choices'][0]['text']

# プロンプトと応答を表示する --- (*2)
def test_ninja(prompt):
    text = ninja_completion(prompt)
    print(prompt, '->', text)

# 簡単な会話でテスト --- (*3)
test_ninja('おはよう')
test_ninja('もう駄目だ')
test_ninja('今日は仕事が忙しくて疲れたよ。')
