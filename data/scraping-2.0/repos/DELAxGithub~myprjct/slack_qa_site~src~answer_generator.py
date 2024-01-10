import openai

def generate_answer(question):
    # OpenAI GPTのAPIキーを設定します
    openai.api_key = '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'
    
    # GPTに質問を送信して回答を生成します
    response = openai.Completion.create(
        engine='text-davinci-003',  # 使用するGPTのエンジンを選択します
        prompt=question,
        max_tokens=100,  # 生成される回答の最大トークン数
        n=1,  # 生成される回答の数
        stop=None,  # 回答の生成を終了するトークン
        temperature=0.7,  # 生成の多様性をコントロールします（0.2から1.0の範囲で調整）
    )
    
    # 回答を取得します
    answer = response.choices[0].text.strip()
    return answer
